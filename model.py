from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.types import Number


class AllMpnetBaseV2:
    def __init__(self) -> None:
        self.sentences = ["This is a sentence", "This is another sentence"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

    # Effectuez une similarité cosinus entre 2 phrases et la retourner
    def perform_cosine_similarity_between_2_sentences(
        self, sentences: list[str] | None = None
    ) -> Number:
        if sentences is None:
            sentences = self.sentences

        if sentences is list[str] and len(sentences) > 2:
            raise Exception("Must provide 2 sentences")

        sentence_embeddings = self.__encode_sentences_and_normalise(sentences)

        # Convertir toutes les intégrations de phrases dans la même dimension à l'aide de .unsqueeze(0)
        same_dim_sentence_embeddings = [
            sentence.unsqueeze(0) for sentence in sentence_embeddings
        ]

        # Performer la similarité cosinus
        similarity = F.cosine_similarity(
            same_dim_sentence_embeddings[0], same_dim_sentence_embeddings[1]
        )

        return similarity.item()

    """
        Effectue une similarité cosinus entre des paires de phrases dans un tableau de phrases, à utiliser si vous souhaitez comparer plus de 2 phrases.
        Renvoie le tuple présentant la plus grande similarité et la meilleure paire de phrases qui se ressemblent

        (max_similarity, best_pair)
    """

    def perform_cosine_similarity_and_return_highest(
        self, sentences: list[str]
    ) -> tuple[Number, tuple[str | None, str | None]]:
        sentence_embeddings = self.__encode_sentences_and_normalise(sentences)

        # Initialisez les variables pour suivre la similarité la plus élevée et les phrases correspondantes
        max_similarity = -1  # Start with the lowest possible similarity
        best_pair: tuple[str | None, str | None] = (None, None)

        # Calculer la similarité par paire
        num_sentences = len(sentences)
        for i in range(num_sentences):
            for j in range(
                i + 1, num_sentences
            ):  # Comparez chaque paire une seule fois
                # Convertir les deux tenseurs d'intégration dans la même dimension et effectuez une similarité cosinus
                similarity = F.cosine_similarity(
                    sentence_embeddings[i].unsqueeze(0),
                    sentence_embeddings[j].unsqueeze(0),
                ).item()

                # Comparer les similarités et attribuer si elles sont supérieures
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair = (sentences[i], sentences[j])

        return (max_similarity, best_pair)

    def encode_sentence_and_normalise(self, sentence: str) -> list:
        """
        La tokenisation est le processus de conversion d'une séquence de caractères en une séquence de jetons(tokens). En PNL, un jeton(token) représente généralement un mot, mais il peut également représenter des sous-mots, des caractères ou d'autres unités, selon le tokeniseur. Ce processus est nécessaire car les modèles d'apprentissage automatique ne peuvent pas comprendre le texte brut. Au lieu de cela, ils travaillent avec des représentations numériques de jetons.

        Le remplissage(padding) est appliqué pour garantir que toutes les séquences d'un lot ont la même longueur, ce qui est une exigence pour de nombreuses architectures de réseaux neuronaux. Des jetons de remplissage sont ajoutés à la fin de la séquence pour étendre sa longueur jusqu'à un maximum prédéfini.
        """
        encoded_input = self.tokenizer(
            sentence, padding=True, truncation=True, return_tensors="pt"
        )

        # Calculer les intégrations de jetons
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Performer le "Mean Pooling"
        sentence_embedding = self.__mean_pooling(
            model_output, attention_mask=encoded_input["attention_mask"]
        )

        # Noramliser les intégrations
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        # Presser(.squeeze()) le tenseur dans un tenseur unidimensionnel, puis le retourner sous forme de liste
        return sentence_embedding.squeeze().tolist()

    """
        Mean Pooling (Regroupement de moyennes) - Tenir compte du masque d'attention (attention_mask) pour une moyenne correcte:
        fonction qui calcule efficacement la moyenne des intégrations de jetons dans chaque phrase, tout en ignorant les jetons de remplissage, ce qui donne un seul vecteur d'intégration qui représente la phrase entière. Il s'agit d'une technique courante utilisée dans les tâches de PNL pour obtenir une représentation de phrase de taille fixe à partir de phrases de longueur variable.

        S'applique au regroupement de moyennes à l'output du modèle, créant une intégration de phrase de taille fixe.

        L'opération de regroupement moyen ne prend en compte que les jetons sans remplissage dans la phrase, en utilisant un masque d'attention. Le masque d'attention a la même longueur que la séquence de jetons, « 1 » indiquant les jetons sans remplissage et « 0 » pour les jetons de remplissage. Cela garantit que l'incorporation de phrase résultante est calculée uniquement à partir du contenu significatif de la séquence.

        Args:
            model_output (torch.Tensor): L'output du modèle, contenant des intégrations de jetons.
            attention_mask: Un masque qui fait la différence entre le contenu et les jetons de remplissage.

        Returns:
            torch.Tensor: L'intégration de phrases regroupées en moyenne représentant la totalité de la séquence d'entrée.
    """

    def __mean_pooling(
        self, model_output: torch.Tensor, attention_mask
    ) -> torch.Tensor:
        # Le premier élément de l'output' du modèle contient toutes les intégrations de jetons
        token_embeddings = model_output[0]
        """
            attention_mask.unsqueeze(-1): 
                une dimension supplémentaire au masque d'attention,
                le transformer à partir d'un tenseur de forme 2D [batch_size, séquence_length]
                à un tenseur 3D de forme [batch_size, sequence_length, 1].

            .expand(token_embeddings.size()): 
                Cela élargit le masque d'attention pour qu'il corresponde à la taille de token_embeddings (intégrations de jetons).
                Le masque a désormais la même forme que les intégrations de jetons [batch_size, séquence_length, embedding_size].

            .float(): 
                Convertit le masque en tenseur des réels(floats). Ceci est nécessaire car le masque est généralement de type int (0 et 1), mais doit être au format à virgule flottante pour une multiplication ultérieure avec les intégrations.
            """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        """
        token_embeddings * input_mask_expanded: 
            Cela multiplie les intégrations par le masque.
            Pour les jetons qui remplissent (valeur de masque 0),
            leurs plongements deviennent nuls et ne contribuent pas à la somme.
            
        torch.sum(..., 1): 
            Additionne les plongements sur la dimension de longueur de séquence,
            ce qui donne un seul vecteur pour chaque phrase du lot(batch).

        torch.clamp(..., min=1e-9):
            Garantit que le diviseur n'est pas nul (ce qui peut arriver si une phrase est entièrement composée de jetons de remplissage). Il fixe une valeur minimale pour éviter la division par zéro.

        input_mask_expanded.sum(1): 
            Fait la somme du masque sur toute la longueur de la séquence, donnant le nombre de jetons réels (sans remplissage) dans chaque phrase.
        """
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    """
        Prend en paramétre une liste de phrases, les code dans un tenseur(vecteur) de 768 dimensions,
        effectue une regroupement de moyennes sur eux, puis les renvoie normalisés.
    """
    def __encode_sentences_and_normalise(
        self, sentences: list[str] | None
    ) -> torch.Tensor:
        # Tokeniser les phrases
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")  # type: ignore

        # Calculer les intégrations de jetons
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Performer la regroupement de moyennes
        sentence_embeddings = self.__mean_pooling(
            model_output=model_output, attention_mask=encoded_input["attention_mask"]
        )

        # Normaliser les intégrations
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    
    def __max_pooling(self, model_output: torch.Tensor, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Set embeddings of padding tokens to a very small value
        masked_embeddings = token_embeddings * input_mask_expanded
        masked_embeddings[masked_embeddings == 0] = -1e9

        # Compute the max across the sequence length dimension
        max_pooled = torch.max(masked_embeddings, dim=1)[0]
        return max_pooled

    def __min_pooling(self, model_output: torch.Tensor, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Set embeddings of padding tokens to a very large value
        masked_embeddings = token_embeddings * input_mask_expanded
        masked_embeddings[masked_embeddings == 0] = 1e9

        # Compute the min across the sequence length dimension
        min_pooled = torch.min(masked_embeddings, dim=1)[0]
        return min_pooled


model = AllMpnetBaseV2()
