from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.types import Number


class AllMpnetBaseV2:
    def __init__(self):
        self.sentences = ["This is a sentence", "This is another sentence"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

    # Perform cosine similarity between 2 sentences, and return it
    def perform_cosine_similarity_between_2_sentences(self, sentences):
        if sentences is None:
            sentences = self.sentences

        if sentences is list[str] and len(sentences) > 2:
            raise Exception("Must provide 2 sentences")

        sentence_embeddings = self.__encode_sentences_and_normalise(sentences)

        # Convert all sentence embeddings to the same dimension using .unsqueeze(0)
        same_dim_sentence_embeddings = [
            sentence.unsqueeze(0) for sentence in sentence_embeddings
        ]

        # Perofrm cosine similarity
        similarity = F.cosine_similarity(
            same_dim_sentence_embeddings[0], same_dim_sentence_embeddings[1]
        )

        return similarity.item()

    """
        Performs cosine similarity between pairs of sentences in an array of sentences, use if you have more than 2 sentences you want to compare.
        Returns a tuple of the highest similarity, and the best pair of sentences that resemble each other

        (max_similarity, best_pair)
    """

    def perform_cosine_similarity_and_return_highest(self, sentences):
        sentence_embeddings = self.__encode_sentences_and_normalise(sentences)

        # Initialize variables to track the highest similarity and corresponding sentences
        max_similarity = -1  # Start with the lowest possible similarity
        best_pair = (None, None)

        # Calculate pair wise similarity
        num_sentences = len(sentences)
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):  # Compare each pair only once
                # Convert both of the embeddings tensors to the same dimension and perform cosine similarity
                similarity = F.cosine_similarity(
                    sentence_embeddings[i].unsqueeze(0),
                    sentence_embeddings[j].unsqueeze(0),
                ).item()

                # Compare similarities and reassign if higher
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair = (sentences[i], sentences[j])

        return (max_similarity, best_pair)

    def encode_sentence_and_normalise(self, sentence):
        # Tokenize the sentence
        encoded_input = self.tokenizer(
            sentence, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute the token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform Pooling
        sentence_embedding = self.__mean_pooling(
            model_output, attention_mask=encoded_input["attention_mask"]
        )

        # Normalise embeddings
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        # Squeeze the tensor into a one dimensional tensor, then return it as a list
        return sentence_embedding.squeeze().tolist()

    """
        Mean Pooling - Take attention mask into account for correct averaging:
        function that effectively computes the average of the token embeddings in each sentence, while ignoring padding tokens, resulting in a single embedding vector that represents the entire sentence. This is a common technique used in NLP tasks to get a fixed-size sentence representation from variable-length sentences.
    """

    def __mean_pooling(self, model_output, attention_mask):
        # First element of the model output contains all token embeddings
        token_embeddings = model_output[0]
        """
            attention_mask.unsqueeze(-1): 
                This adds an extra dimension to the attention mask, 
                transforming it from a 2D tensor of shape [batch_size, sequence_length] 
                to a 3D tensor of shape [batch_size, sequence_length, 1].

            .expand(token_embeddings.size()): 
                This expands the attention mask to match the size of token_embeddings. 
                The mask is now of the same shape as the token embeddings [batch_size, sequence_length, embedding_size].

            .float(): 
                Converts the mask to a float tensor. This is necessary because the mask is usually of type int (0s and 1s), but needs to be in floating-point format for subsequent multiplication with the embeddings.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        """
        token_embeddings * input_mask_expanded: 
            This multiplies the embeddings by the mask. 
            For tokens that are padding (mask value 0), 
            their embeddings become zero and don't contribute to the sum.
            
        torch.sum(..., 1): 
            Sums the embeddings across the sequence length dimension, 
            resulting in a single vector for each sentence in the batch.

        torch.clamp(..., min=1e-9):
            Ensures that the divisor is not zero (which can happen if a sentence consists entirely of padding tokens). It sets a minimum value to avoid division by zero.

        input_mask_expanded.sum(1): 
            Sums the mask across the sequence length, giving the number of actual (non-padding) tokens in each sentence.
        """
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    """
        Takes an input of a list of sentences, encodes them into a tensor(vector) of 768 dimensions, 
        performs mean pooling on them, then returns them normalised.
    """

    def __encode_sentences_and_normalise(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")  # type: ignore

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform Pooling
        sentence_embeddings = self.__mean_pooling(
            model_output=model_output, attention_mask=encoded_input["attention_mask"]
        )

        # Normalise the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


model = AllMpnetBaseV2()
