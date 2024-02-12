from difflib import get_close_matches
import re
from googletrans import Translator


class TunisianTranslator:
    __translation_dict = {
        "tawla": "table",
        "korsi": "chaise",
        "telifoun": "telephone",
        "mtaa": "de",
        "kartabla": "cartable",
        "9raya": "scolarite",
        "9aleya": "poêle",
        "mekina": "machine",
        "9ahwa": "cafe",
        "5obz": "pain",
        "mizen": "balance",
        "koujina": "cuisine",
        "s8ar": "enfant",
        "8ta": "couverture",
        "ghta": "couverture",
        "stilou": "stylo",
        "saboura": "tableau",
        "tey": "the",
        "souria": "chemise",
        "azra9": "bleu",
        "zar9a": "bleu",
        "a7mer": "rouge",
        "7amra": "rouge",
        "ak7al": "noir",
        "ka7la": "noir",
        "a5dhar": "vert",
        "5adhra": "vert",
        "lamba": "lampe",
        "abyedh": "blanche",
        "bidha": "blanche",
        "serwel": "pantalon",
        "shaar": "cheveux",
        "sh3ar": "cheveux",
        "7did": "metal",
        "bdan": "corps",
        "hommes": "rjel",
        "femmes": "nse",
        "mtarcha9": "abîmés",
        "mahlouk": "abîmés",
        "3ilej": "traitement",
        "trikou": "t-shirt",
    }

    def translate(self, tunisian_text):
        words = []

        # Check if the text is non-Latin and attempt translation if necessary
        if not self.__is_latin(tunisian_text):
            try:
                translator = Translator()
                tunisian_text = translator.translate(tunisian_text, dest="fr").text
            except Exception as e:
                # Handle translation failure, possibly log the error or use a fallback
                print(f"Translation failed: {e}")
                return "Translation error"

        words = tunisian_text.split()

        translated_words = []

        for word in words:
            key = self.__find_best_match(word)
            if key:
                translated_words.append(self.__translation_dict[key])
            else:
                translated_words.append(word)

        return " ".join(translated_words)

    def __is_latin(self, s):
        # This regex matches any character that is not in the basic Latin or Latin-1 Supplement range.
        # You might need to adjust the regex to include other Latin ranges depending on your needs.
        return not re.search(r"[^\u0000-\u00FF]", s)

    def __find_best_match(self, tunisian_text):
        matches = get_close_matches(
            tunisian_text, self.__translation_dict.keys(), n=1, cutoff=0.7
        )
        return matches[0] if matches else None


translator = TunisianTranslator()
