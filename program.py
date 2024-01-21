from flask import Flask, request, jsonify
from flask_cors import cross_origin
from model import model
from translator import translator

app = Flask(__name__)


@app.route("/encode", methods=["POST"])
@cross_origin()
def encode():
    data = request.json
    sentence = data.get("sentence")
    translated_sentence = translator.translate(sentence)
    encoded_sentence = model.encode_sentence_and_normalise(translated_sentence)
    return jsonify({"query_vector": encoded_sentence})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
