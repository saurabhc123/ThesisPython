from flask import Flask, jsonify

from word2vec import word2vec, avg_feature_vector

app = Flask(__name__)
model = word2vec.get_model()


@app.route("/getvector/<sentence>")
def get_vector(sentence):
    words = sentence.split()
    result = avg_feature_vector(words, model, 300, model.index2word).tolist()
    return jsonify({"sentence": sentence, "vector": result})

app.run()
