from flask import Flask, jsonify
from flask import request

from word2vec import word2vec, avg_feature_vector

app = Flask(__name__)
model = word2vec.get_model()
file_model = word2vec.get_model_from_file("data/egypt_auxiliary_data_clean.txt")


@app.route("/getvector/<sentence>")
def get_vector(sentence):
    words = sentence.split()
    result = avg_feature_vector(words, model, 300, model.index2word).tolist()
    return jsonify({"sentence": sentence, "vector": result})


@app.route("/distance", methods=['GET'])
def get_similarity():
    document1 = request.args.get('document1').split()
    document2 = request.args.get('document2').split()
    return jsonify({"wmdDistance": model.wmdistance(document1, document2)})


@app.route("/file_model/getvector/<sentence>")
def get_vector_file(sentence):
    words = sentence.split()
    result = avg_feature_vector(words, file_model, 300, file_model.index2word).tolist()
    return jsonify({"sentence": sentence, "vector": result})


@app.route("/file_model/distance", methods=['GET'])
def get_similarity_file():
    document1 = request.args.get('document1').split()
    document2 = request.args.get('document2').split()
    return jsonify({"wmdDistance": file_model.wmdistance(document1, document2)})


@app.route("/synonyms/<word>", methods=['GET'])
def get_synonyms(word):
    similar_words = model.most_similar(positive=[word])
    return jsonify({"synonyms": similar_words})


@app.route("/file_model/synonyms/<word>", methods=['GET'])
def get_synonyms_file(word):
    similar_words = file_model.most_similar(positive=[word])
    return jsonify({"synonyms": similar_words})

app.run()
