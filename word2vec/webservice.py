import csv
import json

import decimal
from flask import Flask, jsonify
from flask import request
from numpy import genfromtxt
from py2app.recipes import numpy
from sklearn.metrics import f1_score

from classifiers import SumWord2VecClassification
from classifiers.CNNEmbedVecClassification import CNNEmbeddedVecClassifier
from classifiers.eboladict import eboladict
from word2vec import word2vec, avg_feature_vector

app = Flask(__name__)
model = word2vec.get_model()
file_model = word2vec.get_model_from_file("data/ebola_auxiliary_data_clean.txt")
sum_classifier = None
cnn_classifier = None


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

@app.route("/wvsum_score/<word>", methods=['GET'])
def wvsum_score(word):
    global sum_classifier
    if sum_classifier is None:
        sum_classifier = SumWord2VecClassification.SumEmbeddedVecClassifier(model, classdict=eboladict)
        sum_classifier.train()
    score = sum_classifier.score(word)
    return jsonify({"scores": score})

@app.route("/cnn_score/<word>", methods=['GET'])
def cnn_score(word):
    global cnn_classifier
    if cnn_classifier is None:
        cnn_classifier = CNNEmbeddedVecClassifier(model,3, classdict=eboladict)
        cnn_classifier.train()
    score = {k:float(v) for k,v in cnn_classifier.score(word).iteritems()}
    #s = get_most_distinguishing_words()
    #f = cnn_train_and_predict("")
    return jsonify(score)


class DataHelper():
    def __init__(self, trainingFileName,validationFileName):
        self.trainingFileName = trainingFileName
        self.validationFileName = validationFileName

    def getValidationData(self):
        reader = csv.reader(open(self.validationFileName, "rb"), delimiter=",")
        validation_data = list(reader)
        return validation_data

    def getTrainingData(self):
        reader = csv.reader(open(self.trainingFileName, "rb"), delimiter=",")
        training_data = list(reader)
        return training_data



@app.route("/cnn_train_and_predict/<filename>", methods=['GET'])
def cnn_train_and_predict(trainingFilename):
    trainingFilename = "data/ebola_validation_data.txt"
    dataHelper = DataHelper(trainingFilename, trainingFilename)
    global cnn_classifier
    trainingDict = dataHelper.getTrainingData()
    if cnn_classifier is None:
        cnn_classifier = CNNEmbeddedVecClassifier(model,3, classdict=eboladict)
        cnn_classifier.train()
    v = dataHelper.getValidationData()
    validation = map(lambda validationDataTuple: validationDataTuple[1], v)
    validationLabels = map(lambda validationDataTuple: int(validationDataTuple[0]), v)
    predictionLabels = map(lambda w: 1 if cnn_classifier.score(w)["ebola"] > cnn_classifier.score(w)["other"] else 0,
                           validation)
    f1 = f1_score(validationLabels, predictionLabels, average='binary')
    return jsonify(f1)



def get_most_distinguishing_words():
    words = []
    for classtype in eboladict:
        for classSamples in eboladict[classtype]:
            words.extend(classSamples.split(" "))
    scores = {(word, cnn_classifier.score(word)["ebola"]) for word in words}
    return sorted(scores, key=lambda tup: -tup[1])



app.run()
