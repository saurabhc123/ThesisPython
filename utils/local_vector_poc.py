from sklearn.metrics import f1_score

from classifiers import eboladict
from word2vec import word2vec

experiment = "egypt"
auxiliary_file = experiment + "_auxiliary_data.txt"
validation_file = experiment + "_validation_data.txt"

def local_vector_cnn_experiment():
    baseFolderName = "/Users/saur6410/Google Drive/VT/Thesis/Source/ThesisPoC/data/python/"
    trainingFileName = baseFolderName +  "/" + "part-00000"
    validationFileName = baseFolderName + validation_file
    file_model = word2vec.word2vec.get_model_from_file("data/"+ auxiliary_file)
    #file_model = word2vec.word2vec.get_model_from_file("data/1k_results_lem.txt")
    similar_words = file_model.most_similar(positive=["egypt"])
    print(similar_words)
    from word2vec.DataHelper import DataHelper
    dataHelper = DataHelper(experiment, validation_file)
    cnn_classifier = None
    if cnn_classifier is None:
        #trainingDict = dataHelper.getTrainingData()
        from classifiers.CNNEmbedVecClassification import CNNEmbeddedVecClassifier
        cnn_classifier = CNNEmbeddedVecClassifier(file_model,3, classdict=eboladict)
        cnn_classifier.train()
    v = dataHelper.getValidationData()
    validation = map(lambda validationDataTuple: (int(validationDataTuple[0],base=10),validationDataTuple[1]), v)
    actualAndPredictedLabels = map(lambda w: {'actual' : w[0],'predicted': 1} if cnn_classifier.score(w[1])['1'] > cnn_classifier.score(w[1])['0'] else {'actual' : w[0],'predicted': 0},
                           validation)
    print(actualAndPredictedLabels)
    print(cnn_classifier.score('ebola threat real allow african conference nyc risky stupid wrong'))
    print(cnn_classifier.score('egypt serious doubt whether morsi control presidency presidency hijack'))
    print(cnn_classifier.score('fukushima npp accident nuclear disaster even one die acute radiation sickness via'))


def local_vector_experiment():
    file_model = word2vec.word2vec.get_model_from_file("data/"+ "1k_results_lem.txt")
    #print(file_model.most_similar(positive=["obesity"]))
    print(file_model.similarity('obesity', 'winterstorm'))
    print(file_model.similarity('flag', 'weight'))
    print(file_model.similarity('japan', 'fukushima'))
    print(file_model.most_similar(positive=["japan"]))
    print(file_model.most_similar(positive=["fukushima"]))
    print(file_model.most_similar(positive=["disease"]))
    print(file_model.most_similar(positive=["obesity"]))


if __name__ == "__main__":
    local_vector_experiment()
    #local_vector_cnn_experiment()