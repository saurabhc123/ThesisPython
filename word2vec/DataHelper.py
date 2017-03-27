import collections
import csv


class DataHelper():
    def __init__(self, trainingFolder,validationFileName):
        self.baseFolderName = "/Users/saur6410/Google Drive/VT/Thesis/Source/ThesisPoC/data/python/"
        self.trainingFileName = self.baseFolderName + trainingFolder + "/" + "part-00000"
        self.validationFileName = self.baseFolderName + validationFileName

    def getValidationData(self):
        reader = csv.reader(open(self.validationFileName, "rb"), delimiter=",")
        validation_data = list(reader)
        return validation_data

    def getTrainingData(self):
        reader = csv.reader(open(self.trainingFileName, "rb"), delimiter=",")
        training_data = list(reader)
        trainingDict = collections.defaultdict(list)
        for line in training_data:
            label = line[0]
            text = line[1]
            trainingDict[label].append(text)
        return trainingDict