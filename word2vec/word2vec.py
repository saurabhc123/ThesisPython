import gensim
import numpy as np

def avg_feature_vector(words, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # list containing names of words in the vocabulary
    # index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def read_from_file(name,model):
    with open(name,"r") as f:
        lines = f.readlines()
        tweets_only = map(lambda line: line.split(';')[1].rstrip().split(" "), lines)
        lables_only = map(lambda line: line.split(';')[0].rstrip(), lines)
        vecs = map(lambda t: avg_feature_vector(t, model, 300, model.index2word),tweets_only)
        return zip(lables_only,vecs)

def write_to_file(lv,file):
    with open(file,"w") as f:
        for l,v in lv:
            f.write(str(l) + "," + ','.join((str(x)for x in np.nditer(v))) + "\n")

class word2vec:
    model = None
    models = {}
    @staticmethod
    def get_model():
        if not word2vec.model:
            model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            model.init_sims(replace=True)  # we are no longer training the model so allow it to trim memory
            word2vec.model = model

        return word2vec.model

    @staticmethod
    def get_model_from_file(name):
        if not name in word2vec.models:
            sentences = []
            with open(name,"r") as f:
                sentences = map(lambda x: x.split(), f.readlines())
            file_model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1)
            word2vec.models[name] = file_model.wv
        return word2vec.models[name]



if __name__ == "__main__":
    model = word2vec.get_model()
    # you can find the terms that are similar to a list of words and different from
    # another list of words like so
    print(model.most_similar(positive=['hurricane'], negative=['isaac']))

    # you can also get the vector for a specific word by doing
    print(model['hurricane'])

    # you can ask for similarity by doing
    print(model.similarity('hurricane', 'shooting'))

    write_to_file(read_from_file("../data/tweets/conn.lem",model),"../data/tweets/conn.vec")
    write_to_file(read_from_file("../data/tweets/fire.lem", model), "../data/tweets/fire.vec")
    write_to_file(read_from_file("../data/tweets/sandy.lem", model), "../data/tweets/sandy.vec")
    write_to_file(read_from_file("../data/tweets/texas.lem", model), "../data/tweets/texas.vec")

    print("done")
