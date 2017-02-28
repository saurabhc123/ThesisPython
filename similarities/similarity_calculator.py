from gensim import corpora, models, similarities
import numpy as np
from scipy import spatial


model = None
def init_sim_model():
    global model
    model = models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)


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

def get_cosine(v1,v2):
    x = spatial.distance.cosine(v1,v2)
    if np.isnan(x):
        x = 0
    return x

def get_similarities(actual_tweets, auxiliary_tweets):
    '''Gives the similarities between actual tweets and the auxilary tweets
        this will return a 2D numpy array given the similarities
        The tweets are expected by be arrays of strings.
    '''
    if not model:
        init_sim_model()
    aux_tw = map(lambda x: x.split(' '), auxiliary_tweets)
    train_tw = map(lambda x: x.split(' '), actual_tweets)
    all_tweets = train_tw + aux_tw
    actual_vectors = [avg_feature_vector(t,model,300,model.index2word).tolist() for t in train_tw]
    aux_vectors = [avg_feature_vector(t,model,300,model.index2word).tolist() for t in aux_tw]
    to_return = [[get_cosine(act,aux) for act in actual_vectors] for aux in aux_vectors]
    to_return_matrix = np.matrix(to_return)
    return to_return_matrix


if __name__ == "__main__":
    actual_tweets = ["United States president Donald Trump eats chicken pizza"
        , "Eric eats chicken pizza"
        , "Hurricane sandy devastates United States",
                     "asdfasdf"]
    aux_tweets = ["pizza is good says Donald Trump, in other news Italian stock market crashes",
                  "sandy loves chicken pizza", "asdfasdf", "Hurricane isaac devastates United States"]
    print(get_similarities(actual_tweets, aux_tweets))
