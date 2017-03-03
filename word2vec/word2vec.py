import gensim



class word2vec:
    model = None

    @staticmethod
    def get_model():
        if not word2vec.model:
            model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            model.init_sims(replace=True)  # we are no longer training the model so allow it to trim memory
            word2vec.model = model

        return word2vec.model


if __name__ == "__main__":
    model = word2vec.get_model()
    # you can find the terms that are similar to a list of words and different from
    # another list of words like so
    print(model.most_similar(positive=['hurricane'], negative=['isaac']))

    # you can also get the vector for a specific word by doing
    print(model['hurricane'])

    # you can ask for similarity by doing
    print(model.similarity('hurricane', 'shooting'))
