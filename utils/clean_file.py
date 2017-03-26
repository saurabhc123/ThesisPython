import string

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def full_pipeline(lem, word):
    word = word.lower()
    word = word.translate(string.punctuation)
    for val in ['a', 'v', 'n']:
        word = lem.lemmatize(word, pos=val)
    return word


def clean_sent(lem, sent):
    sent = unicode(sent,errors='ignore')
    words = sent.replace(","," ").replace(";", " ").replace("#"," ").replace(":", " ").replace("@", " ").split()
    filtered_words = filter(lambda word: word.isalpha() and len(word) > 1 and word != "http", [full_pipeline(lem, word) for word in words])
    return ' '.join(filtered_words)


def clean_file(lem, name):
    with open(name) as f:
        new_sents = filter(lambda x: len(x[1].split()) > 0, [(line.split(',')[0] ,clean_sent(lem, line.split(',')[1].strip())) for line in f.readlines()])
    for lab,sent in new_sents:
        print(str(lab) + "," + str(sent))


print(clean_sent(wordnet_lemmatizer,
                 'Ozawa Ichiro(3):\x0A such as Japanese Bureaucrat and Massmedia.\x0A\x0ASee(Japanese);\x0A1)http://t.co/Goiij1CxE2\x0A2)http://t.co/cKfRcI8Tkb\x0A\x0A#Fukushima'))

clean_file(wordnet_lemmatizer,"data/ebola_auxiliary_data.txt")
