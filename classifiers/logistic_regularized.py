"""
Very simple module for doing logistic regression.

Based on:
- http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html
- http://people.csail.mit.edu/jrennie/writing/lr.pdf
"""

from enum import Enum
from scipy.optimize.optimize import fmin_bfgs, fmin_cg
from scipy.optimize import minimize
from sklearn import linear_model
from sklearn.metrics import f1_score
from similarities import similarity_calculator



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



class Classifier(Enum):
    LR = 1
    LR_TRANSFER = 2

class Model(object):
    """ A simple logistic regression model with L2 regularization (zero-mean
    Gaussian priors on parameters). """

    def __init__(self, train_data, test_data, d, w=0):
        """ Create model for input data consisting of d dimensions. """

        # Initialize parameters to zero, for lack of a better choice.
        if w is not 0:
            self.betas = w
        else:
            self.betas = np.zeros(d)
        self.x_train = np.array(train_data[:,1:])
        self.x_test = np.array(test_data[:,1:])
        self.y_train = np.array(train_data[:,0])
        self.y_test = np.array(test_data[:,0])

        self.n = self.y_train.shape[0]
        self.d = self.x_train.shape[1]



    def likelihood(self, betas, alpha=0):
        """ Likelihood of the data under the given settings of parameters. """

        # Data likelihood
        l = 0
        for i in range(self.n):
             l += log(sigmoid(self.y_train[i] * \
                              np.dot(betas, self.x_train[i,:])))

        # Prior likelihood
        # More like regularization
        for k in range(1, self.x_train.shape[1]):
             l -= (alpha / 2.0) * betas[k]**2

        return l

    def likelihood_alt(self, similarityMatrix, betas, alpha=0):
        """ Likelihood of the data under the given settings of parameters. """

        # Data likelihood
        l = 0
        m = 1
        # import pdb; pdb.set_trace();
        #similarityMatrix = [[1 for i in range(self.x_train.shape[0])] for j in range(m)]
        for i in range(self.n):
            l += log(sigmoid(self.y_train[i] * \
                             np.dot(betas, self.x_train[i,:])))

        # ToDo: Add the training and auxiliary data, without the labels.
        x_total = np.concatenate((self.x_train, self.x_test), axis=0)

        # Prior likelihood
        # More like regularization
        # import pdb;pdb.set_trace();
        reg = 0


        for i in range(self.x_train.shape[0]):
            for j in range(x_total.shape[0]):
                # TODO: change 1 to I_ij
                reg -= similarityMatrix[i,j] * \
                (np.dot(betas, x_total[i]) - \
                 np.dot(betas, x_total[j]))**2

        return l + (alpha / 2.0) * reg

    def train(self, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda B, k: (k > 0) * self.alpha * B[k]  - np.sum([ \
                                    self.y_train[i] * self.x_train[i, k] * \
                                    sigmoid(-self.y_train[i] *\
                                            np.dot(B, self.x_train[i,:])) \
                                    for i in range(self.n)])



        # The full gradient is just an array of componentwise derivatives
        gradient = lambda B: np.array([dl_by_dWk(B, k) \
                                 for k in range(self.x_train.shape[1])])

        # The function to be minimized
        # Use the negative log likelihood for the objective function.
        objectiveFunction = lambda B: -self.likelihood(betas=B, alpha=self.alpha)

        # Optimize
        print('Optimizing for alpha = {}'.format(alpha))
        self.betas = fmin_bfgs(objectiveFunction, self.betas, fprime=gradient)
        #self.betas = fmin_cg(objectiveFunction, self.betas, fprime=gradient, maxiter=10)

    def train_alt(self, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        x_total = np.concatenate((self.x_train, self.x_test), axis=0)
        #similarityMatrix = np.ones((self.x_train.shape[0],x_total.shape[0]))
        similarityMatrix = similarity_calculator.get_similarities_alt(x_total, self.x_train)
        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda W, k: (k > 0) * self.sfRegStep(W, k, similarityMatrix, alpha, x_total)



        # The full gradient is just an array of componentwise derivatives
        gradient = lambda W: np.array([dl_by_dWk(W, k) \
                                       for k in range(self.x_train.shape[1])]).transpose()

        # The function to be minimized
        # Use the negative log likelihood for the objective function.
        objectiveFunction = lambda W: -self.likelihood_alt(similarityMatrix, betas=W, alpha=self.alpha)

        # Optimize
        print('Optimizing for alpha = {}'.format(alpha))
        #self.betas = fmin_bfgs(objectiveFunction, self.betas, fprime=gradient)
        self.betas = fmin_cg(objectiveFunction, self.betas, fprime=gradient, maxiter=10)

    def sfRegStep(self, W, k, similarityMatrix, alpha, x_total):
        # for j in range(self.x_train.shape[0] + self.x_test.shape[0]):
        #     value = self.x_train[j,k]* np.dot(W,data.x_train[j,:]) + self.x_test[k,:]*W*data.x_test[:,:] -\
        #             self.x_test[k,:]*W*data.x_train[:,:] - self.x_train[k,:]*W*data.x_test[:,:]


        n = self.x_train.shape[0]
        m_n = x_total.shape[0]
        dwk = 0
        #print('sfRegStep - First Summation for k={}'.format(k))
        for i in range(n):
            #print('i={}, k={}'.format(i,k))
            dwk +=  -1 * self.y_train[i]*self.x_train[i,k]*sigmoid(-self.y_train[i] * np.dot(W, self.x_train[i,:]))

        #print('sfRegStep - Regularization for k={}'.format(k))
        dL = 0
        # import pdb; pdb.set_trace();
        for i in range(n):
            for j in range(m_n):
                # TODO: replace 1 by I_ij
                dL += alpha* similarityMatrix[i,j] *x_total[i,k]*np.dot(W, x_total[i,:])
                + x_total[j, k]*np.dot(W, x_total[j,:])
                - x_total[j,k]*np.dot(W, x_total[i,:]) - x_total[i,k]*np.dot(W, x_total[j,:])

        return dL+dwk

    def predict(self, x):
        return sigmoid(np.dot(self.betas, x))

    def training_reconstruction(self, data):
        p_y1 = np.zeros(data.n)
        for i in range(data.n):
            p_y1[i] = self.predict(data.x_train[i,:])
        return p_y1

    def test_predictions(self, data):
        p_y1 = np.zeros(data.n)
        for i in range(data.n):
            p_y1[i] = self.predict(data.x_test[i,:])
        return p_y1

    def plot_training_reconstruction(self, data):
        plot(np.arange(data.n), .5 + .5 * data.y_train, 'bo')
        plot(np.arange(data.n), self.training_reconstruction(data), 'rx')
        ylim([-.1, 1.1])

    def plot_test_predictions(self, data):
        plot(np.arange(data.n), .5 + .5 * data.y_test, 'yo')
        plot(np.arange(data.n), self.test_predictions(data), 'rx')
        ylim([-.1, 1.1])


def run_experiment(train, test, validation, w=0, classifier=Classifier.LR):
    lr = Model(train, test, train[:, 1:].shape[1],w)
    # Run for a variety of regularization strengths
    #alphas = [0.01, .11, 1.1, 11.1]
    alphas = [0, .001, .01, .1]
    for j, a in enumerate(alphas):
        print "Initial likelihood:"
        print lr.betas

        # Train the model
        if classifier is Classifier.LR:
            lr.train(alpha=a)
        else:
            lr.train_alt(alpha=a)

        # Display execution info
        print "Final betas:"
        #print lr.betas
        print "Final likelihood:"
        #print lr.betas
        predictions = lr.predict(validation[:, 1:].transpose())
        predictionLabels = map(lambda prediction: 1 if prediction > 0.5 else 0, predictions)
        print "Final Predictions:"
        print predictionLabels
        print f1_score(validation[:, 0], predictionLabels, average='binary')
    predictions = lr.predict(validation[:, 1:].transpose())
    predictionLabels = map(lambda prediction: 1 if prediction > 0.5 else 0, predictions)
    print "Final Predictions:"
    print predictionLabels
    print f1_score(validation[:, 0], predictionLabels, average='binary')
    return lr.betas


if __name__ == "__main__":
    from pylab import *

    source_training_file = 'data/sandy_irene.txt'
    source_auxiliary_file = 'data/sandy_irene_auxiliary.txt'
    source_validation_file = 'data/sandy_irene_validation.txt'

    target_training_file = 'data/target.txt'
    target_auxiliary_file = 'data/target_auxiliary.txt'
    target_validation_file = 'data/target_validation.txt'

    source_training_data = genfromtxt(source_training_file, delimiter=',')
    source_auxiliary_data = genfromtxt(source_auxiliary_file, delimiter=',')
    source_validation_data = genfromtxt(source_validation_file, delimiter=',')
    #np.random.shuffle(source_training_data)

     # Define training, auxiliary and validation filters
    source_train = source_training_data[:, :]
    source_auxiliary = source_auxiliary_data




    target_training_data = genfromtxt(target_training_file, delimiter=',')
    target_auxiliary_data = genfromtxt(target_auxiliary_file, delimiter=',')
    target_validation_data = genfromtxt(target_validation_file, delimiter=',')
    # np.random.shuffle(target_training_data)

    # Define training, auxiliary and validation filters
    target_train = target_training_data[30:50, :]
    target_auxiliary = target_auxiliary_data[14:21,:]

    #lr1 = linear_model.LogisticRegression(C=1e5)
    #lr1.fit(source_training_data[:,1:], source_training_data[:,0])
    #w = lr1.coef_
    #w = run_experiment(source_train[:,:], source_auxiliary, source_validation_data)
    w = run_experiment(source_train[43:44,:], source_auxiliary, source_validation_data,classifier=Classifier.LR)
    run_experiment(source_train[40:41, :], source_auxiliary[:,:], source_validation_data, w, classifier=Classifier.LR_TRANSFER)
