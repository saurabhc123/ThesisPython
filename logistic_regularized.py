"""
Very simple module for doing logistic regression.

Based on:
- http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html
- http://people.csail.mit.edu/jrennie/writing/lr.pdf
"""

from scipy.optimize.optimize import fmin_bfgs
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Data(object):
    """ Abstract base class for data objects. """

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

    def likelihood_alt(self, betas, alpha=0):
        """ Likelihood of the data under the given settings of parameters. """

        # Data likelihood
        l = 0
        m = 1
        similarityMatrix = [[1 for i in range(self.x_train.shape[1])] for j in range(m)]
        for i in range(self.n):
            l += log(sigmoid(self.y_train[i] * \
                             np.dot(betas, self.x_train[i,:])))

        # ToDo: Add the training and auxiliary data, without the labels.
        x_total = self.x_train + self.x_test

        # Prior likelihood
        # More like regularization
        for i in range(1, self.x_train.shape[1]):
            regularizedValue = 0
            for j in range(1, x_total.shape[1]):
                regularizedValue += similarityMatrix[i,j] ** \
                                    (np.dot(betas, self.x_train[i,:]) - \
                                     np.dot(betas, x_total[j,:]))

            l -= (alpha / 2.0) * regularizedValue

        return l



class TsvData(Data):

    def __init__(self, train_data, test_data):


        self.x_train = np.array(train_data[:,1:])
        self.x_test = np.array(test_data[:,1:])
        self.y_train = np.array(train_data[:,0])
        self.y_test = np.array(test_data[:,0])

        self.n = self.y_train.shape[0]
        self.d = self.x_train.shape[1]
        

class Model(object):
    """ A simple logistic regression model with L2 regularization (zero-mean
    Gaussian priors on parameters). """

    def __init__(self, d):
        """ Create model for input data consisting of d dimensions. """

        # Initialize parameters to zero, for lack of a better choice.
        self.betas = np.zeros(d)

    def train(self, data, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda B, k: (k > 0) * self.alpha * B[k] - np.sum([ \
                                    data.y_train[i] * data.x_train[i, k] * \
                                    sigmoid(-data.y_train[i] *\
                                            np.dot(B, data.x_train[i,:])) \
                                    for i in range(data.n)])



        # The full gradient is just an array of componentwise derivatives
        gradient = lambda B: np.array([dl_by_dWk(B, k) \
                                 for k in range(data.x_train.shape[1])])
        
        # The function to be minimized
        # Use the negative log likelihood for the objective function.
        objectiveFunction = lambda B: -data.likelihood(betas=B, alpha=self.alpha)

        # Optimize
        self.betas = fmin_bfgs(objectiveFunction, self.betas, fprime=gradient)

    def train_alt(self, data, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        similarityMatrix = [[1 for m in range(self.x_train.shape[1])] for n in range(self.x_test.shape[1])]

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda W, k: (k > 0) * self.sfRegStep(W, k, i, similarityMatrix) - \
                                 np.sum([data.y_train[i] * data.x_train[i, k] * \
                                            sigmoid(-data.y_train[i] * \
                                                    np.dot(W, data.x_train[i, :])) \
                                    for i in range(data.n)])



        # The full gradient is just an array of componentwise derivatives
        gradient = lambda W: np.array([dl_by_dWk(W, k) \
                                       for k in range(data.x_train.shape[1])])

        # The function to be minimized
        # Use the negative log likelihood for the objective function.
        objectiveFunction = lambda W: -data.likelihood(betas=W, alpha=self.alpha)

        # Optimize
        self.betas = fmin_bfgs(objectiveFunction, self.betas, fprime=gradient)

    def sfRegStep(self, W, k, i, similarityMatrix):
        value = 0
        for j in range(self.x_train.shape[1] + self.x_test.shape[1]):
            value = self.x_train[k,:]*W*data.x_train[:,:] + self.x_test[k,:]*W*data.x_test[:,:] -\
                    self.x_test[k,:]*W*data.x_train[:,:] - self.x_train[k,:]*W*data.x_test[:,:]
        return value * similarityMatrix

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


if __name__ == "__main__":
    from pylab import *
    import sys

    source_file = 'source.txt'

    source_data = genfromtxt(source_file, delimiter=',')
    np.random.shuffle(source_data)

     # Define training and test splits
    train_source = source_data[:150,:]
    train_source_labels = source_data[:150,0]

    test_source = source_data[151:,:]
    test_source_labels = source_data[151:,0]

    data = TsvData(train_source, test_source)
    

    lr = Model(data.d)

    # Run for a variety of regularization strengths
    alphas = [0, .001, .01, .1]
    for j, a in enumerate(alphas):
        print "Initial likelihood:"
        print data.likelihood(lr.betas)
        
        # Train the model
        lr.train(data, alpha=a)
        
        # Display execution info
        print "Final betas:"
        print lr.betas
        print "Final likelihood:"
        print data.likelihood(lr.betas)

        predictions = lr.predict(test_source[:,1:].transpose())
        print predictions
        
        # Plot the results
        #subplot(len(alphas), 2, 2*j + 1)
        #lr.plot_training_reconstruction(data)
        #ylabel("Alpha=%s" % a)
        #if j == 0:
        #   title("Training set reconstructions")
        
        #subplot(len(alphas), 2, 2*j + 2)
        #lr.plot_test_predictions(data)
        #if j == 0:
        #   title("Test set predictions")

    #show()