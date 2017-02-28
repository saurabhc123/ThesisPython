"""
Very simple module for doing logistic regression.

Based on:
- http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html
- http://people.csail.mit.edu/jrennie/writing/lr.pdf
"""

from scipy.optimize.optimize import fmin_bfgs
from sklearn.metrics import f1_score
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))









class Model(object):
    """ A simple logistic regression model with L2 regularization (zero-mean
    Gaussian priors on parameters). """

    def __init__(self, train_data, test_data, d):
        """ Create model for input data consisting of d dimensions. """

        # Initialize parameters to zero, for lack of a better choice.
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
        similarityMatrix = [[1 for i in range(self.x_train.shape[0])] for j in range(m)]
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
                reg += 1* \
                (np.dot(betas, x_total[i]) - \
                 np.dot(betas, x_total[j]))**2

        return (alpha / 2.0) * reg

    def train(self, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda B, k: (k > 0) * self.alpha * B[k] - np.sum([ \
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

    def train_alt(self, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Set alpha so it can be referred to later if needed
        self.alpha = alpha

        similarityMatrix = [[1 for m in range(self.x_train.shape[0])] for n in range(self.x_test.shape[0])]

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        # The following has a dimension of [1 x k] where k = |W|
        dl_by_dWk = lambda W, k: (k > 0) * self.sfRegStep(W, k, similarityMatrix, alpha)



        # The full gradient is just an array of componentwise derivatives
        gradient = lambda W: np.array([dl_by_dWk(W, k) \
                                       for k in range(self.x_train.shape[1])])

        # The function to be minimized
        # Use the negative log likelihood for the objective function.
        objectiveFunction = lambda W: -self.likelihood_alt(similarityMatrix, betas=W, alpha=self.alpha)

        # Optimize
        print('Optimizing for alpha = {}'.format(alpha))
        self.betas = fmin_bfgs(objectiveFunction, self.betas, fprime=gradient)

    def sfRegStep(self, W, k, similarityMatrix, alpha):
        # for j in range(self.x_train.shape[0] + self.x_test.shape[0]):
        #     value = self.x_train[j,k]* np.dot(W,data.x_train[j,:]) + self.x_test[k,:]*W*data.x_test[:,:] -\
        #             self.x_test[k,:]*W*data.x_train[:,:] - self.x_train[k,:]*W*data.x_test[:,:]

        x_total = np.concatenate((self.x_train, self.x_test), axis=0)
        n = self.x_train.shape[0]
        m_n = x_total.shape[0]
        dwk = 0
        #print('sfRegStep - First Summation for k={}'.format(k))
        for i in range(n):
            #print('i={}, k={}'.format(i,k))
            dwk +=  -1 * self.y_train[i]*self.x_train[i,k]*sigmoid(self.y_train[i] * np.dot(W, self.x_train[i,:]))

        #print('sfRegStep - Regularization for k={}'.format(k))
        dL = 0
        # import pdb; pdb.set_trace();
        for i in range(n):
            for j in range(m_n):
                # TODO: replace 1 by I_ij
                dL += alpha*1*x_total[i,k]*np.dot(W, x_total[i,:])\
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


if __name__ == "__main__":
    from pylab import *
    import sys

    source_file = 'source.txt'

    source_data = genfromtxt(source_file, delimiter=',')
    validation_data = genfromtxt(source_file, delimiter=',')[55:75,:]
    np.random.shuffle(source_data)

     # Define training and test splits
    train_source = source_data[:10,:]
    train_source_labels = source_data[:10,0]

    test_source = source_data[30:40,:]
    test_source_labels = source_data[30:40,0]

    lr = Model(train_source, test_source, train_source[:,1:].shape[1])

    # Run for a variety of regularization strengths
    #alphas = [0.01, .11, 1.1, 11.1]
    alphas = [0, .001, .01, .1]
    for j, a in enumerate(alphas):
        print "Initial likelihood:"
        print lr.betas

        # Train the model
        lr.train(alpha=a)

        # Display execution info
        print "Final betas:"
        print lr.betas
        print "Final likelihood:"
        print lr.betas



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
    predictions = lr.predict(validation_data[:,1:].transpose())
    predictionLabels = map (lambda prediction : 1 if prediction > 0.5 else 0, predictions)
    print "Final Predictions:"
    print predictionLabels
    print f1_score(validation_data[:,0], predictionLabels, average='binary')
