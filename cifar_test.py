import logging
logging.basicConfig(level=logging.DEBUG)
import theano.tensor as T
from theanify import theanify, Theanifiable

from deepx.nn import MultilayerPerceptron
from deepx.optimize import RMSProp
from deepx.train import Trainer

import numpy as np
import cPickle

class MLPClassifier(Theanifiable):

    def __init__(self, D, H, C, n_layers):
        super(MLPClassifier, self).__init__()
        self.mlp = MultilayerPerceptron('mlp', D, H, C, n_layers=n_layers)
        self.compile_method('errors')

    @theanify(T.matrix('X'), T.ivector('y'))
    def cost(self, X, y):
        ypred = self.mlp.forward(X)
        return T.nnet.categorical_crossentropy(ypred, y).mean()

    @theanify(T.matrix('X'), T.ivector('y'))
    def errors(self, X, y):
        y_pred = self.mlp.forward(X).argmax(axis=1)
        return T.mean(T.neq(y_pred, y))

    def get_parameters(self):
        return self.mlp.get_parameters()


# Load all 5 Cifar-10 dict-batches into an array
cifar = []

for i in range(1,6):
    f = open('../Data/cifar-10-batches-py/data_batch_{}'.format(i), 'r')
    cifar.append(cPickle.load(f))
    f.close()

# Load the test set as the last element of the array
f = open('../Data/cifar-10-batches-py/test_batch', 'r')
cifar.append(cPickle.load(f))
f.close()


# Concat all batches
x_train = cifar[0]['data']
y_train_real = cifar[0]['labels']
x_test = cifar[5]['data']
y_test_real = cifar[5]['labels']

for i in range(1,5):
    x_train = np.vstack((x_train, cifar[i]['data']))
    y_train_real = y_train_real + cifar[i]['labels']


# Define input and output layers
D, C = 32*32*3, 10

# Use seed for reproducability
np.random.seed(1)

# Redefine y_train and y_test to noise 
y_train_noise = np.random.randint(10, size=len(y_train_real))
y_test_noise = np.random.randint(10, size=len(y_test_real))

Hs = range(100, 1100, 100)
Ls = [1, 2, 3, 4]

for L in Ls:
    for H in Hs:
        for real in [True, False]:
            if real:
                y_train = y_train_real
                y_test = y_test_real
            else:
                y_train = y_train_noise
                y_test = y_test_noise
            
            trn_err = []
            tst_err = []

            mlp = MLPClassifier(D, H, C, L)
            rmsprop = RMSProp(mlp)

            # Training

            iterations = 20000
            learning_rate = 0.01

            batch_size = 20
            for i in xrange(iterations):
                u = np.random.randint(x_train.shape[0] - batch_size)

                if i % 500 == 0:
                    print "Iteration %u: %f" % (i + 1, rmsprop.train(x_train[u:u+batch_size, :],
                                                                     y_train[u:u+batch_size],
                                                                     learning_rate))
                if i % 10 == 0:    
                    trn_err.append(mlp.errors(x_train, y_train))
                    tst_err.append(mlp.errors(x_test, y_test))

                else:
                    rmsprop.train(x_train[u:u+batch_size, :], y_train[u:u+batch_size], learning_rate)

            # Evaluation
            print "Training error:", mlp.errors(x_train, y_train)
            print "Test error:", mlp.errors(x_test, y_test)

            filename = 'MLP_errs_H{}_L{}_R{}.pkl'.format(H,L,real)
            
            f = open(filename, 'w')
            cPickle.dump((trn_err, tst_err, H, L, real), f)
            f.close()

            print "Errors saved to {}".format(filename)
