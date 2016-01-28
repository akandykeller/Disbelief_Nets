from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO)
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *
import numpy as np
from numpy import random
import csv

file_name = 'MLP_VC_H10_Lr_N500-2500.csv'

# Train MLP using SGD until error is 0, or until n_iter
# is reached using a threshold of 0.5 on the sigmoid output
# Return the final error and the number of iterations required
def train(optimizer, X, y, n_iter, lr):
    for i in range(1, n_iter + 1):
        lr = float(lr)*(0.99999)
        loss = optimizer.train(X, y, lr)

        if np.isnan(loss):
            print("Recieved NAN Loss, breaking...")
            return loss, i

        preds = [[int(x >= 0.5)] for x in mlp.predict(X)]

        error = 1 - (preds == y).sum() / float(N)
        if error == 0:
            break
        if i % 1000 == 0:
            print("{}, error:{}, loss:{}".format(i, error, loss))

    return error, i

if __name__ == "__main__":
    # Define the names of the columns for the results CSV
    fieldnames = ['num_layers', 'num_hidden', 'data_dim', 
                  'N_data', 'error', 'iterations']

    # Open the results file and write header
    results_file = open(file_name, 'w')
    writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    writer.writeheader()
    results_file.close()

    # Parameters:
    # Maximum number of iterations before stopping
    n_iter = 300000
    # Dimension of each data point
    d = 100
    # List of number of points to iterate over
    num_pts = range(500, 5000, 100)# + range(500, 1100, 50)
    # Number of hidden units
    Hs = [10]
    # Number of layers
    L = 1
    # Number of times to repeat each result
    num_repeats = 10
    # Learning Rate
    lr = 0.1

    # Seed random for reproducibility
    random.seed(2)

    for repeat in range(num_repeats):
        for H_idx, H in enumerate(Hs):
            for N_idx, N in enumerate(num_pts):
                print("Trying with N = {} points, and H = {}".format(N, H))

                # Generate N d-dimensional data points sampled from a 
                # uniform distribution over [0,1]
                X = random.uniform(low=0, high=1, size=(N, d))

                # Ranomly pick half od the points and assign them a positive label
                labels = np.zeros((N, 1))
                pos_labels = random.choice(np.arange(0,N), np.floor(float(N)/2.0), replace=False)
                labels[pos_labels] = 1;
                labels = labels.astype(np.int32)

                # While we arent getting nan's
                nan_loss = True
                repeat_count = 0
                while (nan_loss and repeat_count < num_repeats):
                    # Define the network with a sigmoid output and LogLoss
                    mlp = Vector(d) >> Repeat(Tanh(H), L) >> Sigmoid(1)
                    sgd_optimizer = SGD(mlp, LogLoss(), clip_gradients=0.01)
                    
                    # Train the network until we reach 0 error or n_iter iterations
                    error, iterations = train(sgd_optimizer, X, labels, n_iter, lr)
                    if not np.isnan(error):
                        nan_loss = False
                    else:
                        print("Trying same network again... With new X data")
                        X = random.uniform(low=0, high=1, size=(N, d))
                        repeat_count += 1

                # Print the final error
                print("Succesfully shattered {} Pts, Error: {}".format(N, error))

                # Write the results as a line to the csv file
                results_file = open(file_name, 'a')
                writer = csv.DictWriter(results_file, fieldnames=fieldnames)
                writer.writerow({'num_layers': L, 'num_hidden': H, 'data_dim': d,
                                 'N_data':N, 'error':error, 'iterations':iterations})
                results_file.close()
