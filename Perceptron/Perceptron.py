#!/usr/bin/env python
# coding: utf-8





def stepFunction(t):
    if t >= 0:
        return 1 
    return 0


def prediction(X, W, b):
    t = (np.matmul(X,W)+b)[0]
    return stepFunction(t)






def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b




def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        
        W, b = perceptronStep(X, y, W, b, learn_rate)
        
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
#         plt.plot((-W[0]/W[1], -b/W[1]), 'r--', label='classification line')
    

    return boundary_lines



if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    np.random.seed(42)
    df = pd.read_csv("data.txt",sep=",",names=["p","q","y"])
    # df.head()
    
    X= df[['p','q']].values
    y = df['y'].values
    boundryLines = trainPerceptronAlgorithm(X,y)
    print("success")




