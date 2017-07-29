import numpy as np
import matplotlib.pyplot as plt
from random import randint
from time import time



class SDCA(svm) :
    name="SDCA"

    def __init__(self, lkernel,E=10,C=100,gamma=2):
        svm.__init__(self)
        self.noyau=lkernel
        self.E=E
        self.C=C
        self.gamma=gamma
        self.x = None
        self.y = None

    def gamma_id(self,x):
        d=0
        n=len(x)
        for i in range(n):
            for j in range(0,i):
                d += np.linalg.norm(x[i]-x[j])

        self.gamma= 2*d/(n*(n-1))
        return self.gamma

    def kernel(self,f,x):

        n = len(x)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = f(x[i], x[j], self.gamma)
        return K


    def fit(self,x,y):
        self.x = x
        self.y = y
        n = np.size(y)
        K=self.kernel(self.noyau, x)


        y_hat = [0] * n
        alpha = [0] * n
        e = 1
        t = 0

        while e <= self.E:
            L = np.random.permutation(range(n))
            for i in L:
                g = 1 - y[i] * y_hat[i]
                if not (g == 0 or (g > 0 and alpha[i] == self.C) or (g < 0 and alpha[i] == 0)):
                    alpha_new = max(0, min(alpha[i] + g / K[i][i], self.C))
                    delta = alpha_new - alpha[i]
                    for j in range(0, n):
                        y_hat[j] = y_hat[j] + delta * y[i] * K[i][j]
                    alpha[i] = alpha_new

            e = e + 1

        self.alpha=alpha
        return alpha

    def predict(self,x):
        y = 0
        for i in range(len(self.x)):
            y+= self.alpha[i]*self.y[i]*self.noyau(x, self.x[i], self.gamma)

        return y
