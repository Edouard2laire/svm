import numpy as np
import tool
from svm import svm
class lsvm(svm) :
    name="lSVM"

    def __init__(self,b0=0,E=10,l=0.1,t0=1.0,p=0.001):
        svm.__init__(self)
        self.b0=b0
        self.E=E
        self.l=l
        self.t0=t0
        self.p=p

    def fit(self,x,y):
        (n,d)=np.shape(x)
        W = np.zeros((1, d))
        b = np.zeros((1, 1))
        b[0,0]=self.b0
        e=1
        t=0
        while(e <= self.E):
            L = np.random.permutation(range(n))
            for i in L:
                x0 = x[i]
                y0 = y[i]
                y_hat= np.dot(W,x0) + b
                if y0*y_hat <  1 :
                    W= (1 - self.p / (t + self.t0))*W + (self.p/(n*self.l*(t+self.t0)))*y0*x0
                    b=b + (self.p/(n*self.l*(t+self.t0)))*y0
                t+=1
            e+=1
        self.W=W
        self.b=b
    def predict(self,X):
        return np.dot(self.W,X) + self.b