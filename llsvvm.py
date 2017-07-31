import numpy as np
import matplotlib.pyplot as plt
from random import randint
from heapq import merge
from time import time
import time
import matplotlib.pyplot as plt

import  tool

from svm import svm

class llsvm(svm) :
    name="llsvm"
    def __init__(self,nb_anchor=10,lkernel=tool.gauss,l=0.00001, t0=100, E=10, skip=5, sigma_inv=2):
        svm.__init__(self)

        self.nb_anchor=nb_anchor
        self.anchor=[]
        self.lkernel=lkernel

        self.l = l
        self.t0 = t0
        self.E = E
        self.skip = skip
        self.sigma_inv = sigma_inv

     # Determine the gamma vector that appromimate x
    def localCoding(self, x, sigma_inv=2):
        gamma=[]
        s=0.0 #compteur pour normalisation
        for v in self.anchor:

            gamma_v=float(self.lkernel(x,v,sigma_inv))
            s+=gamma_v
            gamma.append(gamma_v)

        gamma = np.array(gamma)
        gamma /= max(s,0.001)
        return gamma

    def fit(self, x, y):
        e=1
        k=0
        m=self.nb_anchor
        (n,d)=np.shape(x)
        W=np.zeros((m,d))
        B=np.zeros((m,1))
        count=self.skip

        while(e <= self.E):
            L=np.random.permutation(range(n))
            for i in  L  :
                x0=x[i]
                y0 = y[i]
                gamma_t=self.localCoding(x0, self.sigma_inv)
                H_t=1 - y0*( np.dot(np.dot(gamma_t,W) , x0) + np.dot(gamma_t,B))
                if H_t > 0 :
                    for j in range(m):
                        W[j,:] += y0*gamma_t[j]*x0/(self.l*(k+self.t0))
                    B= B + y0*gamma_t[:,None]/(self.l*(k+self.t0))
                count-=1
                if count <=0 :
                    W=W * ( 1 - self.skip/(k+self.t0))
                    count=self.skip
                k+=1
                
            e=e+1
        self.W=W
        self.B=B
        

    def predict(self,X):
        gamma=self.localCoding(X)
        s = np.dot(np.dot(gamma, self.W), X) + np.dot(gamma, self.B)
        return s
