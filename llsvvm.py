import numpy as np
import matplotlib.pyplot as plt
from random import randint
from heapq import merge
from time import time
import time
import matplotlib.pyplot as plt

import  tool


class llsvm :
    name="llsvm"
    def __init__(self,nb_anchor=10,lkernel=tool.gauss):
        self.nb_anchor=nb_anchor
        self.anchor=[]
        self.lkernel=lkernel

    ## This function computes K-means using the split algorithm.

    def SKMeans(self,X, eps=0.001, iteration=500, eps_norm=1e-6):
        K = 1
        Nc=self.nb_anchor
        X = np.array(X)
        N_sample, features_dim = X.shape
        np.seterr('raise')
        centers = np.zeros((K, features_dim))
        centers[0, :] = np.mean(X, axis=0, keepdims=True)


        while (K < Nc):
            distmatrix = -2 * np.dot(X, centers.transpose())  # n_sample*K
            distmatrix += np.sum(X * X, axis=1)[:, None]
            distmatrix += np.sum(centers * centers, axis=1)[None, :]
            ASSIGNMENTS_OLD = np.argmin(distmatrix, axis=1)
            for k in range(iteration):
                u0 = time.clock()
                j=0
                while (j < K):
                    ind = np.where(ASSIGNMENTS_OLD == j)
                    try:
                        centers[j] = np.mean(X[ind], axis=0)
                    except FloatingPointError:
                        tmp = centers.tolist()
                        del tmp[j]
                        centers = np.array(tmp, dtype=np.float32)
                        K -= 1
                        j -= 1
                    j+=1
                    
                    
                u1 = time.clock() - u0

                distmatrix = -2 * np.dot(X, centers.transpose())  # n_sample*K
                distmatrix += np.sum(X * X, axis=1)[:, None]
                distmatrix += np.sum(centers * centers, axis=1)[None, :]
                ASSIGNMENTS_NEW = np.argmin(distmatrix, axis=1)
                u2 = time.clock() - u0

                #print('{} u1: {} u2 {}'.format(k, u1, u2))

                if np.array_equal(ASSIGNMENTS_OLD, ASSIGNMENTS_NEW):
                    break
                else:
                    ASSIGNMENTS_OLD = ASSIGNMENTS_NEW

            #print('log K: {} log Nc {}'.format(np.log2(K), np.log2(Nc)))
            if np.log2(K) < np.floor(np.log2(Nc)):
                newcenters = np.concatenate((centers, centers), axis=0)

                newcenters[0:K, :] += eps
                newcenters[K:-1, :] -= eps
                centers = newcenters
                K *= 2

            else:
                sumdist = np.zeros(K)
                for j in range(K):
                    ind = np.where(ASSIGNMENTS_NEW == j)
                    sumdist[j] = np.sum(distmatrix[ind[0], j], axis=0) / (len(ind[0]) + eps_norm)
                i = np.argmax(sumdist)

                centers[i] = centers[i] + eps
                newcenters = centers[i, None] - 2 * eps
                centers = np.concatenate((centers, newcenters), axis=0)
                K += 1
                

        # last dist
        distmatrix = -2 * np.dot(X, centers.transpose())  # n_sample*K
        distmatrix += np.sum(X * X, axis=1)[:, None]
        distmatrix += np.sum(centers * centers, axis=1)[None, :]
        ASSIGNMENTS_OLD = np.argmin(distmatrix, axis=1)

        for k in range(iteration):

            for j in range(K):
                ind = np.where(ASSIGNMENTS_OLD == j)
                centers[j] = np.mean(X[ind], axis=0)

            distmatrix = -2 * np.dot(X, centers.transpose())  # n_sample*K
            distmatrix += np.sum(X * X, axis=1)[:, None]
            distmatrix += np.sum(centers * centers, axis=1)[None, :]
            ASSIGNMENTS_NEW = np.argmin(distmatrix, axis=1)

            #print('{} u1: {} u2 {}'.format(k, u1, u2))

            if np.array_equal(ASSIGNMENTS_OLD, ASSIGNMENTS_NEW):
                break
            else:
                ASSIGNMENTS_OLD = ASSIGNMENTS_NEW

        return centers, ASSIGNMENTS_NEW

     # Determine the gamma vector that appromimate x
    def localCoding(self, x, sigma_inv=2):
        gamma=[]
        s=0.0 #compteur pour normalisation
        for v in self.anchor:

            gamma_v=float(self.lkernel(x,v,sigma_inv))
            s+=gamma_v
            gamma.append(gamma_v)

        gamma = np.array(gamma)
        #print(gamma, s)
        gamma /= max(s,0.001)
        return gamma

    def train(self, x, y, l=0.00001, t0=100, E=10, skip=5, sigma_inv=2):
        t=1
        k=0
        m=self.nb_anchor
        (n,d)=np.shape(x)
        W=np.zeros((m,d))
        B=np.zeros((m,1))
        count=skip

        while(t <= E):
            L=np.random.permutation(range(n))
            for i in  L  :
                x0=x[i]
                y0 = y[i]
                gamma_t=self.localCoding(x0, sigma_inv)
                H_t=1 - y0*( np.dot(np.dot(gamma_t,W) , x0) + np.dot(gamma_t,B))
                if H_t > 0 :
                    for j in range(m):
                        W[j,:] += y0*gamma_t[j]*x0/(l*(k+t0))
                    B= B + y0*gamma_t[:,None]/(l*(k+t0))
                count-=1
                if count <=0 :
                    W=W * ( 1 - skip/(k+t0))
                    count=skip
                k+=1
                
            t=t+1
        self.W=W
        self.B=B
        

    def predict(self,X):
        gamma=self.localCoding(X)
        s = np.dot(np.dot(gamma, self.W), X) + np.dot(gamma, self.B)
        return s
    def test(self,x,y):
        c=0.0
        zero=0.0
        faux=0.0
        for i,xi in enumerate(x):
            #print(np.sign(int(self.predict(xi))),np.sign(y[i]))
            if( int(np.sign(self.predict(xi))) == np.sign(y[i])) :
                c+=1.0
        #print(c,zero,faux)
        return c/len(x)

plt.close('all')
plt.ion()


"""
x1,c1=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
#x2,c2=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+ 0)] for k in range(100)],[ -1 for k in range(100)]

x3,c3=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ +1 for k in range(100)]
#x4,c4=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+ 1)] for k in range(100)],[ -1 for k in range(100)]
x5,c5=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ -1 for k in range(100)]
#x6,c6=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+ 0)] for k in range(100)],[ +1 for k in range(100)]
x7,c7=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+-1)] for k in range(100)],[ -1 for k in range(100)]
#x8,c8=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
#x9,c9=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+-0)] for k in range(100)],[ -1 for k in range(100)]

x= x1+x3+x5+x7
y= c1+c3+c5+c7



X= x1+x3+x5+x7


for i in range(len(x)):
    x[i]/=np.linalg.norm(x[i])"""

"""
x,y=tool.load2("/users/edoudela12/PycharmProjects/Edouard/datasets/banana.train")
xtest,ytest=tool.load2("/users/edoudela12/PycharmProjects/Edouard/datasets/banana.test")

x_test=np.linspace(-2,2,150)
y_test=np.linspace(-2,2,150)

M = llsvm(nb_anchor=123, lkernel=tool.linear)
centers, ASSIGNMENTS_NEW = M.SKMeans(np.array(x))
M.anchor = centers
M.train(np.array(x), np.array(y), sigma_inv=5,t0=200, E=20,l=1e-5)
print(M.test(xtest,ytest))
plt.ioff()

t0=time.clock()
tool.draw(x,y,centers,x_test,y_test,M.predict,fig=1)
t1=time.clock()
print(t1-t0)



dataset=["ionosphere"]
n_dataset=len(dataset)
result=[]
for data in dataset :
    print(data)
    x,y=tool.load("/users/edoudela12/PycharmProjects/Edouard/{}_scale".format(data))
    n=np.shape(x)[0]

    p = np.random.permutation(range(n))
    m = int(0.7 * n)
    x_train = x[p[1:m]]
    y_train = y[p[1:m]]
    x_test = x[p[m:-1]]
    y_test = y[p[m:-1]]


    M=llsvm(nb_anchor=20,lkernel=tool.linear)
    ntest=5
    R=[]
    for i in range(ntest):
        centers, ASSIGNMENTS_NEW = M.SKMeans(np.array(x_train))
        M.anchor = centers
        M.train(np.array(x_train), np.array(y_train), sigma_inv=2,t0=20, E=20,l=1e-7)
        r = M.test(x_test,y_test)
        R.append(r)
        print("accuracy{} :{}".format(i,r) )
    result.append(R)
    print("Average accuracy :{}".format(np.mean(R)))

for i in range(n_dataset):
    print("{} average accuracy :{} +- {}".format(dataset[i],np.mean(result[i]),np.std(result[i])))

"""