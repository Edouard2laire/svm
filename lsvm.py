import numpy as np
import tool

class svm :
    name="lSVM"

    def __init__(self):
        print("init")

    def train(self,x,y,b0=0,E=10,l=0.1,t0=1.0,p=0.001):
        (n,d)=np.shape(x)
        W = np.zeros((1, d))
        b = np.zeros((1, 1))
        b[0,0]=b0
        e=1
        t=0
        while(e <= E):
            L = np.random.permutation(range(n))
            for i in L:
                x0 = x[i]
                y0 = y[i]
                y_hat= np.dot(W,x0) + b
                if y0*y_hat <  1 :
                    W= (1 - p / (t + t0))*W + (p/(n*l*(t+t0)))*y0*x0
                    b=b + (p/(n*l*(t+t0)))*y0
                t+=1
            e+=1
        self.W=W
        self.b=b
    def predict(self,X):
        return np.dot(self.W,X) + self.b
"""
x1,c1=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
x3,c3=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ +1 for k in range(100)]
x5,c5=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ -1 for k in range(100)]
x7,c7=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+-1)] for k in range(100)],[ -1 for k in range(100)]

x= np.array(x1+x5+x3+x7)
c=np.array(c1+c5+c3+c7)

M=svm()
M.train(x,c,E=10,p=0.1,l=0.001)


x_test=np.linspace(-2,2,150)
y_test=np.linspace(-2,2,150)
tool.draw(x,c,[],x_test,y_test,M.predict,fig=1)
"""
