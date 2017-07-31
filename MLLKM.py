import numpy as np
import tool
import matplotlib.pyplot as plt
from time import time
from svm import svm

class MLLKM(svm):
    name="MLLKM"

    """
    @brief Init the model
    @params : 
        nb_anchor : number of anchor points
        lkenel : The loccally kernel used in the model ( can be a component wise kernel ) lkenel take the param gamma
        pB,pD,pc are the learnings rates used in the stochastic descent
        l and t0 are also param for the stochastic descent
        E is the number of epoch
        
        anchor : you have to calculate the anchor point manually ( @see tool.SKMeans )
    """
    def __init__(self,nb_anchor=10,lkernel=tool.lgauss,pB=0.001,pD=0.6,pW=0.9,l=1e-6, t0=0.1, E=20,gamma=2.0):
        svm.__init__(self)

        self.nb_anchor=nb_anchor
        self.anchor = []
        self.B=np.array([1.0/nb_anchor for k in range(nb_anchor)])

        self.pB = pB
        self.pD = pD
        self.pW = pW
        self.l = l
        self.t0 = t0
        self.E = E
        self.gamma = gamma

        self.lkernel = lkernel

    def fit(self,x,y):
        t=1
        m=self.nb_anchor
        (n,d)=np.shape(x)
        W=np.zeros((m,d))
        B=self.B
        delta_buf = 0
        e=0

        while(e <= self.E) :
            L=np.random.permutation(range(n))
            for i in  L  :
                x0=x[i]
                y0 = y[i]

                H_t=0.0
                for j in range(len(B)) :
                    H_t+= B[j]* np.dot(W[j],self.lkernel(x0,self.anchor[j],self.gamma))

                ## Methode 1 for updating B : We use a gradient descent, then we project B on the contraint

                """
                if y0*H_t < 1 :
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0,self.anchor[j],self.gamma)
                        Remp = (y0*B[j]/(n*l*(t+t0)))*phi_j
                        W[j] =(1 - B[j]/(t+t0)) *W[j] + Remp

                        norme=np.linalg.norm(W[j])
                        B[j] = B[j] - p/(2*(t+t0))*norme*norme +  (p*y0 /(n*l*(t+t0)))*np.dot(W[j],phi_j)
                        if B[j] <= 1e-4:
                            B[j] = 0
                    B/= np.linalg.norm(B, ord=1)

                """
                #methode 2 for updating B
                if y0*H_t < 1 :
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0,self.anchor[j],self.gamma)
                        Remp = (self.pW * y0 * B[j] / (n * self.l * (t + self.t0))) * phi_j
                        W[j] = (1 - self.pW * B[j] / (t + self.t0)) * W[j] + Remp

                    if e>1 :
                        delta=[]
                        for j in range(len(B)):
                            norme=np.linalg.norm(W[j], ord=2)
                            phi_j = self.lkernel(x0, self.anchor[j], self.gamma)
                            delta.append(0.5*norme*norme - (y0/(n*self.l))*np.dot(W[j],phi_j))
                        delta=np.array(delta)
                        delta_buf = (1-self.pD)*delta_buf + self.pD*delta
                        arg=np.argmax(-delta_buf)
                        D=np.array([1 if k==arg else 0 for k in range(m)])
                        B=(1- self.pB/(t+self.t0))*B + (self.pB/(t+self.t0))*D
                        B[np.where(B<=1e-5)] = 0

                t+=1
                self.W=W
                self.B=B
            e+=1

        self.B=B
        self.W=W

    def predict(self,x):
        s=0
        for j in range(len(self.B)):
            s+= self.B[j] * np.dot(self.W[j], self.lkernel(x, self.anchor[j], self.gamma))
        return s
