import numpy as np
import tool
import matplotlib.pyplot as plt
from time import time

from svm import svm


class MLLKM2(svm):
    name="MLLKM2"

    """
    @brief Init the model
    @params : 
        nb_anchor : number of anchor points
        lkenel : The loccally kernel used in the model ( can be a component wise kernel ) lkenel take the param gamma
        pB,pD,pc,PW are the learnings rates used in the stochastic descent
        l and t0 are also param for the stochastic descent
        E is the number of epoch       
    """
    def __init__(self,nb_anchor=10,lkernel=tool.lgauss,pB=0.001,pD=0.6,pc=0.1,pW=0.9, l=1e-6, t0=0.1, E=20, gamma=2.0):
        svm.__init__(self)

        self.nb_anchor=nb_anchor
        self.anchor = np.array([])
        self.anchorU = np.array([])

        self.B=np.array([1.0/nb_anchor for k in range(nb_anchor)])

        self.pB=pB
        self.pW=pW
        self.pD=pD
        self.pc=pc
        self.l=l
        self.t0=t0
        self.E=E
        self.gamma=gamma
        self.lkernel = lkernel



    def fit(self, x, y):
        t = 1
        m = 0
        n, d = np.shape(x)
        delta_buf = 0
        e = 0

        while (e <= self.E):
            L = np.random.permutation(range(n))
            for i in L:
                x0 = x[i]
                y0 = y[i]

                if m==0 :
                    self.anchor=np.array([x0])
                    self.anchorU = np.array([x0])
                    distance=np.array([0.0])
                    W=np.zeros((1, d))
                    B = np.array([1])
                    m+=1
                elif m < self.nb_anchor:
                    self.anchor = np.append( self.anchor,[x0],axis=0)
                    self.anchorU = np.append( self.anchor,[x0],axis=0)
                    distance=np.concatenate((distance,[0.0]),axis=0)

                    B=np.concatenate((B,[1.0/m]),axis=0)
                    B/=np.linalg.norm(B,ord=1)
                    W=np.concatenate((W,np.zeros((1,d))),axis=0)
                    m+=1

                dmin=np.linalg.norm(x0-self.anchor[0])
                k_min=0

                for k in range(m):
                    de=np.linalg.norm(x0-self.anchor[k])
                    if de<dmin :
                        dmin=de
                        k_min=k
                temp1=(1.0 - self.pc/(t+self.t0) )*self.anchor[k_min]
                temp2=x0*self.pc/(t+self.t0)
                self.anchor[k_min]= temp1 + temp2

                distance_hat=np.linalg.norm(x0-self.anchor[k_min])
                if distance_hat < distance[k_min] :
                    self.anchorU[k_min]=x0
                    distance[k_min]=distance_hat

                # H_t=1 - y0*B*W*K
                H_t = 0.0
                for j in range(len(B)):
                    H_t += B[j] * np.dot(W[j], self.lkernel(x0, self.anchorU[j], self.gamma))

                if y0 * H_t < 1:
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0, self.anchorU[j], self.gamma)
                        Remp = (self.pW*y0 * B[j] / (n * self.l * (t + self.t0))) * phi_j
                        W[j] = (1 - self.pW*B[j] / (t + self.t0)) * W[j] + Remp
                    if t >  self.nb_anchor:
                        delta = []
                        for j in range(len(B)):
                            norme = np.linalg.norm(W[j], ord=2)
                            phi_j = self.lkernel(x0, self.anchorU[j], self.gamma)
                            delta.append(0.5 * norme * norme - (y0 / (n * self.l)) * np.dot(W[j], phi_j))

                        delta = np.array(delta)
                        delta_buf = (1 - self.pD/t) * delta_buf + (self.pD/t)* delta
                        arg = np.argmax(-delta_buf)
                        D = np.array([1 if k == arg else 0 for k in range(m)])
                        B = (1 - self.pB / t) * B + (self.pB / t) * D
                        B[np.where(B <= 1e-5)] = 0
                t += 1
                self.W = W
                self.B = B
            e += 1

        self.B = B
        self.W = W

    def predict(self, x):
        s = 0
        for j in range(len(self.B)):
            s += self.B[j] * np.dot(self.W[j], self.lkernel(x, self.anchor[j], self.gamma))
        return s

