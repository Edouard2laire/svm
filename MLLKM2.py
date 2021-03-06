import numpy as np
#import tool
import matplotlib.pyplot as plt
import time
import numpy as np
#import tool
import matplotlib.pyplot as plt
import time


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
    def __init__(self,nb_anchor=10,lkernel=lgauss,pB=0.001,pD=0.6,pc=0.1,pW=0.9, l=1e-6, t0=0.1, E=20, gamma=0.8):
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
        self.E=E
        self.gamma=gamma
        self.lkernel = lkernel
        
        self.m=0
        self.t=1+t0
    
    def addToAnchor(self,x0,d):
        if self.m==0 :
            self.anchor=np.array([x0])
            self.anchorU = np.array([x0])
            self.distance=np.array([0.0])
            self.W=np.zeros((1, d))
            self.B = np.array([1])
            self.m+=1
        elif self.m < self.nb_anchor:
            self.anchor = np.append( self.anchor,[x0],axis=0)
            self.anchorU = np.append( self.anchorU,[x0],axis=0)
            self.distance=np.concatenate((self.distance,[0.0]),axis=0)
    
            self.B=np.concatenate((self.B,[1.0/self.m]),axis=0)
            self.B/=np.linalg.norm(self.B,ord=1)
            self.W=np.concatenate((self.W,np.zeros((1,d))),axis=0)
            self.m+=1
        
        
            
    def updateAnchors(self,x0):
        dmin=np.linalg.norm(x0-self.anchor[0])
        k_min=0
        
        diff=np.subtract(x0,self.anchorU)
        dist_hat=np.linalg.norm(diff,axis=1)
        k_min=np.argmin(dist_hat)

        self.anchor[k_min]= (1.0 - self.pc/self.t )*self.anchor[k_min] + x0*self.pc/self.t

        distance_hat=np.linalg.norm(x0-self.anchor[k_min])
        if distance_hat < self.distance[k_min] :
            self.anchorU[k_min]=x0
            self.distance[k_min]=distance_hat
    

    def fit(self, x, y):        
        n, d = np.shape(x)
        delta_buf = 0
        e = 0

        while (e <= self.E):
            L = np.random.permutation(range(n))
            for i in L:
                x0 = x[i]
                y0 = y[i]
                                
                if self.m < self.nb_anchor :
                    self.addToAnchor(x0,d)
                else :
                    self.updateAnchors(x0)

                diff=np.subtract(x0,self.anchorU)
                K=self.lkernel(diff,self.gamma)
                
                H_t= np.sum( np.multiply(self.W , K*self.B[:,None]) ) 
                if y0 * H_t < 1:
                    self.W+= self.pW*y0/ (n * self.l *self.t ) *K*self.B[:,None] - (self.pW /self.t) * self.W*self.B[:,None]
                    if self.m >  self.nb_anchor:
                        delta = np.linalg.norm(self.W,axis=1)**2
                        delta-= (y0 / (n*self.l)) * np.einsum('ij,ij->i', self.W, K)
                        
                        delta_buf = (1 - self.pD/self.t) * delta_buf + (self.pD/self.t)* delta
                        arg = np.argmax(-delta_buf)
                        D = np.array([1 if k == arg else 0 for k in range(self.m)])
                        
                        self.B = (1 - self.pB / self.t) * self.B + (self.pB / self.t) * D
                        self.B[np.where(self.B <= 1e-5)] = 0
                self.t += 1
            e += 1
    def predict(self, x):
        diff=np.subtract(x,self.anchorU)
        K=self.lkernel(diff,self.gamma)

        return np.sum( np.multiply(self.W , K*self.B[:,None] ) )
