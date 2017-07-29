import numpy as np
import tool
import matplotlib.pyplot as plt
from time import time

class MLLKM2:
    name="MLLKM2"
    def __init__(self,nb_anchor=10,lkernel=tool.lgauss):
        self.nb_anchor=nb_anchor
        self.anchor = np.array([])
        self.anchorU = np.array([])

        self.B=np.array([1.0/nb_anchor for k in range(nb_anchor)])

        self.lkernel = lkernel



    def train(self, x, y, pB=0.001,pD=0.6,pc=0.1, l=1e-6, t0=0.1, E=20, gamma=2.0):
        t = 1
        m = 0
        self.gamma = gamma
        n, d = np.shape(x)
        delta_buf = 0
        e = 0

        while (e <= E):
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
                temp1=(1.0 - pc/(t+t0) )*self.anchor[k_min]
                temp2=x0*pc/(t+t0)
                self.anchor[k_min]= temp1 + temp2

                distance_hat=np.linalg.norm(x0-self.anchor[k_min])
                if distance_hat < distance[k_min] :
                    self.anchorU[k_min]=x0
                    distance[k_min]=distance_hat

                # H_t=1 - y0*B*W*K
                H_t = 0.0
                for j in range(len(B)):
                    H_t += B[j] * np.dot(W[j], self.lkernel(x0, self.anchorU[j], gamma))

                if y0 * H_t < 1:
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0, self.anchorU[j], gamma)
                        Remp = (y0 * B[j] / (n * l * (t + t0))) * phi_j
                        W[j] = (1 - B[j] / (t + t0)) * W[j] + Remp
                    if t >  self.nb_anchor:
                        delta = []
                        for j in range(len(B)):
                            norme = np.linalg.norm(W[j], ord=2)
                            phi_j = self.lkernel(x0, self.anchorU[j], gamma)
                            delta.append(0.5 * norme * norme - (y0 / (n * l)) * np.dot(W[j], phi_j))

                        delta = np.array(delta)
                        delta_buf = (1 - pD/t) * delta_buf + (pD/t)* delta
                        arg = np.argmax(-delta_buf)
                        D = np.array([1 if k == arg else 0 for k in range(m)])
                        B = (1 - pB / t) * B + (pB / t) * D
                        B[np.where(B <= 1e-5)] = 0
                    # print(B_hat-B)

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

    def test(self, x, y):
        c = 0.0

        for i, xi in enumerate(x):
            if (int(np.sign(self.predict(xi))) == np.sign(y[i])):
                c += 1.0

        return c / len(x)

"""
x1,c1=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
#x2,c2=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+ 0)] for k in range(100)],[ -1 for k in range(100)]

x3,c3=[ [(0.25*np.random.randn()+ -1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ +1 for k in range(100)]
#x4,c4=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+ 1)] for k in range(100)],[ -1 for k in range(100)]
x5,c5=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+ 1)] for k in range(100)],[ +1 for k in range(100)]
#x6,c6=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+ 0)] for k in range(100)],[ +1 for k in range(100)]
x7,c7=[ [(0.25*np.random.randn()+  1),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
#x8,c8=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+-1)] for k in range(100)],[ +1 for k in range(100)]
x9,c9=[ [(0.25*np.random.randn()+  0),(0.25*np.random.randn()+-0)] for k in range(100)],[ -1 for k in range(100)]

x= np.array(x1+x3+x5+x7+x9)
y= np.array(c1+c3+c5+c7+c9)

plt.ion()

M=MLLKM2(nb_anchor=64, lkernel=tool.square)
M.train(x, y, pB=0.7,pc=0.6, l=1e-6, t0=1, E=10, gamma=0.8)


x_test = np.linspace(-2, 2, 100)
y_test = np.linspace(-2, 2, 100)
plt.ioff()
tool.draw(x,y,M.anchor,x_test,y_test,M.predict,fig=1,poids=M.B)
print(M.B)
print(M.W)
"""
