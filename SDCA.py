import numpy as np
import matplotlib.pyplot as plt
from random import randint
from heapq import merge
from time import time
"""
kile



"""
class SDCA :
    name="SDCA"
    """ Attributs : 
        -fonction de noyeau
        -liste des alpha
        -gamma
        
        -fonctions : 
            -train 
            -predict    
    """
    def __init__(self, lkernel):
        self.noyau=lkernel
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


    def train(self,x,y,E=10,C=100,gamma=2):
        self.gamma=gamma
        self.x = x
        self.y = y
        n = np.size(y)
        K=self.kernel(self.noyau, x)

        """
        plt.figure(2)
        plt.imshow(K)
        plt.show()
        """
        y_hat = [0] * n
        alpha = [0] * n
        e = 0
        t = 0

        while e < E:
            L = np.random.permutation(range(n))
            for i in L:
                g = 1 - y[i] * y_hat[i]
                if not (g == 0 or (g > 0 and alpha[i] == C) or (g < 0 and alpha[i] == 0)):
                    alpha_new = max(0, min(alpha[i] + g / K[i][i], C))
                    delta = alpha_new - alpha[i]
                    for j in range(0, n):
                        y_hat[j] = y_hat[j] + delta * y[i] * K[i][j]
                    alpha[i] = alpha_new

                    """
                    self.alpha = alpha
                    
                    fig = plt.figure(1)
                    fig.canvas.set_window_title("iteration {}".format(t))
                    t+=1
                    plt.clf()
                    axes = plt.gca()
                    axes.set_xlim([-3,6])
                    axes.set_ylim([-2,2])

                    plt.stem(x, [y[i] * alpha[i] for i in range(len(alpha))], markerfmt='b+')

                    xtest = np.linspace(-3, 6, 80)
                    ytest = [gaussian.predict(xi) for xi in xtest]
                    plt.plot(xtest, ytest)
                    plt.show()

                    plt.pause(0.3)
                    """
            e = e + 1

        self.alpha=alpha
        return alpha

    def predict(self,x):
        y = 0
        for i in range(len(self.x)):
            y+= self.alpha[i]*self.y[i]*self.noyau(x, self.x[i], self.gamma)

        return y

    def test(self,x,y):
        c=0.0
        for i,xi in enumerate(x):
            if( int(np.sign(self.predict(xi))) == int(np.sign(y[i]))) :
                c+=1.0

        return c/len(x)





"""  liste des fichiers :
"/users/edoudela12/PycharmProjects/Edouard/diabetes_scale"
"/users/edoudela12/PycharmProjects/Edouard/heart_scale"
"/users/edoudela12/PycharmProjects/Edouard/ionosphere_scale"
"/users/edoudela12/PycharmProjects/Edouard/sonar_scale"
"""
"""

x,y=load("/users/edoudela12/PycharmProjects/Edouard/diabetes_scale")

n=np.shape(x)[0]

acc = 0.0
t=0.0
ntest=20
for i in range(ntest):
    p = np.random.permutation(range(n))

    m = int(0.7*n)
    x_train = x[p[1:m]]
    y_train = y[p[1:m]]
    x_test = x[p[m:-1]]
    y_test = y[p[m:-1]]


    gaussian=SDCA(gauss)
    #gaussian.gamma_id(x_train)
    gaussian.gamma = 1.0
    t1=time()
    gaussian.train(x_train,y_train,E=100,C=100)
    t2=time()
    r=gaussian.test(x_test,y_test)
    acc += float(r)/len(x_test)
    t+=(t2-t1)
    print("test n {}/{}".format(i,ntest))
    print("Donnees d'apprentissage : {} ".format(len(x_train)))
    print("Temps d'apprentissage :{} s".format(t2-t1))
    print("Donnees de test : {} ".format(len(x_test)))
    print("Performance :{} % ".format(r*100/len(x_test)))
    print("gamma :{}  ".format(gaussian.gamma))
    print("alpha {}  ".format(gaussian.alpha))

print("acc moyenne:",acc/ntest)
print("temps entrainement moyenn:",t/ntest)


x1,y1=[ (0.5*np.random.randn() -1)    for k in range(10)],[ +1 for k in range(10)]
x2,y2=[ (0.5*np.random.randn()+ 2) for k in range(10)],[ -1 for k in range(10)]
x3,y3=[ (0.5*np.random.randn()+ 4) for k in range(10) ],[+1 for k in range(10) ]

x=x1+x2+x3
y=y1+y2+y3

plt.ion()
plt.figure(1)

gaussian=SDCA(gauss)
gaussian.gamma = 1.0
alpha=gaussian.train(x,y,E=10,C=1)

xtest=np.linspace(-3,6,20)
ytest=[gaussian.predict(xi) for xi in xtest]


plt.figure(1)

plt.stem(x1,y1)
plt.stem(x2,y2)
plt.stem(x3,y3)

print(len(x),len(y))

plt.stem(x,[y[i]*alpha[i] for i in range(len(alpha))],markerfmt='b+')
plt.plot(xtest,ytest)
plt.ioff()
plt.show()
"""