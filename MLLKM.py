import numpy as np
import tool
import matplotlib.pyplot as plt
from time import time


class MLLKM:
    name="MLLKM"
    def __init__(self,nb_anchor=10,lkernel=tool.lgauss):
        self.nb_anchor=nb_anchor
        self.anchor = []
        self.B=np.array([1.0/nb_anchor for k in range(nb_anchor)])

        self.lkernel = lkernel

    def train(self,x,y,p=0.001,l=1e-6, t0=0.1, E=20,gamma=2.0):
        t=1
        m=self.nb_anchor
        self.gamma=gamma
        (n,d)=np.shape(x)
        W=np.zeros((m,d))
        B=self.B
        delta_buf = 0
        e=0

        x_test = np.linspace(-2, 2, 100)
        y_test = np.linspace(-2, 2, 100)

        while(e <= E) :
            L=np.random.permutation(range(n))
            for i in  L  :
                x0=x[i]
                y0 = y[i]

                #H_t=1 - y0*B*W*K
                H_t=0.0
                for j in range(len(B)) :
                    H_t+= B[j]* np.dot(W[j],self.lkernel(x0,self.anchor[j],gamma))

                ## Methode 1
                """
                if y0*H_t < 1 :
                    ## Methode 1
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0,self.anchor[j],gamma)
                        Remp = (y0*B[j]/(n*l*(t+t0)))*phi_j
                        W[j] =(1 - B[j]/(t+t0)) *W[j] + Remp

                        norme=np.linalg.norm(W[j])
                        B[j] = B[j] - p/(2*(t+t0))*norme*norme +  (p*y0 /(n*l*(t+t0)))*np.dot(W[j],phi_j)
                        if B[j] <= 1e-4:
                            B[j] = 0
                    B/= np.linalg.norm(B, ord=1)

                if(t == 5000):
                    print("reset B")
                    B_hat=B
                    B=np.array([1.0/m for k in range(m)])

                    for i in range(m):
                        print(B_hat[i],B[i])
                """
                #methode 2
                if y0*H_t < 1 :
                    for j in range(len(B)):
                        phi_j = self.lkernel(x0,self.anchor[j],gamma)
                        Remp = (y0*B[j]/(n*l*(t+t0)))*phi_j
                        W[j] =(1 - B[j]/(t+t0)) *W[j] + Remp

                    if e>1 :
                        delta=[]
                        for j in range(len(B)):
                            norme=np.linalg.norm(W[j], ord=2)
                            phi_j = self.lkernel(x0, self.anchor[j], gamma)
                            delta.append(0.5*norme*norme - (y0/(n*l))*np.dot(W[j],phi_j))
                        delta=np.array(delta)
                        delta_buf = (1-0.6)*delta_buf + 0.6*delta
                        arg=np.argmax(-delta_buf)
                        D=np.array([1 if k==arg else 0 for k in range(m)])
                        B=(1- p/t)*B + (p/t)*D
                        B[np.where(B<=1e-5)] = 0
                        #print(B_hat-B)

                t+=1
                self.W=W
                self.B=B

                #print(e,t,t%n)
                #if t <= 6 or ( t%200<2):
                #    tool.draw(x,y,M.anchor,x_test,y_test,M.predict,highlight=x0,fig=2,poids=B)
            e+=1

        self.B=B
        self.W=W

    def predict(self,x):
        s=0
        for j in range(len(self.B)):
            s+= self.B[j] * np.dot(self.W[j], self.lkernel(x, self.anchor[j], self.gamma))
        return s
    def test(self,x,y):
        c=0.0

        for i,xi in enumerate(x):
            if( int(np.sign(self.predict(xi))) == np.sign(y[i])) :
                c+=1.0

        return c/len(x)

"""
x,y=tool.load2("/users/edoudela12/PycharmProjects/Edouard/datasets/forestNormalized.train",normalised=False)
xtest,ytest=tool.load2("/users/edoudela12/PycharmProjects/Edouard/datasets/forestNormalized.test",normalised=False)
print("File loaded")
#x_test=np.linspace(-2,2,100)
#y_test=np.linspace(-2,2,100)

#plt.close('all')
#plt.ion()

M=MLLKM(nb_anchor=80,lkernel=tool.lgauss_c)
M.anchor= tool.SKMeans(np.array(x),nb_anchor=M.nb_anchor)
M.train(x,y,p=0.9,l=1e-6, t0=1, E=10,gamma=1.0)
print(M.test(xtest,ytest))

#plt.ioff()

#tool.draw(x,y,M.anchor,x_test,y_test,M.predict,fig=1,poids=M.B)



p = np.random.permutation(range(n))
m = int(0.7 * n)
x_train = x[p[1:m]]
y_train = y[p[1:m]]
x_test = x[p[m:-1]]
y_test = y[p[m:-1]]



M=MLLKM(nb_anchor=20)
M.anchor= tool.SKMeans(np.array(x_train),nb_anchor=M.nb_anchor)
M.train(x_train,y_train,E=40)
print(M.test(x_test,y_test))
"""
"""" 
Resultats de tests : 
#methode 2 

Perf :0.83  Param : nb_anchor=50 / p=0.00001 / l=1e-6 / t0=0.1 / E=10  / gamma=2.0 + sans changer B au debut
Perf :0.84  Param : nb_anchor=50 / p=0.00001 / l=1e-6 / t0=0.1 / E=200 / gamma=2.0 + sans changer B au debut
Perf :0.86  Param : nb_anchor=50 / p=0.00001 / l=1e-6 / t0=0.1 / E=10  / gamma=2.0 + reset B e l'epoque 5
Perf :0.85  Param : nb_anchor=50 / p=0.001 / l=1e-6 / t0=0.1 / E=10  / gamma=2.0 + reset B pour epoque 5
Perf :0.84  Param : nb_anchor=50 / p=0.0001 / l=1e-6 / t0=0.1 / E=10  / gamma=2.0 + reset B pour epoque 5
Perf :0.85  Param : nb_anchor=50 / p=0.005 / l=1e-6 / t0=0.1 / E=10  / gamma=2.0 

"""
"""
(0.0, array([ 0.00506241,  0.01797927]))
(0.00079901267642639004, array([ 1.89538132,  3.70208083]))
(0.013480396720332508, array([-0.90800595, -1.35241806]))
(0.0, array([-0.05090325, -0.0055206 ]))
(0.0010395835823305477, array([ 0.0381666 ,  0.15582389]))
(0.00068105732595362552, array([-2.71931554, -4.18509688]))
(0.0040357601241576427, array([ 0.35569671, -0.61295827]))
(0.00094826227367391855, array([-0.27574054, -0.41401891]))
(0.0030188164751971621, array([ 0.50228465,  1.80847164]))
(0.00031589755195066214, array([-0.58832493,  3.20498809]))
(0.11280475971937229, array([  6.44964489, -11.885888  ]))
(0.17181973439810355, array([ -8.61424597,  14.89619301]))
(7.4328591466472557e-05, array([ 0.00816625,  0.00929214]))
(0.038557720483692118, array([ 1.19612915,  2.39738791]))
(0.059192087786526612, array([ 1.39087994, -5.06841771]))
(7.9112489068187557e-05, array([-0.02042096,  0.00327961]))
(0.0031088483577412641, array([ 0.26108433,  0.43155578]))
(0.15585680475453284, array([-14.60769659, -11.88804062]))
(0.045805228540665023, array([  1.62661783,  12.72148604]))
(0.00018992062517354664, array([ 0.03706176, -0.06899522]))
(0.00038198659596300861, array([-0.12941583,  0.30807271]))
(0.083781735791270365, array([-7.0881559 , -6.81007289]))
(0.0, array([ 0.37384826,  0.12981816]))
(0.019857528582890235, array([-2.13206442,  1.78413211]))
(0.082378797419511318, array([-5.57600944, -4.16086341]))
(0.039465992955788183, array([-5.29399342, -2.03775909]))
(0.10031294502542706, array([ -1.91693541,  16.87582987]))
(0.0015532356953418295, array([-0.1393857 ,  0.01455641]))
(2.0802445706448882e-05, array([-0.00226539,  0.00644054]))
(0.0, array([-0.16015284, -0.42503186]))
(0.0, array([ 0.07718338, -0.04769595]))
(0.00013093811028069908, array([-0.02234096,  0.00587868]))
(0.0, array([-0.38812095, -1.44924822]))
(0.012218795774244893, array([-2.93748134, -3.07840104]))
(0.0, array([ 0.04628173,  0.04572486]))
(0.019466520238286156, array([ 1.22919004,  1.64984119]))
(0.014377051868519118, array([ 2.23079641, -5.56603745]))
(0.00019683256227770587, array([ 0.57045006, -0.58409915]))
(3.2966849936354664e-05, array([ 0.00158075,  0.00710092]))
(4.0376537649031078e-05, array([ 0.00485313,  0.00796583]))
(5.9252675324020312e-05, array([ 0.02779869, -0.00366089]))
(9.2865817539543197e-05, array([ 0.01992082,  0.00190296]))
(0.0027119489208287718, array([ 0.84236772, -0.08042479]))
(0.0094421671911809283, array([-2.81314625, -0.28328796]))
(0.0015031347348546159, array([ 0.13932603,  0.34341965]))
(2.6069750747427601e-05, array([-0.31024994, -1.53594932]))
(8.6371471778819289e-05, array([ 0.0041348 ,  0.01268358]))
(3.7341593055538144e-05, array([ 0.00072321,  0.00644278]))
(1.7008915233612656e-05, array([ 0.00019869,  0.00606177]))
(0.0, array([-0.03668017, -0.20453159]))


"""
"""
dataset=["ionosphere","diabetes","heart","sonar"]
n_dataset=len(dataset)
result=[]
resultLT=[]
resultPT=[]

for data in dataset :
    print(data)
    x,y=tool.load("/users/edoudela12/PycharmProjects/Edouard/datasets/{}_scale".format(data),normalised=False)
    n=np.shape(x)[0]

    p = np.random.permutation(range(n))
    m = int(0.7 * n)
    x_train = x[p[1:m]]
    y_train = y[p[1:m]]
    x_test = x[p[m:-1]]
    y_test = y[p[m:-1]]

    M = MLLKM(nb_anchor=40, lkernel=tool.lgauss_c)
    ntest=30
    R=[]
    LT=[]
    PT=[]
    for i in range(ntest):
        M.anchor = tool.SKMeans(np.array(x), nb_anchor=M.nb_anchor)
        t0=time()
        M.train(np.array(x_train), np.array(y_train), p=0.9, l=1e-6, t0=1, E=15, gamma=10)
        t1=time()
        r = M.test(x_test,y_test)
        t2=time()

        R.append(r)
        LT.append(t1-t0)
        PT.append(t2-t1)

        print("accuracy{} :{}".format(i,r) )
    result.append(R)
    resultLT.append(LT)
    resultPT.append(PT)

    print("Average accuracy :{}".format(np.mean(R)))
    print("Average LT/PT :{}".format(np.mean(LT),np.mean(PT)))

for i in range(n_dataset):
    print("-------")
    print("{} average accuracy :{} +- {}".format(dataset[i],np.mean(result[i]),np.std(result[i])))
    print("{} average LT/PT :{} /{}".format(dataset[i],np.mean(resultLT[i]),np.mean(resultPT[i])))

"""
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

x= np.array(x1+x3+x5+x7)
y= np.array(c1+c3+c5+c7)

plt.ion()

M=MLLKM(nb_anchor=10, lkernel=tool.lgauss)
M.anchor=tool.SKMeans(x,10)
M.train(x, y, p=0.,l=1e-6, t0=1, E=20, gamma=0.1)


x_test = np.linspace(-2, 2, 100)
y_test = np.linspace(-2, 2, 100)
plt.ioff()
tool.draw(x,y,M.anchor,x_test,y_test,M.predict,fig=1,poids=M.B)
print(M.B)
print(M.W)"""