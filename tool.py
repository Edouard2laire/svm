import numpy as np
import matplotlib.pyplot as plt
import time


def load(fichier,normalised=True):
    f=open(fichier,'r')
    y=[]
    x=[]
    n=0 # nombre de coordonees par ligne

    for ligne in f:
        label=ligne.split(' ')
        label.pop()
        y.append(int(label[0]))
        if n==0 :  #On determine le nombre de coordonees
            [n,c]=label[-1].split(':')
            n=int(n)
        coord=[0]*n
        for couple in label[1:]:
            [ind,c]=couple.split(':')
            coord[int(ind)-1]=float(c)
        x.append(coord)
    f.close()

    X=np.array(x,dtype=float)
    Y=np.array(y)
    if(normalised):
        for i in range(len(x)):
            X[i]/=np.linalg.norm(X[i],ord=2)

    return X,Y

def load2(fichier,normalised=True):
    f = open(fichier, 'r')
    y = []
    x = []
    n=0

    for ligne in f:
        if n==0 :
            label = ligne.split(' ')#On determine le nombre de coordonees
            n=int(label[-1])
        else:
            label = ligne.split('  ')
            y.append(float(label[0]))

            coord=label[1].strip('\n').split(' ')
            x.append(coord)

    f.close()

    X=np.array(x,dtype=float)
    Y=np.array(y)
    if normalised:
        for i in range(len(x)):
            X[i]/=np.linalg.norm(X[i],ord=2)


    return X,Y

def load3(fichier,normalised=True,exemple=10000):
    f = open(fichier, 'r')
    y = []
    x = []
    n=0

    for ligne in f:
        if n==0 :
            label = ligne.split(',')#On determine le nombre de coordonees
            n=len(label)-1
            print(label,n)

        else:
            L = ligne.strip('\n').split(',')
            label=float(L[0])
            coord=L[1:]

            if(label== 0):
                y.append(-1.0)
            else:
                y.append(+1.0)
            x.append(coord)
    f.close()

    X=np.array(x,dtype=float)
    Y=np.array(y)


    if exemple > 0:
        n=len(x)
        print(n)
        p = np.random.permutation(range(n))
        print(p[0:int(exemple)])
        X=X[p[0:int(exemple)]]
        Y=Y[p[0:int(exemple)]]

    if normalised:
        for i in range(len(X)):
            X[i]/=np.linalg.norm(X[i],ord=2)

    return X,Y

def load4(fichier,normalised=False,exemple=-1):
    f = open(fichier, 'r')
    y = []
    x = []
    n=0
    k=0
    for ligne in f:
        if n==0 :
            label = ligne.split('\t')#On determine le nombre de coordonees
            n=len(label)-1

        else:
            L = ligne.strip('\n').split('\t')
            label=float(L[-1])
            coord=L[0:n]

            if(label== 1):
                y.append(-1.0)
            else:
                y.append(+1.0)
            x.append(coord)
            k+=1
    f.close()

    X=np.array(x,dtype=float)
    Y=np.array(y,dtype=float)



    if exemple > 0:
        n=len(x)
        p = np.random.permutation(range(n))
        X=X[p[0:int(exemple)]]
        Y=Y[p[0:int(exemple)]]

    if normalised:
        for i in range(len(X)):
            X[i]/=255

    return X,Y




def SKMeans(X,nb_anchor=10, eps=0.001, iteration=500, eps_norm=1e-6):
    K = 1
    Nc = nb_anchor
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
            j = 0
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
                j += 1

            u1 = time.clock() - u0

            distmatrix = -2 * np.dot(X, centers.transpose())  # n_sample*K
            distmatrix += np.sum(X * X, axis=1)[:, None]
            distmatrix += np.sum(centers * centers, axis=1)[None, :]
            ASSIGNMENTS_NEW = np.argmin(distmatrix, axis=1)
            u2 = time.clock() - u0

            # print('{} u1: {} u2 {}'.format(k, u1, u2))

            if np.array_equal(ASSIGNMENTS_OLD, ASSIGNMENTS_NEW):
                break
            else:
                ASSIGNMENTS_OLD = ASSIGNMENTS_NEW

        # print('log K: {} log Nc {}'.format(np.log2(K), np.log2(Nc)))
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

        # print('{} u1: {} u2 {}'.format(k, u1, u2))

        if np.array_equal(ASSIGNMENTS_OLD, ASSIGNMENTS_NEW):
            break
        else:
            ASSIGNMENTS_OLD = ASSIGNMENTS_NEW

    return centers

        # Determine the gamma vector that appromimate x



def gauss(xi,xj,gamma):
    norme = np.linalg.norm(xi-xj)
    return np.exp(-gamma*norme*norme)

def linear(xi,xj,sigma):
    norme= np.linalg.norm(xi-xj)
    #print(norme,1 - gamma*norme*norme)
    return max(0.0, 1 - sigma*norme*norme)

def lgauss(diff,gamma=0.95):
    #iff=xi-xj
    norme = np.linalg.norm(diff,axis=1)
    expV=np.exp(-gamma*norme*norme)[:,None]
    return expV*diff
    
def lgauss_c(diff,gamma=0.95):
    #diff=xi-xj
    return np.multiply(np.exp(-gamma*np.multiply(diff,diff)),diff)


def square(diff,gamma):
    (n,d)=diff.shape
    norme = np.linalg.norm(diff,axis=1)
    cond=np.where( norme < gamma)
    norme[ cond] = np.zeros((1,d))
    norme[ not(cond)] = (xi-xj)
    

    return norme

def square_c(xi,xj,gamma):
    r=np.multiply(xi-xj, xi-xj)
    for k in range(len(r)):
        if r[k]> gamma:
            r[k]=0
        else:
            r[k]=1
    return np.multiply(r,xi-xj)


def draw(x_e,y_e, centers, x_t, y_t, predictor, highlight=[None, None], poids=[],fig=1):

    if( np.shape(x_e)[1] <= 2):
        x, y = np.meshgrid(x_t, y_t)
        f = np.vectorize(lambda x, y: predictor(np.array([x, y])))
        Z = f(x, y)

        x_p=[]
        x_m=[]

        for i  in range(len(x_e)):
            if y_e[i] > 0 :
                x_p.append(x_e[i])
            else:
                x_m.append(x_e[i])

        plt.figure(fig)
        #plt.clf()
        # plt.axis('equal')


        ctr = plt.contour(x, y, Z, [-0.1,0.0, +0.1])
        plt.clabel(ctr,fontsize=10, fmt='%3.2f')

        plt.scatter([abs for [abs, ord] in x_p], [ord for [abs, ord] in x_p], c='red')
        plt.scatter([abs for [abs, ord] in x_m], [ord for [abs, ord] in x_m], c='blue')

        if(len(poids) >= 1):
            size=poids*1000
            plt.scatter([abs for [abs, ord] in centers], [ord for [abs, ord] in centers], s=size, c = 'green')
        else:
            plt.scatter([abs for [abs, ord] in centers], [ord for [abs, ord] in centers], c = 'green')


        if (highlight[0] != None):
            plt.scatter([highlight[0]],[highlight[1]], c = 'yellow')

        plt.show()
        #plt.pause(0.1)

