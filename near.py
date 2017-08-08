import numpy as np

class nearest(svm):
    name="MLLKM2"
    def __init__(self):
        svm.__init__(self)

    def fit(self,x,y):
        self.x=x
        self.y=y
        
    def predict(self,x):
        diff=np.subtract(x,self.x)
        distance=np.linalg.norm(diff,axis=1)
        dmin= np.argmin( distance )
        return self.y[dmin]