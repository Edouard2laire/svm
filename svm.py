
class svm :
    
    """
    
    """
    def fit(X,y):
        raise NotImplementedError
        
    def predict(X):
        raise NotImplementedError
        
    def score(X,y,metric="accuracy"):
        if(metric=="accuracy"):
            c=0.0
            for i,xi in enumerate(X):
                if( int(np.sign(self.predict(xi))) == int(np.sign(y[i]))) :
                    c+=1.0
            return c/len(X)
        else :
            raise NotImplementedError
            
    def get_params():
        raise NotImplementedError