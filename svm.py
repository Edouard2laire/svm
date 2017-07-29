
class svm :
    
    """
    @brief Fit the SVM model according to the given training data.
    """
    def fit(self,X,y):
        raise NotImplementedError
    """
    @brief Make a prediction pour the vector X.
    """    
    def predict(self,X):
        raise NotImplementedError
    
    """
    @brief Returns the accuracy on the given test data and labels. Other metrics will be implemented in the future. 
    """
    def score(self,X,y,metric="accuracy"):
        if(metric=="accuracy"):
            c=0.0
            for i,xi in enumerate(X):
                if  int(np.sign( self.predict(xi) ) ) == int(np.sign(y[i]))  :
                    c+=1.0
            return c/len(X)
        else :
            raise NotImplementedError
    """
    @brief Get parameters for this estimator.
    """
    def get_params(self):
        raise NotImplementedError