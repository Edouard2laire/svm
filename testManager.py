import llsvvm as llsvm
import MLLKM
import MLLKM2
import tool
import SDCA
import numpy as np
import time
import OMLLKM
import linereader
import lsvm

class testManager :
    def __init__(self,svm=None,lkernel=None,datasetPath="",logPath="",nbTest=5):
        self.datasetPath=datasetPath
        self.lkernel=lkernel
        self.logPath=logPath
        self.nbTest=nbTest
        self.svm=svm
        self.dataset=[]
        self.params=[]
    
    def addDataSet(self,files,openMethod,split=True):
        self.dataset+= files
        self.openMethod=openMethod
        self.split=split
        
    def addParam(self,param={},kmean=False):
        
        parametre=dict()
        parametre["constructeur"]=param
        parametre["kmean"]=kmean

        self.params.append(parametre)


    def overview(self, testName=""):
        log=open("{}/{}/overview_{}.txt".format(self.logPath,self.svm.name,testName),'w')
        log.write("test : {}\n".format(testName))    

        for file in self.dataset :
            
            print(file)
            log.write("Dataset : {}\n".format(file))
            if (not self.split):
                x_train, y_train = self.openMethod("{}/{}.train".format(self.datasetPath, file))
                x_test, y_test = self.openMethod("{}/{}.test".format(self.datasetPath, file))
                m=len(y_train)
                n=len(y_test)+m

                print("files open")
            else :
                x,y=self.openMethod("{}/{}".format(self.datasetPath,file))
                print("file open")
                n=np.shape(x)[0]
                m = int(0.7 * n)
            
            
            log.write("Donnee entrainement : {}\n".format(m))
            log.write("Donnee test : {}\n".format(n-m))
            best={"accuracy":0,"vAccuracy":0,"training":0,"vTraining":0,"test":0,"vtest":0,"param":{}}

            for param in self.params :
                print(param)
                log.write("Parametres: {}\n\n".format(param))
                M=self.svm(**param["constructeur"])

                resultat={"accuracy":0,"vAccuracy":0,"training":0,"vTraining":0,"test":0,"vtest":0}
                R=[]
                Rtrain=[]
                Rtest=[]
                for t in range(self.nbTest):
                    if (self.split):
                        p = np.random.permutation(range(n))
                        x_train = x[p[1:m]]
                        y_train = y[p[1:m]]
                        x_test = x[p[m:-1]]
                        y_test = y[p[m:-1]]

                    if(param["kmean"] ):
                        M.anchor=M.anchor=tool.SKMeans(x_train,nb_anchor=M.nb_anchor)
                    t0=time.time()
                    M.fit(x_train,y_train)
                    t1=time.time()
                    r=M.score(x_test,y_test)
                    t2=time.time()
                    
                    R.append(r)
                    Rtrain.append(t1-t0)
                    Rtest.append(t2-t1)
                resultat["accuracy"]=np.mean(R)
                resultat["vAccuracy"]=np.std(R)
                resultat["training"]=np.mean(Rtrain)/len(x_train)
                resultat["vTraining"]=np.std(Rtrain)/len(x_train)
                resultat["test"]=np.mean(Rtest)/len(x_test)
                resultat["vtest"]=np.std(Rtest)/len(x_test)

                if resultat["accuracy"] > best["accuracy"] :
                    best=resultat
                    best["param"]=param

                log.write("Average accuracy :{} +- {} \n".format(resultat["accuracy"],resultat["vAccuracy"]))
                print("Average accuracy :{} +- {} \n".format(resultat["accuracy"],resultat["vAccuracy"]))
                log.write("Training per sample :{} +- {} \n".format(resultat["training"],resultat["vTraining"] ))
                log.write("Test time per sample :{} +- {} \n".format(resultat["test"],resultat["vtest"] ))
                
                log.write("\n -------------- \n")
            log.write("Meilleur test \n")
            log.write("Parametres: {}\n\n".format(best["param"]))
            log.write("Average accuracy :{} +- {} \n".format(best["accuracy"], best["vAccuracy"]))
            log.write("Training per sample :{} +- {} \n".format(best["training"], best["vTraining"]))
            log.write("Test time per sample :{} +- {} \n".format(best["test"], best["vtest"]))
        log.close()

datasetPath="/users/edoudela12/PycharmProjects/Edouard/datasets"
logPath="/users/edoudela12/PycharmProjects/Edouard/logs"


test = testManager(datasetPath=datasetPath,logPath=logPath,nbTest=5,svm=lsvm.lsvm)
test.addDataSet(["sonar_scale","sonar_scale","diabetes_scale","sonar_scale"],tool.load)

for p in np.linspace(0.001,0.5,10):
    print(p)
    test.addParam(param={"b0":0,"E":10,"l":0.1,"t0":1.0,"p":0.001})
test.overview("lsvm")
