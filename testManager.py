import llsvvm as llsvm
import MLLKM
import MLLKM2
import tool
import SDCA
import numpy as np
import time
import OMLLKM
import linereader

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

            for param in self.params :
                print(param)
                log.write("Parametres: {}\n\n".format(param))
                M=self.svm(**param["constructeur"])
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
                    
                log.write("Average accuracy :{} +- {} \n".format(np.mean(R),np.std(R)))
                print("Average accuracy :{} +- {} \n".format(np.mean(R),np.std(R)))
                print("Max/min accuracy :{} / {} \n".format(max(R),min(R)))

                log.write("Max/min accuracy :{} / {} \n".format(max(R),min(R)))
                
                log.write("Training per sample :{} +- {} \n".format(np.mean(Rtrain)/len(x_train),np.std(Rtrain)/len(x_train) ))
                log.write("Test time per sample :{} +- {} \n".format(np.mean(Rtest)/len(x_test),np.std(Rtest)/len(x_test) ))
                
                log.write("\n -------------- \n")
        log.close()

datasetPath="/users/edoudela12/PycharmProjects/Edouard/datasets"
logPath="/users/edoudela12/PycharmProjects/Edouard/logs"


test = testManager(datasetPath="/home/edoudela",logPath=logPath,nbTest=2,svm=MLLKM2.MLLKM2,lkernel=tool.lgauss)
test.addDataSet(["SUSY.csv"],tool.load3)
test.addParam(param={"nb_anchor": 1024, "lkernel": tool.lgauss,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})
test.addParam(param={"nb_anchor": 10096, "lkernel": tool.lgauss ,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 1,"E":1})


test.addParam(param={"nb_anchor": 1024, "lkernel": tool.lgauss_c,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})
test.addParam(param={"nb_anchor": 10096, "lkernel": tool.lgauss_c,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 1,"E":1})


test.addParam(param={"nb_anchor": 1024, "lkernel": tool.square,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})
test.addParam(param={"nb_anchor": 10096, "lkernel": tool.square,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})

test.addParam(param={"nb_anchor": 1024, "lkernel": tool.square_c,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})
test.addParam(param={"nb_anchor": 10096, "lkernel": tool.square_c,"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5,"E":1})

test.overview("susy")
