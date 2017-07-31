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
        
    def addParam(self,constructeur={},kmean=False,train={} ):
        
        param=dict()
        param["constructeur"]=constructeur
        param["kmean"]=kmean
        param["train"]=train
                
        self.params.append(param)


    def test(self, testName="",n=1):
        t4=time.time()
        log = open("{}/{}/overview_{}.txt".format(self.logPath, self.svm.name, testName), 'w')
        log.write("test : {}\n".format(testName))

        for file in self.dataset:
            print(file)
            log.write("Dataset : {}\n".format(file))

            f=open("{}/{}".format(self.datasetPath,file),'r')
            k=0

            log.write("Donnee entrainement : {}\n".format(int(0.7*n)))
            log.write("Donnee test : {}\n".format(int(0.3*n)))

            for param in self.params:
                param["n"]=n
                print(param)
                log.write("Parametres: {}\n\n".format(param))
                M = self.svm(**param["constructeur"])
                R = []
                Rtrain = []
                Rtest = []
                for t in range(self.nbTest):
                    t0 = time.time()
                    r=0
                    k=0
                    f.seek(0)
                    for k in range(n):
                        ligne=f.next()
                        print(float(k)/n)
                        if k > 0 and k < n :
                            x,y=self.openMethod(ligne,k=k)
                            if k < int(0.7*n) :
                                M.train(x, y, **param["train"])
                            elif k == int(0.7*n) :
                                t1 = time.time()
                                if (int(np.sign(M.predict(x))) == np.sign(y)):
                                    r += 1.0
                            else :
                                if( int(np.sign(M.predict(x))) == np.sign(y) ):
                                    r += 1.0
                    t2 = time.time()
                    R.append(r/(0.3*n))
                    Rtrain.append(t1 - t0)
                    Rtest.append(t2 - t1)

                log.write("Average accuracy :{} +- {} \n".format(np.mean(R), np.std(R)))
                print("Average accuracy :{} +- {} \n".format(np.mean(R), np.std(R)))
                print("Max/min accuracy :{} / {} \n".format(max(R), min(R)))

                log.write("Max/min accuracy :{} / {} \n".format(max(R), min(R)))

                log.write("Training per sample :{} +- {} \n".format(np.mean(Rtrain) / int(0.7*n),
                                                                    np.std(Rtrain) / int(0.7*n)))
                log.write("Test time per sample :{} +- {} \n".format(np.mean(Rtest) / int(0.3*n),
                                                                     np.std(Rtest) / int(0.3*n)))

                log.write("\n -------------- \n")
        log.close()



        t5=time.time()
        print(t5-t4)
        
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
                    M.train(x_train,y_train,**param["train"])
                    t1=time.time()
                    r=M.test(x_test,y_test)
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





"""
for k in range(6,13):
    for gamma in np.linspace(0.5,2,5):
        for p in np.linspace(0.1,0.9,3):
            for pc in np.linspace(0.1,0.9,3):
                test.addParam( constructeur={"nb_anchor":2**k,"lkernel":tool.lgauss},train={"pB":p,"pc":pc,"l":1e-6, "t0":1, "E":2,"gamma":gamma})
                test.addParam( constructeur={"nb_anchor":2**k,"lkernel":tool.lgauss_c},train={"pB":p,"pc":pc,"l":1e-6, "t0":1, "E":2,"gamma":gamma})
                test.addParam( constructeur={"nb_anchor":2**k,"lkernel":tool.square},train={"pB":p,"pc":pc,"l":1e-6, "t0":1, "E":2,"gamma":gamma})
                test.addParam( constructeur={"nb_anchor":2**k,"lkernel":tool.square_c},train={"pB":p,"pc":pc,"l":1e-6, "t0":1, "E":2,"gamma":gamma})
"""
"""
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.lgauss_c},train={"pB": 0.1, "pc":0.5, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.square},train={"pB": 0.9, "pc":0.1, "l": 1e-6, "t0": 1, "E": 1, "gamma": 1.625})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.square_c},train={"pB": 0.9, "pc":0.1, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 64, "lkernel": tool.square_c},train={"pB": 0.1, "pc":0.5, "l": 1e-6, "t0": 1, "E": 1, "gamma": 0.875})
"""
"""
test = testManager(datasetPath=datasetPath,logPath=logPath,nbTest=10,svm=OMLLKM.OMLLKM2,lkernel=tool.lgauss)
test.addDataSet(["HEPMASS.csv"],tool.decode3)
test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.test("testO",n=30)"""

"""
test = testManager(datasetPath="/home/edoudela/",logPath=logPath,nbTest=3,svm=OMLLKM.OMLLKM2,lkernel=tool.lgauss)
test.addDataSet(["HIGGS.csv"],tool.decode3)


test.addParam(constructeur={"nb_anchor": 2048, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,  "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.lgauss_c},train={"pB": 0.1, "pc":0.5, "l": 1e-6, "t0": 1,  "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,  "gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.square_c},train={"pB": 0.1, "pc":0.5, "l": 1e-6, "t0": 1, "gamma": 0.875})
test.addParam(constructeur={"nb_anchor": 4096, "lkernel": tool.square_c},train={"pB": 0.1, "pc":0.5, "l": 1e-6, "t0": 1, "gamma": 0.875})



test.test("HIGGS",n=11000000)"""



test = testManager(datasetPath="/home/edoudela",logPath=logPath,nbTest=2,svm=MLLKM2.MLLKM2,lkernel=tool.lgauss)
test.addDataSet(["HEPMASS.csv"],tool.load3)
test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 2048, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 5096, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 10096, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 20096, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 40096, "lkernel": tool.lgauss},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})

test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 2048, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 5096, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 10096, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 20096, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 40096, "lkernel": tool.lgauss_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})

test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 2048, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 5096, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 10096, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 20096, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 40096, "lkernel": tool.square},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})

test.addParam(constructeur={"nb_anchor": 1024, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 2048, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 5096, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 10096, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 20096, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})
test.addParam(constructeur={"nb_anchor": 40096, "lkernel": tool.square_c},train={"pB": 0.5, "pc":0.1, "l": 1e-6, "t0": 1,"gamma": 0.5})

test.overview("HEPPMASS")
