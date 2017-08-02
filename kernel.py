# -*- coding:cp437 -*-

import numpy as np
import tool
import matplotlib.pyplot as plt
import matplotlib
import MLLKM2
import lsvm
import matplotlib.patches as mpatches
print("style:",matplotlib.style.available)
matplotlib.style.use('fivethirtyeight')
plt.close('All')

def conv(x,y) :

    return [ (x+1.)/(13*3) , (y+1.)/(4*6) ]

tab=[]

tab.append({"nom":"MLLKM(SKMEAN-gaussian)","Performance":92.2,"Training":4.7,"Predict":0.16,"color":"#6600FF"})
tab.append({"nom":"MLLKM(SKMEAN-Component Gaussian)","Performance":90.6,"Training":3.6,"Predict":0.15,"color":"#3333FF"})
tab.append({"nom":"MLLKM(gaussian)","Performance":93.6,"Training":9.4,"Predict":0.24,"color":"#6666FF"})
tab.append({"nom":"MLLKM(Component gaussian)","Performance":92.1,"Training":7.6,"Predict":0.14,"color":"#6699FF"})
tab.append({"nom":"SAG","Performance":79.1,"Training":0.0,"Predict":0.01,"color":"#339900"})
tab.append({"nom":"LLSVM","Performance":94.7,"Training":4.3,"Predict":0.2,"color":"#990066"})
tab.append({"nom":"SCDA(gaussian)","Performance":94.6,"Training":2.7,"Predict":1.9,"color":"#993300"})

tab1=[]

tab1.append({"nom":"MLLKM(SKMEAN-gaussian)","Performance":78.6,"Training":5.2,"Predict":0.16,"color":"#6600FF"})
tab1.append({"nom":"MLLKM(SKMEAN-Component Gaussian)","Performance":73.8,"Training":5.1,"Predict":0.17,"color":"#3333FF"})
tab1.append({"nom":"MLLKM(gaussian)","Performance":77.4,"Training":11.9,"Predict":0.24,"color":"#6666FF"})
tab1.append({"nom":"MLLKM(Component gaussian)","Performance":75.0,"Training":7.7,"Predict":0.17,"color":"#6699FF"})
tab1.append({"nom":"SAG","Performance":62.5,"Training":0.0,"Predict":0.01,"color":"#339900"})
tab1.append({"nom":"LLSVM","Performance":81.1,"Training":4.6,"Predict":0.2,"color":"#990066"})
tab1.append({"nom":"SCDA(gaussian)","Performance":86.6,"Training":1.8,"Predict":1.0,"color":"#993300"})

tab2=[]

tab2.append({"nom":"MLLKM(SKMEAN-gaussian)","Performance":81.8,"Training":4.9,"Predict":0.15,"color":"#6600FF"})
tab2.append({"nom":"MLLKM(SKMEAN-Component Gaussian)","Performance":81.6,"Training":3.7,"Predict":0.13,"color":"#3333FF"})
tab2.append({"nom":"MLLKM(gaussian)","Performance":87.8,"Training":10.2,"Predict":0.24,"color":"#6666FF"})
tab2.append({"nom":"MLLKM(Component gaussian)","Performance":82.5,"Training":6.4,"Predict":0.14,"color":"#6699FF"})
tab2.append({"nom":"SAG","Performance":83.0,"Training":0.1,"Predict":0.01,"color":"#339900"})
tab2.append({"nom":"LLSVM","Performance":77.2,"Training":4.6,"Predict":0.2,"color":"#990066"})
tab2.append({"nom":"SCDA(gaussian)","Performance":77.8,"Training":2.1,"Predict":1.3,"color":"#993300"})

tab3=[]

tab3.append({"nom":"MLLKM(SKMEAN-gaussian)","Performance":76.5,"Training":4.9,"Predict":0.15,"color":"#6600FF"})
tab3.append({"nom":"MLLKM(SKMEAN-Component Gaussian)","Performance":77.5,"Training":3.7,"Predict":0.13,"color":"#3333FF"})
tab3.append({"nom":"MLLKM(gaussian)","Performance":76.7,"Training":10.2,"Predict":0.24,"color":"#6666FF"})
tab3.append({"nom":"MLLKM(Component gaussian)","Performance":76.6,"Training":6.4,"Predict":0.14,"color":"#6699FF"})
tab3.append({"nom":"SAG","Performance":83.0,"Training":0.1,"Predict":0.01,"color":"#339900"})
tab3.append({"nom":"LLSVM","Performance":77.2,"Training":4.6,"Predict":0.2,"color":"#990066"})
tab3.append({"nom":"SCDA(gaussian)","Performance":77.8,"Training":2.1,"Predict":1.3,"color":"#993300"})




training=[]
predict=[]
perf=[]
nom=[]
couleur=[]
for resultat in tab :
    predict.append(resultat["Predict"])
    training.append(resultat["Training"])
    perf.append(resultat["Performance"])
    nom.append(resultat["nom"])
    couleur.append(resultat["color"])
poids=[]
PerfMin=np.min(perf)
for performance in perf:
    poids.append( 10**(performance/PerfMin*1)  )

fig=plt.figure(1)
plt.title("Performance des algoritmes sur le dataset IONOSPHERE")
ax=fig.add_subplot(111)
#plt.scatter(training,predict,s=None)
plt.xlabel("<-- Temps d'entrainement rapide(ms)" )
plt.xlim(-1,11)
plt.ylim(-1,3)

plt.ylabel("<-- Temps de prediction rapide(ms)")
e=[]
for i in range(len(perf)):
    print(training[i], predict[i],poids[i])
    x,y=conv(poids[i],poids[i])

    circ=mpatches.Ellipse((training[i], predict[i]), width=x ,height=y, color=couleur[i], fill=True,angle=90)
    e.append(circ)
    plt.annotate("{}".format(i), (training[i],predict[i]),horizontalalignment='center',color="white")
    plt.annotate("{}%".format(perf[i]), (training[i],predict[i] + 0.5*y*(-1)**(i+1) ),horizontalalignment='center')

    ax.add_artist(circ)
    circ.set_clip_box(ax.bbox)

plt.legend(e, nom,
                  loc='upper left', ncol=2, scatterpoints=1,
                  frameon=True, markerscale=2, title='Algorithmes',
                  borderpad=0.5, labelspacing=0.5)

plt.show()