# CLASSIFICATION LOCALEMENT  LINÉAIRE 
Ce github contient les codes sources relatifs à mon stage au sein du laboratoire ETIS. 

Chaque svm est batis sur le modèle de la classe svm codé dans le fichier [svm.py](https://github.com/Edouard2laire/svm/blob/master/svm.py) . 
Chaque svm possède 3 méthodes :  

- init qui initialise les paramètres du modèle. 
- fit qui fait correspondre le modèle aux données
- predict qui fait une prédiction pour la donnée X
- score qui teste le modèle afin de calculer sa performance. 

Les fichiers correspondant aux svm sont 
- [lsvm.py](https://github.com/Edouard2laire/svm/blob/master/lsvm.py) qui code la svm linéaire ( partie 1.1 )
- [SDCA.py](https://github.com/Edouard2laire/svm/blob/master/SDCA.py) qui code la svm  à noyaux ( partie 1.2.1 )
- [llsvvm.py](https://github.com/Edouard2laire/svm/blob/master/llsvvm.py) qui code la svm localement linéaire ( partie 1.2.2) 
- [MLLKM.py](https://github.com/Edouard2laire/svm/blob/master/MLLKM.py) et [MLLKM2.py](https://github.com/Edouard2laire/svm/blob/master/MLLKM2.py) qui codent la svm multi locally linear kernel respectivement avec SKEAM( partie 2) et sans SKMEAN ( partie 2.3.2 )
- [near.py](https://github.com/Edouard2laire/svm/blob/master/near.py) corresponds à l'algorithme vu pour la detaction de peau dans la partie 3.3.1


Les autres fichiers fournissent des outils utile au reste du code : 
- tool.py définit les fonctions permetant d'ouvrir les différents datasets ainsi que les fonctions noyaux
- testManager.py permet de tester les différentes svm sur les différents datasets avec plusieurs set de paramètres. 
- kernel.py permet l'affiche des résultats comme dans la partie 3.2 


## Utilisation du gestionaire de test : 
Afin d'utiliser le gestionnaire de test, il faut commancer par initialiser les variables datasetPath et logPath corrspondant aux chemin d'acces vers les dossiers contenant 
les datasets ainsi que le dossier contenant les logs. 

On peut ensuite initialiser le gestionnaire de test avant de lui assigner les dataset à tester

```
test = testManager(datasetPath=datasetPath,logPath=logPath,nbTest=3,svm=MLLKM2)
test.addDataSet(["Skin_NonSkin.txt"],load4)
```

On renseigne ensuite les paramètres à tester en faisant :
```
test.addParam({'nb_anchor': 128,'pD': 0.3 ,'pB': 0.9, 't0': 666.0, 'pc': 0.01, 'pW': 0.01, 'E': 1, 'gamma': 1200, 'l': 1e-09, 'lkernel': square_c   })
```
puis on lance le test avec ``` test.overview("skin-opti-square3") ```

ce qui produit dans le fichier log : 

```
 -------------- 
Parametres: {'kmean': False, 'constructeur': {'pc': 0.1, 'nb_anchor': 64, 'lkernel': <function lgauss_c at 0x10a815400>, 'l': 1e-06, 'E': 5, 'gamma': 0.5, 'pB': 0.5, 't0': 1}}

Average accuracy :0.9523809523809523 +- 0.012046772038736693 
Training per sample :0.001074785873538158 +- 1.2566780301872397e-05 
Test time per sample :8.271035693940663e-05 +- 6.461228043108929e-06 
```
