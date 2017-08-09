# CLASSIFICATION LOCALEMENT  LINÉAIRE 
Ce github contient les codes sources relatifs à mon stage au sein du laboratoire ETIS. 

Chaque svm est batis sur le modèle de la classe svm codé dans le fichier svm.py. 
Chaque svm possède 3 méthodes :  

- init qui initialise les paramètres du modèle. 
- fit qui fait correspondre le modèle aux données
- predict qui fait une prédiction pour la donnée X
- score qui teste le modèle afin de calculer sa performance. 

Les fichiers correspondant aux svm sont 
- lsvm.py qui code la svm linéaire ( partie 1.1 )
- SDCA.py qui code la svm  à noyaux ( partie 1.2.1 )
- llsvvm.py qui code la svm localement linéaire ( partie 1.2.2) 
- MLLKM.py et MLLKM2.py qui codent la svm multi locally linear kernel respectivement avec SKEAM( partie 2) et sans SKMEAN ( partie 2.3.2 )
- near.py corresponds à l'algorithme vu pour la detaction de peau dans la partie 3.3.1


Les autres fichiers fournissent des outils utile au reste du code : 
- tool.py définit les fonctions permetant d'ouvrir les différents datasets ainsi que les fonctions noyaux
- testManager.py permet de tester les différentes svm sur les différents datasets avec plusieurs set de paramètres. 
- kernel.py permet l'affiche des résultats comme dans la partie 3.2 

