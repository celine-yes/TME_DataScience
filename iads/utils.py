# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import math

from sklearn.utils import shuffle

# ------------------------ 

def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data_label = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])
    
    #data_label = shuffle(data_label)
    
    return (np.random.uniform(binf,bsup,(2*n,p)),data_label)
    
# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    #numpy.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    #numpy.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    np.random.seed(42)
    
    data_desc_n =  np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    data_desc_p =  np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    
    data_desc = np.vstack((data_desc_n, data_desc_p))
    
    data_labels_p = np.asarray([+1 for i in range(nb_points)])
    data_labels_n = np.asarray([-1 for i in range(nb_points)])
    data_labels = np.hstack((data_labels_n, data_labels_p))

    return (data_desc, data_labels)
    
def genere_train_test(desc_set, label_set, n_pos, n_neg):
	""" permet de générer une base d'apprentissage et une base de test
	desc_set: ndarray avec des descriptions
	label_set: ndarray avec les labels correspondants
	n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
	n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
	Hypothèses:
	- desc_set et label_set ont le même nombre de lignes)
	- n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
	"""

	# On commence par mélanger les données:
	desc_set, label_set = shuffle(desc_set, label_set)

	# On récupère les indices des exemples de classe -1:
	indices_negatifs = np.where(label_set == -1)[0]
	# On récupère les indices des exemples de classe +1:
	indices_positifs = np.where(label_set == +1)[0]

	# On tire aléatoirement n_pos indices parmi les indices des exemples de classe +1:
	indices_positifs_train = random.sample(list(indices_positifs), n_pos)
	# On tire aléatoirement n_neg indices parmi les indices des exemples de classe -1:
	indices_negatifs_train = random.sample(list(indices_negatifs), n_neg)

	# On récupère les indices des exemples de classe -1 qui ne sont pas dans la base d'apprentissage:
	indices_negatifs_test = list(set(indices_negatifs) - set(indices_negatifs_train))
	# On récupère les indices des exemples de classe +1 qui ne sont pas dans la base d'apprentissage:
	indices_positifs_test = list(set(indices_positifs) - set(indices_positifs_train))

	# On récupère les exemples de la base d'apprentissage:
	desc_train = np.concatenate((desc_set[indices_negatifs_train], desc_set[indices_positifs_train]))
	label_train = np.concatenate((label_set[indices_negatifs_train], label_set[indices_positifs_train]))

	# On récupère les exemples de la base de test:
	desc_test = np.concatenate((desc_set[indices_negatifs_test], desc_set[indices_positifs_test]))
	label_test = np.concatenate((label_set[indices_negatifs_test], label_set[indices_positifs_test]))

	return (desc_train, label_train), (desc_test, label_test)


# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
   #TODO: A Compléter  
    # Affichage de l'ensemble des exemples :
    # Extraction des exemples de classe -1:
    negatifs =desc[labels == -1]
    # Extraction des exemples de classe +1:
    positifs = desc[labels == +1]
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1
    return

# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

def crossval(X, Y, n_iterations, iteration):

    taille_test = len(X) // n_iterations
    i_test = slice(iteration * taille_test, (iteration + 1) * taille_test)
    i_train = np.concatenate([np.arange(0, iteration * taille_test),
                                np.arange((iteration + 1) * taille_test, len(X))])
    Xtest = X[i_test]
    Ytest = Y[i_test]
    Xapp = X[i_train]
    Yapp = Y[i_train]
    
    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
        # Séparation par classe
    Xp = {}
    Yp = {}
    for i in range(len(X)):
        if Y[i] not in Xp:
            Xp[Y[i]] = []
            Yp[Y[i]] = []
        Xp[Y[i]].append(X[i])
        Yp[Y[i]].append(Y[i])
    
    # Calculer le nombre d'exemples pour chaque ensemble de test
    n_test = {}
    for k, v in Xp.items():
        n_test[k] = int(len(v) / n_iterations)
    
    # Extraire les exemples pour l'ensemble de test courant
    Xtest = []
    Ytest = []
    for k, v in Xp.items():
        start = iteration * n_test[k]
        end = start + n_test[k]
        Xtest += v[start:end]
        Ytest += Yp[k][start:end]
        
    # Extraire les exemples pour l'ensemble d'apprentissage
    Xapp = []
    Yapp = []
    for k, v in Xp.items():
        Xapp += v[:iteration*n_test[k]] + v[(iteration+1)*n_test[k]:]
        Yapp += Yp[k][:iteration*n_test[k]] + Yp[k][(iteration+1)*n_test[k]:]
    
    # Convertir en numpy arrays
    Xapp = np.array(Xapp)
    Yapp = np.array(Yapp)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
   
    return Xapp, Yapp, Xtest, Ytest
    

