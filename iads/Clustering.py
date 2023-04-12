# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import scipy.cluster.hierarchy
from iads import utils as ut




def normalisation(dataframe):
    for c in dataframe:
        mini = min(dataframe[c])
        maxi = max(dataframe[c])
        dataframe[c] = dataframe[c] - mini
        dataframe[c] = dataframe[c]/ maxi
    
    return dataframe


def dist_euclidienne(e1, e2):
    e = e1 - e2
    return np.linalg.norm(e)


def centroide(data):
    X = np.array(data)
    c = np.mean(X, axis=0)
    
    return c


def dist_centroides(data1, data2):
    c1 = centroide(data1)
    c2 = centroide(data2)
    return dist_euclidienne(c1, c2)


def initialise_CHA(DF):
    partition = dict()
    i=0
    for i in range(len(DF)):
        partition[i] = [i]
        i += 1
        
    return partition


def fusionne(DF, P0, verbose=False):
    dist_min = np.inf
    kc1 = -1
    kc2 = -1

    for k1, v1 in P0.items():
        for k2, v2 in P0.items():
            if k1 == k2:
                continue
            l1 = DF.iloc[v1]
            l2 = DF.iloc[v2]
            dist = dist_centroides(l1, l2)
            if dist < dist_min:
                dist_min = dist
                kc1 = k1
                kc2 = k2

    P1 = dict(P0)
    if kc1 != -1:
        del P1[kc1]
        del P1[kc2]
        P1[max(P0) + 1] = P0[kc1] + P0[kc2]

        if verbose:
            print("Distance minimale trouvée entre [" + str(kc1) + ", " + str(kc2) + "] = " + str(dist_min))

    return P1, kc1, kc2, dist_min



def CHA_centroid(DF, verbose=False, dendrogramme=False):
    dico = initialise_CHA(DF)
    res = []
    for i in range(len(DF)):
        dico, kc1, kc2, dist_min = fusionne(DF, dico, verbose)
        nbElem = len(dico[max(dico.keys())])
        res.append([kc1, kc2, dist_min, nbElem])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res[:-1], 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        # Affichage du résultat obtenu:
        plt.show()
        
    return res[:-1]


#pour clustering_hierarchique_complete
def dist_max(data1, data2):
    max_distance = -float('inf')
    for point1 in data1.itertuples(index=False):
        for point2 in data2.itertuples(index=False):
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            max_distance = max(max_distance, distance)
    return max_distance

#pour clustering_hierarchique_simple
def dist_mini(data1, data2):
    min_distance = float('inf')
    for point1 in data1.itertuples(index=False):
        for point2 in data2.itertuples(index=False):
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            min_distance = min(min_distance, distance)
    return min_distance


#pour clustering_hierarchique_average
def dist_mean(data1, data2):
    total_distance = 0
    count = 0
    for point1 in data1.itertuples(index=False):
        for point2 in data2.itertuples(index=False):
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            total_distance += distance
            count += 1
    return total_distance / count



def fusionne_generic(DF, P0, dist_func, verbose=False):
    dist_min = np.inf
    kc1 = -1
    kc2 = -1

    for k1, v1 in P0.items():
        for k2, v2 in P0.items():
            if k1 == k2:
                continue
            l1 = DF.iloc[v1]
            l2 = DF.iloc[v2]
            dist = dist_func(l1, l2)
            if dist < dist_min:
                dist_min = dist
                kc1 = k1
                kc2 = k2

    P1 = dict(P0)
    if kc1 != -1:
        del P1[kc1]
        del P1[kc2]
        P1[max(P0) + 1] = P0[kc1] + P0[kc2]

        if verbose:
            print("Distance minimale trouvée entre [" + str(kc1) + ", " + str(kc2) + "] = " + str(dist_min))

    return P1, kc1, kc2, dist_min




def clustering_hierarchique_complete(DF, verbose=False, dendrogramme=False):
    dico = initialise_CHA(DF)
    res = []
    for i in range(len(DF)):
        dico, kc1, kc2, dist_min = fusionne_generic(DF, dico, dist_max, verbose)
        nbElem = len(dico[max(dico.keys())])
        res.append([kc1, kc2, dist_min, nbElem])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res[:-1], 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        # Affichage du résultat obtenu:
        plt.show()
        
    return res[:-1]


def clustering_hierarchique_simple(DF, verbose=False, dendrogramme=False):
    dico = initialise_CHA(DF)
    res = []
    for i in range(len(DF)):
        dico, kc1, kc2, dist_min = fusionne_generic(DF, dico, dist_mini,verbose)
        nbElem = len(dico[max(dico.keys())])
        res.append([kc1, kc2, dist_min, nbElem])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res[:-1], 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        # Affichage du résultat obtenu:
        plt.show()
        
    return res[:-1]

def clustering_hierarchique_average(DF, verbose=False, dendrogramme=False):
    dico = initialise_CHA(DF)
    res = []
    for i in range(len(DF)):
        dico, kc1, kc2, dist_min = fusionne_generic(DF, dico, dist_mean, verbose)
        nbElem = len(dico[max(dico.keys())])
        res.append([kc1, kc2, dist_min, nbElem])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res[:-1], 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        # Affichage du résultat obtenu:
        plt.show()
        
    return res[:-1]



def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    if linkage == 'centroid':
        return CHA_centroid(DF,verbose,dendrogramme)
    if linkage == 'complete':
        return clustering_hierarchique_complete(DF, verbose, dendrogramme)
    if linkage == 'simple':
        return clustering_hierarchique_simple(DF, verbose, dendrogramme)
    if linkage == 'average':
        return clustering_hierarchique_average(DF, verbose, dendrogramme)


def inertie_cluster(Ens):
    distances = np.sqrt(np.sum((Ens-centroide(Ens))**2))
    return  np.sum(distances ** 2)



def init_kmeans(K,Ens):
    # Tire aléatoirement K indices uniques entre 0 et le nombre d'exemples de la base d'apprentissage
    indices = np.random.choice(len(Ens), K, replace=False)
    
    # Sélectionne les exemples correspondants aux indices tirés aléatoirement
    exemples = Ens.iloc[indices].values
    
    return exemples


def plus_proche(Exe,Centres):
    
    # Convertie l'exemple en un np.array si c'est un pandas.Series
    if isinstance(Exe, pd.Series):
        Exe = Exe.to_numpy()
        
    # Calcule la distance euclidienne entre l'exemple et chaque centre de cluster
    distances = np.linalg.norm(Centres - Exe.reshape((1, -1)), axis=1)
    
    # Trouver l'indice du centroide le plus proche de l'exemple
    res = np.argmin(distances)
    
    return res




def affecte_cluster(Base,Centres):
    # Initialiser le dictionnaire de la matrice d'affectation
    mat_affectation = {k: [] for k in range(len(Centres))}
    
    Exe = Base.values
    
    for i in range(len(Exe)):
        pproche = plus_proche(Exe[i], Centres)
        mat_affectation[pproche].append(i)
    
    return mat_affectation



def nouveaux_centroides(Base,U):
    Centroides = np.zeros((len(U), Base.shape[1]))
    for k in range(len(U)):
        # Extraire les exemples affectés au cluster k
        indices = U[k]
        exemples_k = Base.iloc[indices]
        
        # Calculer la moyenne des exemples affectés au cluster k
        Centroides[k] = np.mean(exemples_k, axis=0)
    return Centroides


def inertie_globale(Base, U):
    
    inertie_globale = 0
    for k, indices in U.items():
        # Extraire les exemples affectés au cluster k
        exemples_k = Base.iloc[indices]
        # Calculer l'inertie du cluster k
        inertie_k = inertie_cluster(exemples_k)
        # Ajouter l'inertie du cluster k à l'inertie globale
        inertie_globale += inertie_k
    return inertie_globale




def kmoyennes(K, Base, epsilon, iter_max):
    centres = init_kmeans(K, Base)
    U = dict()
    inertie_tmp = 0
    
    for i in range(iter_max):
        U = affecte_cluster(Base, centres)
        centres = nouveaux_centroides(Base, U)
        inertie = inertie_globale(Base,U)
        print("iteration " + str(i) + " Inertie : " + str(inertie) + " Difference: " + str(abs(inertie - inertie_tmp)))
        if(abs(inertie - inertie_tmp) < epsilon ):
            break
        
        inertie_tmp = inertie
    
    return (centres, U)


# Librairie pour manipuler les colormaps:
import matplotlib.cm as cm

# on transforme le colormap en couleurs utilisable par plt.scatter:
couleurs = cm.tab20(np.linspace(0, 1, 20))

def affiche_resultat(Base,Centres,Affect):
    for i in range(len(Affect)):
        exemples = (Base.iloc[Affect[i]]).values.tolist()
        for e in exemples:
            plt.scatter(e[0],e[1],color=couleurs[i])
      
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')