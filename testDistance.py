#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:06:35 2022

@author: blaise
"""

#=============================================================================
#
###     Libraries
#
#=============================================================================

import pandas as pd
import numpy as np

from sklearn.neighbors import KDTree
from fibrilParameters import get_parameters, protein_translation, protein_rotation, base, modified_gramschmidt, growth_parameters

#=============================================================================
#=============================================================================
#
###     Basic functions
#
#=============================================================================

def kd_neighbours(data, data2, KNN = 1):
    """
    Return the data with a new column containing the points that are least in a
    sphere of radius searchRadius around each point of the data.

    Parameters:
    -----------
    data: pandas dataframe
        data must contain at least three columns named x, y and z.
    searchRadius: float
        Radius of the sphere.

    Returns:
    --------
    data2: pandas dataframe
        A new dataframe with the neighbours added for each point.
    """
    
    tree = KDTree(data[['x','y','z']], leaf_size=30, metric='euclidean')
    dist, ind = tree.query(data2[['x','y','z']], k = KNN)  
  
    
    data3 = data.copy(deep=True)
    data3['indexNearest'] = list(ind)
    data3['distNearest'] = list(dist)
    data3['nearest'] = data3['indexNearest'].apply(lambda x: [data.iloc[elem][['chain','residue']].to_list() for elem in x])
    
    L = []
    for k in range(len(data3)):
        L += [sorted([[data3.iloc[k]['chain'], data3.iloc[k]['residue']]] + data3.iloc[k]['nearest'])]
    data3['nearestP'] = L
    data3['nearestPair'] = data3['nearestP'].apply(lambda x: str(x[0][0]) + '_' + str(x[0][1]) + '-' + str(x[1][0]) + "_" + str(x[1][1]))
    # data3['chain2'] = data3['nearest'].apply(lambda x: x[0][0])
    # data3['residue2'] = data3['nearest'].apply(lambda x: x[0][1])
    # data3.drop(['nearest'], axis = 1, inplace = True)
    # data3.drop(['indexNearest'], axis = 1, inplace = True)
    data3['index'] = [i for i in range(len(data3))]
    return data3


def kd_neighbours_radius(data, data2, radius = 5.0):
    """
    Return the data with a new column containing the points that are least in a
    sphere of radius searchRadius around each point of the data.

    Parameters:
    -----------
    data: pandas dataframe
        data must contain at least three columns named x, y and z.
    searchRadius: float
        Radius of the sphere.

    Returns:
    --------
    data2: pandas dataframe
        A new dataframe with the neighbours added for each point.
    """
    
    tree = KDTree(data[['x','y','z']], leaf_size=1, metric='euclidean') 
    ind, dist = tree.query_radius(data2[['x','y','z']], r = radius, return_distance=True)
  
    
    data3 = data.copy(deep=True)
    data3['indexNearest'] = list(ind)
    # data3['distNearest'] = list(dist)
    data3['nearest'] = data3['indexNearest'].apply(lambda x: [data2.iloc[elem][['chain','residue']].to_list() for elem in x])
    data3['nearestDist'] = list(dist)
    L = []
    
    for i in range(len(data3)):
        M = data3.iloc[i]['nearest']
        N = []
        for elem in M:
            w = str(elem[0]) + '_' + str(elem[1])
            N += [w]
        L += [sorted(list(set(N)))]
    
    data3['nearestAA'] = L
    
    return data3


def clean(data):
    K = []
    for e1 in data['model'].unique():
        for e2 in data['chain'].unique():
            for e3 in data['residue'].unique():
                df = data.loc[(data['model'] == e1) & (data['chain'] == e2) & (data['residue'] == e3)][['residueName', 'nearest']]
                R = []
                for el in list(df['nearest']):
                    R += el
                    # if el != []:
                    #     print(el, R)
                K += [[e1, e2, e3, df.iloc[0]['residueName'], R]]
    
    data2 = pd.DataFrame(K, columns=['model', 'chain', 'residue', 'residueName', 'nearest'])
    
    A = []
    for i in range(len(data2)):
        L = data2.iloc[i]['nearest']
        S = L.copy()
        #print(S)
        c = 0
        for j in range(len(L)):
            if L[j] in L[:j]:
                S.pop(j-c)
                c += 1
            else:
                c += 0
        S2 = []
        for elem in S:
            S2 += [elem + [L.count(elem)]]
        A += [S2]
   
    data2['neighbours'] = A
    data2.drop(['nearest'], axis=1, inplace=True) 
    return data2


#=============================================================================
#
###     Main function
#
#=============================================================================

def kd_test(data, u, B, diameter, searchRadius):
    vectZ = B[:,0]
    s = np.dot(u,vectZ)
    vectX = B[:,1]
    scalP = np.dot(B[:,1],u)
    scalQ = np.dot(B[:,2],u)
    A = np.asarray(data[['x', 'y', 'z']])
    A_prime = np.dot(A,B[:,1])
    thickness = A_prime.max() - A_prime.min()
    radius = (diameter - thickness)/2
    
    angle = np.linalg.norm([scalP,scalQ])/radius
    data = protein_translation(data, vectX*radius)    
    
    data2 = protein_rotation(data, vectZ, angle)
    data2 = protein_translation(data2, vectZ*s)
    tree2 = KDTree(data2[['x','y','z']], leaf_size=10, metric='euclidean') 
    ind2, dist2 = tree2.query_radius(data[['x','y','z']], r = searchRadius, return_distance=True)
    
    n = np.floor(2*np.pi/angle)
    print('n, angle: {}, {}; n+1, angle: {}, {}'.format(n, np.round(n*angle*180/np.pi,1), n+1, np.round((n+1)*angle*180/np.pi,1)))
    data3 = protein_rotation(data, vectZ, n*angle)
    data3 = protein_translation(data3, vectZ*s*n)
    tree3 = KDTree(data3[['x','y','z']], leaf_size=10, metric='euclidean') 
    ind3, dist3 = tree3.query_radius(data[['x','y','z']], r = searchRadius, return_distance=True)
    
    data4 = protein_rotation(data, vectZ, (n+1)*angle)
    data4 = protein_translation(data4, vectZ*s*(n+1))
    tree4 = KDTree(data4[['x','y','z']], leaf_size=10, metric='euclidean') 
    ind4, dist4 = tree4.query_radius(data[['x','y','z']], r = searchRadius, return_distance=True)
    
    
    dfi1 = data[['model', 'chain', 'residue', 'residueName']].reset_index()
    dfi1['indexNearest'] = [list(x) for x in ind2]
    dfi1['distNearest'] = [list(x) for x in dist2]
    dfi1['nearest'] = dfi1['indexNearest'].apply(lambda x: [data2.iloc[elem][['chain','residue','residueName']].to_list() for elem in x])
    
    dfi3 = data[['model', 'chain', 'residue', 'residueName']].reset_index()
    dfi3['indexNearest'] = [list(x) for x in ind3]
    dfi3['distNearest'] = [list(x) for x in dist3]
    dfi3['nearest'] = dfi3['indexNearest'].apply(lambda x: [data3.iloc[elem][['chain','residue','residueName']].to_list() for elem in x])
    
    dfi2 = data[['model', 'chain', 'residue', 'residueName']].reset_index()
    dfi2['indexNearest'] = [list(x) for x in ind4]
    dfi2['distNearest'] = [list(x) for x in dist4]
    dfi2['nearest'] = dfi2['indexNearest'].apply(lambda x: [data4.iloc[elem][['chain','residue','residueName']].to_list() for elem in x])
    
    return dfi1, dfi2, dfi3


#=============================================================================
#
###     Test
#
#=============================================================================

mol = '3kgs-assembly1'

# Interface
minResi1 = 114
maxResi1 = 124
L1 = [i for i in range(minResi1, maxResi1+1)]

# Complementary Interface
minResi2 = 114
maxResi2 = 124
L2 = [i for i in range(minResi2, maxResi2+1)]

a, b, c = get_parameters(mol, [0, 'B', L1], [0, 'A', L2], 0.0)
treshold = 'default'
minDiameter = 70
maxDiameter = 130
step = 10

parametersList = []

for value in range(minDiameter, maxDiameter+step, step):
    v, t = growth_parameters(a, c, value, treshold)
    if v is not None:
        B = modified_gramschmidt(base(v))
        parametersList += [[value, t, B]]
        d, e, f = kd_test(a, c, B, value, 5.0)
        d2, e2, f2 = clean(d), clean(e), clean(f)
        d2['interface'] = ['i1']*len(d2)
        e2['interface'] = ['i2']*len(e2)
        f2['interface'] = ['i3']*len(f2)
        
        # df = pd.concat([d2,e2,f2])
    #     if L1 == L2:
    #         name = '{}-{}_d'.format(L1[0], L1[-1]) + str(value) + 'treshold-{}'.format(treshold) + '.csv'
    #     else:
    #         name = 'dimer_{}-{}_{}-{}_d'.format(L1[0], L1[-1], L2[0], L2[-1]) + str(value) + 'treshold-{}'.format(treshold) + '.csv'
    #     df.to_csv(name, sep = ';', index = None)
    # else:
    #     print("I skipped case diameter={}".format(value))
