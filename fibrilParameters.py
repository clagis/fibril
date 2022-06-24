#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 09 08:22:57 2022

@author: clement
"""

#=============================================================================
#
###     Libraries
#
#=============================================================================
#=============================================================================

import math
import pandas as pd
import numpy as np
import Bio.PDB

from scipy.spatial import ConvexHull
from GJK import gjk

#=============================================================================
#=============================================================================
#
###     Basic Functions
#
#=============================================================================


def euclidean_distance(point1, point2):
    """
    Return the Euclidean distance between two points in 3D space.
    
    Parameters:
    -----------
    point1: np.array()
        A 3D points.
    point2: np.array()
        A 3D points.
    
    Returns:
    --------
    : float
        The Euclidean distance between point1 and point2.
    """
    
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 +
                (point1[2] - point2[2])**2)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    alpha = math.cos(theta / 2.0)
    beta, gamma, delta = -axis * math.sin(theta / 2.0)
    aa, bb, gg, dd = alpha * alpha, beta * beta, gamma * gamma, delta * delta
    bg, ad, ag, ab, bd, gd = beta * gamma, alpha * delta, alpha * gamma, alpha * beta, beta * delta, gamma * delta
    return np.array([[aa + bb - gg - dd, 2 * (bg + ad), 2 * (bd - ag)],
                     [2 * (bg - ad), aa + gg - bb - dd, 2 * (gd + ab)],
                     [2 * (bd + ag), 2 * (gd - ab), aa + dd - bb - gg]])


def convex_hull(data):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    B: np.array()
        A square matrix of column vectors.
    """
    
    x_list = data['x'].to_list()
    y_list = data['y'].to_list()
    z_list = data['z'].to_list()
    
    points = np.array([[x_list[i],
                        y_list[i],
                        z_list[i]] for i in range(len(data))])
    
    hull = ConvexHull(points)
    
    return hull.points[hull.vertices]


def base(vector):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    Base: np.array()
        A square matrix of column vectors.
    """
    
    Base = np.zeros((3,3))
    Base[:,0] = vector.T
    idx_i = np.where(abs(vector) == abs(vector).min())[0][0]
    idx_j = np.where(abs(vector) == abs(vector).max())[0][0]
    Base[idx_i,1] = 1
    idx_k = [x for x in range(3) if x not in [idx_i, idx_j]][0]
    Base[idx_k,2] = 1
    
    return Base


def modified_gramschmidt(oldBase):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure.
    
    Parameters:
    -----------
    oldBase: np.array()
        A square matrix of column vectors.
    
    Returns:
    --------
    newBase: np.array()
        A square matrix of orthonormal column vectors
    """
    
    # assuming A is a square matrix
    dim = oldBase.shape[0]
    newBase = np.zeros(oldBase.shape, dtype=oldBase.dtype)
    for j in range(0, dim):
        q = oldBase[:,j]
        for i in range(0, j):
            rij = np.dot(q, newBase[:,i])
            q = q - rij*newBase[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            newBase[:,j] = q/rjj
    if np.linalg.det(newBase) < 0:
        newBase[dim-1,:] *= -1
    return newBase


def renamef(elem):
    """
    Return the new name of an element.
    
    Parameters:
    -----------
    elem: list element
    
    Returns:
    --------
    : str
        The new name of an element.
    """
    
    sourceList = ['A', 'B', 'A-2', 'B-2']
    targetList = ['A', 'B', 'C', 'D']
    
    return targetList[sourceList.index(elem)]


def color(AminoAcid):
    """
    Return a number encoding a color.
    
    Parameters:
    -----------
    x: str
        An amino acid three letters code.
    
    Returns:
    --------
    : int
        An integer to encode color.
    """
    
    color_dict = {'HIS': 0, 'VAL': 1, 'GLU': 2, 'TYR': 3, 'SER': 4, 'ARG': 5,
                  'ILE': 6, 'THR': 7, 'LYS': 8, 'PHE': 9, 'GLN': 10, 'PRO': 11,
                  'LEU': 12, 'GLY': 13, 'ALA': 14, 'ASN': 15, 'ASP': 16, 'MET': 17,
                  'TRP': 18, 'CYS': 19, 'SEC': 20}
    
    return color_dict[AminoAcid]


#=============================================================================
#
###     Interaction with PDB
#
#=============================================================================


def atom3d(PDBcode, rename = True):
    """
    Return a dataframe from a PDB code.
    
    Parameters:
    -----------
    code: str
        The code for a PDB file.
    
    Returns:
    --------
    df: pd.DataFrame()
        A dataframe containing the model, chain, residue number, residue name
        and coordinates from the PDB file.
    """
    
    pdbl = Bio.PDB.PDBList() # recruts the retrieval class
    pdbl.retrieve_pdb_file(PDBcode,file_format='mmCif')
    
    parser = Bio.PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(PDBcode,'./' + PDBcode[1:3].lower() + '/' + PDBcode.lower() + '.cif')
    
    Atom = []
    for model in structure:
        for chain in model:
            for residue in chain:
                tags = residue.id
                if tags[0] != " ":
                    # The residue is a heteroatom
                    pass
                else:
                    num = tags[1]
                    resi = residue.get_resname()
                    for atom in residue:
                        Atom += [[model.get_id(), chain.get_id(), num, resi] + atom.get_coord().tolist()]
        
    df = pd.DataFrame(Atom, columns=['model', 'chain', 'residue','residueName','x','y','z'])
    
    if rename:
        S = df['chain'].to_list()
        R = [renamef(x) for x in S]
    
    df['chain'] = R
    return df


#=============================================================================
#
###     Protein Manipulation
#
#=============================================================================


def protein_centering(data):
    """
    Return a dataframe whose mean coordinates are those of the origin.
    
    Parameters:
    -----------
    data: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    
    Returns:
    --------
    data: pd.DataFrame()
        A translated dataframe.
    """
    
    xCM, yCM, zCM = data['x'].mean(), data['y'].mean(), data['z'].mean()
    data['x'] = data['x'].apply(lambda xyz: xyz - xCM)
    data['y'] = data['y'].apply(lambda xyz: xyz - yCM)
    data['z'] = data['z'].apply(lambda xyz: xyz - zCM)
    
    return data


def protein_centering_parameters(data):
    """
    Return the coordinates of the barycenter of the dataframe.
    
    Parameters:
    -----------
    data: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    
    Returns:
    --------
    xCM, yCM, zCM: float
        The coordinates of the barycenter of the dataframe.
    """
    
    xCM, yCM, zCM = data['x'].mean(), data['y'].mean(), data['z'].mean()
    
    return xCM, yCM, zCM


def protein_translation(data, vector):
    """
    Return a dataframe translated by a vector.
    
    Parameters:
    -----------
    data: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    vector: np.array()
        A translation vector.
    
    Returns:
    --------
    data2: pd.DataFrame()
        A translated dataframe.
    """
    
    data2 = data.copy(deep=True)
    data2['x'] = data2['x'].apply(lambda xyz: xyz + vector[0])
    data2['y'] = data2['y'].apply(lambda xyz: xyz + vector[1])
    data2['z'] = data2['z'].apply(lambda xyz: xyz + vector[2])
    
    return data2


def protein_rotation(data, axis, angle):
    """
    Return a dataframe rotated by a certain angle around a certain axis.
    
    Parameters:
    -----------
    data: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    axis: np.array()
        A vector representing the axis around which the data will rotate.
    angle: float
        The angle of rotation in radians.
    
    Returns:
    --------
    data2: pd.DataFrame()
        A rotateded dataframe.
    """
    
    data2 = data.copy(deep=True)
    rMatrix = rotation_matrix(axis, angle)
    dataCoords = np.asarray(data[['x','y','z']])
    data2[['x','y','z']] = np.dot(rMatrix, dataCoords.T).T
    
    return data2


def make_translation(data1, data2, vector, *penetration):
    """
    Return a dataframe, a translated dataframe and the minimal vector such that
    the convexhulls of both dataframe are adjacent but not intersecting
    (using GJK algorithm).
    
    Parameters:
    -----------
    data1: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    data2: pd.DataFrame()
        A dataframe with at least three columns named "x", "y", "z".
    vector: np.array()
        A translation vector giving the direction along which the data must move.
    
    Returns:
    --------
    data1: pd.DataFrame()
        The original dataframe data1.
    protein2: pd.DataFrame()
         The dataframe data2 translated by vect2.   
    vect2: pd.array()
        The minimal translation vector.
    """
    
    proteinHull = convex_hull(data1)
    proteinHull2 = convex_hull(data2)
    Vect = np.tile(vector*100,(len(proteinHull),1))
    proteinHull2 = proteinHull2 + Vect
    I, dist = gjk(proteinHull, proteinHull2)
    #print(I, dist)
    if penetration:
        vect2 = vector*(100-dist/np.linalg.norm(vector))*(1.0-penetration[0])
    else:
        vect2 = vector*(100-dist/np.linalg.norm(vector))
    protein2 = protein_translation(data2, vect2)

    return data1, protein2, vect2


def make_rotation(data, axis, angle):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    B: np.array()
        A square matrix of column vectors.
    """
    
    rMatrix = rotation_matrix(axis, angle)
    
    data2 = data.copy(deep=True)
    dataCoords = np.asarray(data2[['x', 'y', 'z']])
    data2[['x', 'y', 'z']] = np.matmul(rMatrix, dataCoords.T).T
    
    return data2


def metatile(data, group, *penetration):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    B: np.array()
        A square matrix of column vectors.
    """
    
    if group == 'p1':
        return data
    elif group == 'p2':
        data, data2, vect = make_rotation(data, np.pi, 100, *penetration)
        data['pcolor'] = ['green' for i in range(len(data))]
        data2['pcolor'] = ['red' for i in range(len(data))]
        return pd.concat([data,data2],ignore_index=True)
    elif group == 'p3':
        data, data2, vect = make_rotation(data, -np.pi*(2/3), 100, *penetration)
        data, data3, vect2 = make_rotation(data, np.pi*(2/3), 100, *penetration)
        data['pcolor'] = ['green' for i in range(len(data))]
        data2['pcolor'] = ['red' for i in range(len(data))]
        data3['pcolor'] = ['blue' for i in range(len(data))]
        return pd.concat([data,data2,data3],ignore_index=True)
    elif group == 'p4':
        data, data2, vect = make_rotation(data, np.pi*(1/2), 100, *penetration)
        data, data3, vect2 = make_rotation(data, -np.pi*(1/2), 100, *penetration)
        data['pcolor'] = ['green' for i in range(len(data))]
        data2['pcolor'] = ['red' for i in range(len(data))]
        data3['pcolor'] = ['blue' for i in range(len(data))]
        return pd.concat([data,data2,data3],ignore_index=True)
    elif group == 'p6':
        data, data2, vect = make_rotation(data, np.pi*(1/3), 100,*penetration)
        data, data3, vect2 = make_rotation(data, -np.pi*(1/3), 100, *penetration)
        data['pcolor'] = ['green' for i in range(len(data))]
        data2['pcolor'] = ['red' for i in range(len(data))]
        data3['pcolor'] = ['blue' for i in range(len(data))]
        df = pd.concat([data,data2,data3],ignore_index=True)
        df, df2, vect = make_rotation(df, np.pi, *penetration)
        return pd.concat([df,df2],ignore_index=True)


#=============================================================================
#
###     Fibril Growth Functions
#
#=============================================================================


def growth_test(translationVector, axisVector, scale):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    B: np.array()
        A square matrix of column vectors.
    """
    
    normVector = np.linalg.norm(translationVector)
    alpha = np.arccos(min(scale/normVector, 1))
    rMatrix = rotation_matrix(axisVector, alpha)
    
    growthVector = np.dot(rMatrix, translationVector)
    normgrowthVector = np.linalg.norm(growthVector)
    return growthVector/normgrowthVector


def growth_parameters(data, translationVector, diameter, threshold = 'default', tolerance = 0.05):
    """
    Return a base from a vector.
    
    Parameters:
    -----------
    u: np.array()
        A vector.
    
    Returns:
    --------
    B: np.array()
        A square matrix of column vectors.
    """
    
    def get_first_vector(vector):
        """
        Return a base from a vector.
        
        Parameters:
        -----------
        u: np.array()
            A vector.
        
        Returns:
        --------
        B: np.array()
            A square matrix of column vectors.
        """
        
        idx_minNonNull = np.where(abs(vector) > 0, abs(vector), np.Infinity).argmin()
        newVector = np.array([0,0,0])
        newVector[idx_minNonNull] += 1
        return newVector
    
    growthVector = get_first_vector(translationVector)
    
    def get_ratio(data, translationVector, growthVector, diameter, threshold):
        """
        Return a base from a vector.
        
        Parameters:
        -----------
        u: np.array()
            A vector.
        
        Returns:
        --------
        B: np.array()
            A square matrix of column vectors.
        """
        
        dataCoords = np.asarray(data[['x', 'y', 'z']])
        dataCoords_growthVector = np.dot(dataCoords, growthVector)
        dmax, dmin = max(dataCoords_growthVector), min(dataCoords_growthVector)
        growthDiameter = dmax - dmin
        
        growthScale = abs(np.dot(translationVector, growthVector))
        t_prime = translationVector - growthScale*growthVector
        w = np.cross(t_prime, growthVector)
        dataCoords_w = np.dot(dataCoords, w/np.linalg.norm(w))
        tmax, tmin = max(dataCoords_w), min(dataCoords_w)
        thick = tmax - tmin
        
        alpha = 2*np.linalg.norm(t_prime)/(diameter - thick)
        
        if threshold == 'below':
            k = np.floor(2*np.pi/alpha)
        elif threshold == 'above':
            k = np.ceil(2*np.pi/alpha)
        elif threshold == 'default':
            k = 2*np.pi/alpha
        
        return growthScale, growthDiameter, k, thick, w
    
    g, d, k, t, w = get_ratio(data, translationVector, growthVector, diameter, threshold)
    counter = 0
    while (((1-tolerance) > k*g/d) or (k*g/d > (1.0 +tolerance))) and (counter < 100):
        if k != 0:
            s = d / k
            #print(u, w, s)
            growthVector = growth_test(translationVector, w, s)
            g, d, k, t, w = get_ratio(data, translationVector, growthVector, diameter, threshold)
            counter += 1
        else:
            break
    
    if k != 0:
        return growthVector, t
    else:
        pass
        #print("Impossible case")


#=============================================================================
#
###     Main Functions
#
#=============================================================================

        
def get_parameters(pdbFile: str, interface1: list, interface2: list, penetration: float, group: str = 'p1'):
    """
    Return the data (d), the translated data (t) and the translation vector u such
    that the intersection of (d) and (t) is null and u is minimal.

    Parameters:
    -----------
    pdbFile: PDB file either cif or pdb1
        Data must contain at least three columns named x, y and z.
    interface1: list
        Must be of length 3, with model, chain and residue lists.
    interface2: list
        Must be of length 3, with model, chain and residue lists.
    penetration: float
        Must be between 0 and 1.
    group: str
        Must be a crystallographic group name.
        For now only p1 work.
        
    Returns:
    --------
    dt0: pandas dataframe
        Original PDB file in pandas format, centered coordinates.
    dt0: pandas dataframe
        Translated PDB file in pandas format.
    u: np.array
        Translation vector.
    """
    
    L1 = interface1[2]
    L2 = interface2[2]

    df = atom3d(pdbFile)
    df = df.loc[df['chain'].isin(['A', 'B'])]
    df = protein_centering(df)
    df['color'] = df['residueName'].apply(color)

    df0 = df.loc[(df['model'] == interface1[0]) & (df['chain'] == interface1[1])]
    df1 = df.loc[(df['model'] == interface2[0]) & (df['chain'] == interface2[1])]

    x0, y0, z0 = df0.loc[df0['residue'].isin(L1)][['x', 'y', 'z']].mean()
    x1, y1, z1 = df1.loc[df1['residue'].isin(L2)][['x', 'y', 'z']].mean()

    v = np.array([x1-x0, y1-y0, z1-z0])

    mf = metatile(df, group, penetration)
    dt0, dt1, u = make_translation(mf, mf, v)
    return dt0, dt1, u
  
  
def growth_base(v):
     return modified_gramschmidt(base(v))
 

#=============================================================================
#
###     Test
#
#=============================================================================


# Molecule to use
mol = '3kgs-assembly1'

# Interface
chain1 = 'B'
minResi1 = 28
maxResi1 = 36
L1 = [i for i in range(minResi1, maxResi1+1)]

# Complementary Interface
chain2 = 'A'
minResi2 = 28
maxResi2 = 36
L2 = [i for i in range(minResi2, maxResi2+1)]

# Get data, translated data and translation vector
df, translated_df, tVector = get_parameters(mol, [0, chain1, L1], [0, chain2, L2], 0.0)

# Get the centering parameters
df2 = atom3d(mol)
## Use only in case of restriction of protein to some chains
# df2 = df2.loc[df2['chain'].isin([chain1, chain2])]
x_c, y_c, z_c = protein_centering_parameters(df2)

# Parameters for threshold, minimal diameter, maximal diamater and step
threshold = 'above'
minDiameter = 70
maxDiameter = 130
step = 10

parametersList = []

for diameter in range(minDiameter, maxDiameter+step, step):
    gVector, thickness = growth_parameters(df, tVector, diameter, threshold)
    if gVector is not None:
        # Make an orthonormal basis
        B = growth_base(gVector)
        # Add to the parametersList for easier usage
        parametersList += [[diameter, thickness, B]]

