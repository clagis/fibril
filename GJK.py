#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:41:39 2022

@author: clement
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
# from mpl_toolkits.mplot3d import proj3d
# from matplotlib.patches import FancyArrowPatch

#=============================================================================
#
###     Basic Vector Functions
#
#=============================================================================


def subtract(vector1: list, vector2: list) -> list:
    x1, y1, z1 = vector1[0], vector1[1], vector1[2]
    x2, y2, z2 = vector2[0], vector2[1], vector2[2]
    return [x1-x2, y1-y2, z1-z2]


def centroid(polyhedra):
    x = [vertex[0] for vertex in polyhedra]
    y = [vertex[1] for vertex in polyhedra]
    z = [vertex[2] for vertex in polyhedra]
    return np.array([sum(x)/len(polyhedra), sum(y)/len(polyhedra), sum(z)/len(polyhedra)])


def normalize(vector):
    v_norm = np.linalg.norm(vector)
    if v_norm != 0:
        return vector / np.linalg.norm(vector)
    else:
        return vector


def get_support_point(polyhedra1, polyhedra2, direction):
    
    def get_furthest_point(polyhedra, direction):
        furthest_point = np.array(polyhedra[0])
        max_dot = np.dot(furthest_point, direction)
        for i in range(1, len(polyhedra)):
            current_point = np.array(polyhedra[i])
            current_dot = np.dot(current_point, direction)
            if current_dot > max_dot:
                max_dot = current_dot
                furthest_point = current_point
        return furthest_point

    fp_shape1 = get_furthest_point(polyhedra1, direction)
    fp_shape2 = get_furthest_point(polyhedra2, -direction)
    return fp_shape1 - fp_shape2


#=============================================================================
#
###     Distance Functions
#
#=============================================================================


def dist_segment(segment, point):
    P, Q = segment
    O = point
    PQ, PO, QO = np.array(Q) - np.array(P), np.array(O) - np.array(P), np.array(O) - np.array(Q)
    segmentPerp = np.cross(np.cross(PQ, PO), PQ)
    t = np.dot(PQ, PO) / np.dot(PQ, PQ)
    if t < 0:
        dist = np.linalg.norm(PO)
    elif t > 1:
        dist = np.linalg.norm(QO)
    else:
        dist = abs(np.dot(point,segmentPerp))
    
    return dist

def dist_triangle(A, B, C, P):
    # print("P : {}".format(P))
    # print("A : {}".format(A))
    # print("B : {}".format(B))
    # print("C : {}".format(C))
    
    AB, AC = np.array(B) - np.array(A), np.array(C) - np.array(A)
    PA = np.array(A) - np.array(P)
    a = np.dot(AB, AB)
    b = np.dot(AB, AC)
    c = np.dot(AC, AC)
    d = np.dot(AB, PA)
    e = np.dot(AC, PA)
    #f = np.dot(PA, PA)
    
    s = b*e - c*d
    t = b*d - a*e
    det = a*c - b*b
    
    if (s+t) < det:
        if s < 0:
            if t < 0:
                if d < 0:
                    t = 0
                    if -d >= a:
                        s = 1
                    else:
                        s = -d/a
                else:
                    s = 0
                    if e >= 0:
                        t = 0
                    elif -e >= c:
                        t = 1
                    else:
                        t = -e/c
            else:
                s = 0
                if e >= 0:
                    t = 0
                elif -e >= c:
                    t = 1
                else:
                    t = -e/c
        elif t < 0:
            t = 0
            if d >= 0:
                s = 0
            elif -d >= a:
                s = 1
            else:
                s = -d/a
        else:
            s /= det
            t /= det
    else:
        if s < 0:
            u0 = b + d
            u1 = c + e
            if u1 > u0:
                numer = u1 - u0
                denom = a - 2*b + c
                if numer >= denom:
                    s = 1
                else:
                    s = numer/denom
                t = 1 - s
            else:
                s = 0
                if u1 <= 0:
                    t = 1
                elif e > 0:
                    t = 0
                else:
                    t = -e/c
        elif t <0:
            u0 = b + e
            u1 = a + d
            if u1 > u0:
                numer = u1 - u0
                denom = a - 2*b + c
                if numer >= denom:
                    t = 1
                else:
                    t = numer/denom
                s = 1 - t
            else:
                t = 0
                if u1 <= 0:
                    s = 1
                elif d > 0:
                    s = 0
                else:
                    s = -d/a
        else:
            numer = (c + e) - (b + d)
            if numer <= 0:
                s = 0
            else:
                denom = a - 2*b + c
                if numer >= denom:
                    s = 1
                else:
                    s = numer/denom
            t = 1 - s
    
    return np.linalg.norm(A + s*AB + t*AC)


#=============================================================================
#
###     1D, 2D, 3D Case Functions
#
#=============================================================================


def line_case(simplex, d):
    B, A = simplex
    AB, AO = np.array(B) - np.array(A), -np.array(A)
    ABperp = np.cross(np.cross(AB, AO), AB)
    
    if np.dot(ABperp, AO) > 0:
        d = normalize(ABperp)
        dist = dist_segment([A, B], np.array([0,0,0]))
        return False, simplex, d, dist
    else:
        simplex = [simplex[1]]
        if np.linalg.norm(AO) == 0:
            d = normalize(AO)
            return True, simplex, d, 0.0
        else:
            d = normalize(AO)
            dist = np.linalg.norm(AO)
            return False, simplex, d, dist


def triangle_case(simplex, d):
    C, B, A = simplex
    AB, AC, AO = np.array(B) - np.array(A), np.array(C) - np.array(A), -np.array(A)
    ABC = np.cross(AB, AC)
    ACperp = np.cross(ABC, AC)
    ABperp = np.cross(AB, ABC)
    
    if np.dot(ACperp, AO) > 0:
        if np.dot(AC, AO) > 0:
            simplex = [simplex[0]] + [simplex[2]]
            d = normalize(np.cross(np.cross(AC, AO), AC))
            dist = dist_segment(simplex, np.array([0,0,0]))
            return False, simplex, d, dist
        else:
            if np.dot(AB, AO) > 0:
                simplex = [simplex[1]] + [simplex[2]]
                d = normalize(np.cross(np.cross(AB, AO), AB))
                dist = dist_segment(simplex, np.array([0,0,0]))
                return False, simplex, d, dist
            else:
                simplex = [simplex[2]]
                d = normalize(AO)
                dist = np.linalg.norm(A)
                return False, simplex, d, dist
    else:
        if np.dot(ABperp, AO) > 0:
            if np.dot(AB, AO) > 0:
                simplex = [simplex[1]] + [simplex[2]]
                d = normalize(np.cross(np.cross(AB, AO), AB))
                dist = dist_segment(simplex, np.array([0,0,0]))
                return False, simplex, d, dist
            else:
                simplex = [simplex[2]]
                d = normalize(AO)
                dist = np.linalg.norm(A)
                return False, simplex, d, dist
        else:
            if np.dot(ABC, AO) > 0:
                d = normalize(ABC)
                return False, simplex, d, np.dot(d,AO)
            else:
                if np.dot(ABC, AO) < 0:
                    simplex = [simplex[1]] + [simplex[0]] + [simplex[2]]
                    d = normalize(-ABC)
                    dist = dist_triangle(A, B, C, np.array([0,0,0]))
                    return False, simplex, d, np.dot(d,AO)
                else:
                    d = normalize(ABC)
                    return True, simplex, d, 0.0

def tetrahedral_case(simplex, d):
    D, C, B, A = simplex
    AB, AC, AD, AO = np.array(B) - np.array(A), np.array(C) - np.array(A), np.array(D) - np.array(A), -np.array(A)
    ABCperp = np.cross(AB, AC)
    ACDperp = np.cross(AC, AD)
    ADBperp = np.cross(AD, AB)
    
    if np.dot(ABCperp, AO) > 0:
        if np.dot(ACDperp, AO) > 0:
            if np.dot(AC,AO) > 0:
                simplex = [simplex[1]] + [simplex[3]]
                d = normalize(np.cross(np.cross(AC, AO), AC))
                dist = dist_segment(simplex, np.array([0,0,0]))
                return False, simplex, d, dist
            else:
                simplex = [simplex[3]]
                dist = np.linalg.norm(A)
                return False, simplex, d, dist
        else:
            if np.dot(ADBperp, AO) > 0:
                if np.dot(AB,AO) > 0:
                    simplex = [simplex[2]] + [simplex[3]]
                    d = normalize(np.cross(np.cross(AB, AO), AB))
                    dist = dist_segment(simplex, np.array([0,0,0]))
                    return False, simplex, d, dist
                else:
                    simplex = [simplex[3]]
                    dist = np.linalg.norm(A)
                    return False, simplex, d, dist
            else:
                simplex = simplex[1:]
                dist = dist_triangle(A, B, C, np.array([0,0,0]))
                return False, simplex, d, np.dot(d,AO)
    else:
        if np.dot(ACDperp, AO) > 0:
            if np.dot(ADBperp, AO) > 0:
                if np.dot(AD,AO) > 0:
                    simplex = [simplex[0]] + [simplex[3]]
                    d = normalize(AD)
                    dist = dist_segment(simplex, np.array([0,0,0]))
                    return False, simplex, d, dist
                else:
                    simplex = [simplex[3]]
                    dist = np.linalg.norm(A)
                    return False, simplex, d, dist
            else:
                simplex = [simplex[0]] + [simplex[1]] + [simplex[3]]
                d = normalize(ACDperp)
                dist = dist_triangle(A, C, D, np.array([0,0,0]))
                return False, simplex, d, dist
        else:
            if np.dot(ADBperp, AO) > 0:
                if (np.dot(AD, AO) > 0) and (np.dot(AB, AO) > 0):
                    simplex = [simplex[0]] + [simplex[2]] + [simplex[3]]
                    d = normalize(ADBperp)
                    dist = dist_triangle(A, B, D, np.array([0,0,0]))
                    return False, simplex, d, dist
                else:
                    simplex = [simplex[3]]
                    d = normalize(AO)
                    dist = np.linalg.norm(A)
                    return False, simplex, d, dist
            else:
                d = normalize(AO)
                dist = np.linalg.norm(A)
                return True, simplex, d, dist


#=============================================================================
#
###     Help Functions
#
#=============================================================================


def handle_simplex(simplex, d):
    # L = ["line", "triangle", "tetrahedra"]
    # print(L[len(simplex)-2])
    if len(simplex) == 2:  
        return line_case(simplex, d)
    elif len(simplex) == 3:
        return triangle_case(simplex, d)
    elif len(simplex) == 4:
        return tetrahedral_case(simplex, d)


def handle_distance(simplex, P):
    # L = ["line", "triangle", "tetrahedra"]
    # print(L[len(simplex)-2])
    if len(simplex) == 2:  
        return dist_segment(simplex, P)
    elif len(simplex) == 3:
        C, B, A = simplex
        #print(simplex)
        return dist_triangle(A, B, C, P)
    elif len(simplex) == 4:
        D, C, B, A = simplex
        return min(dist_triangle(A, B, C, P), dist_triangle(A, B, D, P), dist_triangle(A, C, D, P), dist_triangle(B, C, D, P))
  

#=============================================================================
#
###     Main Function
#
#=============================================================================

      
def gjk(polyhedra1, polyhedra2):
    d = normalize(centroid(polyhedra2) - centroid(polyhedra1))
    simplex = [get_support_point(polyhedra1, polyhedra2, d)]
    d *= -1
    dist = -1
    
    while True:
        A = get_support_point(polyhedra1, polyhedra2, d)
        simplex += [A]
        if np.dot(A,d) < 0:
            return False, handle_distance(simplex, np.array([0,0,0]))
        if handle_simplex(simplex, d)[0]:
            return True, 0.0
        else:
            simplex, d, dist = handle_simplex(simplex, d)[1:]
            

#=============================================================================
#
###     Test Function
#
#=============================================================================


def test(polyhedra1, polyhedra2):
    d = normalize(centroid(polyhedra2) - centroid(polyhedra1))
    simplex = [get_support_point(polyhedra1, polyhedra2, d)]
    
    V = [list(get_support_point(polyhedra1, polyhedra2, d))]
    d *= -1
    dist = [np.linalg.norm(V[0])]
    
    while True:
        A = get_support_point(polyhedra1, polyhedra2, d)
        V += [list(A)]
        if np.dot(A,d) < 0:
            print(simplex + [A])
            return dist
        simplex += [A]
        if handle_simplex(simplex, d)[0]:
            return True
        else:
            simplex, d, neo_dist = handle_simplex(simplex, d)[1:]
            dist += [neo_dist]
            n = len(V)
            W = np.array(V)
            W = np.append(W,[[i/n] for i in range(n)], axis = 1)
            fig = plt.figure()
            ax = p3.Axes3D(fig)
            sc = ax.scatter(polyhedra1[:,0],polyhedra1[:,1],polyhedra1[:,2],
                            color = 'red',
                            alpha=0.8)
            sc = ax.scatter(polyhedra2[:,0],polyhedra2[:,1],polyhedra2[:,2],
                            color = 'blue',
                            alpha=0.8)
            sc = ax.scatter(W[:,0],W[:,1],W[:,2],
                            c = W[:,3],
                            cmap = "cool",
                            alpha=0.8)

#=============================================================================
#
###     Test
#
#=============================================================================
        
C = np.array([[0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [1,1,1]],dtype="float64")

P = np.array([[0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1]],dtype="float64")

C2 = np.array([[0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [1,1,1]],dtype="float64")

T = np.ones((8,3))

#P += T*1.8
C2 += T*2

# fig = plt.figure()
# ax = p3.Axes3D(fig)
#ax.set_axis_off()
# sc1 = ax.scatter(C[:,0],C[:,1],C[:,2],
#                 color = 'red',
#                 alpha=0.8)
# sc2 = ax.scatter(P[:,0],P[:,1],P[:,2],
#                 color = 'blue',
#                 alpha=0.8)




    