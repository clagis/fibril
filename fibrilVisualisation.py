#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:36:17 2021

@author: clement
"""

import numpy as np
from pymol.cgo import *
from pymol import cmd

# # create an axis, with red as x-, green as y- and blue as z-axis
# w = 1 # cylinder width
# l = 30 # cylinder length
# h = 5 # cone hight
# d = w * 1.5 # cone base diameter

# obj = [CYLINDER, 0.0, 0.0, 0.0,   l, 0.0, 0.0, w, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
#        CYLINDER, 0.0, 0.0, 0.0, 0.0,   l, 0.0, w, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
#        CYLINDER, 0.0, 0.0, 0.0, 0.0, 0.0,   l, w, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
#        CONE,   l, 0.0, 0.0, h+l, 0.0, 0.0, d, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
#        CONE, 0.0,   l, 0.0, 0.0, h+l, 0.0, d, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
#        CONE, 0.0, 0.0,   l, 0.0, 0.0, h+l, d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# cmd.load_cgo(obj, 'axes')
cmd.delete('all')

# basic parameters
change = 180/np.pi
mol = '3kgs'
molNumber = 30

# coordinates of the center to center the mol and translation vector (get them from parameters)
x_c, y_c, z_c = 21.693999987499517, 42.731500013222856, 48.15921182310983
x_t, y_t, z_t =  44.45585392, -5.11456601,  0. #0.,  68.01475148, -18.69298347 # 
trect = [x_t, y_t, z_t]
thickness = 53.849002838134766
radius = (100-thickness)/2

# load molecule representation retrieve by logical path (often need pdb1 file with is not obtainable trough BIOPython)
cmd.load('/Users/blaise/Desktop/Fibre/kg/3kgs.pdb1')
#cmd.load('/Users/laurentvuillon/Desktop/Fibre/3kgs.pdb1')


# construction
cmd.split_states(mol)
cmd.delete(mol)

cmd.create('a1', mol + '_0001')
cmd.create('a2', mol + '_0002')

cmd.delete('3kgs_0001')
cmd.delete('3kgs_0002')

cmd.translate([x_c, -y_c, -z_c], "a1")
cmd.translate([x_c, -y_c, -z_c], "a2")


B = np.array([[ 0.35499154,  0.        ,  0.93486951],
       [ 0.93486951,  0.        , -0.35499154],
       [ 0.        ,  1.        ,  0.        ]])

vectG = B[:,0]
vectP = B[:,1]
vectQ = B[:,2]
scalG = np.dot(vectG,trect)
scalP = np.dot(vectP,trect)
scalQ = np.dot(vectQ,trect)


for k in range(1, molNumber+1):
    name1 = 'obj1_' + str(k)
    name2 = 'obj2_' + str(k)

    cmd.copy(name1, 'a1')
    cmd.copy(name2, 'a2')
    # np.array obtained in parameters script, use only the (x,y)-coordinates
    norm = np.linalg.norm(np.array([scalP*k, scalQ*k]))
    # the next loop need to know if you use radians or degree, so be sure to use the good one
    angle, radians = norm/radius, True
    if radians:
        cmd.rotate([vectG[0], vectG[1], vectG[2]], angle*change, name1)
        cmd.rotate([vectG[0], vectG[1], vectG[2]], angle*change, name2)
        # np.array obtained in parameters script, use only the z-coordinates
        # plus radius and angle of fiber
        
        v_t = radius*np.cos(angle)*vectP + radius*np.sin(angle)*vectQ + k*scalG*vectG
        cmd.translate([v_t[0], v_t[1], v_t[2]], name1)
        cmd.translate([v_t[0], v_t[1], v_t[2]], name2)
    else:
        cmd.rotate(vectG, angle, name1)
        cmd.rotate(vectG, angle, name2)
        # np.array obtained in parameters script, use only the z-coordinates
        # plus radius and angle of fiber
        cmd.translate([radius*np.cos(angle/change), radius*np.sin(angle/change), z_t*k], name1)
        cmd.translate([radius*np.cos(angle/change), radius*np.sin(angle/change), z_t*k], name2)
    # alternating colors
    if k % 2 == 0:
        cmd.color('marine', name1)
        cmd.color('marine', name2)

cmd.delete('a1')
cmd.delete('a2')


