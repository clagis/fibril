#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:36:17 2021

@author: clement
"""

#=============================================================================
#
###     Libraries
#
#=============================================================================

import numpy as np
from pymol import cmd

#=============================================================================


cmd.delete('all')

# basic parameters
change = 180/np.pi
mol = '3kgs-assembly1'
molNumber = 30

# coordinates of the center to center the mol and translation vector (get them from fibrilParameters)
x_c, y_c, z_c = 21.06059661589228, 29.63259774433093, 48.15921182310983
x_t, y_t, z_t =  -22.25133585,   1.09601211, -53.44273143
trect = [x_t, y_t, z_t]
thickness = 45.275787381373746
radius = (120-thickness)/2

# load molecule representation retrieve by logical path (often need pdb1 file with is not obtainable through BIOPython)
cmd.load('/Users/blaise/Desktop/Fibre/kg/3kgs-assembly1.cif')
#cmd.load('/Users/laurentvuillon/Desktop/Fibre/3kgs.pdb1')


# construction
cmd.create('a', mol)
cmd.delete(mol)
# cmd.remove("chain A-2")
# cmd.remove("chain B-2")

cmd.translate([-x_c, -y_c, -z_c], "a")

# Orthonormal basis obtained through Modified Gram-Schmidt
B = np.array([[-5.63535429e-02,  9.98410876e-01, -8.75443317e-19],
       [ 9.89194133e-01,  5.58333200e-02,  1.35564033e-01],
       [ 1.35348605e-01,  7.63951352e-03, -9.90768587e-01]])

# Axis are formed using column vectors from the basis
vectG = B[:,0]
vectP = B[:,1]
vectQ = B[:,2]
scalG = np.dot(vectG,trect)
scalP = np.dot(vectP,trect)
scalQ = np.dot(vectQ,trect)


for k in range(1, molNumber+1):
    name = 'obj_' + str(k)

    cmd.copy(name, 'a')
    # np.array obtained in parameters script, use only the (x,y)-coordinates
    norm = np.linalg.norm(np.array([scalP*k, scalQ*k]))
    # the next loop need to know if you use radians or degree, so be sure to use the good one
    angle, radians = norm/radius, True
    if radians:
        cmd.rotate([vectG[0], vectG[1], vectG[2]], angle*change, name)
        # np.array obtained in parameters script, use only the z-coordinates
        # plus radius and angle of fiber
        
        v_t = radius*np.cos(angle)*vectP + radius*np.sin(angle)*vectQ + k*scalG*vectG
        cmd.translate([v_t[0], v_t[1], v_t[2]], name)
    else:
        cmd.rotate(vectG, angle, name)
        # np.array obtained in parameters script, use only the z-coordinates
        # plus radius and angle of fiber
        cmd.translate([radius*np.cos(angle/change), radius*np.sin(angle/change), z_t*k], name)
    # alternating colors
    if k % 2 == 0:
        cmd.color('marine', name)

cmd.delete('a')
#cmd.save('28-36_r130.pdb', 'all')

