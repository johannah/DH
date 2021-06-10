import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import pickle
import math
from datetime import datetime as date
from glob import glob
import torch
from robosuite.utils.transform_utils import mat2quat
from IPython import embed
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# Params for Denavit-Hartenberg Reference Frame Layout (DH)
Panda_DH_lengths = {'D1':0.333, 'D2':0, 
               'D3':0.316, 'D4':0,
               'D5':0.384, 'D6':0, 
               'D7':0, 'DF':0.1065, 
               'e1':0.0825, 'e2':0.088}

 
DH_attributes_Panda = {
          'DH_a':[0, 0, 0, Panda_DH_lengths['e1'], -Panda_DH_lengths['e1'], 0, Panda_DH_lengths['e2'], 0],
           'DH_alpha':[0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, 0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1, 0],
           'DH_theta_offset':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           'DH_d':(Panda_DH_lengths['D1'], 
                   Panda_DH_lengths['D2'],
                   Panda_DH_lengths['D3'], 
                   Panda_DH_lengths['D4'],
                   Panda_DH_lengths['D5'],
                   Panda_DH_lengths['D6'],
                   Panda_DH_lengths['D7'],
                   Panda_DH_lengths['DF'])
           }


Sawyer_DH_lengths = {'D1':0.237, 'D2':0.1925, 
               'D3':0.4, 'D4':0.1685,
               'D5':0.4, 'D6':0.1363, 
               'D7':0.11, 'e1':0.081}

 
DH_attributes_Sawyer = {
          'DH_a':[0,Sawyer_DH_lengths['e1'], 0, 0, 0, 0, 0, 0],
           'DH_alpha':[0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, -np.pi/2.0, 0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1, 0],
           'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
           'DH_d':(Sawyer_DH_lengths['D1'], 
                   Sawyer_DH_lengths['D2'],
                   Sawyer_DH_lengths['D3'], 
                   Sawyer_DH_lengths['D4'],
                   Sawyer_DH_lengths['D5'],
                   Sawyer_DH_lengths['D6'],
                   Sawyer_DH_lengths['D7'])
           }

Baxter_DH_lengths = {'D1':0.27035, 'D2':0.102, 
               'D3':0.26242, 'D4':0.10359,
               'D5':0.2707,  'D6':0.115975,
               'D7':0.11355, 'e1':0.069, 'e2':0.010}

 
DH_attributes_Baxter = {
          'DH_a':[Baxter_DH_lengths['e1'], 0, Baxter_DH_lengths['e1'], 0, Baxter_DH_lengths['e2'], 0, 0],
           'DH_alpha':[-np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0,0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
           'DH_theta_offset':[0.0, np.pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           'DH_d':(Baxter_DH_lengths['D1'], 
                   0,
                   Baxter_DH_lengths['D2']+Baxter_DH_lengths['D3'], 
                   0,
                   Baxter_DH_lengths['D4']+Baxter_DH_lengths['D5'],
                   0,
                   Baxter_DH_lengths['D6']+Baxter_DH_lengths['D7'])
           }