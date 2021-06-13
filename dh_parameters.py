import numpy as np

"""
DH params found by Yuying Blair Huang
"""
# Params for Denavit-Hartenberg Reference Frame Layout (DH)
jaco27DOF_DH_lengths = {'D1':0.2755, 'D2':0.2050, 
                        'D3':0.2050, 'D4':0.2073,
                        'D5':0.1038, 'D6':0.1038, 
                        'D7':0.1600, 'e2':0.0098, 'D_grip':0.1775} # .001775e2 is dist to grip site


 
DH_attributes_jaco27DOF = {
          'DH_a':[0, 0, 0, 0, 0, 0, 0],
          'DH_alpha':[np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi],
          'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
          'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
          'DH_d':(-jaco27DOF_DH_lengths['D1'], 
                    0, 
                    -(jaco27DOF_DH_lengths['D2']+jaco27DOF_DH_lengths['D3']), 
                    -jaco27DOF_DH_lengths['e2'], 
                    -(jaco27DOF_DH_lengths['D4']+jaco27DOF_DH_lengths['D5']), 
                    0, 
                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D_grip']))
           }
# Params for Denavit-Hartenberg Reference Frame Layout (DH)
Panda_DH_lengths = {'D1':0.333, 'D2':0, 
               'D3':0.316, 'D4':0,
               'D5':0.384, 'D6':0, 
               'D7':0, 'DF':0.1065, 'e1':0.0825}

 
DH_attributes_Panda = {
          'DH_a':[0, 0, Panda_DH_lengths['e1'], -Panda_DH_lengths['e1'], 0, 0, 0, 0],
           'DH_alpha':[0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0, np.pi/2.0,0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
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
               'D3':0.4, 'D4':-0.1685,
               'D5':0.4, 'D6':0.1363, 
               'D7':0.11, 'e1':0.081}

 
DH_attributes_Sawyer = {
          'DH_a':[Sawyer_DH_lengths['e1'], 0, 0, 0, 0, 0, 0],
           'DH_alpha':[-np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0,0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
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

DH_attributes_dm_reacher = {
     'DH_a':[0.12,0.12], # arm is .12, hand is .1 + .01 sphere finger
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0], 
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}

DH_attributes_dm_reacher_long_wrist = {
     'DH_a':[0.12,0.22], 
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0], 
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}

DH_attributes_dm_reacher_double = {
     'DH_a':[0.22,0.22], 
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0], 
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}



       
robot_attributes = {
                    'reacher':DH_attributes_dm_reacher, 
                    'reacher_long_wrist':DH_attributes_dm_reacher_long_wrist, 
                    'reacher_double':DH_attributes_dm_reacher_double, 
                    'Jaco':DH_attributes_jaco27DOF, 
                    'Baxter':DH_attributes_Baxter, 
                    'Sawyer':DH_attributes_Sawyer, 
                    'Panda':DH_attributes_Panda, 
                   }


