import numpy as np
# Params for Denavit-Hartenberg Reference Frame Layout (DH)
jaco27DOF_DH_lengths = {'D1':0.2755, 'D2':0.2050, 
               'D3':0.2050, 'D4':0.2073,
               'D5':0.1038, 'D6':0.1038, 
               'D7':0.1600, 'e2':0.0098}

 
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
                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D7']))
           }
DH_attributes_dm_reacher = {
     'DH_a':[0.12,0.12], # arm is .12, hand is .1 + .01 sphere finger
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0], 
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}
       
robot_attributes = {'reacher':DH_attributes_dm_reacher, 
                    'Jaco':DH_attributes_jaco27DOF, 
                   }


