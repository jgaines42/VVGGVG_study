# Create Dunbrack predicted values

# Input:
# xvg file
# column ids for phi, psi and chi values
# list of rotamer values corresponding to dunbrack rotamer centers
# rotamer bin widths
# number of unique rotamers

# Output: 
# Columns
#   1: phi bin
#   2: psi bin
#   3: # of frames in rotamer 1
#   4: # of frames in rotamer 2, etc


import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import math


folder = 'cluster7/'

data_CP = np.loadtxt(folder+'s1_cluster7.xvg',skiprows=17)
data = np.loadtxt('../Sidechain_bbDependent/Dunbrack_smoothed_5/val.bbdep.rotamers.lib',skiprows=28, usecols=(1,2,3,4,8,9))

for val_loop in range(0,3):


    if (val_loop== 0):
        Val_chi1 = data_CP[:,14]
        CP_phi = data_CP[:,2]
        CP_psi = data_CP[:,3]
    elif (val_loop == 1):
        Val_chi1 = data_CP[:,15]
        CP_phi = data_CP[:,4]
        CP_psi = data_CP[:,5]
    elif (val_loop == 2):
        Val_chi1 = data_CP[:,16]
        CP_phi = data_CP[:,10]
        CP_psi = data_CP[:,11]
    sidechain_bin_width = 30
    Nrot = 3
    # Columns of input
    # 0: phi
    # 1: psi
    # 2: count
    # 3: rotamer number
    # 4: probability
    # 5: chi1 Value

    # Creat phi/psi probability distribution for CP

    CP_phi_psi = np.zeros([36, 36])
    counts = np.zeros([36,36])
    for i in range(0,CP_phi.shape[0]):
        phi = CP_phi[i]
        phi_index = (math.floor(phi/10)+17)
        psi =CP_psi[i]
        psi_index = (math.floor(psi/10)+17)
        # if (psi_index == 36):
        #     psi_index = 0
        # if (phi_index == 36):
        #     phi_index = 0
        CP_phi_psi[psi_index,phi_index] = CP_phi_psi[psi_index,phi_index] + 1
    print(sum(sum(CP_phi_psi)))

    # Loop over all rotamers
    p_each_rot = np.zeros([Nrot,1])
    for rotamer_loop in range(0,Nrot):
        ind0 = data[:,3] == (rotamer_loop+1)
        this_rot_Dun = data[ind0,:].copy()
        expected_values = np.zeros([36*36, 3])
        to_plot = np.zeros([36,36])
        for bb_loop in range(0,(35*35+1)):
            
            phi = this_rot_Dun[bb_loop,0]
            phi_index = math.floor(phi/10)+17
            psi = this_rot_Dun[bb_loop,1]
            psi_index = math.floor(psi/10)+17
            CP_count = CP_phi_psi[psi_index,phi_index]

            prob = this_rot_Dun[bb_loop,4]

            if (phi < 180 and psi < 180):
                expected_values[bb_loop,0] = phi
                expected_values[bb_loop,1] = psi
                expected_values[bb_loop,2] = prob*CP_count
                to_plot[psi_index, phi_index] = prob*CP_count
                if (prob*CP_count > 0):
                    p_each_rot[rotamer_loop] =  p_each_rot[rotamer_loop]+prob*CP_count

        to_plot = to_plot/CP_psi.shape[0]

        ind0 = to_plot <= 4.0E-7# 0.000001
        to_plot[ind0] = 'nan'   
        print(np.amin(to_plot[~ind0]))
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(to_plot/100, origin='lower',cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.clim(0,0.0005)
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 

        if (val_loop == 0):
            save_start = folder+'Val_1'
           
        elif (val_loop == 1):
            plt.title('Val 2',fontsize=20)
            save_start = folder+'Val_2'
        else:
            plt.title('Val 5',fontsize=20)
            save_start = folder+'Val_5'
        if (rotamer_loop == 0):
            plt.title('$\chi_1$ = 60',fontsize=20)
            plt.savefig(save_start + '_60_PChi_Dun.png', bbox_inches='tight')
            np.savetxt(save_start + '_60_Dun_temp.txt', to_plot)
        if (rotamer_loop == 1):
            plt.title('$\chi_1$ = 180',fontsize=20)
            plt.savefig(save_start + '_180_PChi_Dun.png', bbox_inches='tight')
        if (rotamer_loop == 2):
            plt.title('$\chi_1$ = 300',fontsize=20)
            plt.savefig(save_start + '_300_PChi_Dun.png', bbox_inches='tight')

        plt.show()
       
    print(p_each_rot/CP_psi.shape[0])

 


