############################################################
# Val_phi_psi_chi.py
#
# Plots data for VVGGVG Valine residues 
#
# Input:
#   VVGGVG_s1_angle.xvg
#   Columns:
#       2: V1 phi
#       3: V1 psi
#       4: V2 phi
#       5: V2 psi 
#      10: V5 phi
#      11: V5 psi
#      14: V1 chi
#      15: V2 chi 
#      16: V5 chi
#
# Summary of plot:
#   phi/psi distribution for each residue
#   phi/psi distribution for each rotamer (normalized across all 3, and within)
#   phi/psi distribution for each rotamer, Normalize by if-phi-psi to match Dynamomics data
############################################################

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import math


# Define phi/psi regions
beta_x = np.array([-180, -180, -50, -50, -115, -115, -180])/10
beta_y = np.array([80, 180, 180, 80, 80,100, 100])/10
beta_x1 = np.array([-180, -180, -50, -50, -180])/10
beta_y1 = np.array([-180, -170, -170, -180, -180])/10

pII_x = np.array([-110, -110, -50, -50, -110])/10
pII_y = np.array([120, 180, 180, 120, 120])/10

pII_x1 = np.array([-180, -180, -115, -115, -180])/10
pII_y1 = np.array([50, 100, 100, 50, 50])/10

alphaL_x = np.array([5, 5, 75, 75, 5])/10
alphaL_y = np.array([25, 120, 120, 25, 25])/10

# grouped alphaR and near alphaR
alphaR_x  = np.array([-180, -180, -30, -30, -180])/10
alphaR_y = np.array([-80, -5, -5, -80, -80])/10

# Load VVGGVG data
data = np.loadtxt('VVGGVG_s1_angle.xvg',skiprows=17)

# Loop over all 3 Valines
for val_loop in range(0,3):

    # Extract phi, psi and chi data for this residue
    if (val_loop== 0):
        Val_chi1 = data[:,14]
        Val_phi = data[:,2]
        Val_psi = data[:,3]
    elif (val_loop == 1):
        Val_chi1 = data[:,15]
        Val_phi = data[:,4]
        Val_psi = data[:,5]
    elif (val_loop == 2):
        Val_chi1 = data[:,16]
        Val_phi = data[:,10]
        Val_psi = data[:,11]

    # Make chi between 0 and 359.9
    ind0 = Val_chi1 < 0
    Val_chi1[ind0] = Val_chi1[ind0]+360

    # Get number of frames in dataset
    num_steps = Val_chi1.shape[0]

    # Plot phi/psi heatmap of full trajectory
    square_chi1 = np.zeros([36, 36])
    counts = np.zeros([36,36])

    # Loop over all frames and create phi/psi heatmap
    for i in range(0,Val_phi.shape[0]):
        phi = Val_phi[i]
        if (phi >= -180):
            phi_index = (math.floor(phi/10)+17)
            psi = Val_psi[i]
            if (psi >= -180):
                psi_index = (math.floor(psi/10)+17)
                square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
            else:
                print(i)
        else:
            print(i)

    # Convert counts into probability (divide by 100 because of box size)
    square_chi1 = square_chi1/Val_phi.shape[0]/100
    
    # Make all 0s into nan to show up as white
    ind0 = square_chi1 == 0
    square_chi1[ind0] = 'nan'   

    # Plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plt.imshow(square_chi1, origin='lower', cmap='jet')#, interpolation='nearest')
    
    # Plot backbone region outlines
    plt.plot(beta_x+17.5, (beta_y+17.5), 'r')
    plt.plot(beta_x1+17.5, (beta_y1+17.5), 'r')
    plt.plot(pII_x+17.5, (pII_y+17.5), 'r')
    plt.plot(pII_x1+17.5, (pII_y1+17.5), 'r')
    plt.plot(alphaL_x+17.5, (alphaL_y+17.5), 'r')
    plt.plot(alphaR_x+17.5, (alphaR_y+17.5), 'r')
    
    plt.colorbar()
    plt.clim(0,0.0007)
    plt.xlabel('$\phi$',fontsize=20)
    plt.ylabel('$\psi$',fontsize=20)    
    locs, labels = plt.xticks()            
    plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
    locs, labels = plt.yticks()            
    plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
    plt.axis([-.5, 35.5, -.5, 35.5])

    # Save figure
    if (val_loop == 0):
        plt.title('Val 1',fontsize=20)
        plt.savefig('Val_1_phipsiAll.png', bbox_inches='tight')
    elif (val_loop == 1):
        plt.title('Val 2',fontsize=20)
        plt.savefig('Val_2_phipsiAll.png', bbox_inches='tight')
    else:
        plt.title('Val 5',fontsize=20)
        plt.savefig('Val_5_phipsiAll.png', bbox_inches='tight')

    plt.close()


    # Seperate phi/psi data by chi rotamer bin
    ind0 = Val_chi1 < 120
    ind1 = Val_chi1 >= 0
    Val_60_phi = Val_phi[ind0&ind1]
    Val_60_psi = Val_psi[ind0&ind1]

    ind0 = Val_chi1 < 240
    ind1 = Val_chi1 >= 120
    Val_180_phi = Val_phi[ind0&ind1]
    Val_180_psi = Val_psi[ind0&ind1]

    ind0 = Val_chi1 <= 360
    ind1 = Val_chi1 >= 240
    Val_300_phi = Val_phi[ind0&ind1]
    Val_300_psi = Val_psi[ind0&ind1]

    # Plot each region 

    # Loop over the 3 rotamers
    for loop_rotamers in range(0,3):

        if (loop_rotamers == 0):
            this_rot_phi = Val_60_phi.copy()
            this_rot_psi = Val_60_psi.copy()
            str_save = '60'
        elif (loop_rotamers == 1):
            this_rot_phi = Val_180_phi.copy()
            this_rot_psi = Val_180_psi.copy()
            str_save = '180'
        else:
            this_rot_phi = Val_300_phi.copy()
            this_rot_psi = Val_300_psi.copy()
            str_save = '300'

        # Bin phi/psi data
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,this_rot_phi.shape[0]):
            phi = this_rot_phi[i]
            phi_index = (math.floor(phi/10)+17)
            psi =this_rot_psi[i]
            psi_index =(math.floor(psi/10)+17)
            square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1

        if (loop_rotamers == 0):
            square_chi1_60 = square_chi1.copy()
        elif (loop_rotamers == 1):
            square_chi1_180 = square_chi1.copy()
        else:
            square_chi1_300 = square_chi1.copy()

        # Make 0 show up white
        ind0 = square_chi1 == 0
        square_chi1[ind0] = 'nan'  

        # Plot 
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        # Convert counts into probability (divide by 100 because of box size)
        plt.imshow(square_chi1/num_steps/100,origin='lower', cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = '+str_save+'$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.0001)
        plt.axis([-.5, 35.5, -.5, 35.5])

        if (val_loop == 0):
            plt.savefig('Val_'+  str_save + '_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig('Val_'+  str_save + '_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig('Val_'+  str_save + '_V5.png', bbox_inches='tight')

        plt.close()

        # Normalize in plot
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1/this_rot_phi.shape[0]/100, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = '+str_save+'$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.0007)
        plt.axis([-.5, 35.5, -.5, 35.5])
        if (val_loop == 0):
            plt.savefig('Val_'+  str_save + '_normIn_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig('Val_'+  str_save + '_normIn_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig('Val_'+  str_save + '_normIn_V5.png', bbox_inches='tight')

        plt.close()

        

    ## End of loop_rotamers ##
   

    ## Normalize by if-phi-psi to match Dynamomics data
    counts = np.zeros([36,36])
    for i in range(0,36):
        for j in range(0,36):
            count1 = square_chi1_60[i,j] + square_chi1_180[i,j] + square_chi1_300[i,j]
            if (count1 > 0):
                square_chi1_60[i,j] = square_chi1_60[i,j]/count1
                square_chi1_180[i,j] = square_chi1_180[i,j]/count1
                square_chi1_300[i,j] = square_chi1_300[i,j]/count1
                counts[i,j] = count1

    for loop_rotamers in range(0,3):
        if (loop_rotamers == 0):
            square_chi1 = square_chi1_60.copy()
            str_save = '60'
        if (loop_rotamers == 1):
            square_chi1 = square_chi1_180.copy()
            str_save = '180'
        if (loop_rotamers == 2):
            square_chi1 = square_chi1_300.copy()
            str_save = '300'

        # Make 0s into 'nan' to show up white
        ind0 = square_chi1 == 0
        square_chi1[ind0] = 'nan'  

        # Plot data
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, origin='lower',cmap='YlGnBu')#, interpolation='nearest')
       
        # Plot backbone region outlines
        plt.plot(beta_x+17.5, (beta_y+17.5), 'r')
        plt.plot(beta_x1+17.5, (beta_y1+17.5), 'r')
        plt.plot(pII_x+17.5, (pII_y+17.5), 'r')
        plt.plot(pII_x1+17.5, (pII_y1+17.5), 'r')
        plt.plot(alphaL_x+17.5, (alphaL_y+17.5), 'r')
        plt.plot(alphaR_x+17.5, (alphaR_y+17.5), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = ' + str_save + '$^{\circ}$',fontsize=20)

        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,1)

        plt.axis([-.5, 35.5, -.5, 35.5])
        if (val_loop == 0):
            plt.savefig('Val_' + str_save + '_compareDYN_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig('Val_' + str_save + '_compareDYN_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig('Val_' + str_save + '_compareDYN_V5.png', bbox_inches='tight')

        plt.show()
