import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

for cluster_loop in range(4,5):
    if (cluster_loop == 0):
        folder = 'cluster4/'
        data = np.loadtxt(folder+'s1_cluster4.xvg',skiprows=17)
    if (cluster_loop == 1):
        folder = 'cluster5/'
        data = np.loadtxt(folder+'s1_cluster5.xvg',skiprows=17)
    if (cluster_loop == 2):
        folder = 'cluster6/'
        data = np.loadtxt(folder+'s1_cluster6.xvg',skiprows=17)
    if (cluster_loop == 3):
        folder = 'cluster7/'
        data = np.loadtxt(folder+'s1_cluster7.xvg',skiprows=17)
    if (cluster_loop == 4):
        folder = 'cluster8/'
        data = np.loadtxt(folder+'s1_cluster8.xvg',skiprows=17)
    print(data.shape[0])
    for val_loop in range(0,3):
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

        ind0 = Val_chi1 < 0
        Val_chi1[ind0] = Val_chi1[ind0]+360

        print(max(Val_phi))
        print(max(Val_psi))
        print(max(Val_chi1))
        print(min(Val_phi))
        print(min(Val_psi))
        print(min(Val_chi1))
        num_steps = Val_chi1.shape[0]

        #Val_chi_binned = np.zeros(Val_chi1.shape,dtype=int)

        ind0 = Val_chi1 <= 120
        ind1 = Val_chi1 >= 0
        Val_60_phi = Val_phi[ind0&ind1]
        Val_60_psi = Val_psi[ind0&ind1]
        print(Val_60_phi.shape[0]/Val_phi.shape[0])
        ind0 = Val_chi1 <= 240
        ind1 = Val_chi1 >= 120
        Val_180_phi = Val_phi[ind0&ind1]
        Val_180_psi = Val_psi[ind0&ind1]
        print(Val_180_phi.shape[0]/Val_phi.shape[0])
        ind0 = Val_chi1 <= 360
        ind1 = Val_chi1 >= 240
        Val_300_phi = Val_phi[ind0&ind1]
        Val_300_psi = Val_psi[ind0&ind1]
        print(Val_300_phi.shape[0]/Val_phi.shape[0])
        beta_x = np.array([-180, -180, -50, -50, -115, -115, -180])/10
        beta_y = np.array([80, 180, 180, 80, 80,100, 100])/10
        #beta_x = np.array(beta_x)
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

        # Plot heatmap of full trajectory
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_phi.shape[0]):
            phi = Val_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1

        square_chi1 = square_chi1/Val_phi.shape[0]/100
        print(Val_phi.shape[0])
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'   
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='jet')#, interpolation='nearest')
        plt.plot(beta_x+18, 35-(beta_y+17), 'r')
        plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
        plt.plot(pII_x+18, 35-(pII_y+17), 'r')
        plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
        plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
        plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.001)
        plt.axis([0, 36, 36, 0])

        if (val_loop == 0):
            plt.title('Val 1',fontsize=20)
            plt.savefig(folder+'Val_1_phipsiAll.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.title('Val 2',fontsize=20)
            plt.savefig(folder+'Val_2_phipsiAll.png', bbox_inches='tight')
        else:
            plt.title('Val 5',fontsize=20)
            plt.savefig(folder+'Val_5_phipsiAll.png', bbox_inches='tight')

        plt.close()#plt.show()

        # Plot Val distribution
        ind0 = Val_chi1 < 0
        Val_chi1[ind0] = Val_chi1[ind0]+360

        bins = np.arange(0,360,10)+5
        for loop_chi in range(0,1):
            this_chi = Val_chi1
            
            n, bins, patches = plt.hist(this_chi, bins,density=True, histtype='stepfilled', alpha=0.3)
            plt.close()#plt.show()
            plt.plot(bins[0:35], n,'k')
            plt.xticks(np.arange(0,360, step=60))
            if (loop_chi == 0):
                plt.xlabel('$\chi_1$')
                plt.ylabel('P($\chi_1$)')
            elif (loop_chi == 1):
                plt.xlabel('$\chi_2$')
                plt.ylabel('P($\chi_2$)')
            elif (loop_chi == 2):
                plt.xlabel('$\chi_3$')
                plt.ylabel('P($\chi_3$)')
            elif (loop_chi == 3):
                plt.xlabel('$\chi_4$')
                plt.ylabel('P($\chi_4$)')
        if (val_loop == 0):
            plt.title('Val 1',fontsize=20)
            plt.savefig(folder+'Val_1_PChi.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.title('Val 2',fontsize=20)
            plt.savefig(folder+'Val_2_PChi.png', bbox_inches='tight')
        else:
            plt.title('Val 5',fontsize=20)
            plt.savefig(folder+'Val_5_PChi.png', bbox_inches='tight')
        plt.close()#plt.show()
        
        # Plot 60
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_60_phi.shape[0]):
            phi = Val_60_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_60_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1


        square_chi1 = square_chi1/num_steps/100
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'   
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='jet')#, interpolation='nearest')
        # plt.plot(beta_x+18, 35-(beta_y+17), 'r')
        # plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
        # plt.plot(pII_x+18, 35-(pII_y+17), 'r')
        # plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
        # plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
        # plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 60$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.001)
        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_60_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_60_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_60_V5.png', bbox_inches='tight')

        plt.close()#plt.show()

        # Plot 180
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_180_phi.shape[0]):
            phi = Val_180_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_180_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
                    
        square_chi1 = square_chi1/num_steps/100
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 180$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.001)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_180_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_180_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_180_V5.png', bbox_inches='tight')

        plt.close()#plt.show()

        # Plot 300

        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_300_phi.shape[0]):
            phi = Val_300_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_300_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
                    
        square_chi1 = square_chi1/num_steps/100
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 300$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.001)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_300_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_300_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_300_V5.png', bbox_inches='tight')
        plt.close()#plt.show()


        ## Normalize in Plot
        # Plot 60
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_60_phi.shape[0]):
            phi = Val_60_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_60_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
                    
        square_chi1_60 = square_chi1.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1/Val_60_phi.shape[0]/100, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 60$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.0007)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_60_normIn_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_60_normIn_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_60_normIn_V5.png', bbox_inches='tight')

        plt.close()#plt.show()

        # Plot 180
        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_180_phi.shape[0]):
            phi = Val_180_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_180_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
                    

        square_chi1_180 = square_chi1.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1/Val_180_phi.shape[0]/100, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 180$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.0007)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_180_normIn_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_180_normIn_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_180_normIn_V5.png', bbox_inches='tight')
        plt.close()#plt.show()

        # Plot 300

        square_chi1 = np.zeros([36, 36])
        counts = np.zeros([36,36])
        for i in range(0,Val_300_phi.shape[0]):
            phi = Val_300_phi[i]
            if (phi > -180):
                
                phi_index = (int(phi/10)+17)
                psi =Val_300_psi[i]
                if (psi > -180):
                #    psi = psi + 360
                    psi_index = 35-(int(psi/10)+17)
                    #print(phi_index, psi_index)
                    square_chi1[psi_index,phi_index] = square_chi1[psi_index,phi_index] + 1
                    

        square_chi1_300 = square_chi1.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1/Val_300_phi.shape[0]/100, cmap='jet')#, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 300$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,0.0007)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_300_normIn_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_300_normIn_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_300_normIn_V5.png', bbox_inches='tight')
        plt.close()#plt.show()

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

        ind0 = counts == 0# 0.000001
        counts[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(counts, cmap='jet')#, interpolation='nearest')
        plt.close()#plt.show()


        square_chi1 = square_chi1_60.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='YlGnBu')#, interpolation='nearest')
        plt.plot(beta_x+18, 35-(beta_y+17), 'r')
        plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
        plt.plot(pII_x+18, 35-(pII_y+17), 'r')
        plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
        plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
        plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 60$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,1)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_60_compareDYN_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_60_compareDYN_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_60_compareDYN_V5.png', bbox_inches='tight')

        plt.close()#plt.show()

        square_chi1 = square_chi1_180.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='YlGnBu')#, interpolation='nearest')
        plt.plot(beta_x+18, 35-(beta_y+17), 'r')
        plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
        plt.plot(pII_x+18, 35-(pII_y+17), 'r')
        plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
        plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
        plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 180$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,1)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_180_compareDYN_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_180_compareDYN_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_180_compareDYN_V5.png', bbox_inches='tight')
        plt.close()#plt.show()

        square_chi1 = square_chi1_300.copy()
        ind0 = square_chi1 == 0# 0.000001
        square_chi1[ind0] = 'nan'  
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.imshow(square_chi1, cmap='YlGnBu')#, interpolation='nearest')
        plt.plot(beta_x+18, 35-(beta_y+17), 'r')
        plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
        plt.plot(pII_x+18, 35-(pII_y+17), 'r')
        plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
        plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
        plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
        plt.colorbar()
        plt.xlabel('$\phi$',fontsize=20)
        plt.ylabel('$\psi$',fontsize=20)
        plt.title('$\chi_1$ = 300$^{\circ}$',fontsize=20)
        locs, labels = plt.xticks()            # Get locations and labels
        plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
        plt.clim(0,1)

        plt.axis([0, 36, 36, 0])
        if (val_loop == 0):
            plt.savefig(folder+'Val_300_compareDYN_V1.png', bbox_inches='tight')
        elif (val_loop == 1):
            plt.savefig(folder+'Val_300_compareDYN_V2.png', bbox_inches='tight')
        elif (val_loop == 2):
            plt.savefig(folder+'Val_300_compareDYN_V5.png', bbox_inches='tight')
        plt.close()#plt.show()