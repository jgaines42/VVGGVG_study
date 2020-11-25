import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

data = np.loadtxt('VVGGVG_s1_angle.xvg',skiprows=17)

print(data[0,13])
for val_loop in range(0,3):
    if (val_loop== 0): # Val
        Val_chi1 = data[:,14].copy()
        Val_phi = data[:,4].copy() 
        Val_psi = data[:,5].copy()
        nChi = 1
    elif (val_loop == 1): # Val
        Val_chi1 = data[:,15].copy()
        Val_phi = data[:,6].copy()
        Val_psi = data[:,7].copy()
        nChi = 1
    elif (val_loop == 2): # Val
        Val_chi1 = data[:,16].copy()
        nChi = 1
   
    ind0 = Val_chi1 < 0
    Val_chi1[ind0] = Val_chi1[ind0]+360
    print(data[0,13])
    if (nChi >= 2):
        ind0 = Val_chi2 < 0
        Val_chi2[ind0] = Val_chi2[ind0]+360
    if (nChi >= 3):
        ind0 = Val_chi3 < 0
        Val_chi3[ind0] = Val_chi3[ind0]+360
    if (nChi >= 4):
        ind0 = Val_chi4 < 0
        Val_chi4[ind0] = Val_chi4[ind0]+360
    print(data[0,13])

    bins = np.arange(0,360,10)+5
    for loop_chi in range(0,nChi):
        if (loop_chi == 0):
            this_chi = Val_chi1
        elif (loop_chi == 1):
            this_chi = Val_chi2
        elif (loop_chi == 2):
            this_chi = Val_chi3
        elif (loop_chi == 3):
            this_chi = Val_chi4
        n, bins, patches = plt.hist(this_chi, bins,density=True, histtype='stepfilled', alpha=0.3)
        plt.show()
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
        
        
        plt.show()
        if (val_loop== 0): # Asn
            if (loop_chi == 1):
                Val_chi2 = data[:,13].copy()
                this_chi = Val_chi2
                bins = np.arange(-180,180,10)+5
                n, bins, patches = plt.hist(this_chi, bins,density=True, histtype='stepfilled', alpha=0.3)
                plt.show()
                print(bins)
                print(min(Val_chi2))
                plt.plot(bins[0:35], n,'k')
                plt.legend()
                plt.xticks(np.arange(-180,180, step=60))
                plt.xlabel('$\chi_2$')
                plt.ylabel('P($\chi_2$)')
                plt.show()
                bins = np.arange(0,360,10)+5

