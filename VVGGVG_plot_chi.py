import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

data = np.loadtxt('VVGGVG_s1_angle.xvg',skiprows=17)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for i in range(0,3):
    if (i== 0):
        Val_chi1 = data[:,14]
        Val_phi = data[:,2]
        Val_psi = data[:,3]
    elif (i == 1):
        Val_chi1 = data[:,15]
        Val_phi = data[:,4]
        Val_psi = data[:,5]
    elif (i == 2):
        Val_chi1 = data[:,16]
        Val_phi = data[:,10]
        Val_psi = data[:,11]

    ind0 = Val_chi1 < 0
    Val_chi1[ind0] = Val_chi1[ind0]+360
    bins = np.arange(0,360,15)+7.5
    n, bins, patches = plt.hist(Val_chi1, bins,density=True, histtype='step')
    if (i== 0):
        V1 = n
    elif (i==1):
        V2 = n
    else:
        V5 = n

plt.show()
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
plt.plot(bins[0:23], V1,  label='V1')
plt.plot(bins[0:23], V2,  label='V2')
plt.plot(bins[0:23], V5,  label='V5')
plt.legend()
plt.minorticks_on()
plt.xticks(np.arange(0,360, step=60))
plt.xlabel('$\chi_1$')
plt.ylabel('P($\chi_1$)')
plt.show()


