
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

data = np.loadtxt('VVGGVG_s1_angle.xvg',skiprows=17)

print(data[0,:])

Val_chi1 = data[:,14]
ind0 = Val_chi1 < 0
Val_chi1[ind0] = Val_chi1[ind0]+360

Val_phi = data[:,2]
Val_psi = data[:,3]
data[:,0] = data[:,0]/100
Val_data = np.stack((Val_phi, Val_psi, Val_chi1), axis=-1)

# plt.rcParams.update({'font.size': 15})
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(data[:,0],Val_chi1, '.', markersize=2)
# axs[1].plot(data[:,0],Val_phi, '.', markersize=2)
# axs[2].plot(data[:,0],Val_psi, '.', markersize=2)
# axs[0].axis([500, 1000, 0, 360])
# axs[1].axis([500, 1000, -180, 180])
# axs[2].axis([500, 1000, -180, 180])
# axs[0].set_ylabel('$\chi_1$')
# axs[1].set_ylabel('$\phi$')
# axs[2].set_ylabel('$\psi$')
# axs[2].set_xlabel('Time (ns)')
# axs[0].set_yticks(np.arange(0,360, step=60))
# axs[1].set_yticks(np.arange(-180, 180, step=60))
# axs[2].set_yticks(np.arange(-180, 180, step=60))

# plt.show()

max_show = 1000
Val_data = Val_data[0:max_show]
ind0 = Val_chi1[0:max_show] < 90
ind1 = Val_chi1[0:max_show] > 30
Val_data_60 = Val_data[ind0&ind1,:]
Val_rest = Val_data[~(ind0&ind1),:]
plt.plot(Val_data_60[:,0], Val_data_60[:,1], '.r',markersize=1)

ind0 = Val_rest[0:max_show,2] < 210
ind1 = Val_rest[0:max_show,2] > 150
Val_data_180 = Val_rest[ind0&ind1,:]
Val_rest = Val_rest[~(ind0&ind1),:]
plt.plot(Val_data_180[:,0], Val_data_180[:,1], '.b',markersize=1)

ind0 = Val_rest[0:max_show,2] < 330
ind1 = Val_rest[0:max_show,2] > 270
Val_data_300 = Val_rest[ind0&ind1,:]
Val_rest = Val_rest[~(ind0&ind1),:]
plt.plot(Val_data_300[:,0], Val_data_300[:,1], '.g',markersize=1)


plt.plot(Val_rest[:,0], Val_rest[:,1],'xk',markersize=2)
plt.axis([-180, 0, -180, 180])
plt.xticks(np.arange(-180, 0, step=60))
plt.yticks(np.arange(-180, 180, step=60))
plt.minorticks_on()



plt.show()