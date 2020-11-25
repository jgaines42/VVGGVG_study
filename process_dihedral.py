# Process GNSRV


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
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(3, 1)
axs[0].plot(data[:,0],Val_chi1, '.', markersize=2)
axs[1].plot(data[:,0],Val_phi, '.', markersize=2)
axs[2].plot(data[:,0],Val_psi, '.', markersize=2)
axs[0].axis([500, 1000, 0, 360])
axs[1].axis([500, 1000, -180, 180])
axs[2].axis([500, 1000, -180, 180])
axs[0].set_ylabel('$\chi_1$')
axs[1].set_ylabel('$\phi$')
axs[2].set_ylabel('$\psi$')
axs[2].set_xlabel('Time (ns)')
axs[0].set_yticks(np.arange(0,360, step=60))
axs[1].set_yticks(np.arange(-180, 180, step=60))
axs[2].set_yticks(np.arange(-180, 180, step=60))

plt.show()

plt.plot(Val_phi[0:100], Val_psi[0:100])
plt.show()
# plt.plot(Val_phi, Val_psi, '.', 'MarkerSize',5)
# plt.show()

# Group by phi/psi regions
Val_data = np.stack((Val_phi, Val_psi, Val_chi1), axis=-1)



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
ind0 = square_chi1 == 0# 0.000001
square_chi1[ind0] = 'nan'   
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
plt.imshow(square_chi1, cmap='YlGnBu')#, interpolation='nearest')
# plt.plot(beta_x+18, 35-(beta_y+17), 'r')
# plt.plot(beta_x1+18, 35-(beta_y1+17), 'r')
# plt.plot(pII_x+18, 35-(pII_y+17), 'r')
# plt.plot(pII_x1+18, 35-(pII_y1+17), 'r')
# plt.plot(alphaL_x+18, 35-(alphaL_y+17), 'r')
# plt.plot(alphaR_x+18, 35-(alphaR_y+17), 'r')
plt.colorbar()
plt.xlabel('$\phi$',fontsize=20)
plt.ylabel('$\psi$',fontsize=20)
plt.title('Val 1',fontsize=20)
locs, labels = plt.xticks()            # Get locations and labels
plt.xticks(np.arange(0,36,6), np.arange(-180,180,60),fontsize=20) 
locs, labels = plt.yticks()            # Get locations and labels
plt.yticks(np.arange(0,36,6), np.arange(180,-180,-60),fontsize=20) 
#plt.clim(0,0.0001)
plt.axis([0, 36, 36, 0])
plt.savefig('Val_60_phipsiAll.png', bbox_inches='tight')

# PII:
#   -100 <= phi <= -30
#   psi >= 100 or psi <= -160
ind0 = Val_phi <= -50
ind1 = Val_phi >= -120
ind2 = Val_psi >= 80
ind3 = Val_psi <= -170
PII = Val_data[ind0&ind1&ind2,:]
PIIb = Val_data[ind0&ind1&ind3,:]
PII = np.concatenate([PII, PIIb])


# beta:
#   -180 <= phi <= -100
#   psi >= 100 or psi <= -160
ind0 = Val_phi >= -180
ind1 = Val_phi <= -120
ind2 = Val_psi >= 80
ind3 = Val_psi <= -170
Beta = Val_data[ind0&ind1&ind2,:]
Beta2 = Val_data[ind0&ind1&ind3,:]
Beta = np.concatenate([Beta, Beta2])

# alpha_R:
#   -180 <= phi <= -30
#   -70 <= psi <= 40
ind0 = Val_phi <= -30
ind1 = Val_phi >= -180
ind2 = Val_psi <= -5
ind3 = Val_psi >= -80
alphaR = Val_data[ind0&ind1&ind2&ind3,:]


# alpha_L
#   30 <= phi <= 100
#   -20 <= psi <= 80
ind0 = Val_phi <= 75
ind1 = Val_phi >= 5
ind2 = Val_psi >= 25
ind3 = Val_psi <= 120
alphaL = Val_data[ind0&ind1&ind2&ind3,:]


ind0 = Val_phi <= -30
ind2 = Val_psi >= 40
ind3 = Val_psi <= 80
C7 = Val_data[ind0&ind2&ind3,:]

print(PII.shape)
print(Beta.shape)
print(alphaR.shape)
print(alphaL.shape)
print(C7.shape)
print(Val_data.shape)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
plt.rcParams.update({'font.size': 15})

plt.plot(Val_data[:,0], Val_data[:,1],'.k', markersize=2)
plt.plot(PII[:,0], PII[:,1],'.', markersize=2, label='PII')
plt.plot(Beta[:,0], Beta[:,1],'.', markersize=2, label = 'Beta')
plt.plot(alphaR[:,0], alphaR[:,1],'.', markersize=2, label = 'alphaR')
plt.plot(alphaL[:,0], alphaL[:,1],'.', markersize=2, label = 'alphaL')
plt.plot(C7[:,0], C7[:,1],'.', markersize=2, label = 'C7')
plt.axis([-180, 180, -180, 180])
plt.xticks(np.arange(-180, 180, step=60))
plt.yticks(np.arange(-180, 180, step=60))
plt.minorticks_on()

plt.legend()
plt.xlabel('$\phi$')
plt.ylabel('$\psi$')
plt.savefig('Val1_phi_psi_colored.png', bbox_inches='tight')

plt.show()
bins = np.arange(0,360,15)+7.5

n, bins, patches = plt.hist(PII[:,2], bins,density=True, histtype='step')
nB, bins, patches = plt.hist(Beta[:,2], bins,density=True, histtype='step')
nR, bins, patches = plt.hist(alphaR[:,2], bins,density=True, histtype='step')
nL, bins, patches = plt.hist(alphaL[:,2], bins,density=True, histtype='step')
nC, bins, patches = plt.hist(C7[:,2], bins,density=True, histtype='step')
plt.show()
print(bins)
bins = np.arange(15,360,15)
print(bins)

plt.rcParams.update({'font.size': 15})
plt.plot(bins[0:23], n, 'k', label='PII')
plt.plot(bins[0:23], nB, 'r', label='beta')
plt.plot(bins[0:23], nR, 'g',label='alpha_R')
plt.legend()
plt.minorticks_on()
plt.xticks(np.arange(0,360, step=60))
plt.xlabel('$\chi_1$')
plt.ylabel('P($\chi_1$)')
plt.axis([0, 360, 0, 0.035])
plt.show()


# ##################################################################
# # SER
# ##################################################################


# Val_chi1 = data[:,14]
# ind0 = Val_chi1 < 0
# Val_chi1[ind0] = Val_chi1[ind0]+360

# Val_phi = data[:,6]
# Val_psi = data[:,7]
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

# # plt.plot(Val_phi, Val_psi, '.', 'MarkerSize',5)
# # plt.show()

# # Group by phi/psi regions
# Val_data = np.stack((Val_phi, Val_psi, Val_chi1), axis=-1)

# # PII:
# #   -100 <= phi <= -30
# #   psi >= 100 or psi <= -160
# ind0 = Val_phi <= -30
# ind1 = Val_phi >= -100
# ind2 = Val_psi >= 100
# ind3 = Val_psi <= -160
# PII = Val_data[ind0&ind1&ind2,:]
# PIIb = Val_data[ind0&ind1&ind3,:]
# PII = np.concatenate([PII, PIIb])


# # beta:
# #   -180 <= phi <= -100
# #   psi >= 100 or psi <= -160
# ind0 = Val_phi >= -180
# ind1 = Val_phi <= -100
# ind2 = Val_psi >= 100
# ind3 = Val_psi <= -160
# Beta = Val_data[ind0&ind1&ind2,:]
# Beta2 = Val_data[ind0&ind1&ind3,:]
# Beta = np.concatenate([Beta, Beta2])

# # alpha_R:
# #   -180 <= phi <= -30
# #   -70 <= psi <= 40
# ind0 = Val_phi <= -30
# ind1 = Val_phi >= -180
# ind2 = Val_psi <= 40
# ind3 = Val_psi >= -70
# alphaR = Val_data[ind0&ind1&ind2&ind3,:]


# # alpha_L
# #   30 <= phi <= 100
# #   -20 <= psi <= 80
# ind0 = Val_phi <= 100
# ind1 = Val_phi >= 30
# ind2 = Val_psi >= -20
# ind3 = Val_psi <= 80
# alphaL = Val_data[ind0&ind1&ind2&ind3,:]


# ind0 = Val_phi <= -30
# ind2 = Val_psi >= 40
# ind3 = Val_psi <= 100
# C7 = Val_data[ind0&ind2&ind3,:]


# print(PII.shape)
# print(Beta.shape)
# print(alphaR.shape)
# print(alphaL.shape)
# print(C7.shape)
# print(Val_data.shape)

# plt.rcParams.update({'font.size': 15})

# plt.plot(Val_data[:,0], Val_data[:,1],'.k', markersize=2)
# plt.plot(PII[:,0], PII[:,1],'.', markersize=2, label='PII')
# plt.plot(Beta[:,0], Beta[:,1],'.', markersize=2, label = 'Beta')
# plt.plot(alphaR[:,0], alphaR[:,1],'.', markersize=2, label = 'alphaR')
# plt.plot(alphaL[:,0], alphaL[:,1],'.', markersize=2, label = 'alphaL')
# plt.plot(C7[:,0], C7[:,1],'.', markersize=2, label = 'C7')
# plt.axis([-180, 180, -180, 180])
# plt.xticks(np.arange(-180, 180, step=60))
# plt.legend()
# plt.xlabel('$\phi$')
# plt.ylabel('$\psi$')
# plt.show()
# bins = np.arange(0,360,15)+7.5

# n, bins, patches = plt.hist(PII[:,2], bins,density=True, histtype='step')
# nB, bins, patches = plt.hist(Beta[:,2], bins,density=True, histtype='step')
# nR, bins, patches = plt.hist(alphaR[:,2], bins,density=True, histtype='step')
# nL, bins, patches = plt.hist(alphaL[:,2], bins,density=True, histtype='step')
# nC, bins, patches = plt.hist(C7[:,2], bins,density=True, histtype='step')
# plt.show()
# print(bins)
# bins = np.arange(15,360,15)
# print(bins)

# plt.rcParams.update({'font.size': 15})
# plt.plot(bins[0:23], n,'k',label='PII')
# plt.plot(bins[0:23], nB, 'r',label='beta')
# plt.plot(bins[0:23], nR, 'g',label='alpha_R')
# plt.legend()
# plt.xticks(np.arange(0,360, step=60))
# plt.xlabel('$\chi_1$')
# plt.ylabel('P($\chi_1$)')
# plt.axis([0, 360, 0, 0.035])
# plt.show()
