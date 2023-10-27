import pickle
import numpy as np
import matplotlib.pyplot as plt

path = '../simulations/2D_fitting/ICN1000_M100_long/ICN1000_M100_long'
aspect = 'auto'

def log_fit(x, a, b):
    return x**a * np.exp(b)

# read the 2d grid simulation
with open(path+'_rs.pl', 'rb') as f:
    rs = pickle.load(f)
with open(path+'_pars.pl', 'rb') as f:
    pars = pickle.load(f)

ICN, M1, M2, _ = rs.shape
pars1, pars2 = pars
rs_healthy = rs[:, :, :, 0].reshape(ICN, M1, M2)
rs_glioma =  rs[:, :, :, 1].reshape(ICN, M1, M2)
reg_healthy = rs[:, :, :, 2].reshape(ICN, M1, M2)
reg_glioma =  rs[:, :, :, 3].reshape(ICN, M1, M2)

# PLOT THE RESULTS
# Plot the healthy 2D grid using imshow
rs_healthy_mean = np.mean(rs_healthy,axis=0)
rs_glioma_mean = np.mean(rs_glioma,axis=0)
reg_healthy_mean = np.mean(reg_healthy,axis=0)
reg_glioma_mean = np.mean(reg_glioma,axis=0)


# create log fit
healthy_coupling = log_fit(pars2, 0.49, 1.38)
glioma_coupling = log_fit(pars2, 0.48, 1.59)


# plot
plt.figure()
plt.imshow(rs_healthy_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.plot(pars2, healthy_coupling, linestyle='--', color='blue')
plt.colorbar(label='Pearson correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.title('Control group fit')
plt.savefig('../plots/optimal_curve_on_grid_healthy.pdf', dpi=300)

plt.figure()
plt.imshow(rs_glioma_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.plot(pars2, glioma_coupling, linestyle='--', color='red')
plt.colorbar(label='Pearson correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.title('Patient group fit')
plt.savefig('../plots/optimal_curve_on_grid_glioma.pdf', dpi=300)
#plt.show()

