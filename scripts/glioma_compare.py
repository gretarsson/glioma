import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------
# Here we read data for optimization of control and craniotomy
# for average healthy and average patient functional connectivity
# --------------------------------------------------------------
# where to save figures
fig_save = './figures/glioma_FINAL/CRAN_NORMAL_DIFF/'

# where to find craniotomoy data and control
data_orig_path = './figures/glioma/fitting/pearson_patients/normal_big/'
data_cran_path = './figures/glioma/fitting/pearson_patients/craniotomy_big/'
#data_orig_path = './figures/glioma/fitting/pearson_patients/h_decay_p90/'
#data_cran_path = './figures/glioma/fitting/pearson_patients/h_decay_p90_allT/'

# load data
orig_healthy_pars = np.array(pickle.load( open( data_orig_path + 'healthy_pars.pl', "rb" ) ))[0]
orig_patient_pars = np.array(pickle.load( open( data_orig_path + 'patient_pars.pl', "rb" ) ))[0] 
cran_healthy_pars = np.array(pickle.load( open( data_cran_path + 'healthy_pars.pl', "rb" ) ))[0] 
cran_patient_pars = np.array(pickle.load( open( data_cran_path + 'patient_pars.pl', "rb" ) ))[0] 

# compute differences
ctrl = cran_healthy_pars - orig_healthy_pars 
patn = cran_patient_pars - orig_patient_pars 

# plot the damn thing
M = orig_healthy_pars.shape[0]
colours = sns.color_palette("hls", 8)
size=20

# with baseline
fig1 = plt.figure()
ax1 = fig1.add_subplot()
yvals = np.array([i for i in range(M)])
ax1.scatter(patn[:,0]-ctrl[:,0], yvals, color='grey', s=size)
ax1.set_xlim(xmin=-max(np.abs(plt.xlim())), xmax=max(np.abs(plt.xlim())))
ax1.set_ylabel('Initial condition')
ax1.set_xlabel('Coupling strength increase after craniotomy from baseline')
fig1.savefig(fig_save+'parDiffBase.pdf',dpi=300)

# without baseline
fig2 = plt.figure()
ax2 = fig2.add_subplot()
yvals = np.array([i for i in range(M)])
ax2.scatter(ctrl[:,0], yvals, color=colours[-3],s=size)
ax2.scatter(patn[:,0], yvals+M, color=colours[0],s=size)
ax2.set_xlim(xmin=-max(np.abs(plt.xlim())), xmax=max(np.abs(plt.xlim())))
ax2.set_ylabel('Initial condition')
ax2.set_xlabel('Coupling strength increase after craniotomy')
fig2.savefig(fig_save+'parDiff.pdf',dpi=300)

# print the average
print('------------------------------------------')
print('Healty original')
print(f'Average h: {np.nanmean(orig_healthy_pars[:,0])}')
print(f'Average decay: {np.nanmean(orig_healthy_pars[:,1])}')
print('------------------------------------------')
print('Glioma original')
print(f'Average h: {np.nanmean(orig_patient_pars[:,0])}')
print(f'Average decay: {np.nanmean(orig_patient_pars[:,1])}')
print('------------------------------------------')
print('Healty craniotomy')
print(f'Average h: {np.nanmean(cran_healthy_pars[:,0])}')
print(f'Average decay: {np.nanmean(cran_healthy_pars[:,1])}')
print('------------------------------------------')
print('Glioma craniotomy')
print(f'Average h: {np.nanmean(cran_patient_pars[:,0])}')
print(f'Average decay: {np.nanmean(cran_patient_pars[:,1])}')

