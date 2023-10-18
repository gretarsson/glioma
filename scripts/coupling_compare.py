import symengine as sym
import numpy as np
from solve_brain.brain_models import compile_hopf, solve_dde, threshold_matrix, random_initial
from solve_brain.brain_analysis import PLI, butter_bandpass_filter, compute_phase_coherence, compute_phase_coherence_old
from scipy.stats import pearsonr
from tqdm import tqdm
from glioma_helpers import create_directory
import pickle
from math import pi
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------
# in this script, we compute the pearson correlation
# between simulated and experimental FC along 1 parameter
# -----------------------------------------------------------
np.random.seed(0)
run = True
# save paths
folder_name = 'hopf_compare'
path = '../plots/' + folder_name
W_path = '../data/glioma_struct_conns_avg.p'
exp_PLI_path = '../data/exp_PLI_updated.p'
gli_PLI_path = '../data/exp_PLI_glioma.p'
freq_path = '../data/exp_frequencies.csv'
create_directory('../plots/'+folder_name+'/')
create_directory('../simulations/'+folder_name+'/')

# read tumor file (to fit tumor region parameters)
tumor_indss_f = '../data/patients_tumor_overlaps.csv'
tumor_indss = pd.read_csv(tumor_indss_f, sep=';').to_numpy()
tumor_indss = tumor_indss[:,1:]
tumor_inds = np.array([])
n_patients, N = tumor_indss.shape
for k in range(n_patients):
    tumor_inds = np.concatenate((tumor_inds,  np.nonzero(tumor_indss[k,:])[0]))
tumor_inds = list(set(list(tumor_inds))) 
tumor_inds = [int(tumor_inds[k]) for k in range(len(tumor_inds))]

# parameters to vary
parmin=-10;parmax=60;M=100  # (100)
ICN=1  # number of initial conditions (50)

# tumor region parameter
#h = []
#hT = sym.var('h_tumor')
#for n in range(N):
#    if n in tumor_inds:
#        h.append(hT)
#    else:
#        h.append(13.4)

# ODE parameters
kappa = 20
#decay = 14.9
decay = sym.var('Excitability')
#h = sym.var('h')  # uncomment for changing h
h = 13.4  # optimal healthy value
#control_pars = [hT]
control_pars = [decay]

# solver settings
t_span = (0,14.5)
step = 1/1250
atol = 10**-6
rtol = 10**-3
cutoff = 1

# PLI settings
band = [8,12]
threshold = 0.972
normalize = True
true_coherence = 0.67

# load experimental PLI
exp_PLIs = pickle.load( open( exp_PLI_path, "rb" ) )
exp_PLI = np.mean(exp_PLIs, axis=0)
gli_PLIs = pickle.load( open( gli_PLI_path, "rb" ) )
gli_PLI = np.mean(gli_PLIs, axis=0)
if threshold:
    exp_PLI = threshold_matrix(exp_PLI, threshold)
    gli_PLI = threshold_matrix(gli_PLI, threshold)
if normalize:
    exp_PLI = exp_PLI / np.amax(exp_PLI)
    gli_PLI = gli_PLI / np.amax(gli_PLI)

# read adjacency matrix and experimental frequencies
W = pickle.load( open( W_path, "rb" ) )
N = W.shape[0]  
freqss = np.genfromtxt(freq_path, delimiter=',')[1:,1:]
freqs = np.mean(freqss, axis=0)
w = freqs[0:N] * 2*pi

# create pars array, and store pearson and coherence error
pars = np.linspace(parmin,parmax,M)
rs_healthy = np.zeros((ICN,M))
rs_glioma = np.zeros((ICN,M))
coh_errors = np.zeros((ICN,M))
parvar = control_pars[0]  # for plotting and saving

if run:
    # compile ODE
    y0s = [random_initial(N) for _ in range(ICN)]
    DE = compile_hopf(N, kappa=kappa, w=w, decay=decay, h=h, \
                 control_pars=control_pars)

    # loop through parameter
    for i,y0 in tqdm(enumerate(y0s), total=len(y0s)):
        for j,par in enumerate(pars):
            # solve ODE
            sol = solve_dde(DE, y0, W, t_span=t_span, \
                     parameterss=[[par]], discard_y=True, cutoff=cutoff, step=step, atol=atol, rtol=rtol)

            # extract solution
            x = sol[0]['x']
            t = sol[0]['t']

            # bandpass and compute PLI
            fs = 1/(t[1]-t[0])
            x = butter_bandpass_filter(x, band[0], band[1], fs)
            sim_PLI = PLI(x)

            # compute pearson correlation and coherence error
            r_healthy, _ = pearsonr(sim_PLI.flatten(), exp_PLI.flatten())
            r_glioma, _ = pearsonr(sim_PLI.flatten(), gli_PLI.flatten())
            #coherence_error = np.abs(np.mean(compute_phase_coherence(x)) - 0.1)
            coherence_error = np.abs(np.mean(compute_phase_coherence(x)) - true_coherence)

            # store results
            rs_healthy[i,j] = r_healthy
            rs_glioma[i,j] = r_glioma
            coh_errors[i,j] = coherence_error
    
    # save files
    with open('../simulations/'+folder_name+f'/rs_healthy_{parvar}.pl', 'wb') as f:
        pickle.dump(rs_healthy, f)
    with open('../simulations/'+folder_name+f'/rs_glioma_{parvar}.pl', 'wb') as f:
        pickle.dump(rs_glioma, f)
    with open('../simulations/'+folder_name+f'/coh_errors_{parvar}.pl', 'wb') as f:
        pickle.dump(coh_errors, f)
    with open('../simulations/'+folder_name+f'/pars_{parvar}.pl', 'wb') as f:
        pickle.dump(pars, f)

# load files
with open('../simulations/'+folder_name+f'/rs_healthy_{parvar}.pl', 'rb') as f:
    rs_healthy = pickle.load(f)
with open('../simulations/'+folder_name+f'/rs_glioma_{parvar}.pl', 'rb') as f:
    rs_glioma = pickle.load(f)
with open('../simulations/'+folder_name+f'/coh_errors_{parvar}.pl', 'rb') as f:
    coh_errors = pickle.load(f)
with open('../simulations/'+folder_name+f'/pars_{parvar}.pl', 'rb') as f:
    pars = pickle.load(f)
ICN,M = rs_healthy.shape

# plot the results
fig_pearson, ax_pearson = plt.subplots(1,1)
fig_coh, ax_coh = plt.subplots(1,1)
fig_obj, ax_obj = plt.subplots(1,1)
for i in range(ICN):
    # plot pearson correlation
    ax_pearson.plot(pars, -rs_healthy[i,:], color='blue')
    ax_pearson.plot(pars, -rs_glioma[i,:], color='red')
    # plot coherence error
    ax_coh.plot(pars, coh_errors[i,:])
    # plot objective function
    ax_obj.plot(pars, -(rs_healthy - 5*coh_errors)[i,:], color='blue')
    ax_obj.plot(pars, -(rs_glioma - 5*coh_errors)[i,:], color='red')



# add plot with mean and standard deviation of pearson correlation
plt.figure()
mean_rs_healthy = np.mean(-rs_healthy, axis=0)
std_rs_healthy = np.std(-rs_healthy, axis=0)
mean_rs_glioma = np.mean(-rs_glioma, axis=0)
std_rs_glioma = np.std(-rs_glioma, axis=0)
# Calculate the minimum indices for mean_rs_healthy and mean_rs_glioma
min_rs_healthy_idx = np.argmin(mean_rs_healthy)
min_rs_glioma_idx = np.argmin(mean_rs_glioma)
# Plot the mean line
plt.plot(pars, mean_rs_healthy, color='blue')
plt.plot(pars, mean_rs_glioma, color='red')
# Fill the area around the mean with the standard deviation
plt.fill_between(pars, mean_rs_healthy - std_rs_healthy, mean_rs_healthy + std_rs_healthy, color=(0.8, 0.8, 1.0))
plt.fill_between(pars, mean_rs_glioma - std_rs_glioma, mean_rs_glioma + std_rs_glioma, color=(1.0, 0.8, 0.8))
# Add stippled vertical lines at the minimum points
plt.axvline(pars[min_rs_healthy_idx], color='blue', linestyle='dashed', linewidth=1)
plt.axvline(pars[min_rs_glioma_idx], color='red', linestyle='dashed', linewidth=1)
# Customize the plot
plt.xlabel('Coupling strength')
plt.ylabel('Pearson correlation')
plt.savefig(path+'/pearson_std.pdf', dpi=300)
plt.close()

# pearson figure
parvar = control_pars[0]
ax_pearson.set_xlabel(parvar)
ax_pearson.set_ylabel('Pearson correlation')
fig_pearson.savefig(path+f'/pearson_{parvar}.pdf', dpi=300)
# coherence figure
ax_coh.set_xlabel(parvar)
ax_coh.set_ylabel('Coherence error (1-norm)')
fig_coh.savefig(path+f'/coherence_{parvar}.pdf', dpi=300)
# objective figure
ax_obj.set_xlabel(parvar)
ax_obj.set_ylabel('Objective function')
fig_obj.savefig(path+f'/ojective_{parvar}.pdf', dpi=300)
