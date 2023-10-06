import symengine as sym
import numpy as np
from solve_brain import compile_hopf, solve_dde, threshold_matrix, random_initial

from scipy.stats import pearsonr
from network_dynamics import PLI
from fourier import butter_bandpass_filter
from feedback_helpers import compute_phase_coherence
from tqdm import tqdm
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
path = './figures/1d_fitting/'
W_path = './connectome/glioma/glioma_struct_conns_avg.p'
exp_PLI_path = './connectome/glioma/exp_PLI_updated.p'
freq_path = './connectome/glioma/exp_frequencies.csv'

# read tumor file (to fit tumor region parameters)
tumor_indss_f = './connectome/glioma/patients_tumor_overlaps.csv'
tumor_indss = pd.read_csv(tumor_indss_f, sep=';').to_numpy()
tumor_indss = tumor_indss[:,1:]
tumor_inds = np.array([])
n_patients, N = tumor_indss.shape
for k in range(n_patients):
    tumor_inds = np.concatenate((tumor_inds,  np.nonzero(tumor_indss[k,:])[0]))
tumor_inds = list(set(list(tumor_inds))) 
tumor_inds = [int(tumor_inds[k]) for k in range(len(tumor_inds))]

# parameters to vary
parmin=0;parmax=30;M=200  # (100)
ICN=50  # number of initial conditions (50)

# tumor region parameter
h = []
hT = sym.var('h_tumor')
for n in range(N):
    if n in tumor_inds:
        h.append(hT)
    else:
        h.append(13.4)

# ODE parameters
kappa = 20
decay = 14.9
#h = 13.4
control_pars = [hT]

# solver settings
t_span = (0,14.5)
cutoff = 1

# PLI settings
band = [8,12]
threshold = 0.972
normalize = True

# load experimental PLI
exp_PLIs = pickle.load( open( exp_PLI_path, "rb" ) )
exp_PLI = np.mean(exp_PLIs, axis=0)
if threshold:
    exp_PLI = threshold_matrix(exp_PLI, threshold)
if normalize:
    exp_PLI = exp_PLI / np.amax(exp_PLI)

# read adjacency matrix and experimental frequencies
W = pickle.load( open( W_path, "rb" ) )
N = W.shape[0]  
freqss = np.genfromtxt(freq_path, delimiter=',')[1:,1:]
freqs = np.mean(freqss, axis=0)
w = freqs[0:N] * 2*pi

# create pars array, and store pearson and coherence error
pars = np.linspace(parmin,parmax,M)
rs = np.zeros((ICN,M))
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
                     parameterss=[[par]], discard_y=True, cutoff=cutoff)

            # extract solution
            x = sol[0]['x']
            t = sol[0]['t']

            # bandpass and compute PLI
            fs = 1/(t[1]-t[0])
            x = butter_bandpass_filter(x, band[0], band[1], fs)
            sim_PLI = PLI(x)

            # compute pearson correlation and coherence error
            r, _ = pearsonr(sim_PLI.flatten(), exp_PLI.flatten())
            coherence_error = np.abs(np.mean(compute_phase_coherence(x)) - 0.1)

            # store results
            rs[i,j] = r
            coh_errors[i,j] = coherence_error
    
    # save files
    with open(path + f'rs_{parvar}.pl', 'wb') as f:
        pickle.dump(rs, f)
    with open(path + f'coh_errors_{parvar}.pl', 'wb') as f:
        pickle.dump(coh_errors, f)
    with open(path + f'pars_{parvar}.pl', 'wb') as f:
        pickle.dump(pars, f)

# load files
with open(path + f'rs_{parvar}.pl', 'rb') as f:
    rs = pickle.load(f)
with open(path + f'coh_errors_{parvar}.pl', 'rb') as f:
    coh_errors = pickle.load(f)
with open(path + f'pars_{parvar}.pl', 'rb') as f:
    pars = pickle.load(f)
ICN,M = rs.shape

# plot the results
fig_pearson, ax_pearson = plt.subplots(1,1)
fig_coh, ax_coh = plt.subplots(1,1)
fig_obj, ax_obj = plt.subplots(1,1)
for i in range(ICN):
    # plot pearson correlation
    ax_pearson.plot(pars, rs[i,:])
    # plot coherence error
    ax_coh.plot(pars, coh_errors[i,:])
    # plot objective function
    ax_obj.plot(pars, (rs - 5*coh_errors)[i,:])

# pearson figure
parvar = control_pars[0]
ax_pearson.set_xlabel(parvar)
ax_pearson.set_ylabel('Pearson correlation')
fig_pearson.savefig(path+f'pearson_{parvar}.pdf', dpi=300)
# coherence figure
ax_coh.set_xlabel(parvar)
ax_coh.set_ylabel('Coherence error (1-norm)')
fig_coh.savefig(path+f'coherence_{parvar}.pdf', dpi=300)
# objective figure
ax_obj.set_xlabel(parvar)
ax_obj.set_ylabel('Objective function')
fig_obj.savefig(path+f'ojective_{parvar}.pdf', dpi=300)
