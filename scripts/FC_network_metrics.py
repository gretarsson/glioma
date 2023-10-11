import symengine as sym
import numpy as np
from solve_brain.brain_models import compile_hopf, solve_dde, threshold_matrix, random_initial
from scipy.stats import pearsonr
from solve_brain.brain_analysis import PLI, butter_bandpass_filter, bandpower, compute_phase_coherence
from tqdm import tqdm
import pickle
from math import pi, ceil
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------
# in this script, we compute network metrics
# between simulated and experimental FC along 1 parameter
# -----------------------------------------------------------
np.random.seed(0)
run = True
run_metrics = True
# save paths
path = '../plots/network_metrics/'
W_path = '../data/glioma_struct_conns_avg.p'
exp_PLI_path = '../data/exp_PLI_updated.p'
freq_path = '../data/exp_frequencies.csv'

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
parmin=0;parmax=20;M=100  # (100)
ICN=50  # number of initial conditions (50)

# ODE parameters
kappa = 20
decay = 14.9
h = sym.var('h')
control_pars = [h]

# solver settings
t_span = (0,14.5)
cutoff = 1

# PLI settings
band = [8,12]
threshold = 0.972
threshold_sim = 0.972
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
sim_PLIs = np.zeros((ICN,M,N,N))
powers = np.zeros((ICN,M,N))
parvar = control_pars[0]  # for plotting and saving

if run:
    # compile ODE
    y0s = [random_initial(N) for _ in range(ICN)]
    DE = compile_hopf(N, kappa=kappa, w=w, decay=decay, h=h, \
                 control_pars=control_pars)

    # loop through parameter
    print('\nSimulating brain dynamics...')
    for i,y0 in tqdm(enumerate(y0s), total=len(y0s)):
        for j,par in tqdm(enumerate(pars), total=len(pars), leave=False):
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

            # compute bandpower
            power = [bandpower(x[n,:], fs, band) for n in range(N)]

            # store results
            sim_PLIs[i,j,:,:] = sim_PLI
            powers[i,j,:] = power
    # save files
    with open('../simulations/network_metrics_sim.pl', 'wb') as f:
        pickle.dump((sim_PLIs,powers), f)
    with open('../simulations/network_metrics_sim_pars.pl', 'wb') as f:
        pickle.dump(pars, f)

# load files
with open('../simulations/network_metrics_sim_pars.pl', 'rb') as f:
    pars = pickle.load(f)
with open('../simulations/network_metrics_sim.pl', 'rb') as f:
    (sim_PLIs,powers) = pickle.load(f)

# initialize network metrics
ICN,M,N,_ = sim_PLIs.shape
triangles = np.zeros((ICN,M))
avg_clusterings = np.zeros((ICN,M))
clusterings = np.zeros((ICN,M,N))
centralities = np.zeros((ICN,M))
# loop through to compute metrics
if run_metrics:
    print('\nComputing network metrics...')
    for i in tqdm(range(ICN)):
        for j in tqdm(range(M),leave=False):
            # extract PLI
            sim_PLI = sim_PLIs[i,j]

            # create network from PLI
            G = nx.from_numpy_matrix(sim_PLI)
            G_thresh = nx.from_numpy_matrix(threshold_matrix(sim_PLI,threshold_sim))

            # compute global clustering coefficient, by triangle method
            triangle = nx.transitivity(G_thresh)

            # compute local clustering coefficient (and their average)
            clustering = nx.clustering(G, weight='weight')  #  a dictionary    
            avg_clustering = sum(clustering.values())/len(clustering)

            # compute eigenvector centrality
            centrality = nx.eigenvector_centrality(G, weight='weight')
            avg_centrality = sum(centrality.values())/len(centrality)

            # store results
            sim_PLIs[i,j,:,:] = sim_PLI
            triangles[i,j] = triangle
            avg_clusterings[i,j] = avg_clustering
            clusterings[i,j,:] = list(clustering.values())
            centralities[i,j] = avg_centrality
        with open('../simulations/network_metrics.pl', 'wb') as f:
            pickle.dump((triangles, avg_clusterings, clusterings, centralities), f)
with open('../simulations/network_metrics.pl', 'rb') as f:
    triangles, avg_clusterings, clusterings, centralities = pickle.load(f)
            
# PLOT THE RESULTS
fig_triangle, ax_triangle = plt.subplots(1,1)
fig_avg_clustering, ax_avg_clustering = plt.subplots(1,1)
fig_centrality, ax_centrality = plt.subplots(1,1)
for i in range(ICN):
    # plot transitivity
    ax_triangle.plot(pars, triangles[i,:])
    # plot average clustering
    ax_avg_clustering.plot(pars, avg_clusterings[i,:])
    # plot average clustering
    ax_centrality.plot(pars, centralities[i,:])


# vertical line
healthy_v = 13.44
glioma_v = 15.18

# transitivity figure
parvar = control_pars[0]
ax_triangle.set_xlabel(parvar)
ax_triangle.set_ylabel('Transitivity')
ax_triangle.axvline(x=healthy_v, c='blue', alpha=0.6, linestyle='--', linewidth=3)
ax_triangle.axvline(x=glioma_v, c='red', alpha=0.6, linestyle='--', linewidth=3)
fig_triangle.savefig(path+f'triangle.pdf', dpi=300)
# average clustering figure
ax_avg_clustering.set_xlabel(parvar)
ax_avg_clustering.set_ylabel('Average clustering')
ax_avg_clustering.axvline(x=healthy_v, c='blue', alpha=0.6, linestyle='--', linewidth=3)
ax_avg_clustering.axvline(x=glioma_v, c='red', alpha=0.6, linestyle='--', linewidth=3)
fig_avg_clustering.savefig(path+f'avg_clustering.pdf', dpi=300)
# average centrality figure
ax_centrality.set_xlabel(parvar)
ax_centrality.set_ylabel('Eigenvector centrality')
ax_centrality.axvline(x=healthy_v, c='blue', alpha=0.6, linestyle='--', linewidth=3)
ax_centrality.axvline(x=glioma_v, c='red', alpha=0.6, linestyle='--', linewidth=3)
fig_centrality.savefig(path+f'centrality.pdf', dpi=300)

# plotting local clustering per regional activity (bandpower)
# average over initial conditions
powers = np.mean(powers, axis=0)
clusterings = np.mean(clusterings, axis=0)

# down-sample to down_N parameter runs
down_N = 6
down_inds = np.floor(np.linspace(0, len(pars)-1, down_N)).astype(int)
powers = powers[down_inds,:]
clusterings = clusterings[down_inds,:]

# plot clustering against activity per down-sampled iteration
for i in range(down_N):
    power = powers[i]
    clustering = clusterings[i]
    
    # sort after highest power
    sort_inds = power.argsort()
    power = power[sort_inds]
    clustering = clustering[sort_inds]

    # node versus cluster
    plt.figure()
    plt.scatter(range(N),clustering)
    plt.title(f'h = {round(pars[down_inds[i]],2)}')
    plt.xlabel('Node')
    plt.ylabel('Local clustering')
    plt.savefig('../plots/network_metrics/node_vs_cluster_{i+1}.png', dpi=300)
    plt.close()

    plt.figure()
    power_sub = power[power >= 0]
    clustering_sub = clustering[power >= 0]
    plt.scatter(power_sub,clustering_sub)
    plt.title(f'h = {round(pars[down_inds[i]],2)}')
    plt.xlabel('Bandpower')
    plt.ylabel('Local clustering')
    plt.savefig('../plots/network_metrics/power_vs_cluster_{i+1}.png', dpi=300)
    plt.close()
    
# we're done
