import symengine as sym
import numpy as np
from solve_brain.brain_models import compile_hopf, solve_dde, threshold_matrix, random_initial
from solve_brain.brain_analysis import PLI, butter_bandpass_filter, compute_phase_coherence, PLI_from_complex
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from tqdm import tqdm
import pickle
from math import pi
from jitcdde import jitcdde
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from glioma_helpers import remove_rows_and_columns, clustering_coefficient, create_directory, compute_eigenvector_centrality, amplitude_matrix
import networkx as nx
from matplotlib.ticker import LogFormatter, LogLocator
import time
plt.style.use('seaborn-muted')
plt.rcParams['figure.max_open_warning'] = 50  

# -----------------------------------------------------------
# in this script, we compute the pearson correlation
# between simulated and experimental FC along 1 parameter
# -----------------------------------------------------------
np.random.seed(5)
run = False
craniotomy = False
loglog = True
bandpass = True

# save paths
n_jobs=100;ICN=100;M=100
file_name = f'{ICN}IC_{M}M'
plot_path = '../plots/2D_fitting/' + file_name + '/'
simu_path = '../simulations/2D_fitting/' + file_name + '/'
W_path = '../data/glioma_struct_conns_avg.p'
exp_PLI_path = '../data/exp_PLI_updated.p'
gli_PLI_path = '../data/exp_PLI_glioma.p'
exp_reg_path = '../data/exp_ampl_matrix.p'
gli_reg_path = '../data/gli_ampl_matrix.p'
freq_path = '../data/exp_frequencies.csv'
DE_file = simu_path+'hopf.so'
create_directory(plot_path)
create_directory(simu_path)

# parameters to vary
parmin1=0;parmax1=50;M1=M  
parmin2=0;parmax2=50;M2=M

# ODE parameters
kappa = 20
decay = sym.var('decay')
h = sym.var('h')  
control_pars = [h, decay]

# solver settings
t_span = (0,14.5)
cutoff = 1
step = 1/1250
atol = 10**-6
rtol = 10**-3

# PLI settings
band = [8,12]
threshold = 0.972 # 0.972
normalize = False  # True
aspect = 'auto'

# load experimental PLI
exp_PLIs = pickle.load( open( exp_PLI_path, "rb" ) )
gli_PLIs = pickle.load( open( gli_PLI_path, "rb" ) )
exp_reg = pickle.load( open( exp_reg_path, "rb" ) )
gli_reg = pickle.load( open( gli_reg_path, "rb" ) )
exp_PLI = np.mean(exp_PLIs, axis=0)
gli_PLI = np.mean(gli_PLIs, axis=0)
if threshold:
    exp_PLI = threshold_matrix(exp_PLI, threshold)
    gli_PLI = threshold_matrix(gli_PLI, threshold)
if normalize:
    exp_PLI = exp_PLI / np.amax(exp_PLI)
    gli_PLI = gli_PLI / np.amax(gli_PLI)
G_exp = nx.from_numpy_array(exp_PLI)
exp_clustering = np.mean(list(nx.clustering(G_exp, weight='weight').values()))
exp_centrality = np.mean(list(nx.eigenvector_centrality(G_exp, weight='weight').values()))

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

# read adjacency matrix and experimental frequencies
W = pickle.load( open( W_path, "rb" ) )
N = W.shape[0]; n=2*N
freqss = np.genfromtxt(freq_path, delimiter=',')[1:,1:]
freqs = np.mean(freqss, axis=0)
w = freqs[0:N] * 2*pi

# create pars array, and store pearson and coherence error
pars1 = np.linspace(parmin1,parmax1,M1)
pars2 = np.linspace(parmin2,parmax2,M2)
pars = [pars1, pars2]
rs = np.zeros((ICN,M1,M2,2+2*N+2))

if run:
    # compile ODE
    DE = compile_hopf(N, kappa=kappa, w=w, decay=decay, h=h, \
                 control_pars=control_pars)
    DE.save_compiled(destination=DE_file, overwrite=True)

    # remove tumor regions if asked
    if craniotomy:  
        exp_PLI = remove_rows_and_columns(exp_PLI, tumor_inds)
        gli_PLI = remove_rows_and_columns(gli_PLI, tumor_inds)

    for IC in range(ICN):
        # initial conditions
        y0 = random_initial(N)

       # Function to compute for a single parameter pair
        def compute_single(i, j, DE_file, n):
            par1 = pars1[i]
            par2 = pars2[j]

            # compile jitcode object from file
            DE = jitcdde(module_location=DE_file, control_pars=control_pars, max_delay=0, n=n)

            # solve ODE
            sol = solve_dde(DE, y0, W, t_span=t_span,
                            parameterss=[[par1, par2]], discard_y=False, cutoff=cutoff, step=step, atol=atol, rtol=rtol)

            # extract solution
            x = sol[0]['x']
            y = sol[0]['y']
            t = sol[0]['t']
            hil = x+1j*y

            # bandpass and compute PLI
            fs = 1 / (t[1] - t[0])
            if bandpass:
                x = butter_bandpass_filter(x, band[0], band[1], fs)
            sim_PLI = PLI(x)
            sim_reg = amplitude_matrix(x,2*fs)
            if np.amax(sim_PLI) > 10**-6:
                sim_PLI = sim_PLI / np.amax(sim_PLI)
            if np.amax(sim_reg) > 10**-6:
                sim_reg = sim_reg / np.amax(np.abs(sim_reg))

            # remove tumor regions if asked
            if craniotomy:  
                sim_PLI = remove_rows_and_columns(sim_PLI, tumor_inds)

            # compute pearson correlation 
            r_healthy, _ = pearsonr(sim_PLI.flatten(), exp_PLI.flatten())
            r_glioma, _ = pearsonr(sim_PLI.flatten(), gli_PLI.flatten())
            reg_healthy, _ = pearsonr(sim_reg.flatten(), exp_reg.flatten())
            reg_glioma, _ = pearsonr(sim_reg.flatten(), gli_reg.flatten())

            # compute clustering coefficient
            G = nx.from_numpy_array(sim_PLI)
            
            clustering = list(nx.clustering(G, weight='weight').values())
            centrality = compute_eigenvector_centrality(G)

            return (r_healthy, r_glioma, *clustering, *centrality, reg_healthy, reg_glioma)

        # Run the computation in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(compute_single)(i, j, DE_file, n) for i in tqdm(range(M1), desc=f'Initial condition {IC+1} of {ICN}') for j in range(M2)) 
        # store results 
        rs[IC] = np.array(results).reshape((M1,M2,2+2*N+2))

    # save files
    with open(simu_path+file_name+'_rs.pl', 'wb') as f:
        pickle.dump(rs, f)
    with open(simu_path+file_name+'_pars.pl', 'wb') as f:
        pickle.dump(pars, f)

# load files
with open(simu_path+file_name+'_rs.pl', 'rb') as f:
    rs = pickle.load(f)
with open(simu_path+file_name+'_pars.pl', 'rb') as f:
    pars = pickle.load(f)
ICN, M1, M2, _ = rs.shape
pars1, pars2 = pars
rs_healthy = rs[:, :, :, 0].reshape(ICN, M1, M2)
rs_glioma =  rs[:, :, :, 1].reshape(ICN, M1, M2)
clustering =  rs[:, :, :, 2:2+N].reshape(ICN, M1, M2, N)
centrality =  rs[:, :, :, 2+N:2+2*N].reshape(ICN, M1, M2, N)
reg_healthy = rs[:, :, :, 2+2*N].reshape(ICN, M1, M2)
reg_glioma =  rs[:, :, :, 2+2*N+1].reshape(ICN, M1, M2)

# PLOT THE RESULTS
# Plot the healthy 2D grid using imshow
rs_healthy_mean = np.mean(rs_healthy,axis=0)
rs_glioma_mean = np.mean(rs_glioma,axis=0)
reg_healthy_mean = np.mean(reg_healthy,axis=0)
reg_glioma_mean = np.mean(reg_glioma,axis=0)

# averages of network metrics over initial conditions and nodes
clustering = np.mean(clustering,axis=0)
clustering_mean = np.mean(clustering,axis=2)
centrality = np.mean(centrality,axis=0)
centrality_mean = np.mean(centrality,axis=2)

# healthy grid 
plt.figure()
plt.imshow(rs_healthy_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Pearson correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+ file_name+'_pearson_grid_healthy.png',dpi=300)

# glioma grid
plt.figure()
plt.imshow(rs_glioma_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Pearson correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_pearson_grid_glioma.png',dpi=300)

# healthy regularization grid 
plt.figure()
plt.imshow(reg_healthy_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Regularization correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+ file_name+'_regularization_grid_healthy.png',dpi=300)

# glioma regularization grid
plt.figure()
plt.imshow(reg_glioma_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Regularization correlation')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_regularization_grid_glioma.png',dpi=300)

# clustering grid
plt.figure()
plt.imshow(clustering_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Mean clustering')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_clustering_grid.png',dpi=300)

# centrality grid
plt.figure()
plt.imshow(centrality_mean, cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Mean centrality')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_centrality_grid.png',dpi=300)

# clustering error
plt.figure()
plt.imshow(np.abs(clustering_mean-exp_clustering), cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Mean clustering error')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_clustering_err_grid.png',dpi=300)

# centrality error
plt.figure()
plt.imshow(np.abs(centrality_mean-exp_centrality), cmap='magma', extent=[pars2.min(), pars2.max(), pars1.min(), pars1.max()],
           interpolation='nearest', origin='lower', aspect=aspect)
plt.colorbar(label='Mean centrality error')
plt.xlabel('Excitability')
plt.ylabel('Coupling Strength')
plt.savefig(plot_path+file_name+'_centrality_err_grid.png',dpi=300)

# plot difference in optimal coupling strength
plt.figure()
max_coupling_healthy_all = np.zeros((ICN,M2))
max_coupling_glioma_all = np.zeros((ICN,M2))
for IC in range(ICN):
    rs_healthy_i = rs_healthy[IC]
    rs_glioma_i = rs_glioma[IC]
    # Find the coupling strength index with the highest value for each excitability value
    max_coupling_idx_healthy = np.argmax(rs_healthy_i, axis=0)
    max_coupling_idx_glioma = np.argmax(rs_glioma_i, axis=0)
    max_coupling_healthy = pars1[max_coupling_idx_healthy]
    max_coupling_glioma = pars1[max_coupling_idx_glioma]
    max_coupling_healthy_all[IC,:] = max_coupling_healthy
    max_coupling_glioma_all[IC,:] = max_coupling_glioma

    # Plot different in optimal coupling
    plt.scatter(max_coupling_glioma - max_coupling_healthy, [IC for _ in range(M2)], alpha=0.5, c='grey')
plt.savefig(plot_path+file_name+'_optimal_coupling.png',dpi=300)

# plot mean and std of all inital conditions
if loglog:
    plt.figure()
    mean_coupling_healthy = np.mean(max_coupling_healthy_all, axis=0)
    std_coupling_healthy = np.std(max_coupling_healthy_all, axis=0)
    mean_coupling_glioma = np.mean(max_coupling_glioma_all, axis=0)
    std_coupling_glioma = np.std(max_coupling_glioma_all, axis=0)
    # Plot the mean line
    plt.plot(pars2, mean_coupling_healthy, color='blue')
    plt.plot(pars2, mean_coupling_glioma, color='red')
    # Fill the area around the mean with the standard deviation
    plt.fill_between(pars2, mean_coupling_healthy - std_coupling_healthy, mean_coupling_healthy + std_coupling_healthy, color=(0.8, 0.8, 1.0), alpha=0.7)
    plt.fill_between(pars2, mean_coupling_glioma - std_coupling_glioma, mean_coupling_glioma + std_coupling_glioma, color=(1.0, 0.8, 0.8), alpha=0.7)
    # Customize the plot
    plt.xlabel('Excitability')
    plt.ylabel('Optimal coupling strength')
    plt.savefig(plot_path+file_name+'_optimal_coupling_std.png',dpi=300)

    # remove 0 from the data due to log transformation
    pars2 = pars2[1:]
    mean_coupling_healthy = mean_coupling_healthy[1:]
    mean_coupling_glioma = mean_coupling_glioma[1:]
    # Apply log transformations to the data
    log_pars2 = np.log(pars2)
    log_mean_coupling_healthy = np.log(mean_coupling_healthy)
    log_mean_coupling_glioma = np.log(mean_coupling_glioma)

    plt.figure(figsize=(8, 6)) 
    plt.loglog(pars2, mean_coupling_healthy, color='blue')
    plt.loglog(pars2, mean_coupling_glioma, color='red')

    # Define a linear fitting function
    def linear_fit(x, a, b):
        return a * x + b
    def log_fit(x, a, b):
        return x**a * np.exp(b)
    # Fit a straight line to the log-log plot for the healthy data
    params_h, covariance_h = curve_fit(linear_fit, log_pars2, log_mean_coupling_healthy)
    a_h, b_h = params_h
    slope_h = a_h
    intercept_h = b_h

    # Fit a straight line to the log-log plot for the glioma data
    params_g, covariance_g = curve_fit(linear_fit, log_pars2, log_mean_coupling_glioma)
    a_g, b_g = params_g
    slope_g = a_g
    intercept_g = b_g

    # Plot the fitted lines
    plt.plot(pars2, log_fit(pars2, *params_h), '--', color='blue', label=f'Fit (Healthy): y = {slope_h:.2f}x + {intercept_h:.2f}')
    plt.plot(pars2, log_fit(pars2, *params_g), '--', color='red', label=f'Fit (Glioma): y = {slope_g:.2f}x + {intercept_g:.2f}')
    plt.xlabel('Log(excitability)')
    plt.ylabel('Log(optimal coupling)')
    plt.legend()
    plt.savefig(plot_path+file_name+'_loglog.png',dpi=300)



# now that we have the scaling, we can look at network metrics and nodes
excs = [5, 15, 25, 35, 45]
for exc in excs:
    # pick an excitability parameter value and find corresponding coupling strengths
    cou_h = log_fit(exc, *params_h)
    cou_g = log_fit(exc, *params_g)


    # find the closest grid point
    index_exc = min(range(len(pars2)), key=lambda i: abs(pars2[i] - exc))
    index_cou_h = min(range(len(pars1)), key=lambda i: abs(pars1[i] - cou_h))
    index_cou_g = min(range(len(pars1)), key=lambda i: abs(pars1[i] - cou_g))
    index_h = [index_exc, index_cou_h]
    index_g = [index_exc, index_cou_g]


    # iterate through the nodes, plotting node degree vs clustering/centrality
    GW = nx.from_numpy_array(W)
    node_degrees = []
    clustering_hs = []
    clustering_gs = []
    centrality_hs = []
    centrality_gs = []
    for i in range(N):
        # healthy
        clustering_h = clustering[index_cou_h, index_exc, i]
        centrality_h = centrality[index_cou_h, index_exc, i]
        clustering_hs.append(clustering_h)
        centrality_hs.append(centrality_h)

        # toxic
        clustering_g = clustering[index_cou_g, index_exc, i]
        centrality_g = centrality[index_cou_g, index_exc, i]
        clustering_gs.append(clustering_g)
        centrality_gs.append(centrality_g)

        # compute node_degre
        node_degree = GW.degree(i, weight='weight')
        node_degrees.append(node_degree)


    # plotting clustering
    plt.figure()
    node_degrees=np.array(node_degrees);clustering_hs=np.array(clustering_hs)
    clustering_gs=np.array(clustering_gs);centrality_hs=np.array(centrality_hs)
    centrality_gs=np.array(centrality_gs)
    #plt.scatter(node_degrees, clustering_hs, c='blue')
    #plt.scatter(node_degrees, clustering_gs, c='red')
    clust_diff = clustering_gs-clustering_hs
    plt.scatter(node_degrees, clust_diff, c='grey')
    plt.axhline(y=np.mean(clust_diff), color='black', linestyle='--')
    plt.xlabel('Weighted node degree')
    plt.ylabel('Clustering coefficient cohort difference')
    plt.savefig(plot_path+file_name+f'_clustering_nodes_exc{exc}.png',dpi=300, bbox_inches='tight')

    # plotting centrality
    plt.figure()
    #plt.scatter(node_degrees, centrality_hs, c='blue')
    #plt.scatter(node_degrees, centrality_gs, c='red')
    centr_diff = centrality_gs-centrality_hs
    plt.scatter(node_degrees, centr_diff, c='grey')
    plt.scatter(node_degrees, centr_diff, c='grey')
    plt.axhline(y=np.mean(centr_diff), color='black', linestyle='--')
    plt.xlabel('Weighted node degree')
    plt.ylabel('Eigenvector centrality cohort difference')
    plt.savefig(plot_path+file_name+f'_centrality_nodes_exc{exc}.png',dpi=300, bbox_inches='tight')

