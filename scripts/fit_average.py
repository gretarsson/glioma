import pickle
import numpy as np
import symengine as sym
from solve_brain.brain_models import compile_hopf, solve_dde, error_FC, random_initial, threshold_matrix
from solve_brain.brain_analysis import PLI, butter_bandpass_filter, plot_functional_connectomes
#from solve_brain.brain_analysis import plot_functional_connectomes, PLI, butter_bandpass_filter
from scipy.optimize import fmin, differential_evolution
from scipy.stats import ttest_ind, kstest
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D 
import networkx as nx
import time
import pandas as pd
import colorcet as cc
from joblib import Parallel, delayed
import dill as pickle
from math import pi
from glioma_helpers import parallell_optimize
from jitcdde import jitcdde
plt.style.use('seaborn')
np.random.seed(2)
#np.random.seed(2)
#np.random.seed(2)


# ---------------------------------------------------
# in this script we optimize the hopf model
# using an optimization procedure (fmin).
# ---------------------------------------------------

# run twice for now
random_init = True
run = False
craniotomy = False  # whether to inclde/exclude tumor regions

# run multiple times (to compare between runs)
M = 100
n_jobs = min(70,M)
maxiter = 100
objective = 'pearson'

# PATH NAMES
DE_file = '../simulations/hopf.so'
mean_struct_conn_path = '../data/glioma_struct_conns_avg.p'
exp_PLI_path = '../data/exp_PLI_updated.p'
exp_PLI_glioma_path = '../data/exp_PLI_glioma.p'

# PROCESSING SETTINGS
normalize_exp = True  # inside minimize function
threshold_exp = 0.972  # inside minimize function (0.972 healthy, 0.977 avg patient)
threshold_plot = 0.9  # threshold for plotting exp. FC

# ODE SETTINGS
step = 1/1250
atol = 10**-6
rtol = 10**-3
tspan = (0,14.5)  # (0,14)
cutoff = 1
band = [8,12]

# OPTIMIZATION SETTINGS
tol = 0.01  #0.01
recombination = 0.3  #0.7
popsize = 50  #15
mutation = (0.5,1.0)  #(0.5,1)

# READ STRUCTURAL CONNECTOME, AND FIND #NODES AND #SUBJECTS
mean_W = pickle.load( open( mean_struct_conn_path, "rb" ) )
G = nx.from_numpy_matrix(mean_W)
N = mean_W.shape[0]  # number of nodes

# SET INITIAL CONDITIONS
if random_init:
    y0 = [random_initial(N) for _ in range(M)]
else:
    y0_m = random_initial(N)
    y0 = [y0_m for _ in range(M)]

# READ EXPERIMENTAL FREQUENCIES
freq_f = '../data/exp_frequencies.csv'
freqss = np.genfromtxt(freq_f, delimiter=',')[1:,1:]
freqs = np.mean(freqss, axis=0)
freqs = freqs[0:N]

# READ TUMOR INDICES
tumor_indss_f = '../data/patients_tumor_overlaps.csv'
tumor_indss = pd.read_csv(tumor_indss_f, sep=';').to_numpy()
tumor_indss = tumor_indss[:,1:]
n_patients, _ = tumor_indss.shape
print('Number of tumor regions in patients:')
print([ np.count_nonzero(tumor_indss[i]) for i in range(len(tumor_indss)) ])


# READ EXPERIMENTAL CONTROL PLI
print(f'\nLoading experimental PLI connectomes at {exp_PLI_path}...')
exp_PLIs = pickle.load( open( exp_PLI_path, "rb" ) )
mean_exp_PLI = np.mean(exp_PLIs, axis=0)
n_subjects = exp_PLIs.shape[0]
N = exp_PLIs.shape[1]
print('Done.')

# READ EXPERIMENTAL PATIENT PLI
print(f'\nLoading experimental patient PLI connectomes at {exp_PLI_glioma_path}...')
exp_PLIs_ps = pickle.load( open( exp_PLI_glioma_path, "rb" ) )
mean_exp_PLI_p = np.mean(exp_PLIs_ps, axis=0)
p_inds = [1,4,6,8,3,9,7,5,2,0]
print('Done.')

# IF TOLD, COMPUTE MEAN EXPERIMENTAL PLI, NORMALIZE AND THRESHOLD
## Create graph from adjacency matrix
#G = nx.from_numpy_matrix(mean_exp_PLI/np.amax(mean_exp_PLI), create_using=nx.Graph)
#clustering_coefficients = nx.clustering(G, weight='weight')
#avg_clustering_coefficient = np.mean(list(clustering_coefficients.values()))
#print("Average clustering coefficient healthy:", avg_clustering_coefficient)
#G = nx.from_numpy_matrix(mean_exp_PLI_p/np.amax(mean_exp_PLI), create_using=nx.Graph)
#clustering_coefficients = nx.clustering(G, weight='weight')
#avg_clustering_coefficient = np.mean(list(clustering_coefficients.values()))
#print("Average clustering coefficient patients:", avg_clustering_coefficient)
#plt.figure()
#plt.imshow(threshold_matrix(mean_exp_PLI, 0.975), cmap='magma')
#plt.figure()
#plt.imshow(threshold_matrix(mean_exp_PLI_p,0.975), cmap='magma')
#plt.show()

# PLOTTING PLI SETTINGS
lobe_names = ['visual', 'sensorimotor', 'NaN', 'attention', 'limbic', 'frontoparietal', 'default-mode']
colours = sns.color_palette('hls', len(lobe_names))

# READ MNI COORDINATES
mni_coords = np.loadtxt('../data/mni_coordinates.csv')
coordinates = np.zeros((N,3))
for n in range(N):
    mni_coord = mni_coords[n]
    coordinates[n,0] = mni_coord[0] 
    coordinates[n,1] = mni_coord[1] 
    coordinates[n,2] = mni_coord[2] 

# define delays 
delays = np.zeros((N,N))

# READ REGION NAMES
roi_names = np.genfromtxt('../data/roi_names.csv', dtype='str')
region_names = []
for n in range(N):
    region_names.append(roi_names[n])

# DEFINE BRAIN REGIONS 
regions = [[] for _ in range(len(lobe_names))]
region_data = np.genfromtxt('../data/regions.csv')
for n in range(N):
    lobe = int(region_data[n])-1
    regions[lobe].append(n)


# HOPF PARAMETERS
a = [1 for n in range(N)]
b = [1 for n in range(N)]
w = freqs * 2*pi   # set frequencies as mean of experimental ones
kappa = 20
decay = sym.var('decay')
h = sym.var('h')

# symbolic hopf parameters
control_pars = [h, decay]
bounds = [(10,18),(12,17)]
if threshold_exp == -1:
    bounds.append((0,1.0))

# compile hopf
print('begin compiling...')
DE = compile_hopf(N, a=a, b=b, delays=delays, t_span=tspan, \
             kappa=kappa, w=w, decay=decay, random_init=True, \
             h=h,\
             control_pars=control_pars, \
             only_a=True)
DE.save_compiled(destination=DE_file, overwrite=True)



# NEW INITIALIZATIONS
healthy_pars = []
patient_pars = []
healthy_vals = []
patient_vals = []

# OPTIMIZE
start_time = time.time()
if run:
    print(f'\nInitiating fitting...')
    start_time_patient = time.time()

    # add all tumor regions together (used for craniotomy)
    if craniotomy:
        tumor_inds = np.array([])
        for k in range(n_patients):
            tumor_inds = np.concatenate((tumor_inds,  np.nonzero(tumor_indss[k,:])[0]))
        tumor_inds = list(set(list(tumor_inds))) 
        tumor_inds = [int(tumor_inds[k]) for k in range(len(tumor_inds))]
    else:
        tumor_inds = []  

    # experimental FC
    exp_PLI_p = mean_exp_PLI_p

    ## HOPF PARAMETERS
    #a = [1 for n in range(N)]
    #b = [1 for n in range(N)]
    #w = freqs * 2*pi   # set frequencies as mean of experimental ones
    #kappa = 20
    #decay = sym.var('decay')
    #h = sym.var('h')

    ## symbolic hopf parameters
    #control_pars = [h, decay]
    #bounds = [(10,18),(12,17)]
    #if thres_h == -1:
    #    bounds.append((0,1.0))

    ## compile hopf
    #print('begin compiling...')
    #DE = compile_hopf(N, a=a, b=b, delays=delays, t_span=tspan, \
    #             kappa=kappa, w=w, decay=decay, random_init=True, \
    #             h=h,\
    #             control_pars=control_pars, \
    #             only_a=True)
    #DE.save_compiled(destination=DE_file, overwrite=True)
        
    # FIT AVERAGE CONTROL IN PARALLEL
    minimizes = parallell_optimize(mean_W, DE_file, control_pars, bounds, M, n_jobs, mean_exp_PLI, y0, tspan=tspan, atol=atol, rtol=rtol, cutoff=cutoff, band=band, normalize_exp=normalize_exp, threshold_exp=threshold_exp, objective=objective, popsize=popsize, opt_tol=tol, recombination=recombination, mutation=mutation, maxiter=maxiter, step=step, n=2*78, inds=tumor_inds)


    # SAVE OPTIMAL PARAMETERS
    for minimize in minimizes:
        healthy_pars.append(minimize.x)
        healthy_vals.append(minimize.fun)
    print(f'Completed fitting control')
    
    # FIT AVERAGE PATIENT IN PARALLEL
    minimizes_p = parallell_optimize(mean_W, DE_file, control_pars, bounds, M, n_jobs, mean_exp_PLI_p,  y0, tspan=tspan, atol=atol, rtol=rtol, cutoff=cutoff, band=band, normalize_exp=normalize_exp, threshold_exp=threshold_exp, objective=objective, popsize=popsize, opt_tol=tol, recombination=recombination, mutation=mutation, maxiter=maxiter, step=step, n=2*78)

    # SAVE OPTIMAL PARAMETERS
    for minimize_p in minimizes_p:
        patient_pars.append(minimize_p.x)
        patient_vals.append(minimize_p.fun)
    print(f'Completed fitting patient')


# SAVE OPTIMAL PARAMETERS
if run:
    print(f'\nSaving optimal parameters...')
    pickle.dump( healthy_pars, open( '../simulations/average_healthy_pars.pl', "wb" ) )
    pickle.dump( patient_pars, open( '../simulations/average_patient_pars.pl', "wb" ) )
    pickle.dump( healthy_vals, open( '../simulations/average_healthy_vals.pl', "wb" ) )
    pickle.dump( patient_vals, open( '../simulations/average_patient_vals.pl', "wb" ) )
    pickle.dump( y0, open( '../simulations/average_y0.pl', "wb" ) )
    print('Done.')

# print time elapsed
print(f'Whole script took: {time.time() - start_time} seconds')

# LOAD OPTIMAL PARAMETERS
print('\nLoading optimal parameters...')
healthy_pars = pickle.load( open( '../simulations/average_healthy_pars.pl', "rb" ) )
patient_pars = pickle.load( open( '../simulations/average_patient_pars.pl', "rb" ) )
healthy_vals = pickle.load( open( '../simulations/average_healthy_vals.pl', "rb" ) )
patient_vals = pickle.load( open( '../simulations/average_patient_vals.pl', "rb" ) )
y0 = pickle.load( open( '../simulations/average_y0.pl', "rb" ) )
print('Done.')

# PLOT NEW PARAMETERS
print('\n--------------------------------------------------------------------------------------------------------------------')
print('Patient\t\tParameter\tHealthy optima\tPatient optima\tAvg. difference\t\tp-value\t\thealthy success\t\tpatient success')
healthy_par = np.array(healthy_pars)
patient_par = np.array(patient_pars)
healthy_val = np.array(healthy_vals)
patient_val = np.array(patient_vals)
n_pars = healthy_par.shape[1]

# for simulating FC later
avg_par_healthy = []
avg_par_glioma = []

# ITERATE THROUGH PARAMETERS
for npar in range(n_pars):
    # extracting
    healthy_pari_full = healthy_par[:,npar]
    patient_pari_full = patient_par[:,npar]


    # remove failed optimizations
    healthy_pari = healthy_pari_full[~np.isnan(healthy_pari_full)]
    patient_pari = patient_pari_full[~np.isnan(patient_pari_full)]
    if npar == 0:
        healthy_val = healthy_val[~np.isnan(healthy_pari_full)]
        patient_val = patient_val[~np.isnan(patient_pari_full)]
    if patient_pari.size * healthy_pari.size == 0:
        break

    # SIGNIFICANCE TEST
    ks_test = kstest(healthy_pari, patient_pari)
    p_val = ks_test.pvalue
    avg_diff = np.mean(patient_pari) - np.mean(healthy_pari)
    print(f'{npar+1}\t\t{np.mean(healthy_pari)}\t\t{np.mean(patient_pari)}\t\t{round(avg_diff,2)}\t\t\t{round(p_val,10)}\t{healthy_pari.size}/{M}\t\t\t{patient_pari.size}/{M}')
    avg_par_healthy.append(np.mean(healthy_pari))
    avg_par_glioma.append(np.mean(patient_pari))

    # PLOT OBJECTIVE VALUE WITH PARAMETER VALUE
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('parameter') 
    ax.set_ylabel('objective value') 
    ax.scatter(healthy_pari, healthy_val, color=colours[-3])
    ax.scatter(patient_pari, patient_val, color=colours[0])
    ax.set_xlim([0,30])
    fig.savefig(f'../plots/fit_average/obj_par{npar}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT HEALTHY AND PATIENT POINTS
    npars_h = npar*np.ones((healthy_pari.size))
    npars_p = npar*np.ones((patient_pari.size))
    ax.scatter(npars_h, healthy_pari, color=colours[-3], label='control', alpha=0.5)
    ax.scatter(npars_p, patient_pari, color=colours[0], label='patient', alpha=0.5)

    # PLOT DIFFERENCE
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Difference in parameter') 
    ax.set_ylabel('Initial condition') 
    diff = patient_pari_full - healthy_pari_full
    diff = diff[~np.isnan(diff)]
    yvals = np.arange(diff.size)
    ax.scatter(diff, yvals, color='black')
    fulldiff = patient_par - healthy_par
    fulldiff = fulldiff[~np.isnan(fulldiff)]
    lim = np.amax([np.abs(np.amax(fulldiff)), np.abs(np.amin(fulldiff))])
    ax.set_xlim([-lim-1,lim+1])
    fig.savefig(f'../plots/fit_average/diff_par{npar}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # histograms
    plt.figure()
    binwidth = None
    sns.histplot(data=healthy_pari, binwidth=binwidth, color=colours[-3])       
    sns.histplot(data=patient_pari, binwidth=binwidth, color=colours[0], alpha=0.6)       
    pval_leg = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                              markersize=10, label=f'KS p-value {round(p_val,3)}')
    diff_leg = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label=f'mean difference {round(avg_diff,3)}')
    plt.legend(handles=[pval_leg, diff_leg])
    plt.xlim([0,30])
    plt.savefig(f'../plots/fit_average/distr_par{npar}.png', dpi=300, bbox_inches='tight')
    plt.close()
print('--------------------------------------------------------------------------------------------------------------------')
    
# Threshold experimental functional connectomes for plotting
mean_exp_PLI = threshold_matrix(mean_exp_PLI, threshold_plot) / np.amax(mean_exp_PLI)
mean_exp_PLI_p = threshold_matrix(mean_exp_PLI_p, threshold_plot) / np.amax(mean_exp_PLI_p)

# PLOT FUNCTIONAL CONNECTOMES
figs, brain_figs = plot_functional_connectomes(mean_exp_PLI, \
                     coordinates=coordinates, \
                     region_names=region_names, regions=regions, colours=colours, title=' ')
print('Done.')

# SAVE AND CLOSE EXP. AVG. PLI
figs[0].savefig('../plots/fit_average/exp_control.png', dpi=300, bbox_inches='tight')
brain_figs[0].savefig('../plots/fit_average/mni_exp_control.png', dpi=300)
plt.close(figs[0])

# PLOT STRUCTURAL CONNECTIVITY
print('\nPlotting SC...')
figs, brain_figs = plot_functional_connectomes(mean_W, coordinates=coordinates, \
                             region_names=region_names, regions=regions, \
                             colours=colours, \
                             edge_threshold='0.0%', title=' ')
print('Done.')

# SAVE AND CLOSE STRUCTURAL CONNECTIVITY
figs[0].savefig('../plots/fit_average/structural.png', dpi=300, bbox_inches='tight')
brain_figs[0].savefig('../plots/fit_average/structural_mni.png', dpi=300)
plt.close('all')

# Plot exp glioma PLI
figs, brain_figs = plot_functional_connectomes(mean_exp_PLI_p, coordinates=coordinates, \
                     region_names=region_names, regions=regions, colours=colours, title=' ')
print('Done.')

# SAVE AND CLOSE EXP. AVG. PLI
figs[0].savefig('../plots/fit_average/exp_patients.png', dpi=300, bbox_inches='tight')
brain_figs[0].savefig('../plots/fit_average/mni_exp_patients.png', dpi=300)
plt.close(figs[0])


# FIND OPTIMAL SIMULATED PLI CONTROL
print(f'\nSolving for optimal healthy dynamical model parameters..')
sol = solve_dde(DE, DE.y0, mean_W, t_span=tspan, step=step, atol=atol, rtol=rtol, \
 parameterss=np.array([avg_par_healthy]), cutoff=cutoff)
print('Done.')

# EXTRACT SOLUTION
x = sol[0]['x']
t = sol[0]['t']

# SAMPLING RATE
fs = 1/(t[1]-t[0])

# BANDPASS
x = butter_bandpass_filter(x, band[0], band[1], fs)

# COMPUTE PLI MATRIX
opt_sim_PLI = PLI(x)
opt_sim_PLI = opt_sim_PLI / np.amax(opt_sim_PLI)

# PLOT OPTIMAL SIMULATED AVERAGE PLI 
print('\nPlotting optimal simulated FC...')
# MAKE TITLE
title = ' '
figs, brain_figs = plot_functional_connectomes(opt_sim_PLI, coordinates=coordinates, \
        region_names=region_names, regions=regions, colours=colours, title=title)
print('Done.')

# SAVE AND CLOSE SIM. AVG. PLI
figs[0].savefig('../plots/fit_average/sim_PLI_patients.png', \
             dpi=300, bbox_inches='tight')
brain_figs[0].savefig('../plots/fit_average/mni_sim_PLI_patients.png', \
             dpi=300)
plt.close('all')

# FIND OPTIMAL SIMULATED PLI CONTROL
print(f'\nSolving for optimal patient dynamical model parameters..')
sol = solve_dde(DE, DE.y0, mean_W, t_span=tspan, step=step, atol=atol, rtol=rtol, \
 parameterss=np.array([avg_par_glioma]), cutoff=cutoff)
print('Done.')

# EXTRACT SOLUTION
x = sol[0]['x']
t = sol[0]['t']

# SAMPLING RATE
fs = 1/(t[1]-t[0])

# BANDPASS
x = butter_bandpass_filter(x, band[0], band[1], fs)

# COMPUTE PLI MATRIX
opt_sim_PLI = PLI(x)
opt_sim_PLI = opt_sim_PLI / np.amax(opt_sim_PLI)

# PLOT OPTIMAL SIMULATED AVERAGE PLI PATIENT P 
print('\nPlotting optimal patient simulated FC...')
# MAKE TITLE
title = ' '
figs, brain_figs = plot_functional_connectomes(opt_sim_PLI, coordinates=coordinates, \
            region_names=region_names, regions=regions, colours=colours, title=title)
print('Done.')

# SAVE AND CLOSE SIM. AVG. PLI
figs[0].savefig('../plots/fit_average/sim_PLI_control.png', \
                 dpi=300, bbox_inches='tight')
brain_figs[0].savefig('../plots/fit_average/mni_sim_PLI_control.png', dpi=300)
plt.close('all')
