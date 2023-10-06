import pickle
import numpy as np
import symengine as sym
from solve_brain import compile_hopf, compile_hopf_diff, solve_dde, error_FC, random_initial, threshold_matrix
from network_dynamics import plot_functional_connectomes, PLI
from fourier import butter_bandpass_filter
from scipy.optimize import fmin, differential_evolution
from scipy.stats import ttest_ind, kstest
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D 
import networkx as nx
import time
#import skopt 
import pandas as pd
import colorcet as cc
from joblib import Parallel, delayed
import dill as pickle
from math import pi
plt.style.use('seaborn')
np.random.seed(2)
#np.random.seed(2)
#np.random.seed(2)


# ---------------------------------------------------
# in this script we optimize the hopf model
# using an optimization procedure (fmin).
# ---------------------------------------------------

# run twice for now
fig_save_path = './figures/glioma/fitting/pearson_patients/h_hT_decay/'
random_freqs = False
freq_normal = False
random_init = True
run = True
plot = False

# run multiple times (to compare between runs)
M = 500
n_jobs = min(10,M)
maxiter = 200
objective = 'pearson'

# PATH NAMES
mean_struct_conn_path = './connectome/glioma/glioma_struct_conns_avg.p'
exp_PLI_path = './connectome/glioma/exp_PLI_updated.p'
exp_PLI_glioma_path = './connectome/glioma/exp_PLI_glioma.p'
normalize = False  # normalize exp. data
threshold = False  # threshold exp. data
threshold_perc = 0.90 # of individual matrices
normalize_exp = True  # inside minimize function
threshold_exp = 0.975  # inside minimize function (0.972 healthy, 0.977 avg patient)
thres_h = 0.972
thres_p = 0.972
#threshold_exp = -1
#threshold_exp = 0.05  # inside minimize function
if objective == 'jaccard':
    threshold_exp = -1

# SETTINGS
step = 1/1250
atol = 10**-6
rtol = 10**-3
tspan = (0,14.5)  # (0,14)
cutoff = 1
band = [8,12]

# optimization parameters
tol = 0.01  #0.01
recombination = 0.3  #0.7
popsize = 50  #15
mutation = (0.5,1.0)  #(0.5,1)
max_delay_c = 20
zero_scale = 0
#tol = 0.001  #0.01
#recombination = 0.3  #0.7
#popsize = 30  #15
#mutation = (0.5,1.0)  #(0.5,1)
#max_delay_c = 20
#zero_scale = 0

# READ STRUCTURAL CONNECTOME, AND FIND #NODES AND #SUBJECTS
mean_W = pickle.load( open( mean_struct_conn_path, "rb" ) )
G = nx.from_numpy_matrix(mean_W)
N = mean_W.shape[0]  # number of nodes

# random single initial condition
if random_init:
    y0 = [random_initial(N) for _ in range(M)]
else:
    y0_m = random_initial(N)
    y0 = [y0_m for _ in range(M)]

# READ EXPERIMENTAL FREQUENCIES
freq_f = './connectome/glioma/exp_frequencies.csv'
freqss = np.genfromtxt(freq_f, delimiter=',')[1:,1:]
freqs = np.mean(freqss, axis=0)
freqs = freqs[0:N]

# READ TUMOR INDICES
tumor_indss_f = './connectome/glioma/patients_tumor_overlaps.csv'
tumor_indss = pd.read_csv(tumor_indss_f, sep=';').to_numpy()
tumor_indss = tumor_indss[:,1:]
print('Number of tumor regions in patients:')
print([ np.count_nonzero(tumor_indss[i]) for i in range(len(tumor_indss)) ])

n_patients, _ = tumor_indss.shape
opt_pars_p = [ [] for _ in range(n_patients)]  # storing fitted parameters for each patient
opt_vals_p = [ [] for _ in range(n_patients)]  # similarily for optimal objective values
opt_pars = [ [] for _ in range(n_patients)]
opt_vals = [ [] for _ in range(n_patients)]


# FIND INTERHEMISPHERIC EDGE INDICES
inter_idx = []
roi_names= np.genfromtxt('./connectome/glioma/roi_names.csv', dtype='str')
edgesx, edgesy = np.nonzero(mean_W)
for i in range(edgesx.size):
    ind1 = edgesx[i]
    ind2 = edgesy[i]
    node1 = roi_names[ind1]
    node2 = roi_names[ind2]
    if node1[-1] != node2[-1]:
        inter_idx.append((ind1,ind2))

# READ EXPERIMENTAL CONTROL PLIS
print(f'\nLoading experimental PLI connectomes at {exp_PLI_path}...')
exp_PLIs = pickle.load( open( exp_PLI_path, "rb" ) )
n_subjects = exp_PLIs.shape[0]
N = exp_PLIs.shape[1]
print('Done.')

# READ EXPERIMENTAL PATIENT PLIS
print(f'\nLoading experimental patient PLI connectomes at {exp_PLI_glioma_path}...')
exp_PLIs_ps = pickle.load( open( exp_PLI_glioma_path, "rb" ) )
p_inds = [1,4,6,8,3,9,7,5,2,0]
print('Done.')

# IF TOLD, NORMALIZE AND THRESHOLD EACH SUBJECT'S PLI BEFORE AVERAGING
if normalize:
    for s in range(n_subjects):
        # controls
        exp_max = np.amax(exp_PLIs[s]) 
        exp_PLIs[s] = exp_PLIs[s] / exp_max
    for p in range(n_patients):
        # patients
        exp_max = np.amax(exp_PLIs_ps[p]) 
        exp_PLIs_ps[p] = exp_PLIs_ps[p] / exp_max
if threshold:
    for s in range(n_subjects):
        exp_PLI_flat = exp_PLIs[s].flatten()
        exp_PLI_flat = np.sort(exp_PLI_flat)  # sort lowest to highest
        threshold_val = exp_PLI_flat[int( threshold_perc * exp_PLI_flat.size ) ]
        exp_PLIs[s][exp_PLIs[s] < threshold_val] = 0
    for p in range(n_patients):
        exp_PLI_ps_flat = exp_PLIs_ps[p].flatten()
        exp_PLI_ps_flat = np.sort(exp_PLI_ps_flat)  # sort lowest to highest
        threshold_val = exp_PLI_ps_flat[int( threshold_perc * exp_PLI_ps_flat.size ) ]
        exp_PLIs_ps[p][exp_PLIs_ps[p] < threshold_val] = 0

# IF TOLD, COMPUTE MEAN EXPERIMENTAL PLI, NORMALIZE AND THRESHOLD
mean_exp_PLI = np.mean(exp_PLIs, axis=0)
mean_exp_PLI_p = np.mean(exp_PLIs_ps, axis=0)
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
mni_coords = np.loadtxt('./connectome/glioma/mni_coordinates.csv')
coordinates = np.zeros((N,3))
for n in range(N):
    mni_coord = mni_coords[n]
    coordinates[n,0] = mni_coord[0] 
    coordinates[n,1] = mni_coord[1] 
    coordinates[n,2] = mni_coord[2] 

# define delays (from distances in mni coordinates) and bin them
delay_dim = 40
delays = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if mean_W[i][j] > 0:
            delays[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
bins = np.linspace(0, np.amax(delays), delay_dim)
binned_delays = np.digitize(delays, bins) 
delays = (bins[1] - bins[0]) * binned_delays
delays = (0.02 / np.mean(delays)) * delays  # set average delay at 0.02 (Bick's avg)
max_delay = np.amax(delays)*max_delay_c
delays = np.zeros((N,N))

# READ REGION NAMES
roi_names = np.genfromtxt('./connectome/glioma/roi_names.csv', dtype='str')
region_names = []
for n in range(N):
    region_names.append(roi_names[n])

# DEFINE BRAIN REGIONS 
regions = [[] for _ in range(len(lobe_names))]
region_data = np.genfromtxt('./connectome/glioma/regions.csv')
for n in range(N):
    lobe = int(region_data[n])-1
    regions[lobe].append(n)

# NEW INITIALIZATIONS
healthy_pars = []
patient_pars = []
healthy_vals = []
patient_vals = []
patient_ids = [1,2,4,5,6,7,8,9]
#patient_ids = [5]
patient_ids = [4]

# ITERATE THROUGH PATIENTS
start_time = time.time()
if run:
    #for p in range(n_patients):  
    for i, p in enumerate(patient_ids):  
        healthy_pars.append([])  # adding list to fill for patient p
        patient_pars.append([])
        healthy_vals.append([])  # adding list to fill for patient p
        patient_vals.append([])
        print(f'\nInitiating fitting of patient {p}...')
        start_time_patient = time.time()
        tumor_inds = np.nonzero(tumor_indss[p,:])[0]
        # add all tumor regions together
        # ----------------------------------------
        tumor_inds = np.array([])
        for k in range(n_patients):
            tumor_inds = np.concatenate((tumor_inds,  np.nonzero(tumor_indss[k,:])[0]))
        tumor_inds = list(set(list(tumor_inds))) 
        tumor_inds = [int(tumor_inds[k]) for k in range(len(tumor_inds))]
        print(f'with {len(tumor_inds)} parameters...')
        # ----------------------------------------

        #exp_PLI_p = exp_PLIs_ps[p_inds[p],:,:]
        exp_PLI_p = mean_exp_PLI_p

        # HOPF PARAMETERS
        if random_freqs:
            freq_var = 0.0
            freqs_dist = np.random.normal(10, freq_var, size=N)
            w = [freqs[n]*(2*sym.pi) for n in range(N)]
        else:
            w = freqs * 2*pi   # set frequencies as mean of experimental ones

        # free up a and b tumor parameters
        #a = []
        #a_tumor = []
        #b = []
        #b_tumor = []
        #for n in range(N):
        #    if n in np.nonzero(tumor_inds)[0]:
        #        a_n = sym.var(f'a_{n}')
        #        b_n = sym.var(f'b_{n}')
        #        a.append(a_n)
        #        b.append(b_n)
        #        a_tumor.append(a_n)
        #        b_tumor.append(b_n)
        #    else:
        #        a.append(1)
        #        b.append(1)

        # free up decay tumor parameters
        #a = [1 for n in range(N)]
        #b = [1 for n in range(N)]
        #decay_tumor = []
        #decay = []
        #for n in range(N):
        #    if n in tumor_inds:
        #        decay_n = sym.var(f'decay_{n}')
        #        decay_tumor.append(decay_n)
        #        decay.append(decay_n)
        #    else:
        #        decay.append(14.5)

        # free up h tumor parameters as one
        a = [1 for n in range(N)]
        b = [1 for n in range(N)]
        h = []
        hT = sym.var('hT')
        hG = sym.var('hG')
        for n in range(N):
            if n in tumor_inds:
                h.append(hT)
            else:
                h.append(hG)
                
        #h = 12.9
        #h = sym.var('h')
        #kappa = 36.2
        #kappa = sym.var('kappa')
        kappa = 20
        #decay = 14.5
        decay = sym.var('decay')
        #decay = 15
        #h = sym.var('h')
        #decay = 14.89
        #decay = sym.var('decay')
        inter_c = 1
        delay_c = 1
        #print(f'with {len(a_tumor)} parameters to tune...')

        # symbolic hopf parameters
        control_pars = [hG, hT, decay]
        #bounds = [ (0.01, 10) for _ in range(len(control_pars)) ]  # for a and b
        #bounds = [ (-50, 50) for _ in range(len(control_pars)) ]  # for decays
        #bounds = [ (0,30), (10,20) ]
        bounds = [(0,30),(0,30),(13,17)]
        if thres_h == -1:
            bounds.append((0,1.0))
        #popsize = 15
        #opt_init = 'latinhypercube'
        #opt_init = np.random.normal(loc=0.5, scale=0.25, size=(15,len(control_pars)))
        #opt_init = np.clip(opt_init,0.1,1.0)
        #tol = 1e-2  #0.01
        # define simpler function for parallelizing 
        def optimize(varl): 
            exp_PLI = varl[0]
            y0 = varl[1]
            threshold_exp = varl[2]
            DE = compile_hopf(N, a=a, b=b, delays=delays, t_span=tspan, \
                         kappa=kappa, w=w, decay=decay, random_init=True, \
                         h=h, inter_c=inter_c, \
                        inter_idx=inter_idx, delay_c=delay_c, control_pars=control_pars, \
                        only_a=True, max_delay=max_delay)
            minimize = differential_evolution(error_FC, bounds, args=(DE, mean_W, tspan, step, \
                            atol, rtol, cutoff, band, exp_PLI, normalize_exp, threshold_exp,  \
                            False, False, zero_scale, y0, [], objective, freq_normal), \
                            disp=False, strategy='best1bin', popsize=popsize, init='latinhypercube', tol=tol, \
                            recombination=recombination, mutation=mutation, maxiter=maxiter)
            del DE
            print('One down!')
            print(f'success: {minimize.success}')
            print(f'number of iterations: {minimize.nit}')
            print(f'minimize.x = {minimize.x}')
            if minimize.success:
                return minimize
            else:
                npars = len(minimize.x)
                sols = [ np.nan for _ in range(npars) ] 
                minimize.x = sols
                return minimize

        # FIT CONTROLS IN PARALLEL
        print('\n\nBegin control parallelization...')
        start_time_h = time.time()
        minimizes = Parallel(n_jobs=n_jobs, prefer=None)(delayed(optimize)([mean_exp_PLI, y0[m], thres_h]) for m in range(M))        
        print(f'Healthy control optimization took: {time.time() - start_time_h} seconds')
        print(f'One average optimization took: {(time.time() - start_time_h)/M} seconds')


        # SAVE OPTIMAL PARAMETERS
        for minimize in minimizes:
            healthy_pars[i].append(minimize.x)
            healthy_vals[i].append(minimize.fun)
        print(f'Fitted controls for patient {p}')

        # free up h tumor parameters as one
        a = [1 for n in range(N)]
        b = [1 for n in range(N)]
        #h = []
        #for n in range(N):
        #    if n in tumor_inds:
        #        h.append(sym.var('hT'))
        #    else:
        #        h.append(15.23)
        #decay = 14.56
        #decay = sym.var('decay')
        #h = sym.var('h')
        
        # FIT PATIENT IN PARALLEL
        print(f'\n\nBegin patient {p} parallelization...')
        start_time_h = time.time()
        minimizes_p = Parallel(n_jobs=n_jobs, prefer=None)(delayed(optimize)([exp_PLI_p, y0[m], thres_p]) for m in range(M))        
        print(f'Patient optimization took: {time.time() - start_time_h} seconds')
        print(f'One average optimization took: {(time.time() - start_time_h)/M} seconds')

        # SAVE OPTIMAL PARAMETERS
        for minimize_p in minimizes_p:
            patient_pars[i].append(minimize_p.x)
            patient_vals[i].append(minimize_p.fun)
        print(f'Fitted controls for patient {p}')

        # PLOT OPTIMAL AVERAGE PLI AND EXPERIMENTAL PLI
        # PLOT EXPERIMENTAL AVERAGE PLI 
        if plot:
            print('\nPlotting optimal experimental FC...')
            if m == 0 and p == 0:
                figs, brain_figs = plot_functional_connectomes(mean_exp_PLI, \
                                     coordinates=coordinates, \
                                     region_names=region_names, regions=regions, colours=colours)
                print('Done.')

                # SAVE AND CLOSE EXP. AVG. PLI
                figs[0].savefig(fig_save_path + f'exp_control.png', dpi=300, bbox_inches='tight')
                brain_figs[0].savefig(fig_save_path + f'mni_exp_control.png', dpi=300)
                plt.close(figs[0])

                # PLOT STRUCTURAL CONNECTIVITY
                print('\nPlotting SC...')
                figs, brain_figs = plot_functional_connectomes(mean_W, coordinates=coordinates, \
                                             region_names=region_names, regions=regions, \
                                             colours=colours, \
                                             edge_threshold='0.0%')
                print('Done.')

                # SAVE AND CLOSE STRUCTURAL CONNECTIVITY
                figs[0].savefig(fig_save_path + 'structural.png', dpi=300, bbox_inches='tight')
                brain_figs[0].savefig(fig_save_path + 'structural_mni.png', dpi=300)
                plt.close('all')
            if m == 0:
                figs, brain_figs = plot_functional_connectomes(exp_PLI_p, coordinates=coordinates, \
                                     region_names=region_names, regions=regions, colours=colours)
                print('Done.')

                # SAVE AND CLOSE EXP. AVG. PLI
                figs[0].savefig(fig_save_path + f'exp_patient_{p}.png', dpi=300, bbox_inches='tight')
                brain_figs[0].savefig(fig_save_path + f'mni_exp_patient_{p}.png', dpi=300)
                plt.close(figs[0])
            

            # FIND OPTIMAL SIMULATED PLI CONTROL
            print(f'\nSolving for optimal healthy dynamical model parameters..')
            sol = solve_dde(DE, DE.y0, mean_W, t_span=tspan, step=step, atol=atol, rtol=rtol, \
                 parameterss=np.array([minimize.x]), cutoff=cutoff)
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

            # PLOT OPTIMAL SIMULATED AVERAGE PLI 
            print('\nPlotting optimal simulated FC...')
            # MAKE TITLE
            title = ''
            for i, str_par in enumerate(control_pars):
                title += f'{str_par} = {round(minimize.x[i],3)} '
            title += f'optima = {minimize.fun}'
            figs, brain_figs = plot_functional_connectomes(opt_sim_PLI, coordinates=coordinates, \
                        region_names=region_names, regions=regions, colours=colours, title=title)
            print('Done.')

            # SAVE AND CLOSE SIM. AVG. PLI
            figs[0].savefig(fig_save_path + f'sim_run_{m}_patient_{p}_control.png', \
                             dpi=300, bbox_inches='tight')
            brain_figs[0].savefig(fig_save_path + f'mni_sim_run_{m}_patient_{p}_control.png', \
                             dpi=300)
            plt.close('all')

            # FIND OPTIMAL SIMULATED PLI CONTROL
            print(f'\nSolving for optimal patient dynamical model parameters..')
            sol = solve_dde(DE, DE.y0, mean_W, t_span=tspan, step=step, atol=atol, rtol=rtol, \
                 parameterss=np.array([minimize_p.x]), cutoff=cutoff)
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

            # PLOT OPTIMAL SIMULATED AVERAGE PLI PATIENT P 
            print('\nPlotting optimal patient simulated FC...')
            # MAKE TITLE
            title = ''
            for i, str_par in enumerate(control_pars):
                title += f'{str_par} = {round(minimize_p.x[i],3)} '
            title += f'optima = {minimize_p.fun}'
            figs, brain_figs = plot_functional_connectomes(opt_sim_PLI, coordinates=coordinates, \
                            region_names=region_names, regions=regions, colours=colours, title=title)
            print('Done.')

            # SAVE AND CLOSE SIM. AVG. PLI
            figs[0].savefig(fig_save_path + f'sim_run_{m}_patient{p}.png', \
                                 dpi=300, bbox_inches='tight')
            brain_figs[0].savefig(fig_save_path + f'mni_sim_run_{m}_patient{p}.png', dpi=300)
            plt.close('all')

    # SAVE OPTIMAL PARAMETERS
    if run:
        print(f'\nSaving optimal parameters for patient {p+1}...')
        pickle.dump( healthy_pars, open( fig_save_path + 'healthy_pars.pl', "wb" ) )
        pickle.dump( patient_pars, open( fig_save_path + 'patient_pars.pl', "wb" ) )
        pickle.dump( healthy_vals, open( fig_save_path + 'healthy_vals.pl', "wb" ) )
        pickle.dump( patient_vals, open( fig_save_path + 'patient_vals.pl', "wb" ) )
        print('Done.')

    print(f'Patient {p} took {time.time() - start_time_patient} seconds')

# print time elapsed
print(f'Whole script took: {time.time() - start_time} seconds')

# LOAD OPTIMAL PARAMETERS
print('\nLoading optimal parameters...')
healthy_pars = pickle.load( open( fig_save_path + 'healthy_pars.pl', "rb" ) )
patient_pars = pickle.load( open( fig_save_path + 'patient_pars.pl', "rb" ) )
healthy_vals = pickle.load( open( fig_save_path + 'healthy_vals.pl', "rb" ) )
patient_vals = pickle.load( open( fig_save_path + 'patient_vals.pl', "rb" ) )
print('Done.')

# PLOT NEW PARAMETERS
print('\n--------------------------------------------------------------------------------------------------------------------')
print('Patient\t\tParameter\tHealthy optima\tPatient optima\tAvg. difference\t\tp-value\t\thealthy success\t\tpatient success')
n_patients = len(healthy_pars) # in case we only look at subsets of patients
for p in range(n_patients):
    healthy_par = np.array(healthy_pars[p])
    patient_par = np.array(patient_pars[p])
    healthy_val = np.array(healthy_vals[p])
    patient_val = np.array(patient_vals[p])
    n_pars = healthy_par.shape[1]
    
    # plot settings
    fig = plt.figure()
    ax = fig.add_subplot()
    colours = sns.color_palette("hls", 8)
    plt.xlabel('parameters')
    plt.ylabel('fitted values')
    plt.suptitle(f'Patient {p+1}')
    plt.close()

    # ITERATE THROUGH PARAMETERS OF PATIENT P
    for npar in range(n_pars):
        # extracting
        healthy_pari_full = healthy_par[:,npar]
        patient_pari_full = patient_par[:,npar]
        #healthy_vali = healthy_val[:,npar]
        #patient_vali = patient_val[:,npar]


        healthy_pari = healthy_pari_full[~np.isnan(healthy_pari_full)]
        patient_pari = patient_pari_full[~np.isnan(patient_pari_full)]
        if npar == 0:
            healthy_val = healthy_val[~np.isnan(healthy_pari_full)]
            patient_val = patient_val[~np.isnan(patient_pari_full)]
        if patient_pari.size * healthy_pari.size == 0:
            break

        # SIGNIFICANCE TEST
        #t_stat, p_val = ttest_ind(healthy_par[:,npar], patient_par[:,npar], equal_var=False)
        ks_test = kstest(healthy_pari, patient_pari)
        p_val = ks_test.pvalue
        avg_diff = np.mean(patient_pari) - np.mean(healthy_pari)
        print(f'{p}\t\t{npar+1}\t\t{np.mean(healthy_pari)}\t\t{np.mean(patient_pari)}\t\t{round(avg_diff,2)}\t\t\t{round(p_val,10)}\t{healthy_pari.size}/{M}\t\t\t{patient_pari.size}/{M}')

        # PLOT OBJECTIVE VALUE WITH PARAMETER VALUE
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('parameter') 
        ax.set_ylabel('objective value') 
        ax.scatter(healthy_pari, healthy_val, color=colours[-3])
        ax.scatter(patient_pari, patient_val, color=colours[0])
        ax.set_xlim([0,30])
        fig.savefig(fig_save_path + f'obj_{p}_par{npar}.png', dpi=300, bbox_inches='tight')
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

        fig.savefig(fig_save_path + f'diff_par{npar}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # create histogram
        binwidth = None
        # healthy
        #plt.figure()
        #sns.histplot(data=healthy_pari, binwidth=binwidth)       
        #plt.savefig(fig_save_path + f'distr_{p}_par{npar}h.png', dpi=300, bbox_inches='tight')

        # patient
        #plt.figure()
        #sns.histplot(data=patient_pari, binwidth=binwidth)       
        #plt.savefig(fig_save_path + f'distr_{p}_par{npar}p.png', dpi=300, bbox_inches='tight')

        # both
        plt.figure()
        sns.histplot(data=healthy_pari, binwidth=binwidth, color=colours[-3])       
        sns.histplot(data=patient_pari, binwidth=binwidth, color=colours[0], alpha=0.6)       
        pval_leg = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                                  markersize=10, label=f'KS p-value {round(p_val,3)}')
        diff_leg = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                  markersize=10, label=f'mean difference {round(avg_diff,3)}')
        plt.legend(handles=[pval_leg, diff_leg])
        plt.xlim([0,30])
        plt.savefig(fig_save_path + f'distr_{p}_par{npar}.png', dpi=300, bbox_inches='tight')
        plt.close()
    print('--------------------------------------------------------------------------------------------------------------------')
        
    # save and close    
    #plt.legend()
    #plt.savefig(fig_save_path + f'fits_patient_{p}.png', dpi=300, bbox_inches='tight')
    plt.close()

