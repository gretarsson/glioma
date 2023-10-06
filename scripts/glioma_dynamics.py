import pickle
import os
import csv
import time
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from network_dynamics import delay_matrix, mean_network_rhythms_series, spectral_properties, plot_spectral_properties, envelopes, plot_envelopes_connectome, create_rhythms, plot_oscillatory_activity, summed_power, average_MPC, functional_connectomes, plot_functional_connectomes, amplitude_distr, portion_ampl, plot_portions, freq_distr, plot_freq_distr, plot_amplitude_distr, plot_global_functional, hopf_trajectory, wilson_cowan_trajectory
from math import pi
from datetime import datetime
from scipy.signal import periodogram
# change maximum number of figures
mpl.rcParams['figure.max_open_warning'] = 30


# -----------------------------------------------------------------------------------------------
# Here we model the dynamics of neural oscillations
# on a discrete time-series of degrading connectomes
# -----------------------------------------------------------------------------------------------


## import connectome
connectome_file = './connectome/connectome_33.graphml'
G = nx.read_graphml(connectome_file)
G = nx.convert_node_labels_to_integers(G)
N = G.number_of_nodes()

# get the names of each region
region_names = []
for n in range(N):
    node = G.nodes[n]
    name = node["dn_name"]
    region_names.append(name)

# read MNI coordinates
mni_coords = np.loadtxt('./connectome/mni_coordinates.csv')
coordinates = np.zeros((N,3))
for n in range(N):
    mni_coord = mni_coords[n]
    coordinates[n,0] = mni_coord[0] 
    coordinates[n,1] = mni_coord[1] 
    coordinates[n,2] = mni_coord[2] 

## dynamics parameters
kappa = 0.4

osc_freq = 8
taux = 0.346 / osc_freq 
tauy = 0.346 / osc_freq 

Cxx = 24
Cxy = -20 

Cyy = 0 
Cyx = 35

P = 1.0
Q = -2

h = 1
Sa = 1
theta = 4

transmission_speed = 130.0

## settings
save_name = "glioma_satelite_ex_up_short"
figure_folder = f"./figures/glioma/{save_name}/"
#figure_folder = f"./figures/glioma/sampling/"
start = 0
end = 50
points = 1
L = 10
#t_dyn = [0, 50]
#fourier_t = 6.5  # 1/t gives the frequency resolution
fourier_cutoff = 1
n_epochs = 10
l_epochs = 6.5
t_dyn = [0,  n_epochs*l_epochs + fourier_cutoff] 
bands = [[0.5,12]]
delay_dim = 40
step = 1/1000
freq_tol = 0
wiggle = 0.1
scale = None
spread_file = './simulations/spread/spread_glioma_satelite_ex_up'
lobe_names = ['frontal', 'parietal', 'occipital', 'temporal', 'limbic', 'basal-ganglia', 'brainstem', 'tumor']
#lobe_names = ['visual', 'sensorimotor', 'NaN', 'attention', 'limbic', 'frontoparietal', 'default-mode']
tumor_seed = [74]  # left insula
#fourier_cutoff = t_dyn[-1] - fourier_t
plt.style.use('seaborn-muted')
#plt.style.use('seaborn')
sns.despine()
colours = sns.color_palette('hls', len(lobe_names)+4)
save_time = True

# create figure folder if it does not exist
isExist = os.path.exists(figure_folder)
if not isExist:
    os.makedirs(figure_folder)
    print(f"Created folder: {figure_folder}")

## create delay matrix per Bick & Goriely
distances = []
with open('./connectome/LengthFibers33.csv', 'r') as file:
    reader = csv.reader(file)
    node_i = 0
    for row in reader:
        node_j = 0
        for col in row:
            if float(col) > 0:
                distances.append((node_i,node_j, float(col)/10))
            node_j += 1
        node_i += 1
delays = delay_matrix(distances, transmission_speed, N, discretize=delay_dim)
#delays = np.zeros((N,N))

## define brain regions (LobeIndex_I.txt)
regions = [[] for _ in range(len(lobe_names))]
regions[-1] = tumor_seed
with open('./connectome/LobeIndex_I.txt') as f:
    node = 0
    for line in f:
        if node in tumor_seed:
            node += 1
            continue
        lobe = int(float(line.strip()))-1
        regions[lobe].append(node)
        node += 1

# import spreading simulation 
spread_sol = pickle.load( open( spread_file+'.p', "rb" ) )
spread_par = pickle.load( open( spread_file+'_meta.p', "rb" ) )
rhythms = create_rhythms(spread_sol, start, end, points)
W = rhythms[0][0]
t_stamps = np.linspace(start, end, points)

# -----------------------------------------------------------------------------------------------
## solve dynamics
## solve
#print(f'\nSolving rhythm dynamics...')
#start = time.time()
#sol = wilson_cowan_trajectory(L, rhythms, taux=taux, tauy=tauy, Cxx=Cxx, Cxy=Cxy, \
#        Cyx=Cyx, Cyy=Cyy, kappa=kappa, \
#        h=h, Sa=Sa, theta=theta, P=P, Q=Q, delays=delays, \
#        t_span=t_dyn,  random_init=True, atol=10**-6, rtol=10**-3)
#end = time.time()
#print(f'\nElapsed time for dynamics simulation: {end-start} seconds\n')
#
## save dynamics result
## put current date and time in filename
#filename = './simulations/dynamics/' + 'dynamics_' + save_name + '.p'
#filename_par = './simulations/dynamics/' + 'dynamics_' + save_name + '_meta' + '.p'
#
## dump
#pickle.dump( sol, open( filename, "wb" ) )
#
## add metadata (parameters)
#par_dict = {'L':L, 't_stamps':t_stamps,\
#        'kappa':kappa, \
#        'connectome_file': connectome_file, 'connectome': G}
#
## dump metadata
#pickle.dump( par_dict, open( filename_par, "wb" ) )
## -----------------------------------------------------------------------------------------------
## analysis and plotting
# load
filename = './simulations/dynamics/' + 'dynamics_' + save_name + '.p'
parsname = './simulations/dynamics/' + 'dynamics_' + save_name + '_meta' + '.p'
sol = pickle.load( open( filename, "rb") )
pars = pickle.load( open( parsname, "rb") )
t_stamps = pars['t_stamps']
rhythms = create_rhythms(spread_sol, t_stamps[0], t_stamps[-1], len(t_stamps))

## envelopes
#n_of_points = round((np.mean(mean_w)-2*np.mean(dw))/(2*pi)) * t_dyn[-1]
#t_avg, maxima_N, minima_N = envelopes(rhythms, sol, n_of_points)
# 
### plot envelopes
#sns.set_context(font_scale=2, rc={"axes.labelsize":18,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":8})   
#figs, axs = plot_envelopes_connectome(t_stamps, t_avg, maxima_N, minima_N, colours, regions=regions, t_cutoff=fourier_cutoff, wiggle=wiggle)
#
## save envelope plots
#env_names = [save_name + f'_{i}' for i in range(len(figs))]
#for i, fig in enumerate(figs):
#    fig.savefig("../figures/alzh_paper/envelopes/" + env_names[i] + '.pdf', dpi=300)
#
### plot average oscillatory (integral)
##sns.set_context(font_scale=2, rc={"axes.labelsize":18,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":8})   
##fig, ax = plot_oscillatory_activity(t_stamps, sol, regions=regions[:-1], colours=colours, legends=lobe_names[:-1], wiggle=wiggle)
#
## save average oscillatory 
##fig.savefig("../figures/alzh_paper/" + save_name + "_osc_activity.pdf", dpi=300)
#
## spectral analysis
if len(bands)>1:
    relative = True
else:
    relative = False
bandpowers, freq_peaks = spectral_properties(sol, bands, fourier_cutoff, freq_tol=freq_tol, relative=relative)

# plot spectral analysis
sns.set_context(font_scale=2, rc={"axes.labelsize":18,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":8})   
figs_PSD, figs_peaks = plot_spectral_properties(t_stamps, bandpowers, freq_peaks, bands, wiggle, '', lobe_names, colours, regions=regions, only_average=False)
#plt.show()

## save spectral plots
sns.despine()
spec_names = ["_power", "_peaks"]
if relative:
    specif = 'relative'
else:
    specif = 'absolute'
for i in range(len(figs_PSD)):
    figs_PSD[i].savefig(figure_folder + save_name + spec_names[0] + f'_band={bands[i]}' +\
            specif + '.pdf', dpi=300, bbox_inches='tight')
    figs_peaks[i].savefig(figure_folder + save_name + spec_names[1] + f'_band={bands[i]}' \
            + '.pdf', dpi=300, bbox_inches='tight')
    plt.close('all')

#### functional connectivity
## compute functional connectivity matrices
method = 'PLI'
filename_mats = './simulations/dynamics/' + 'dynamics_' + save_name + '_matrices' + '.p'
#avg_F = functional_connectomes(sol, fourier_cutoff=fourier_cutoff, bands=bands, method=method, \
#    n_epochs=n_epochs, l_epochs=l_epochs)
#pickle.dump( avg_F, open( filename_mats, "wb" ) )
avg_F = pickle.load( open( filename_mats, "rb") )

### plot global functional connectivity
normalize = True
figs_glob = plot_global_functional(t_stamps, avg_F, bands, regions=regions, \
        lobe_names=lobe_names, colours=colours, normalize=normalize, wiggle=wiggle)
#plt.show()

## save global functional plots
sns.despine()
if normalize:
    specif = 'normalized'
else:
    specif = 'nonnormalized'
env_names = []
for b in range(len(bands)):
    env_names.append(save_name + f'_{method}_functional_{bands[b]}_'+str(specif))
    env_names.append(save_name + f'_{method}_clustering_{bands[b]}_'+str(specif))
    env_names.append(save_name + f'_{method}_path_{bands[b]}_'+str(specif))
    env_names.append(save_name + f'_{method}_modularity_{bands[b]}_'+str(specif))
    env_names.append(save_name + f'_{method}_synchronizability_{bands[b]}_'+str(specif))
for i, fig in enumerate(figs_glob):
    fig.savefig(figure_folder + env_names[i] + '.pdf', dpi=300, bbox_inches='tight')
    plt.close('all')

#### plot functional connectomes
figs3, figs4 = plot_functional_connectomes(avg_F, t_stamps=t_stamps, bands=bands, \
        regions=regions, region_names=region_names, colours=colours, coordinates=coordinates)
#plt.show()
sns.despine()
env_names = []
env_names_brain = []
for b in range(len(bands)):
    for i in reversed(range(len(t_stamps))):
        env_names.append(save_name + f'_{bands[b]}_t={round(t_stamps[i],1)}')
        env_names_brain.append(save_name + f'_brain_{bands[b]}_t={round(t_stamps[i],1)}')
for i, fig in enumerate(figs3):
    fig.savefig(figure_folder + env_names[i] + '.png', dpi=300, bbox_inches='tight')
    figs4[i].savefig(figure_folder + env_names_brain[i] + '.png', dpi=300)
    figs4[i].close()
    plt.close('all')
# find amplitude distribution
#plt.style.use('seaborn-muted')
#sns.despine()
#bins = amplitude_distr(sol, cutoff=fourier_cutoff)
#fig_bins = plot_amplitude_distr(bins, t_stamps=t_stamps, xlim=[0,40], binwidth=2.5)
#plt.show()
#portions = portion_ampl(bins, 2.5)
#fig = plot_portions(portions, t_stamps)
#plt.show()

# plot amplitude distribution of chosen simulation
#plt.figure()
#sns.histplot(data=bins[0][0], bins=20, stat='probability')
#plt.figure()
#sns.histplot(data=bins[1][0], bins=20, stat='probability')
#plt.figure()
#sns.histplot(data=bins[2][0], bins=20, stat='probability')
#plt.show()

# plot all time-series of a single run at time zero 
save_time = True
for i in reversed(range(len(t_stamps))):
    (t0, x0, y0) = sol[i]
    x = x0[0]  # first trial
    y = y0[0]  # first trial
    tot_t = 100  # last secs to show
    show_n = 83
    inds = [i for i in range(len(t0)) if t0[i] > (t0[-1]-tot_t)]
    t0 = t0[inds]
    x = x[:,inds]
    y = y[:,inds]
    plt.figure()
    plt.title(f't={round(t_stamps[i],1)}')
    for j in range(show_n):
        for r, region in enumerate(regions):
            if j in region:
                colour = colours[r]
        plt.plot(t0, x[j], c=colour)
        plt.ylim([0,1])
    if save_time:
        plt.savefig(figure_folder+save_name+f'_series_t={round(t_stamps[i],1)}'+'.png')
        plt.close('all')

# look at spectra of a single node in at given time in spreading
l = 0
psd_max = 0
psds = [ [ [] for _ in range(N)] for _ in range(len(t_stamps)) ]
freqss = [ [ [] for _ in range(N)] for _ in range(len(t_stamps)) ]
for i in reversed(range(len(t_stamps))):
    t0,x,y = sol[i]
    plt.figure()
    for n in range(N):
        xn = x[l,n,:]
        tot_t = t0[-1] - t0[0]
        sf = xn.shape[0]/tot_t
        freqs, psd = periodogram(xn, sf)
        freqss[i][n] = freqs
        psds[i][n] = psd
        n_psd_max = np.amax(psd)
        if n_psd_max > psd_max:
            psd_max = n_psd_max

for i in reversed(range(len(t_stamps))):
    plt.figure()
    for n in range(N):
        plt.title(f't={round(t_stamps[i],1)}')
        plt.plot(freqss[i][n], psds[i][n])
        plt.xlim([0,12])
        plt.ylim([0,psd_max])
    if save_time:
        plt.savefig(figure_folder+save_name+f'_PSD_t={round(t_stamps[i],1)}'+'.png')
        plt.close('all')
#plt.show()
#
## compute distributions of peak frequencies
#freq_peaks = freq_distr(sol, cutoff=fourier_cutoff, freq_tol=freq_tol)
#freq_figs = plot_freq_distr(freq_peaks, t_stamps=t_stamps, xlim=[-1,12], binwidth=0.1)
#plt.show()


# we're done
print(f'Parameters: {pars}')
#plt.show()

