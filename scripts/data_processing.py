import os
import numpy as np
import pickle
from scipy import signal
from pprint import pprint
from solve_brain.brain_analysis import PLI, compute_phase_coherence, butter_bandpass_filter
import matplotlib.pyplot as plt


# --------------------------------------
# in this script, we read MEG data, as
# as provided by Linda, and compute
# a PLI functional connectome
# ------------------------------------

# settings
fs = 1250  # sampling frequency
filename = 'test'
save_F = '../data/exp_PLI_'+filename+'.p'  # where to save PLI
save_ampls = '../data/gli_ampls_'+filename+'.p'  # where to save amplitude
save_ampl_matrix = '../data/gli_ampl_matrix_'+filename+'.p'  
band = [8, 12]

# find path names for each subject's folder
rootdir = '../data/meg_glioma/'
#rootdir = '../data/meg_healthy/'
subject_folders = []
subject_epochs = []
find_N = False  # set to true to use all ROIs 
N = 78  # if find_N=False, we look only at first N ROIs
for subdir, dirs, files in os.walk(rootdir):
    if files:
        subject_folders.append(subdir)
        subject_epochs.append(files)
        # find the number of regions, N
        if find_N:  
            epoch_file = files[0]
            subject_folder = subdir
            f = open(subject_folder + '/' + epoch_file, 'r')
            data = np.loadtxt(f).T
            f.close()
            N = data.shape[0]
            find_N = False

# print folders to make sure we are reading right ones
print('\nFound the following folders:')
print(subject_folders)
print()

# initialize peaks with NaNs (#subjects, max#epochs)
n_subjects = len(subject_folders)
len_epochs = [len(subject_epoch) for subject_epoch in subject_epochs]
F = np.empty((n_subjects,N,N))
F[:] = np.NaN
ampls = np.empty((n_subjects,N))

# iterate through each subject
for s, subject_folder in enumerate(subject_folders):
    # display
    print(f'Now at subject {s+1} of {n_subjects}')

    # store subject's epochs momentarily
    F_s = np.empty((len_epochs[s],N,N))
    amplss = np.empty((len_epochs[s],N))

    # iterate through each epoch
    for epi, epoch_file in enumerate(subject_epochs[s]):
        # display
        print(f'\tat epoch {epi+1} of {len_epochs[s]}')

        # read raw MEG data
        f = open(subject_folder + '/' + epoch_file, 'r')
        data = np.loadtxt(f).T
        f.close()
        
        # extract shape and time
        _, M = data.shape
        T = 1/fs * M
        t = np.linspace(0, T, M)

        # potentially subsample nodes
        data = data[0:N,:]

        # bandpass signal (all ROIs)
        data_passed = butter_bandpass_filter(data, band[0], band[1], fs)

        # find amplitude
        sec_int = 2
        for i in range(N):
            datan = np.array(data_passed[i,:])
            amplss[epi,i] = find_avg_diff(datan,sec_int*fs)

        # PLI
        F_s[epi] = PLI(data_passed)

    # average across epochs
    F[s,:,:] = np.mean(F_s, axis=0)
    ampls[s,:] = np.mean(amplss,axis=0)

# save epoch-averaged PLI connectomes with pickle
pickle.dump(F, open( save_F, "wb" ))

# compute average amplitude for each roi and the amplitude matrix
ampls = np.mean(ampls,axis=0) 
ampls = ampls / np.amax(np.abs(ampls))
pickle.dump(ampls, open( save_ampls, "wb" ))
ampl_matrix = np.array([[ampls[i] - ampls[j] for j in range(N)] for i in range(N)])
pickle.dump(ampl_matrix, open( save_ampl_matrix, "wb" ))

# plot the average PLI and average amplitude matrix
plt.figure()
plt.imshow(np.mean(F,axis=0))
plt.figure()
plt.imshow(ampl_matrix)
plt.show()

