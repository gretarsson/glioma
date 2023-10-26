import os
import numpy as np
import pickle
from scipy import signal
from pprint import pprint
from solve_brain.brain_analysis import PLI, compute_phase_coherence, butter_bandpass_filter, PLI_from_complex, amplitude_coupling_from_complex
import matplotlib.pyplot as plt


# --------------------------------------
# in this script, we read MEG data, as
# as provided by Linda, and compute
# a PLI functional connectome
# ------------------------------------

# settings
fs = 1250  # sampling frequency
filename = 'test_healthy'
save_pli = '../data/'+filename+'_PLI_.p'  # where to save PLI
save_ampl = '../data/'+filename+'_ampl.p'  
band = [8, 12]

# find path names for each subject's folder
#rootdir = '../data/meg_glioma/'
rootdir = '../data/meg_healthy/'
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
pli = np.empty((n_subjects,N,N))
ampl = np.empty((n_subjects,N,N))
pli[:] = np.NaN
ampl[:] = np.NaN

# iterate through each subject
for s, subject_folder in enumerate(subject_folders):
    # display
    print(f'Now at subject {s+1} of {n_subjects}')

    # store subject's epochs momentarily
    pli_s = np.empty((len_epochs[s],N,N))
    ampl_s = np.empty((len_epochs[s],N,N))

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

        # hilbert transform and FC metrics
        data_hilbert = signal.hilbert(data_passed)
        pli_s[epi] = PLI_from_complex(data_hilbert)
        ampl_s[epi] = amplitude_coupling_from_complex(data_hilbert)


    # average across epochs
    pli[s,:,:] = np.mean(pli_s, axis=0)
    ampl[s,:,:] = np.mean(ampl_s, axis=0)

# save epoch-averaged functional connectomes with pickle
pickle.dump(pli, open( save_pli, "wb" ))
pickle.dump(ampl, open( save_ampl, "wb" ))

# plot the average PLI and average amplitude matrix
plt.figure()
plt.imshow(np.mean(pli,axis=0))
plt.figure()
plt.imshow(np.mean(ampl,axis=0))
plt.show()

