import os.path as op

import numpy as np

from surfer import Brain
import mne


# IMPORTANT: run it with ipython --gui=qt

parcellation = 'aparc_sub'

# This will download the data if it not already on your machine. We also set
# the subjects directory so we don't need to give it to functions.
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'fsaverage'

mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose=True)

# First, we get an info structure from the test subject.
evoked_fname = op.join(data_path, 'MEG', 'sample', 'sample'+'_audvis-ave.fif')
# double check if correct path
info = mne.io.read_info(evoked_fname)
sel = mne.pick_types(info, meg=False, eeg=True, stim=True)
info = mne.pick_info(info, sel)
tstep = 1. / info['sfreq']

# To simulate sources, we also need a source space. It can be obtained from the
# forward solution of the sample subject.
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample'+'_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

# To select a region to activate, we use the caudal middle frontal to grow
# a region of interest.
# randomly select a label of interest

selected_label = mne.read_labels_from_annot(
    subject, parc='aparc_sub', subjects_dir=subjects_dir)
# regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)

label = selected_label[0]
# Define the time course of the activity for each source of the region to
# activate. Here we use a sine wave at 18 Hz with a peak amplitude
# of 10 nAm.
source_time_series = np.sin(2. * np.pi * 18. * np.arange(100) * tstep) * 10e-9

# Define when the activity occurs using events. The first column is the sample
# of the event, the second is not used, and the third is the event id. Here the
# events occur every 200 samples.
n_events = 50
events = np.zeros((n_events, 3))
events[:, 0] = 100 + 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

# Create simulated source activity. Here we use a SourceSimulator whose
# add_data method is key. It specified where (label), what
# (source_time_series), and when (events) an event type will occur.
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
source_simulator.add_data(label, source_time_series, events)

# Project the source time series to sensor space and add some noise. The source
# simulator can be given directly to the simulate_raw function.
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])
raw.plot()

# Plot evoked data to get another view of the simulated raw data.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
evoked.plot()

# plot where the signal originates from
brain = Brain(subject, 'lh', 'inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('aparc_sub', color='k')
brain.add_label(label, borders=False, color='b')
