import os.path as op

import numpy as np

from surfer import Brain
import mne
from mne import random_parcellation


# IMPORTANT: run it with ipython --gui=qt

# same variables
n = 100
random_state = 0
hemi = 'lh'
subject = 'sample'

# This will download the data if it not already on your machine. We also set
# the subjects directory so we don't need to give it to functions.
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

random_annot_name = hemi + '.random' + str(n) + '.annot'
random_annot_path = op.join(subjects_dir, subject, 'label', random_annot_name)
# we will randomly create a parcellation of n parcels in left hemisphere
def make_random_parcellation(path_annot, n, hemi, subjects_dir, random_state,
                             subject):
    parcel = random_parcellation(subject, n, hemi, subjects_dir=subjects_dir,
                             surface='white', random_state=random_state)

    mne.write_labels_to_annot(parcel, subjects_dir=subjects_dir,
                              subject=subject,
                              annot_fname=path_annot,
                              overwrite=True)

# check if the annotation already exists, if not create it

make_random_parcellation(random_annot_path, n, hemi, subjects_dir, random_state,
                         subject)

# First, we get an info structure from the test subject.
evoked_fname = op.join(data_path, 'MEG', 'sample', subject+'_audvis-ave.fif')
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

selected_label = mne.read_labels_from_annot(subject=subject,
                                            annot_fname=random_annot_path,
                                            hemi=hemi,
                                            subjects_dir=subjects_dir)

# calculate center of mass for the labels
label1_center_of_mass = selected_label[0].center_of_mass(restrict_vertices=True,
                                          surf='white',
                                          subjects_dir=subjects_dir)

label1 = selected_label[0].copy()
label2 = selected_label[8].copy()
label = label1 #+ label2

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
source_simulator.add_data(label1, source_time_series, events)
source_simulator.add_data(label2, -source_time_series, events)

# Project the source time series to sensor space and add some noise. The source
# simulator can be given directly to the simulate_raw function.
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])
#raw.plot()

# Plot evoked data to get another view of the simulated raw data.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
#evoked.plot()

# visualize the brain with the parcellations and the source of the signal
brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))

#brain.add_annotation('random' + str(n), color='k')

#brain.add_label(label, alpha=0.2)
# 0 if lh, 1 if rh
#l = mne.vertex_to_mni(label1_center_of_mass, 0, subject, subjects_dir)
for label in selected_label:
    brain.add_label(label, borders=True, color = 'k')
brain.add_foci([0,0,0], color='k')


#brain.add_foci(l, map_surface="white", color="gold")
brain.add_foci(label1_center_of_mass, coords_as_verts=True, map_surface="white", color="red")
file_save_brain = 'fig/brain.png'
brain.save_image(file_save_brain)
