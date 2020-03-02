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
subject = 'sample' #'fsaverage'


mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose=True)

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


selected_label = mne.read_labels_from_annot(
    'fsaverage', parc='aparc_sub', subjects_dir=subjects_dir)
"""
# transform to the 'sample' subject
for idx in range(len(selected_label)):
    print(str(idx+1)+'/'+str(len(selected_label)))
    selected_label[idx].morph(subject_to="sample",
             subject_from='fsaverage',
             subjects_dir=subjects_dir,
             verbose=True)
    '''
    hemi = selected_label[idx].hemi
    try:
        if hemi == 'lh':
            lh_labels += selected_label[idx].copy()
        elif hemi == 'rh':
            rh_labels += selected_label[idx].copy()
    except:
        if hemi == 'lh':
            lh_labels = selected_label[idx].copy()
        elif hemi == 'rh':
            rh_labels = selected_label[idx].copy()
    '''
"""
# save to the file
#rh_labels.save('data/annot_sub_rh_sample.label')
#lh_labels.save('data/annot_sub_lh_sample.label)

# regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)
#import pdb; pdb.set_trace()
label1 = selected_label[0].copy()
label2 = selected_label[8].copy()
label = label1 + label2

# morph labels to other subjects coordinates
#label1.morph(subject_to="sample",
#             subject_from='fsaverage',
#             subjects_dir=subjects_dir,
#             verbose=True)
#label2.morph(subject_to="sample",
#             subject_from='fsaverage',
#             subjects_dir=subjects_dir,
#             verbose=True)



###
#raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
#trans_fname = op.join(data_path, 'MEG', 'sample',
#                      'sample_audvis_raw-trans.fif')
#raw = mne.io.read_raw_fif(raw_fname)
#trans = mne.read_trans(trans_fname)

#fig = mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
#                             subjects_dir=subjects_dir, surfaces='head-dense',
#                             show_axes=True, dig=True, eeg=[], meg='sensors',
#                             coord_frame='meg')
#import pdb; pdb.set_trace()



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
raw.plot()

# Plot evoked data to get another view of the simulated raw data.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
evoked.plot()


# png jpg bmp tiff ps eps pdf rib oogl iv vrml obj
# plot where the signal originates from
brain = Brain('sample', 'both', 'inflated', subjects_dir=subjects_dir,
            cortex='low_contrast', background='white', size=(800, 600))
#brain.add_annotation('aparc_sub', color='k')
# draws where is the source
brain.add_label(label, borders=False, color='b')

#brain.add_label(full_annot, borders=True, color='k')
#brain.add_label(lh_labels, borders=False, color='b')
#brain.add_label(rh_labels, borders=False, color='b')
file_save_brain='fig/brain.png'

for label in selected_label:
    brain.add_label(label, borders= True, color='k')
brain.save_image(file_save_brain)

#visualize_loc(subjects_dir, label, file_save_brain='fig/brain.png')

brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', background='white', size=(800, 600))

from mne import random_parcellation
parcel = random_parcellation('sample', 80, 'lh', subjects_dir=subjects_dir,
                        surface='white', random_state=0)

mne.write_labels_to_annot(parcel,subjects_dir=subjects_dir, subject='sample')
                            annot_fname='random40', hemi='lh')
