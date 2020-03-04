import os.path as op
import random

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

# Here we are creating the directories/files for left and right hemisphere data
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

if ((hemi == 'both') or (hemi == 'lh')):
    random_annot_name_lh = 'lh.random' + str(n) + '.annot'
    random_annot_path_lh = op.join(subjects_dir, subject, 'label',
                                   random_annot_name_lh)
if ((hemi == 'both') or (hemi == 'rh')):
    random_annot_name_rh = 'rh.random' + str(n) + '.annot'
    random_annot_path_rh = op.join(subjects_dir, subject, 'label',
                                   random_annot_name_rh)


# we will randomly create a parcellation of n parcels in one hemisphere
def make_random_parcellation(path_annot, n, hemi, subjects_dir, random_state,
                             subject):
    parcel = random_parcellation(subject, n, hemi, subjects_dir=subjects_dir,
                                 surface='white', random_state=random_state)

    mne.write_labels_to_annot(parcel, subjects_dir=subjects_dir,
                              subject=subject,
                              annot_fname=path_annot,
                              overwrite=True)


def find_centers_of_mass(parcellation, subjects_dir):
    centers = np.zeros([len(parcellation)])
    # calculate center of mass for the labels
    for idx, parcel in enumerate(parcellation):
        centers[idx] = parcel.center_of_mass(restrict_vertices=True,
                                             surf='white',
                                             subjects_dir=subjects_dir)
    return centers.astype('int')


# check if the annotation already exists, if not create it
if ((hemi == 'both') or (hemi == 'lh')) and not op.exists(random_annot_path_lh):
    make_random_parcellation(random_annot_path_lh, n, 'lh', subjects_dir,
                             random_state, subject)

if ((hemi == 'both') or (hemi == 'rh')) and not op.exists(random_annot_path_rh):
    make_random_parcellation(random_annot_path_rh, n, 'rh', subjects_dir,
                             random_state, subject)

# read the labels from annot
if ((hemi == 'both') or (hemi == 'lh')):
    parcels_lh = mne.read_labels_from_annot(subject=subject,
                                            annot_fname=random_annot_path_lh,
                                            hemi='lh',
                                            subjects_dir=subjects_dir)
    cm_lh = find_centers_of_mass(parcels_lh, subjects_dir)
if ((hemi == 'both') or (hemi == 'rh')):
    parcels_rh = mne.read_labels_from_annot(subject=subject,
                                            annot_fname=random_annot_path_rh,
                                            hemi='rh',
                                            subjects_dir=subjects_dir)
    cm_rh = find_centers_of_mass(parcels_rh, subjects_dir)


# randomly choose how many parcels will be activated, left or right hemisphere
# and exact parcels
n_parcels = random.randint(1, 1)
if hemi == 'both':
    hemi_selected = random.choices(['lh', 'rh'], weights=[1, 1])[0]
else:
    hemi_selected = hemi
parcel_selected = random.randint(0, n-1)

if hemi_selected == 'lh':
    l1_center_of_mass = parcels_lh[parcel_selected].copy()
    l1_center_of_mass.vertices = [cm_lh[parcel_selected]]
    parcel_used = parcels_lh[parcel_selected]
elif hemi_selected == 'rh':
    l1_center_of_mass = parcels_rh[parcel_selected].copy()
    l1_center_of_mass.vertices = [cm_rh[parcel_selected]]
    parcel_used = parcels_rh[parcel_selected]


def find_corpus_callosum():
    import os
    import numpy as np
    import nibabel as nib
    from mne.datasets import sample

    data_path = sample.data_path()
    fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
    fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
    subjects_dir = data_path + '/subjects'

    subject_id = 'sample'
    hemi = 'lh'

    aparc_file = os.path.join(subjects_dir,
                         subject_id, "label",
                         hemi + ".aparc.a2009s.annot")
    #labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

    labels = mne.read_labels_from_annot(subject=subject_id,
                               annot_fname=aparc_file,
                                            hemi=hemi,
                                            subjects_dir=subjects_dir)

    #vertices_in_label_unknown = np.where(labels == 1)
    #assert names[0] == b'Unknown'  # label 0
    #what I mean by label 0 is the first label in the aparc
    #let me know if it's not clear
    return labels
labels = find_corpus_callosum()

def generate_signal(data_path, subject, parcel):
    # Generate the signal
    # First, we get an info structure from the test subject.
    evoked_fname = op.join(data_path, 'MEG', subject,
                           subject+'_audvis-ave.fif')
    info = mne.io.read_info(evoked_fname)
    sel = mne.pick_types(info, meg=False, eeg=True, stim=True)
    info = mne.pick_info(info, sel)
    tstep = 1. / info['sfreq']

    # To simulate sources, we also need a source space. It can be obtained from
    # the forward solution of the sample subject.
    fwd_fname = op.join(data_path, 'MEG', subject,
                        subject+'_audvis-meg-eeg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    # Define the time course of the activity for each source of the region to
    # activate. Here we use a sine wave at 18 Hz with a peak amplitude
    # of 10 nAm.
    source_time_series = np.sin(2. * np.pi * 18. * np.arange(100) * tstep
                                ) * 10e-9

    # Define when the activity occurs using events. The first column is the
    # sample of the event, the second is not used, and the third is the event
    # id. Here the events occur every 200 samples.
    n_events = 50
    events = np.zeros((n_events, 3))
    events[:, 0] = 100 + 200 * np.arange(n_events)  # Events sample.
    events[:, 2] = 1  # All events have the sample id.

    # Create simulated source activity. Here we use a SourceSimulator whose
    # add_data method is key. It specified where (label), what
    # (source_time_series), and when (events) an event type will occur.
    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    source_simulator.add_data(parcel, source_time_series, events)

    # Project the source time series to sensor space and add some noise.
    # The source simulator can be given directly to the simulate_raw function.
    raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
    cov = mne.make_ad_hoc_cov(raw.info)
    mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.02])

    return events, source_time_series, raw


events, source_time_series, raw = generate_signal(data_path, subject,
                                                  parcel=l1_center_of_mass)

raw.plot()

# Plot evoked data to get another view of the simulated raw data.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
evoked.plot()

# visualize the brain with the parcellations and the source of the signal
brain = Brain('sample', hemi, 'inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))

#brain.add_label(parcel_used, alpha=0.5, color='r')
#brain.add_label(parcels_rh[0], alpha=0.5, color='b')
brain.add_label(labels[-1], alpha=0.5, color='b')
#brain.add_label(parcels_rh[0], alpha=0.5, color='b')
# 0 if lh, 1 if rh
# l = mne.vertex_to_mni(l1_center_of_mass.vertices, 0, subject, subjects_dir)

#for center in cm_lh:
#    brain.add_foci(center, coords_as_verts=True, map_surface="white",
#                   color="gold", hemi=hemi_selected)

file_save_brain = 'fig/brain.png'
brain.save_image(file_save_brain)
