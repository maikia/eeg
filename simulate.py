import os.path as op
import random

import numpy as np


import mne
from mne import random_parcellation

from simulation.parcels import find_centers_of_mass
from simulation.raw_signal import generate_signal
from simulation.plot_signal import visualize_brain

# IMPORTANT: run it with ipython --gui=qt

def prepare_parcels():
    pass

def init_signal():
    # same variables
    n = 100 # initial number of parcels (corpsu callosum will be excluded)
    random_state = 0
    hemi = 'both'
    subject = 'sample'
    recalculate_parcels = False # initiate new random parcels

    # Here we are creating the directories/files for left and right hemisphere
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

    # check if the annotation already exists, if not create it
    if (recalculate_parcels or not op.exists(random_annot_path_lh)) and \
    ((hemi == 'both') or (hemi == 'lh')):
        make_random_parcellation(random_annot_path_lh, n, 'lh', subjects_dir,
                                random_state, subject,
                                remove_corpus_callosum=True)

    if (recalculate_parcels or not op.exists(random_annot_path_rh)) and \
    ((hemi == 'both') or (hemi == 'rh')):
        make_random_parcellation(random_annot_path_rh, n, 'rh', subjects_dir,
                                random_state, subject,
                                remove_corpus_callosum=True)

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
    n_parcels = random.randint(1, 3)
    to_activate = []
    parcels_selected = []
    # do this so that the same label is not selected twice
    deck_lh = list(range(0, len(parcels_lh)))
    random.shuffle(deck_lh)
    deck_rh = list(range(0, len(parcels_rh)))
    random.shuffle(deck_rh)
    for idx in range(n_parcels):
        if hemi == 'both':
            hemi_selected = random.choices(['lh', 'rh'], weights=[1, 1])[0]
        else:
            hemi_selected = hemi

        if hemi_selected == 'lh':
            parcel_selected = deck_lh.pop()
            l1_center_of_mass = parcels_lh[parcel_selected].copy()
            l1_center_of_mass.vertices = [cm_lh[parcel_selected]]
            parcel_used = parcels_lh[parcel_selected]
        elif hemi_selected == 'rh':
            parcel_selected = deck_rh.pop()
            l1_center_of_mass = parcels_rh[parcel_selected].copy()
            l1_center_of_mass.vertices = [cm_rh[parcel_selected]]
            parcel_used = parcels_rh[parcel_selected]
        to_activate.append(l1_center_of_mass)
        parcels_selected.append(parcel_used)

    # activate selected parcels
    for idx in range(n_parcels):
        events, source_time_series, raw = generate_signal(data_path, subject,
                                                        parcels=to_activate)

    raw.plot()

    visualize_brain(subject, hemi, 'random' + str(n), subjects_dir,
                    parcels_selected)

    # as the signal given give a single point at
    data = raw.get_data() # 59 electrodes + 10 sti channels

    e_data = data[9:,:]
    get_data_at = 100
    e_data[:,get_data_at]
    print(parcels_selected)
    print(len(parcels_rh))
    print(len(parcels_lh))



init_signal()
