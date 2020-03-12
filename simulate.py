import numpy as np
import os.path as op
import pandas as pd
import random
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import mne

from simulation.parcels import find_centers_of_mass
from simulation.raw_signal import generate_signal
from simulation.plot_signal import visualize_brain
from simulation.parcels import make_random_parcellation

# IMPORTANT: run it with ipython --gui=qt


def prepare_parcels(subject, subjects_dir, hemi, n_parcels,
                    recalculate_parcels = True):
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
        make_random_parcellation(random_annot_path_lh, n_parcels,
                                 'lh', subjects_dir,
                                 random_state, subject,
                                 remove_corpus_callosum=True)

    if (recalculate_parcels or not op.exists(random_annot_path_rh)) and \
       ((hemi == 'both') or (hemi == 'rh')):
        make_random_parcellation(random_annot_path_rh, n_parcels, 'rh',
                                 subjects_dir,
                                 random_state, subject,
                                 remove_corpus_callosum=True)

    # read the labels from annot
    if ((hemi == 'both') or (hemi == 'lh')):
        parcels_lh = mne.read_labels_from_annot(subject=subject,
                                                annot_fname=random_annot_path_lh,
                                                hemi='lh',
                                                subjects_dir=subjects_dir)
        cm_lh = find_centers_of_mass(parcels_lh, subjects_dir)
        # remove the last, unknown label which is corpus callosum
        assert parcels_lh[-1].name[:7] == 'unknown'
        parcels_lh = parcels_lh[:-1]
    if ((hemi == 'both') or (hemi == 'rh')):
        parcels_rh = mne.read_labels_from_annot(subject=subject,
                                                annot_fname=random_annot_path_rh,
                                                hemi='rh',
                                                subjects_dir=subjects_dir)
        # remove the last, unknown label which is corpus callosum
        assert parcels_rh[-1].name[:7] == 'unknown'
        parcels_rh = parcels_rh[:-1]
        cm_rh = find_centers_of_mass(parcels_rh, subjects_dir)

    if hemi == 'both':
        return [parcels_lh, parcels_rh], [cm_lh, cm_rh]
    elif hemi == 'rh':
        return [parcels_rh], [cm_rh]
    elif hemi == 'lh':
        return [parcel_lh], [cm_lh]


def init_signal(parcels, cms, hemi):
    # randomly choose how many parcels will be activated, left or right
    # hemisphere and exact parcels
    if hemi == 'both':
        parcels_lh, parcels_rh = parcels
        cm_lh, cm_rh = cms
    elif hemi == 'rh':
        [parcels_rh] = parcels
        [cm_rh] = cms
    elif hemi == 'lh':
        [parcels_lh] = parcels
        [cm_lh] = cms

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

    #visualize_brain(subject, hemi, 'random' + str(n), subjects_dir,
    #                parcels_selected)

    # as the signal given give a single point at
    data = raw.get_data()  # 59 electrodes + 10 sti channels

    e_data = data[9:, :]
    get_data_at = 100
    names_parcels_selected = [parcel.name for parcel in parcels_selected]
    return e_data[:, get_data_at], names_parcels_selected


def targets_to_sparse(target_list, parcel_names):
    targets = []

    for idx, tar in enumerate(target_list):
        row = np.zeros(len(parcel_names))
        for t in tar:
            row[np.where(parcel_names == t)[0][0]] = 1
        targets.append(row)
    targets_sparse = csr_matrix(targets)
    return targets_sparse

# same variables
n = 100  # initial number of parcels (corpsu callosum will be excluded
         # afterwards)
random_state = 10
hemi = 'both'
subject = 'sample'
recalculate_parcels = False  # initiate new random parcels
number_of_train = 10
number_of_test = 5

# Here we are creating the directories/files for left and right hemisphere
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

parcels, cms = prepare_parcels(subject, subjects_dir, hemi=hemi, n_parcels=n,
                               recalculate_parcels=recalculate_parcels)
parcel_names = [item for sublist in parcels for item in sublist]
parcel_names = [parcel.name for parcel in parcel_names]
parcel_names = np.array(parcel_names)

data_labels = ['e'+str(idx+1) for idx in range(0, 59)]

# prepare train data
signal_list = []
target_list = []
for sample in range(number_of_train):
    signal, parcels_used = init_signal(parcels, cms, hemi)
    signal_list.append(signal)
    target_list.append(parcels_used)

signal_list = np.array(signal_list)
df = pd.DataFrame(signal_list, columns=list(data_labels))
#df['parcels'] = target_list
train_target = targets_to_sparse(target_list, parcel_names)

df.to_csv('data/train.csv', index=False)
save_npz('data/train_target.npz', sparse_matrix)
print(str(len(df)), ' train samples were saved')

# prepare test data
signal_list = []
target_list = []
for sample in range(number_of_test):
    signal, parcels_used = init_signal(parcels, cms, hemi)
    signal_list.append(signal)
    target_list.append(parcels_used)

signal_list = np.array(signal_list)
df = pd.DataFrame(signal_list, columns=list(data_labels))
# df['parcels'] = target_list
test_target = targets_to_sparse(target_list, parcel_names)

df.to_csv('data/test.csv', index=False)
save_npz('data/test_target.npz', sparse_matrix)
print(str(len(df)), ' test samples were saved')








# data to give to the participants:
# labels with their names and vertices: parcels
# ? centers of mass: cms
# datapoints generated along with the target labels: df

