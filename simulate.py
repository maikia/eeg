import os

import numpy as np
import pandas as pd
import pickle
# import random
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import mne
from mne.utils import check_random_state

from joblib import Memory, Parallel, delayed
from tqdm import tqdm

from simulation.parcels import find_centers_of_mass
from simulation.raw_signal import generate_signal
from simulation.parcels import make_random_parcellation
# from simulation.plot_signal import visualize_brain # TODO: add check if
# running from the server do not import, causes errors

# IMPORTANT: run it with ipython --gui=qt


mem = Memory('./')
N_JOBS = -1
# N_JOBS = 1

make_random_parcellation = mem.cache(make_random_parcellation)


@mem.cache
def prepare_parcels(subject, subjects_dir, hemi, n_parcels, random_state):
    if ((hemi == 'both') or (hemi == 'lh')):
        annot_fname_lh = 'lh.random' + str(n_parcels) + '.annot'
        annot_fname_lh = os.path.join(subjects_dir, subject, 'label',
                                      annot_fname_lh)
    if ((hemi == 'both') or (hemi == 'rh')):
        annot_fname_rh = 'rh.random' + str(n_parcels) + '.annot'
        annot_fname_rh = os.path.join(subjects_dir, subject, 'label',
                                      annot_fname_rh)

    make_random_parcellation(annot_fname_lh, n_parcels,
                             'lh', subjects_dir,
                             random_state, subject,
                             remove_corpus_callosum=True)

    make_random_parcellation(annot_fname_rh, n_parcels, 'rh',
                             subjects_dir,
                             random_state, subject,
                             remove_corpus_callosum=True)

    # read the labels from annot
    if ((hemi == 'both') or (hemi == 'lh')):
        parcels_lh = mne.read_labels_from_annot(subject=subject,
                                                annot_fname=annot_fname_lh,
                                                hemi='lh',
                                                subjects_dir=subjects_dir)
        cm_lh = find_centers_of_mass(parcels_lh, subjects_dir)
        # remove the last, unknown label which is corpus callosum
        assert parcels_lh[-1].name[:7] == 'unknown'
        parcels_lh = parcels_lh[:-1]
    if ((hemi == 'both') or (hemi == 'rh')):
        parcels_rh = mne.read_labels_from_annot(subject=subject,
                                                annot_fname=annot_fname_rh,
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
        return [parcels_lh], [cm_lh]


# @mem.cache
def init_signal(parcels, cms, hemi, n_parcels_max=3, random_state=None,
                source_at_cm=False):
    '''
    source_at_cm: the source will always be center of mass of the parcel
    '''
    # randomly choose how many parcels will be activated, left or right
    # hemisphere and exact parcels
    rng = check_random_state(random_state)

    if hemi == 'both':
        parcels_lh, parcels_rh = parcels
        cm_lh, cm_rh = cms
    elif hemi == 'rh':
        [parcels_rh] = parcels
        [cm_rh] = cms
    elif hemi == 'lh':
        [parcels_lh] = parcels
        [cm_lh] = cms

    n_parcels = rng.randint(n_parcels_max, size=1)[0] + 1
    to_activate = []
    parcels_selected = []

    # do this so that the same label is not selected twice
    deck_lh = list(rng.permutation(len(parcels_lh)))
    deck_rh = list(rng.permutation(len(parcels_rh)))
    for idx in range(n_parcels):
        if hemi == 'both':
            hemi_selected = ['lh', 'rh'][rng.randint(2, size=1)[0]]
        else:
            hemi_selected = hemi

        if hemi_selected == 'lh':
            parcel_selected = deck_lh.pop()
            l1_source = parcels_lh[parcel_selected].copy()
            if source_at_cm:
                l1_source.vertices = [cm_lh[parcel_selected]]
            parcel_used = parcels_lh[parcel_selected]
        elif hemi_selected == 'rh':
            parcel_selected = deck_rh.pop()
            l1_source = parcels_rh[parcel_selected].copy()
            if source_at_cm:
                l1_source.vertices = [cm_rh[parcel_selected]]
            parcel_used = parcels_rh[parcel_selected]
        if not source_at_cm:
            l1_source.vertices = [rng.choice(parcel_used.vertices)]
        to_activate.append(l1_source)
        parcels_selected.append(parcel_used)

    # activate selected parcels
    events, _, raw = generate_signal(data_path, subject,
                                     parcels=to_activate)

    evoked = mne.Epochs(raw, events, tmax=0.3).average()
    data = evoked.data[:, np.argmax((evoked.data ** 2).sum(axis=0))]
    # visualize_brain(subject, hemi, 'random' + str(n), subjects_dir,
    #                parcels_selected)

    names_parcels_selected = [parcel.name for parcel in parcels_selected]
    return data, names_parcels_selected, to_activate


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
n_parcels = 10  # number of parcels per hemisphere
# (will be reduced by corpus callosum)
random_state = 10
hemi = 'both'
subject = 'sample'
n_samples_train = 1000
n_samples_test = 300
n_parcels_max = 2

# Here we are creating the directories/files for left and right hemisphere
data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

parcels, cms = prepare_parcels(subject, subjects_dir, hemi=hemi,
                               n_parcels=n_parcels,
                               random_state=42)
parcels_flat = [item for sublist in parcels for item in sublist]
parcel_names = [parcel.name for parcel in parcels_flat]
parcel_names = np.array(parcel_names)


if 0:
    visualize_brain(subject, hemi, 'random' + str(n_parcels), subjects_dir,
                    parcels_flat)


len_parcels_flat = len(parcels_flat)
# save label names with their corresponding vertices
parcel_vertices = {}
for idx, parcel in enumerate(parcels_flat, 1):
    parcel_name = str(idx) + parcel.name[-3:]
    parcel_vertices[parcel_name] = parcel.vertices

# prepare train and test data
signal_list = []
target_list = []
rng = np.random.RandomState(42)
n_samples = n_samples_train + n_samples_test
seeds = rng.randint(np.iinfo('int32').max, size=n_samples)


train_data = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(
    delayed(init_signal)(parcels, cms, hemi, n_parcels_max, seed,
                         source_at_cm=False)
    for seed in tqdm(seeds)
)

signal_list, target_list, activated = zip(*train_data)

signal_list = np.array(signal_list)
data_labels = ['e%d' % (idx + 1) for idx in range(signal_list.shape[1])]
df = pd.DataFrame(signal_list, columns=list(data_labels))
target = targets_to_sparse(target_list, parcel_names)

df_train = df.iloc[:n_samples_train]
train_target = target[:n_samples_train]

df_test = df.iloc[n_samples_train:]
test_target = target[n_samples_train:]

data_dir_specific = 'data_' + str(len_parcels_flat) + '_' + str(n_parcels_max)
if not os.path.isdir(data_dir_specific):
    os.mkdir(data_dir_specific)

with open(os.path.join(data_dir_specific, 'labels.pickle'), 'wb') as outfile:
    pickle.dump(parcel_vertices, outfile)
outfile.close()

df_train.to_csv(os.path.join(data_dir_specific, 'train.csv'), index=False)
save_npz(os.path.join(data_dir_specific, 'train_target.npz'), train_target)
print(str(len(df_train)), ' train samples were saved')

df_test.to_csv(os.path.join(data_dir_specific, 'test.csv'), index=False)
save_npz(os.path.join(data_dir_specific, 'test_target.npz'), test_target)
print(str(len(df_test)), ' test samples were saved')


# Visualize
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
info = mne.read_evokeds(fname)[0].pick('eeg').info
evoked = mne.EvokedArray(df.values.T, info, tmin=0)
evoked.plot_topomap()

# data to give to the participants:
# labels with their names and vertices: parcels

# reading forward matrix and saving
fwd_fname = os.path.join(data_path, 'MEG', subject,
                         subject + '_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
lead_field = fwd['sol']['data']

picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True, exclude=[])
lead_field = lead_field[picks_eeg, :]

# now we make a vector of size n_vertices for each surface of cortex
# hemisphere and put a int for each vertex that says it which label
# it belongs to.
parcel_indices_lh = np.zeros(len(fwd['src'][0]['inuse']), dtype=int)
parcel_indices_rh = np.zeros(len(fwd['src'][1]['inuse']), dtype=int)
for label_name, label_idx in parcel_vertices.items():
    label_id = int(label_name[:-3])
    if '-lh' in label_name:
        parcel_indices_lh[label_idx] = label_id
    else:
        parcel_indices_rh[label_idx] = label_id

# Make sure label numbers different for each hemisphere
parcel_indices = np.concatenate((parcel_indices_lh,
                                 parcel_indices_rh), axis=0)

# Now pick vertices that are actually used in the forward
inuse = np.concatenate((fwd['src'][0]['inuse'],
                        fwd['src'][1]['inuse']), axis=0)
parcel_indices_l = parcel_indices[np.where(inuse)[0]]

assert len(parcel_indices_l) == lead_field.shape[1]

# Remove from parcel_indices and from the leadfield all the indices == 0 (not
# used by our brain)
lead_field = lead_field[:, parcel_indices_l != 0]
parcel_indices_l = parcel_indices_l[parcel_indices_l != 0]
assert len(parcel_indices_l) == lead_field.shape[1]

np.savez(os.path.join(data_dir_specific, 'lead_field.npz'),
         lead_field=lead_field, parcel_indices=parcel_indices_l)
