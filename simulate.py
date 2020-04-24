import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import mne
from mne.utils import check_random_state

from joblib import Memory, Parallel, delayed
from tqdm import tqdm

from simulation.parcels import find_centers_of_mass
from simulation.raw_signal import generate_signal
from simulation.parcels import make_random_parcellation

import config

if os.environ.get('DISPLAY'):  # display exists
    from simulation.plot_signal import visualize_brain
    visualize = True
    N_JOBS = 1
else:
    # running on the server, no display
    visualize = False
    N_JOBS = -1

# IMPORTANT: run it with ipython --gui=qt

mem = Memory('./')

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
def init_signal(parcels, raw_fname, fwd_fname, subject,
                n_parcels_max=3, random_state=None, signal_type='eeg'
                ):
    '''
    '''
    # randomly choose how many parcels will be activated between 1 and
    # n_parcels_max and which index at the parcel
    rng = check_random_state(random_state)

    n_parcels = rng.randint(n_parcels_max, size=1)[0] + 1
    to_activate = []
    parcels_selected = []

    # do this so that the same label is not selected twice
    deck = list(rng.permutation(len(parcels)))
    # deck_rh = list(rng.permutation(len(parcels_rh)))
    for idx in range(n_parcels):
        parcel_selected = deck.pop()
        parcel_used = parcels[parcel_selected]
        l1_source = parcels[parcel_selected].copy()
        l1_source.vertices = [rng.choice(parcel_used.vertices)]

        to_activate.append(l1_source)
        parcels_selected.append(parcel_used)

    # activate selected parcels
    events, _, raw = generate_signal(raw_fname, fwd_fname, subject,
                                     parcels=to_activate,
                                     signal_type=signal_type)

    evoked = mne.Epochs(raw, events, tmax=0.3).average()
    data = evoked.data[:, np.argmax((evoked.data ** 2).sum(axis=0))]

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
n_parcels = 20  # number of parcels per hemisphere
# (will be reduced by corpus callosum)
random_state = 42
hemi = 'both'
subject = 'sample'
# subject = 'CC120008'
n_samples = 2000
n_parcels_max = 3
signal_type = 'grad' # 'eeg', 'meg', 'mag' or 'grad'

# Here we are creating the directories/files for left and right hemisphere
data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

if subject == 'sample':
    raw_fname = os.path.join(data_path, 'MEG', subject,
                             subject + '_audvis_raw.fif')
    fwd_fname = os.path.join(data_path, 'MEG', subject,
                             subject + '_audvis-meg-eeg-oct-6-fwd.fif')
else:
    raw_fname = config.get_raw_fname(subject)
    fwd_fname = config.get_fwd_fname(subject)

assert os.path.exists(raw_fname)
assert os.path.exists(fwd_fname)

# The parcelation is done on the average brain
subjects_dir = os.path.join(data_path, 'subjects')
parcels, cms = prepare_parcels('fsaverage', subjects_dir, hemi=hemi,
                               n_parcels=n_parcels,
                               random_state=random_state)
parcels_flat = [item for sublist in parcels for item in sublist]

# morph labels to the subject we are using
parcels_flat = mne.morph_labels(parcels_flat, subject, 'fsaverage',
                                subjects_dir,
                                'white')

parcel_names = [parcel.name for parcel in parcels_flat]
parcel_names = np.array(parcel_names)

if visualize:
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
seeds = rng.randint(np.iinfo('int32').max, size=n_samples)


train_data = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(
    delayed(init_signal)(parcels_flat, raw_fname, fwd_fname, subject,
                         n_parcels_max, seed, signal_type)
    for seed in tqdm(seeds)
)

signal_list, target_list, activated = zip(*train_data)

signal_list = np.array(signal_list)
data_labels = ['e%d' % (idx + 1) for idx in range(signal_list.shape[1])]
df = pd.DataFrame(signal_list, columns=list(data_labels))
target = targets_to_sparse(target_list, parcel_names)

case_specific = subject + '_' + str(len_parcels_flat) + '_' + str(n_parcels_max)
data_dir_specific = 'data_' + case_specific
if not os.path.isdir(data_dir_specific):
    os.mkdir(data_dir_specific)

with open(os.path.join(data_dir_specific, 'labels.pickle'), 'wb') as outfile:
    pickle.dump(parcel_vertices, outfile)
outfile.close()

df.to_csv(os.path.join(data_dir_specific, 'X.csv'), index=False)
save_npz(os.path.join(data_dir_specific, 'target.npz'), target)
print(str(len(df)), ' samples were saved')

# Visualize
fig, axes = plt.subplots(figsize=(7.5, 2.5), ncols=5)
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
info = mne.read_evokeds(fname)[0].pick(signal_type).info
evoked = mne.EvokedArray(df.values.T, info, tmin=0)
evoked.plot_topomap(axes=axes)
plt.savefig(os.path.join('figs', 'evoked' + case_specific + '.png'))

# data to give to the participants:
# labels with their names and vertices: parcels

# reading forward matrix and saving
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
lead_field = fwd['sol']['data']

if signal_type == 'eeg':
    picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True, exclude=[])
    lead_field = lead_field[picks_eeg, :]
elif signal_type == 'meg':
    picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False, exclude=[])
    lead_field = lead_field[picks_meg, :]
elif signal_type == 'mag' or signal_type == 'grad':
    picks_meg = mne.pick_types(fwd['info'], meg=signal_type,
                               eeg=False, exclude=[])
    lead_field = lead_field[picks_meg, :]

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
         lead_field=lead_field, parcel_indices=parcel_indices_l,
         signal_type=signal_type)
