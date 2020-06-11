import pytest

import mne

import numpy as np
import os
import pandas as pd


from sklearn import linear_model
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import hamming_loss, jaccard_score

from simulation.sparse_regressor import SparseRegressor, ReweightedLasso

from mne import read_labels_from_annot

SEED = 42


@pytest.fixture(scope="session")
def make_dataset_from_sample():
    # assign paths
    data_path = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_path, 'subjects')

    # get parcels and remove corpus callosum
    parcels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined',
                                         'both', subjects_dir=subjects_dir)
    # corpus callosum labels
    aparc_file_lh = os.path.join(subjects_dir,
                                 'fsaverage', "label",
                                 'lh.aparc.a2009s.annot')
    aparc_file_rh = os.path.join(subjects_dir,
                                 'fsaverage', "label",
                                 'rh.aparc.a2009s.annot')

    labels_corpus_lh = read_labels_from_annot(subject='fsaverage',
                                              annot_fname=aparc_file_lh,
                                              hemi='lh',
                                              subjects_dir=subjects_dir)
    labels_corpus_rh = read_labels_from_annot(subject='fsaverage',
                                              annot_fname=aparc_file_rh,
                                              hemi='rh',
                                              subjects_dir=subjects_dir)

    assert labels_corpus_lh[-1].name[:7] == 'Unknown'  # corpus callosum
    assert labels_corpus_rh[-1].name[:7] == 'Unknown'  # corpus callosum
    corpus_callosum = [labels_corpus_lh[-1],
                       labels_corpus_rh[-1]]

    # remove from parcels all the vertices from corpus callosum
    to_remove = []
    for idx, parcel in enumerate(parcels):
        if parcel.hemi == 'lh':
            cc_free = set(parcel.vertices) - set(corpus_callosum[0].vertices)
        elif parcel.hemi == 'rh':
            cc_free = set(parcel.vertices) - set(corpus_callosum[1].vertices)
        parcel.vertices = np.array(list(cc_free))
        if len(parcel.vertices) == 0:
            to_remove.append(idx)

    # morph from fsaverage to sample
    parcels = mne.morph_labels(parcels, 'sample', 'fsaverage', subjects_dir,
                               'white')

    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                            verbose=True)

    raw_fname = os.path.join(data_path, 'MEG', 'sample',
                             'sample_audvis_raw.fif')
    fwd_fname = os.path.join(data_path, 'MEG', 'sample',
                             'sample_audvis-meg-eeg-oct-6-fwd.fif')
    assert os.path.exists(raw_fname)
    assert os.path.exists(fwd_fname)

    info = mne.io.read_info(raw_fname)
    sel = mne.pick_types(info, meg='grad', eeg=False, stim=True, exclude=[])
    info = mne.pick_info(info, sel)
    tstep = 1. / info['sfreq']

    # read forward solution
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    rng = np.random.RandomState(42)

    n_samples = 2
    signal_len = 10
    n_events = 50
    add_noise = False
    source_time_series = np.sin(2. * np.pi *
                                18. * np.arange(signal_len) * tstep) * 10e-9

    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = signal_len * len(parcels) + 200 * np.arange(n_events)
    events[:, 2] = 1  # All events have the sample id.

    signal_list = []
    true_idx = np.empty(n_samples, dtype=np.int16)
    for idx, source in enumerate(range(n_samples)):
        idx_source = rng.choice(np.arange(len(parcels)))
        true_idx[idx] = int(idx_source)
        source = parcels[idx_source]

        source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
        source_simulator.add_data(source, source_time_series, events)

        raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)

        if add_noise:
            cov = mne.make_ad_hoc_cov(raw.info)
            mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.02])

        evoked = mne.Epochs(raw, events, tmax=0.3).average()
        data = evoked.data[:, np.argmax((evoked.data ** 2).sum(axis=0))]
        signal_list.append(data)
    # data <=signal_list
    # names_parcels_selected <= target_list
    # to_activate <= activated

    signal_list = np.array(signal_list)
    data_labels = ['e%d' % (idx + 1) for idx in range(signal_list.shape[1])]

    X = pd.DataFrame(signal_list, columns=list(data_labels))
    X['subject_id'] = 0
    X['subject'] = '0'

    y = np.zeros((n_samples, len(parcels)))
    y[np.arange(n_samples), true_idx] = 1

    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    lead_field = fwd['sol']['data']
    picks_meg = mne.pick_types(fwd['info'], meg='grad',
                               eeg=False, exclude=[])
    lead_field = lead_field[picks_meg, :]

    parcel_vertices = {}
    for idx, parcel in enumerate(parcels, 1):
        parcel_name = str(idx) + parcel.name[-3:]
        parcel_vertices[parcel_name] = parcel.vertices
        parcel.name = parcel_name

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

    parcel_indices = parcel_indices[np.where(inuse)[0]]
    assert len(parcel_indices) == lead_field.shape[1]

    lead_field = lead_field[:, parcel_indices != 0]
    parcel_indices = parcel_indices[parcel_indices != 0]

    return X, y, [lead_field], [parcel_indices]


def make_dataset(n_subjects=1, n_samples_per_subj=2, n_parcels=10,
                 n_sources=300, n_sensors=100, max_true_sources=1):
    rng = np.random.RandomState(SEED)

    electrode_names = [f'e{i}' for i in range(1, n_sensors + 1)]

    L, parcel_indices = [], []
    X = []
    y = []

    for k in range(n_subjects):
        # make random Lead Field
        Lk = rng.randn(n_sensors, n_sources)
        L.append(Lk)

        parcel_indices_k = (np.arange(n_sources) % n_parcels) + 1
        parcel_indices.append(parcel_indices_k)

        mask = np.zeros((n_samples_per_subj, n_sources), dtype=bool)
        yk = np.zeros((n_samples_per_subj, n_parcels), dtype=int)
        for i in range(n_samples_per_subj):
            mask[i, rng.randint(0, n_sources, size=max_true_sources)] = True
            yk[i, np.unique(parcel_indices_k[mask[i]]) - 1] = 1

        beta = rng.randn(n_samples_per_subj, n_sources)
        beta[~mask] = 0

        Xk = (Lk @ beta.T).T

        Xk = pd.DataFrame(Xk, columns=electrode_names)
        Xk['subject_id'] = k
        X.append(Xk)
        y.append(yk)

    X = pd.concat(X, axis=0)
    X['subject'] = X['subject_id'].apply(str)
    y = np.concatenate(y, axis=0)

    return X, y, L, parcel_indices


lasso = linear_model.LassoLars(max_iter=2, normalize=False,
                               fit_intercept=False, alpha=0.01)
rwl1 = ReweightedLasso(alpha_fraction=.01, max_iter=20,
                       max_iter_reweighting=1, tol=1e-4)
rwl10 = ReweightedLasso(alpha_fraction=.01, max_iter=20,
                        max_iter_reweighting=10, tol=1e-4)


@pytest.mark.parametrize('model, hl_max',
                         [(lasso, 0.02),
                          (rwl1, 0),
                          (rwl10, 0.002)
                          ])
def test_sparse_regressor(model, hl_max):
    n_subjects = 1
    n_samples_per_subj = 100
    n_parcels = 10
    n_sources = 500
    n_sensors = 100
    max_true_sources = 2
    X, y, L, parcel_indices = make_dataset(
        n_subjects, n_samples_per_subj, n_parcels, n_sources,
        n_sensors, max_true_sources
    )

    # assert that all the dimensions correspond
    assert X.shape == (n_samples_per_subj * n_subjects, n_sensors + 2)
    assert X['subject_id'].unique() == np.arange(n_subjects)
    assert X.shape[0] == y.shape[0]

    assert len(L) == n_subjects == len(parcel_indices)
    assert L[0].shape == (n_sensors, n_sources)
    assert y.shape[1] == n_parcels
    assert np.mean(y.sum(1) == max_true_sources) > 0.9

    sparse_regressor = SparseRegressor(
        L, parcel_indices, model
    )

    y_pred = sparse_regressor.predict(X)
    hl = hamming_loss(y_pred, y)
    assert hl <= hl_max


@pytest.mark.parametrize('model, hl_max',
                         [(lasso, 0.02),
                          (rwl1, 0),
                          (rwl10, 0.002)
                          ])
def test_sparse_regressor_on_sample(make_dataset_from_sample, model, hl_max):
    n_subjects = 1
    n_samples_per_subj = 2
    n_parcels = 10
    n_sources = 500
    n_sensors = 100
    max_true_sources = 1
    X, y, L, parcel_indices = make_dataset(
        n_subjects, n_samples_per_subj, n_parcels, n_sources,
        n_sensors, max_true_sources
    )
    X, y, L, parcel_indices = make_dataset_from_sample
    assert X.shape[0] == y.shape[0]

    assert len(L) == n_subjects == len(parcel_indices)
    assert np.mean(y.sum(1) == max_true_sources) > 0.9

    sparse_regressor = SparseRegressor(
        L, parcel_indices, model
    )

    y_pred = sparse_regressor.predict(X)
    hl = hamming_loss(y_pred, y)
    assert hl <= hl_max
