from surfer import Brain
import matplotlib.pylab as plt
import mne
import numpy as np
import os
import pandas as pd


def plot_evoked(raw, events, fig_dir='figs', ext='.png'):
    # Plot evoked data to get another view of the simulated raw data.
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
    evoked = epochs.average()
    evoked.plot()
    evoked.save_image(os.path.join(fig_dir, 'evoked' + ext))


def visualize_brain(subject, hemi, annot_name, subjects_dir, parcels_selected,
                    fig_dir='figs', ext='.png'):
    # visualize the brain with the parcellations and the source of the signal
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    brain = Brain(subject, hemi, 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white', size=(800, 600))

    for parcel in parcels_selected:
        brain.add_label(parcel, alpha=1, color=parcel.color)

    save_fig = os.path.join(fig_dir, 'brain_' + annot_name + ext)
    brain.save_image(save_fig)
    print('brain image saved in {}'.format(save_fig))


def plot_sources_at_activation(X, y, signal_type, fig_dir='figs', ext='.png'):
    # plots each parcel (careful if too many parcels) at the moment of the
    # activation indicated by y == 1 at 5 different samples
    # signal type must be either 'eeg', 'meg', 'mag' or 'grad'

    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    username = os.environ.get('USER')
    if "hjana" in username:
        data_path = "/storage/store/data/mne_data/MNE-sample-data"
    else:
        data_path = mne.datasets.sample.data_path()
    data_path = mne.datasets.sample.data_path()

    fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    info = mne.read_evokeds(fname)[0].pick(signal_type).info

    n_classes = y.shape[1]
    fig, axes = plt.subplots(5, n_classes, figsize=(16, 4))

    X = X.iloc[:, :-2]
    for k in range(n_classes):
        X_k = X.iloc[np.argmax(y, axis=1) == k]
        info_temp = info.copy()
        info_temp['bads'] = []
        for i, ax in enumerate(axes[:, k]):
            if X_k.shape[0] > i:
                mne.viz.plot_topomap(X_k.iloc[i].values, info_temp, axes=ax,
                                     show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'visualize' + ext))


def plot_samples_vs_score(scores_all, data_samples, fig_dir='figs',
                          ext='.png'):
    '''
    scores_all: dataframe where each row is the result for one data, with
        the keys:
        'n_parcels': number of parcels in the given simulated data
        'max_sources': maximum number of parcels activated simulteneoulsy
        'scores': scores for running the data on the data_samples (if
        len(scores) might be less than len(data_samples)). Then only first
        data_samples are considered
    data_samples: np.array with the number of smaples run to obtain each score
    '''
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    plt.figure()
    max_sources_all = len(scores_all['max_sources'].unique())

    for max_sources in range(1, max_sources_all + 1):
        plt.subplot(max_sources_all, 1, max_sources)
        plt.title('max sources: ' + str(max_sources))
        scores_used = scores_all[scores_all['max_sources'] == max_sources]

        for index, row in scores_used.iterrows():
            plt.plot(data_samples[:len(row['scores'])], row['scores'],
                     label='parcels: ' + str(row['n_parcels']))
        plt.ylabel('score')
        plt.legend()
    plt.xlabel('number of samples used')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'score_data' + ext))


def plot_y_pred_true_parcels(subject_name, labels_pred, labels_true,
                             colors={'pred': 'y', 'true': 'b',
                                     'overlap': 'g'}):
    # fig_name = (subject_name + '_' + str(len(parcels_subject)) + '_' +
    #            str(n_parcels_max))
    # TODO: make the switch automatic depending on who/where it runs
    data_path = 'mne_data/MNE-sample-data'  # Maja
    # mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')  # Alex
    # data_path = mne_dir + '/MNE-sample-data'
    subjects_dir = os.path.join(data_path, 'subjects')
    hemi = 'both'
    brain = Brain(subject_name, hemi, 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white')

    for parcel in labels_pred:
        brain.add_label(parcel, alpha=1, color=colors['pred'])
    for parcel in labels_true:
        brain.add_label(parcel, alpha=1, color=colors['true'])
    for parcel in set(labels_pred) & set(labels_true):
        brain.add_label(parcel, alpha=1, color=colors['overlap'])
    import pdb
    pdb.set_trace()


def plot_distance(subject, data_dir, parcels, parcel_name='1-lh'):
    print('displaying brain of subject {}, distances from label {}'.format(
                                                    subject, parcel_name))
    data_path = 'mne_data/MNE-sample-data'  # Maja
    subjects_dir = os.path.join(data_path, 'subjects')

    hemi = 'both'
    brain = Brain(subject, hemi, 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white')

    dist_path_lh = os.path.join(data_dir, subject + '_dist_matrix_lh.csv')
    dist_matrix_lh = pd.read_csv(dist_path_lh, index_col=0)
    dist_matrix_lh_norm = dist_matrix_lh.divide(dist_matrix_lh.max().max())

    if 'lh' in parcel_name:
        hemi = 'lh'
    elif 'rh' in parcel_name:
        hemi = 'rh'
    for parcel in parcels:
        if parcel_name == parcel.name:
            brain.add_label(parcel, alpha=1, color='blue')
        elif hemi in parcel.name:
            dist = dist_matrix_lh_norm.loc[parcel_name][parcel.name]
            brain.add_label(parcel, alpha=1, color=str(1 - dist))
    import pdb; pdb.set_trace()
    cm = parcel.center_of_mass(subject, subjects_dir = subjects_dir)

    brain.add_foci(cm, coords_as_verts=True, hemi='lh', color='blue')
