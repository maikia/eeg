from surfer import Brain
import matplotlib.pylab as plt
import mne
import numpy as np
import os


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

    brain.save_image(os.path.join(fig_dir, 'brain' + ext))


def plot_sources_at_activation(X, y, fig_dir='figs', ext='.png'):
    # plots each parcel (careful if too many parcels) at the moment of the
    # activation indicated by y == 1 at 5 different samples

    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    data_path = mne.datasets.sample.data_path()
    fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    info = mne.read_evokeds(fname)[0].pick('eeg').info

    n_classes = y.shape[1]
    fig, axes = plt.subplots(5, n_classes, figsize=(16, 4))

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
    print('saved in ' + os.path.join(fig_dir, 'visualize' + ext))
