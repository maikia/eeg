import mne
import numpy as np
from mne import SourceEstimate as STC
from matplotlib import cm
import config


if __name__ == "__main__":
    M = np.load("data/ground_metric.npy")
    subjects_dir = config.get_subjects_dir_subj("sample")
    subject = "fsaverage"
    annot = "aparc_sub"

    parcels = mne.read_labels_from_annot(subject, annot, hemi='both',
                                         subjects_dir=subjects_dir)

    # pick some parcel
    idx = 18
    xx = parcels[idx]
    dists = M[idx]
    colors = cm.Reds(np.linspace(0., 1., len(parcels)))

    # create some empty stc
    stc0 = STC(data=np.zeros((1, 1)), vertices=[np.array([]), np.array([0])],
               tmin=0, tstep=1)
    brain = stc0.plot(subject, hemi="both", subjects_dir=subjects_dir)
    for ii, color in zip(np.argsort(dists), colors):
        hemi = parcels[ii].name[-2:]
        brain.add_label(parcels[ii], hemi=hemi, borders=True, color=color)
    brain.show_view("frontal")
    brain.save_image("data/ground_metric.png")
