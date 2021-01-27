import numpy as np
from mne import SourceEstimate as STC
from matplotlib import cm
import config

from simulate import get_ready_parcels

if __name__ == "__main__":
    M = np.load("data/ground_metric.npy")
    subjects_dir = config.get_subjects_dir_subj("sample")
    subject = "fsaverage"
    annot = "aparc_sub"

    parcels = get_ready_parcels(subjects_dir, annot)

    # pick some parcel
    idx = 17
    xx = parcels[idx]
    dists = M[idx]
    # set all max_dists to the same dark red color
    n_ones = (dists == 1.).sum()
    colors = cm.Reds(np.linspace(0., 1., len(parcels) - n_ones))
    colors = np.concatenate([colors, cm.Reds(np.ones(n_ones))])

    # create some empty stc
    stc0 = STC(data=np.zeros((1, 1)), vertices=[np.array([]), np.array([0])],
               tmin=0, tstep=1)
    brain = stc0.plot(subject, hemi="both", subjects_dir=subjects_dir)
    for ii, color in zip(np.argsort(dists), colors):
        hemi = parcels[ii].name[-2:]
        brain.add_label(parcels[ii], hemi=hemi, borders=True, color=color)
    brain.show_view("frontal")
    brain.save_image("data/ground_metric.png")
