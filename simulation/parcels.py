import numpy as np
import os.path as op

import mne
from raw_signal import random_parcellation


def find_corpus_callosum(subject, subjects_dir, hemi='lh'):
    aparc_file = op.join(subjects_dir,
                         subject, "label",
                         hemi + ".aparc.a2009s.annot")

    labels = mne.read_labels_from_annot(subject=subject,
                                        annot_fname=aparc_file,
                                        hemi=hemi,
                                        subjects_dir=subjects_dir)

    assert labels[-1].name[:7] == 'Unknown'  # corpus callosum
    return labels[-1]


# remove those parcels which overlap with corpus callosum
def remove_overlapping(parcels, xparcel):
    not_overlapping = []
    for parcel in parcels:
        if not np.any(np.isin(parcel.vertices, xparcel.vertices)):
            not_overlapping.append(parcel)
    return not_overlapping


# we will randomly create a parcellation of n parcels in one hemisphere
def make_random_parcellation(path_annot, n, hemi, subjects_dir, random_state,
                             subject, remove_corpus_callosum=False):
    parcel = random_parcellation(subject, n, hemi, subjects_dir=subjects_dir,
                                 surface='white', random_state=random_state)

    if remove_corpus_callosum:
        xparcel = find_corpus_callosum(subject, subjects_dir, hemi=hemi)
        parcel = remove_overlapping(parcel, xparcel)
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
