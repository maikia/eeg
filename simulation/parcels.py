import gdist
import nibabel as nib
import numpy as np
import os
import pandas as pd

from mne import random_parcellation
from mne import read_labels_from_annot
from mne import write_labels_to_annot


def find_corpus_callosum(subject, subjects_dir, hemi='lh'):
    aparc_file = os.path.join(subjects_dir,
                         subject, "label",
                         hemi + ".aparc.a2009s.annot")

    labels = read_labels_from_annot(subject=subject,
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
    write_labels_to_annot(parcel, subjects_dir=subjects_dir,
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


def calc_dist_matrix_for_sbj(data_dir, subject):
    '''
    '''
    base_dir = 'mne_data/MNE-sample-data/subjects/' + subject
    surf_lh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                               'surf/lh.pial'))
    surf_rh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                               'surf/rh.pial'))
    labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                               allow_pickle=True)

    labels_x = labels_x['arr_0']
    labels_x_lh = [s for s in labels_x if s.hemi == 'lh']
    labels_x_rh = [s for s in labels_x if s.hemi == 'rh']
    distance_matrix_lh = calc_dist_matrix_labels(surf=surf_lh, source_nodes=labels_x_lh,
                                   dist_type = "min", nv = 20)
    distance_matrix_rh = calc_dist_matrix_labels(surf=surf_rh, source_nodes=labels_x_rh,
                                   dist_type = "min", nv = 20)
    return distance_matrix_lh, distance_matrix_rh


def calc_dist_matrix_labels(surf, source_nodes, dist_type='min', nv = 0):
    '''
       extract all the necessary information from the given brain surface and
       labels and calculate the distance
       source_nodes : list of labels
       nv = every how many vertices will be skipped (useful if a lot of
            vertices)
       returns distance matrix, pandas dataframe
    '''

    vertices, triangles = surf
    new_triangles = triangles.astype('<i4')
    cn = [label.name for label in source_nodes]
    dist_matrix = pd.DataFrame(columns = cn, index = cn)
    np.fill_diagonal(dist_matrix.values, 0)

    # TODO: parallel?
    # NOTE: very slow
    for i in range(len(source_nodes)-1):
        prev_source = source_nodes[i].vertices.astype('<i4')
        prev_name = source_nodes[i].name

        for j in range(i+1, len(source_nodes)):
            loading =  ("i: " + str(i) + '/' + str(len(source_nodes)) + ':' +
                        "." * j + ' ' * (len(source_nodes)-j-1) + '|')
            print(loading, end="\r")

            # computes the distance between the targets and the source (gives as
            # many values as targets)
            next_source = source_nodes[j].vertices.astype('<i4')
            next_name = source_nodes[j].name
            distance = gdist.compute_gdist(vertices, new_triangles,
                       source_indices=np.array(prev_source, ndmin=1)[::nv],
                       target_indices=np.array(next_source, ndmin=1)[::nv])
            if dist_type == 'min':
                dist = np.min(distance)
            elif dist_type == 'mean':
                dist = np.mean(distance)

            dist_matrix.loc[prev_name][next_name] = dist
            dist_matrix.loc[next_name][prev_name] = dist

    # import seaborn as sns
    # sns.heatmap(dist_matrix, annot=True)
    return dist_matrix


def dist_calc(surf, source, target):

    """
    source and target are arrays of vertices, surf, surface of the brain,
    return min distance between the two arrays
    """
    vertices, triangles = surf
    new_triangles = triangles.astype('<i4')

    distance = gdist.compute_gdist(vertices, new_triangles,
                       source_indices=np.array(source, ndmin=1),
                       target_indices=np.array(target, ndmin=1))
    return np.min(distance)

