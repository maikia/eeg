import numpy as np
import os.path as op

from mne import random_parcellation
from mne import read_labels_from_annot
from mne import write_labels_to_annot


def find_corpus_callosum(subject, subjects_dir, hemi='lh'):
    aparc_file = op.join(subjects_dir,
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


def dist_calc(surf, cortex, source_nodes, dist_type = "min"):

    """
        Calculates the minimum distance between all of the given parcels
        returns the distance matrix
    """

    import gdist
    #from utils import surf_keep_cortex, translate_src, recort


    # cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    # translated_source_nodes = translate_src(source_nodes, cortex)

    # if len(src_new) == 1:
    #     
    #     dist_type = "min"
    #    print("calculating min for single node ROI")
        
    # if dist_type == "min":
    vertices, triangles = surf
    
    new_triangles = triangles.astype('<i4')
    # sn_converted = []
    # for sn in source_nodes:
    #    new_sn = sn.vertices.astype('<i4')
    #    sn_converted.append(new_sn)
    # from joblib import Memory, Parallel, delayed
    distance_matrix = np.zeros((len(source_nodes), len(source_nodes)))
    data_nodes = []
    # N_JOBS = -1
    # 
    # distance_matrix = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(
    #    delayed(init_signal)(parcels_subject, raw_fname, fwd_fname, subject,
    #                         n_parcels_max, seed, signal_type)
    #    for seed in tqdm(seeds)
    #)
    # take only every n-th vertex
    nv = 20
    # TODO: parallel?
    # NOTE: very slow
    for i in range(len(source_nodes)-1):
        previous_source = source_nodes[i].vertices.astype('<i4')
        # print('i: {}/{}'.format(i, len(source_nodes)-1), end =" ")
        for j in range(i+1, len(source_nodes)):
            loading =  "i: " + str(i) + '/' + str(len(source_nodes)) + ':' + "." * j + ' ' * (len(source_nodes)-j-1) + ';'
            print(loading, end="\r")

            # computes the distance between the targets and the source (gives as
            # many values as targets)
            next_source = source_nodes[j].vertices.astype('<i4')
            distance = gdist.compute_gdist(vertices, new_triangles,
                       source_indices=np.array(previous_source, ndmin=1)[::nv],
                       target_indices=np.array(next_source, ndmin=1)[::nv])
            min_dist = np.min(distance)
            distance_matrix[i, j] = distance_matrix[j, i] = min_dist

    return distance_matrix
