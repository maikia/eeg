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
    """

    import gdist
    #from utils import surf_keep_cortex, translate_src, recort
    import numpy as np

    source_nodes = source_nodes['arr_0']
    source_nodes[0].vertices
    # cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    # translated_source_nodes = translate_src(source_nodes, cortex)

    # if len(src_new) == 1:
    #     
    #     dist_type = "min"
    #    print("calculating min for single node ROI")
        
    # if dist_type == "min":
    vertices, triangles = surf
    
    new_triangles = triangles.astype('<i4')
    sn_converted = []
    # for sn in source_nodes:
    #    new_sn = sn.vertices.astype('<i4')
    #    sn_converted.append(new_sn)

    distance_matrix = np.zeros((len(sn_converted), len(sn_converted)))
    data_nodes = []
    previous_source = source_nodes[i].vertices.astype('<i4')
    for i in range(len(sn_converted)):
        for j in range(i, len(sn_converted)):
            # computes the distance between the targets and the source (gives as
            # many values as targets)
            next_source = source_nodes[j].vertices.astype('<i4')
            distance = gdist.compute_gdist(vertices, new_triangles,
                       source_indices=np.array(previous_source, ndmin=1),
                       target_indices=np.array(next_source, ndmin=1))
            min_dist = np.min(distance)
            distance_matrix[i, j] = distance_matrix[j, i] = min_dist
            previous_source = next_source
            import pdb; pdb.set_trace()

    # data = gdist.compute_gdist(vertices, new_triangles, source_indices = np.array(sn_converted)) #, target_indices = sn_converted[2])
    import pdb; pdb.set_trace()

    return distance_matrix
