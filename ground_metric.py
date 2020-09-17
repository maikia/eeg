import numpy as np
import mne

from numba import njit
import time

from ot import emd2

import config as config


def mesh_all_distances(points, tris):
    """Compute all pairwise distances on the mesh based on edges lengths
    using Floyd-Warshall algorithm
    """
    A = mne.surface.mesh_dist(tris, points)
    A = A.toarray()
    print('Running Floyd-Warshall algorithm')
    A[A == 0.] = 1e6
    A.flat[::len(A) + 1] = 0.
    D = floyd_warshall(A)
    return D


@njit(nogil=True, cache=True)
def floyd_warshall(dist):
    npoints = dist.shape[0]
    for k in range(npoints):
        for i in range(npoints):
            for j in range(npoints):
                # If i and j are different nodes and if
                # the paths between i and k and between
                # k and j exist, do
                # d_ikj = min(dist[i, k] + dist[k, j], dist[i, j])
                d_ikj = dist[i, k] + dist[k, j]
                if ((d_ikj != 0.) and (i != j)):
                    # See if you can't get a shorter path
                    # between i and j by interspacing
                    # k somewhere along the current
                    # path
                    if ((d_ikj < dist[i, j]) or (dist[i, j] == 0)):
                        dist[i, j] = d_ikj
    return dist


def compute_ground_metric(subject, subjects_dir, annot, grade):
    """Computes pairwise distance matrix between the parcels"""
    spacing = "ico%d" % grade
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir)
    ground_metrics = []
    n_labels = []
    for hemi, s in zip(["lh", "rh"], src):
        print("Doing hemi %s ..." % hemi)
        tris = s["use_tris"]
        vertno = s["vertno"]
        points = s["rr"][vertno]
        D = mesh_all_distances(points, tris)
        n_vertices = len(vertno)

        mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                                  verbose=True)
        labels = mne.read_labels_from_annot(subject, annot, hemi,
                                            subjects_dir=subjects_dir)
        n_labels.append(len(labels))

        print("Morphing labels ...")
        labels = [label.morph(subject_to=subject, subject_from=subject,
                              grade=grade, subjects_dir=subjects_dir)
                  for label in labels]
        n_parcels = len(labels)
        ground_metric_hemi = np.zeros((n_parcels, n_parcels))
        for ii, label_i in enumerate(labels):
            a = np.zeros(n_vertices)
            a[label_i.vertices] = 1
            a /= a.sum()
            for jj in range(ii + 1, n_parcels):
                b = np.zeros(n_vertices)
                b[labels[jj].vertices] = 1
                b /= b.sum()
                ground_metric_hemi[ii, jj] = emd2(a, b, D)
        ground_metric_hemi = 0.5 * (ground_metric_hemi + ground_metric_hemi.T)
        ground_metrics.append(ground_metric_hemi)
    across_hemi_mat = np.ones((n_labels[0], n_labels[1]))
    across_hemi_mat *= ground_metric_hemi.max() * 2
    ground_metric = np.block([[ground_metrics[0], across_hemi_mat],
                              [across_hemi_mat.T, ground_metrics[1]]])

    ground_metric *= 1000  # change units to mm

    return ground_metric


if __name__ == "__main__":
    start_time = time.time()
    subjects_dir = config.get_subjects_dir_subj("sample")
    grade = 3
    annot = "aparc_sub"
    ground_metric = compute_ground_metric("fsaverage",
                                          subjects_dir=subjects_dir,
                                          annot=annot,
                                          grade=grade)
    np.save("data/ground_metric.npy", ground_metric)

    print("It took %s seconds to execute" % (time.time() - start_time))
