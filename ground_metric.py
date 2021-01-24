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
    # Compute 2 large ground metrics between all vertices on ico3
    # one for each hemisphere
    for hemi, s in zip(["lh", "rh"], src):
        print("Doing hemi %s ..." % hemi)
        tris = s["use_tris"]
        vertno = s["vertno"]
        points = s["rr"][vertno]
        D = mesh_all_distances(points, tris)
        n_vertices = len(vertno)
        ground_metrics.append(D)

    largest_distance = max(ground_metrics[0].max(), ground_metrics[1].max())
    labels = mne.read_labels_from_annot(
        'fsaverage', annot, hemi='both', subjects_dir=subjects_dir)

    labels = [label.morph(subject_to=subject, subject_from=subject,
                          grade=grade, subjects_dir=subjects_dir)
              for label in labels]
    n_parcels = len(labels)

    # fill the final ground_metric between parcels by comparing
    # uniform measures with the parcels as supports
    ground_metric = np.zeros((n_parcels, n_parcels))
    for ii, label_i in enumerate(labels):
        hemi_i = label_i.name[-2:]
        hemi_indx = int(hemi_i == "rh")
        a = np.zeros(n_vertices)
        a[label_i.vertices] = 1
        a /= a.sum()
        for jj in range(ii, n_parcels):
            # if in the same hemi, compute emd
            # else use the largest distance in ground_metrics
            if hemi_i in labels[jj].name:
                b = np.zeros(n_vertices)
                b[labels[jj].vertices] = 1
                b /= b.sum()
                ground_metric[ii, jj] = emd2(a, b, ground_metrics[hemi_indx])
            else:
                ground_metric[ii, jj] = largest_distance
    ground_metric *= 1000  # change units to mm
    # fill the lower part by symmetry
    ground_metric = 0.5 * (ground_metric + ground_metric.T)

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
