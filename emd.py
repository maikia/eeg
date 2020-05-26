import os.path as op
import warnings

import numpy as np
import jit
from joblib import Memory

from ot import emd2

import mne

from mne.datasets import testing

import config
from simulation.parcels import find_centers_of_mass

mem = Memory('./')


def _mesh_all_distances(points, tris, verts=None):
    """Compute all pairwise distances on the mesh."""
    A = mne.surface.mesh_dist(tris, points)
    if verts is not None:
        A = A[verts][:, verts]
    A = A.toarray()
    A[A == 0.] = 1e6
    A.flat[::len(A) + 1] = 0.
    A = _floyd_warshall(A)
    return A


@jit(nogil=True, cache=True, nopython=True)
def _floyd_warshall(dist):
    """Run Floyd-Warshall algorithm to find shortest path on a mesh."""
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


def _get_src_space(subject, subjects_dir):
    if subject == "fsaverage":
        data_path = testing.data_path()
        subjects_dir = op.join(data_path, 'subjects')
        src = mne.setup_source_space(subject="fsaverage",
                                     spacing="ico4",
                                     subjects_dir=subjects_dir,
                                     add_dist=False)
    else:
        fwd_fname = config.get_fwd_fname(subject)
        src = mne.read_forward_solution(fwd_fname)["src"]
    return src


@mem.cache
def _compute_full_ground_metric(subject, hemi_indices, subjects_dir):
    """Compute geodesic distance matrix on the triangulated mesh of src."""
    src = _get_src_space(subject, subjects_dir)
    Ds = []
    for i in hemi_indices:
        tris = src[i]["use_tris"]
        vertno = src[i]["vertno"]
        points = src[i]["rr"][vertno]

        D = _mesh_all_distances(points, tris)
        Ds.append(D)
    if len(hemi_indices) == 1:
        return D

    n1, n2 = len(Ds[0]), len(Ds[1])

    D = (Ds[0]).max() * np.ones((n1 + n2, n1 + n2))
    D[:n1, :n1] = Ds[0]
    D[n1:, n1:] = Ds[1]
    return D


def emd_score(y_true, y_score, parcels, subjects_dir):
    """Compute Earth-Mover-Distance.

    parameters:
    -----------

    y_true: binary array (n_classes,)
    y_score: array (n_classes,)
    parcels: parcellation list
    subjects_dir: str

    Returns:
    --------
    float, emd value
    """
    assert len(y_true) == len(parcels)
    if y_score.any() is False:
        warnings.warn("Cannot compute EMD with a null y_score. Returned inf")
        return float("inf")
    subject = parcels[0].subject
    hemis = [p.hemi for p in parcels]
    hemi_indices = [["lh", "rh"].index(h) for h in np.unique(hemis)]
    # compute a ground metric on a ico4 src space
    ground_metric = _compute_full_ground_metric(subject, hemi=hemi_indices,
                                                subjects_dir=subjects_dir)

    # get nearest vertices to the parcel centers in the src space
    src = _get_src_space(subject, subjects_dir)
    src_coords = np.concatenate((src[0]["rr"], src[1]["rr"]))
    parcel_positions = find_centers_of_mass(parcels, subjects_dir,
                                            return_positions=True)
    distances = ((src_coords[:, None, :] -
                  parcel_positions[None, :, :]) ** 2).sum(axis=-1)
    nearest_vertices = np.argmin(distances, axis=1)

    # shift the vertices in the right hemi by n_sources_lh
    n_sources_lh = src[0]["nuse"]
    shift = (np.array(hemis) == "rh").astype(int) * n_sources_lh
    nearest_vertices += shift

    # keep only the vertices of the parcel in the ground metric
    ground_metric = ground_metric[nearest_vertices, :][:, nearest_vertices]

    # compute emd
    y_true = y_true / y_true.sum()
    y_score = y_score / y_score.sum()
    score = emd2(y_true, y_score, ground_metric)

    return score
