import os.path as op
import warnings

import numpy as np
from numba import jit
from joblib import Memory

from ot import emd2

import mne

from mne.datasets import sample

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
    print("Running floyd-warshall")
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
    if subject == "fsaverage" or subject == "sample":
        data_path = sample.data_path()
        subjects_dir = op.join(data_path, 'subjects')
        src = mne.setup_source_space(subject=subject,
                                     spacing="ico4",
                                     subjects_dir=subjects_dir,
                                     add_dist=False)
    else:
        fwd_fname = config.get_fwd_fname(subject)
        src = mne.read_forward_solution(fwd_fname)["src"]
    return src


@mem.cache
def _compute_full_ground_metric(subject, hemi, subjects_dir):
    """Compute geodesic distance matrix on the triangulated mesh of src."""
    if hemi == "both":
        hemi_indices = [0, 1]
    else:
        hemi_indices = [hemi == "rh"]
    src = _get_src_space(subject, subjects_dir)
    Ds = []
    for i in hemi_indices:
        tris = src[i]["use_tris"]
        vertno = src[i]["vertno"]
        points = src[i]["rr"][vertno]
        if np.max(tris) > len(np.unique(tris)):
            tris = tris.copy()
            for ii, v in enumerate(vertno):
                tris[tris == v] = ii

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
    n_simu, n_classes = y_true.shape
    assert n_classes == len(parcels)
    if y_score.any() is False:
        warnings.warn("Cannot compute EMD with a null y_score. Returned inf")
        return float("inf")
    subject = parcels[0].subject
    hemis = [p.hemi for p in parcels]
    if len(np.unique(hemis)) == 2:
        hemi = "both"
    else:
        hemi = hemis[0]
    # compute a ground metric on a ico4 src space
    ground_metric = _compute_full_ground_metric(subject, hemi=hemi,
                                                subjects_dir=subjects_dir)

    # get nearest vertices to the parcel centers in the src space
    src = _get_src_space(subject, subjects_dir)
    inuse_lh = src[0]["inuse"].astype(bool)
    inuse_rh = src[1]["inuse"].astype(bool),

    src_coords = np.concatenate((src[0]["rr"][inuse_lh],
                                 src[1]["rr"][inuse_rh]))
    src_coords = src_coords * 1000
    parcel_positions = find_centers_of_mass(parcels, subjects_dir,
                                            return_positions=True)
    distances = ((src_coords[:, None, :] -
                  parcel_positions[None, :, :]) ** 2).sum(axis=-1)
    nearest_vertices = np.argmin(distances, axis=0)

    # keep only the vertices of the parcel in the ground metric
    ground_metric = ground_metric[nearest_vertices, :][:, nearest_vertices]
    ground_metric = np.ascontiguousarray(ground_metric)

    # change unit to cm
    ground_metric = ground_metric * 100
    # compute emd
    score = 0.
    for yt, ys in zip(y_true, y_score):
        yt = yt / yt.sum()
        ys = ys / ys.sum()
        score += emd2(yt, ys, ground_metric)
    score /= n_simu

    return score
