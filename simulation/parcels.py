import gdist
import nibabel as nib
import numpy as np
import os
import pandas as pd
import sys

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
    subjects_dir = 'mne_data/MNE-sample-data/subjects'
    base_dir = os.path.join(subjects_dir, subject)

    surf_lh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                           'surf/lh.pial'))
    surf_rh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                           'surf/rh.pial'))
    labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                       allow_pickle=True)
    surf_lh[0][cms_lh[0]]
    labels_x = labels_x['arr_0']
    labels_x_lh = [s for s in labels_x if s.hemi == 'lh']
    labels_x_rh = [s for s in labels_x if s.hemi == 'rh']

    distance_matrix_lh = calc_dist_matrix_labels(surf=surf_lh,
                                                 source_nodes=labels_x_lh,
                                                 dist_type="min", nv=20)
    distance_matrix_rh = calc_dist_matrix_labels(surf=surf_rh,
                                                 source_nodes=labels_x_rh,
                                                 dist_type="min", nv=20)
    return distance_matrix_lh, distance_matrix_rh


def find_shortest_path_between_hemi(data_dir, subject):
    """
       1. calculates the center of mass (cms) for each parcel
       2. calculates the euclidian distance between each cms
       3. adds exponential punishment for each connection across the two
          hemishperes
       4. calculates the shortest path between each parcel from one hemisphere
          to each parcel from the second hemisphere
       5. returns nested dictionary of the shortest paths
    """
    subjects_dir = 'mne_data/MNE-sample-data/subjects'
    base_dir = os.path.join(subjects_dir, subject)

    # load the brain anatomy for both hemispheres
    surf_lh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                           'surf/lh.pial'))
    surf_rh = nib.freesurfer.read_geometry(os.path.join(base_dir,
                                           'surf/rh.pial'))

    # load parcels for both hemi
    labels_x = np.load(os.path.join(data_dir, subject + '_labels.npz'),
                       allow_pickle=True)
    labels_x = labels_x['arr_0']
    labels_x_lh = [s for s in labels_x if s.hemi == 'lh']
    labels_x_rh = [s for s in labels_x if s.hemi == 'rh']

    # calculate center of mass
    cms_lh = [parcel.center_of_mass(subject, subjects_dir = subjects_dir) for
              parcel in labels_x_lh]
    cms_rh = [parcel.center_of_mass(subject, subjects_dir = subjects_dir) for
              parcel in labels_x_rh]
    
    vertices_lh, triangles_lh = surf_lh
    vertices_rh, triangles_rh = surf_rh

    vertices = np.concatenate((vertices_lh, vertices_rh))

    # vertices = vertices_rh + np.max(vertices_lh)
    max_lh = np.max(triangles_lh)
    #triangles = np.concatenate((triangles_lh,
    #                           triangles_rh + max_lh + 1))
    #triangles = triangles.astype('<i4')

    cms_rh = list(cms_rh + max_lh)
    cms = np.array(cms_lh + cms_rh)
    cms = cms.astype('<i4')
    # import pdb; pdb.set_trace()
    #labels_x_lh

    # distance = gdist.compute_gdist(vertices, triangles,source_indices=np.array(labels_x_lh[0].vertices, ndmin=1),target_indices=np.array(labels_x_lh[0].vertices, ndmin=1))

    # calculate the shortest distance between all the parcels
    #dist_euclidian = np.empty((len(cms), len(cms)))
    dist_euclidian = {}
    for i in range(len(cms)):
        for j in range(len(cms)):
            x1, y1, z1 = vertices[cms[i]]
            x2, y2, z2 = vertices[cms[j]]
            e_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2- z1)**2)
            # dist_euclidian[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2
            # - z1)**2)
            if i < len(labels_x_lh):
                name_i = labels_x_lh[i].name
            else:
                name_i = labels_x_rh[i-len(labels_x_lh)].name

            if j < len(labels_x_lh):
                name_j = labels_x_lh[j].name
            else:
                name_j = labels_x_rh[j-len(labels_x_lh)].name
            if not name_i in dist_euclidian:
                dist_euclidian[name_i] = {}
            # punish distance accross the hemi
            if (i < len(labels_x_lh) and j >= len(labels_x_lh)) or (i >= len(labels_x_lh) and j < len(labels_x_lh)):
                e_dist = np.exp(e_dist)
            dist_euclidian[name_i][name_j] = e_dist

    # calculate the shortest path for all the parcels from different hemis
    shortestpath(dist_euclidian,'3-lh','14-rh')
    import pdb; pdb.set_trace()
    # distance = gdist.compute_gdist(vertices, triangles,source_indices=np.array(cms[20], ndmin=1),target_indices=cms)


def calc_dist_matrix_labels(surf, source_nodes, dist_type='min', nv=0):
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
    dist_matrix = pd.DataFrame(columns=cn, index=cn)
    np.fill_diagonal(dist_matrix.values, 0)

    # TODO: parallel?
    # NOTE: very slow
    for i in range(len(source_nodes)-1):
        prev_source = source_nodes[i].vertices.astype('<i4')
        prev_name = source_nodes[i].name

        for j in range(i+1, len(source_nodes)):
            loading = ("i: " + str(i) + '/' + str(len(source_nodes)) + ':' +
                       "." * j + ' ' * (len(source_nodes)-j-1) + '|')
            print(loading, end="\r")

            # computes the distance between the targets and the source
            # (gives as many values as targets)
            next_source = source_nodes[j].vertices.astype('<i4')
            next_name = source_nodes[j].name
            distance = gdist.compute_gdist(vertices, new_triangles,
                                           source_indices=np.array(
                                               prev_source, ndmin=1)[::nv],
                                           target_indices=np.array(
                                               next_source, ndmin=1)[::nv])
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

# Dijkstra's algorithm for shortest paths
# adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228
def shortestpath(graph,start,end,visited=[],distances={},predecessors={}):
    """Find the shortest path between start and end nodes in a graph"""
    # we've found our end node, now find the path to it, and return
    if start==end:
        path=[]
        while end != None:
            path.append(end)
            end=predecessors.get(end,None)
        return distances[start], path[::-1]
    # detect if it's the first time through, set current distance to zero
    if not visited: distances[start]=0
    # process neighbors as per algorithm, keep track of predecessors
    for neighbor in graph[start]:
        if neighbor not in visited:
            neighbordist = distances.get(neighbor,sys.maxsize)
            tentativedist = distances[start] + graph[start][neighbor]
            if tentativedist < neighbordist:
                distances[neighbor] = tentativedist
                predecessors[neighbor]=start
    # neighbors processed, now mark the current node as visited
    visited.append(start)
    # finds the closest unvisited node to the start
    unvisiteds = dict((k, distances.get(k,sys.maxsize)) for k in graph if k not
                      in visited)
    closestnode = min(unvisiteds, key=unvisiteds.get)
    # now we can take the closest node and recurse, making it current
    return shortestpath(graph,closestnode,end,visited,distances,predecessors)
