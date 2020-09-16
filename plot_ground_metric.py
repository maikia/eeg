import numpy as np

import mne

from mayavi import mlab
from surfer import Brain

import config


grade = 3
annot = "aparc_sub"
subject = "fsaverage"
subjects_dir = config.get_subjects_dir_subj("sample")
ground_metric = np.load("data/ground_metric.npy")

hemi = "lh"
label_index = 0
labels = mne.read_labels_from_annot(subject, annot, hemi,
                                    subjects_dir=subjects_dir)
labels = [label.morph(subject_to=subject, subject_from=subject,
                      grade=grade, subjects_dir=subjects_dir)
          for label in labels]
distances = np.zeros(642)
label_vertices = labels[label_index].vertices
for ii, label_i in enumerate(labels):
    v_i = label_i.vertices
    distances[v_i] = ground_metric[0][ii]

f = mlab.figure(size=(700, 600))
brain = Brain(subject, hemi, "white", subjects_dir=subjects_dir, figure=f,
              background="white", foreground='black')
vertices = np.arange(642)
brain.add_data(distances, vertices=vertices,
               hemi="lh", transparent=True, smoothing_steps=2,
               colormap="RdBu", alpha=1, mid=12)
