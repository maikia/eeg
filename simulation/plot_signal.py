from surfer import Brain
import mne


def plot_evoked(raw, events):
    # Plot evoked data to get another view of the simulated raw data.
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
    evoked = epochs.average()
    evoked.plot()


def visualize_brain(subject, hemi, annot_name, subjects_dir, parcels_selected):
    # visualize the brain with the parcellations and the source of the signal
    brain = Brain(subject, hemi, 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white', size=(800, 600))

    for parcel in parcels_selected:
        brain.add_label(parcel, alpha=1, color='r')
    # brain.add_label(parcels_rh[0], alpha=0.5, color='b')
    # brain.add_label(corpus_callosum, alpha=0.5, color='b')
    # brain.add_label(parcels_rh[0], alpha=0.5, color='b')
    # 0 if lh, 1 if rh
    # l = mne.vertex_to_mni(l1_center_of_mass.vertices, 0,subject,subjects_dir)
    if hemi == 'lh' or hemi == 'both':
        brain.add_annotation(annot_name, borders=True, color='r',
                             alpha=0.2)
    if hemi == 'rh' or hemi == 'both':
        brain.add_annotation(annot_name, borders=True, color='r',
                             alpha=0.2)
    # for center in cm_lh:
    #    brain.add_foci(center, coords_as_verts=True, map_surface="white",
    #                   color="gold", hemi=hemi_selected)

    file_save_brain = 'fig/brain.png'
    brain.save_image(file_save_brain)
