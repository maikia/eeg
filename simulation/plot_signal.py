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
        brain.add_label(parcel, alpha=1, color=parcel.color)

    brain.save_image('fig/brain.png')
