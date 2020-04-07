import numpy as np
import mne


def generate_signal(raw_fname, fwd_fname, subject, parcels, n_events=30,
                    signal_type='eeg'):
    signal_len = 10
    # Generate the signal
    info = mne.io.read_info(raw_fname)
    if signal_type == 'eeg':
        sel = mne.pick_types(info, meg=False, eeg=True, stim=True, exclude=[])
    elif signal_type == 'meg':
        sel = mne.pick_types(info, meg=True, eeg=False, stim=True, exclude=[])
    elif signal_type == 'mag' or signal_type == 'grad':
        sel = mne.pick_types(info, meg=signal_type,
                             eeg=False, stim=True, exclude=[])
    info = mne.pick_info(info, sel)
    tstep = 1. / info['sfreq']

    # To simulate sources, we also need a source space. It can be obtained from
    # the forward solution of the sample subject.
    # fwd_fname = op.join(data_path, 'MEG', subject,
    #                     subject + '_audvis-meg-eeg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    # Define the time course of the activity for each source of the region to
    # activate. Here we use a sine wave at 18 Hz with a peak amplitude
    # of 10 nAm.
    source_time_series = np.sin(2. * np.pi * 18. *
                                np.arange(signal_len) * tstep
                                ) * 50e-9

    # Define when the activity occurs using events. The first column is the
    # sample of the event, the second is not used, and the third is the event
    # id. Here the events occur every 200 samples.
    events = np.zeros((n_events, 3), dtype=int)
    # Events sample
    events[:, 0] = signal_len * len(parcels) + 200 * np.arange(n_events)
    events[:, 2] = 1  # All events have the sample id.

    # Create simulated source activity. Here we use a SourceSimulator whose
    # add_data method is key. It specified where (label), what
    # (source_time_series), and when (events) an event type will occur.
    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    for idx, parcel in enumerate(parcels):
        # each signal will be shifted by 2 data point in each next parcel
        source_simulator.add_data(
            parcel,
            source_time_series[2 * idx:signal_len + 5 * idx],
            events
        )

    # Project the source time series to sensor space and add some noise.
    # The source simulator can be given directly to the simulate_raw function.
    raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
    if signal_type == 'eeg':
        raw.set_eeg_reference(projection=True)
    cov = mne.make_ad_hoc_cov(raw.info)
    mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.02])
    return events, source_time_series, raw
