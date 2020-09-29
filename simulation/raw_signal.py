import mne
import numpy as np


def generate_signal(raw_fname, fwd_fname, subject, parcels, n_events=30,
                    signal_type='eeg'):
    signal_len = 0.1  # in sec
    # Generate the signal
    info = mne.io.read_info(raw_fname)
    if signal_type == 'eeg':
        sel = mne.pick_types(info, meg=False, eeg=True, stim=False, exclude=[])
    elif signal_type == 'meg':
        sel = mne.pick_types(info, meg=True, eeg=False, stim=False, exclude=[])
    elif signal_type == 'mag' or signal_type == 'grad':
        sel = mne.pick_types(info, meg=signal_type,
                             eeg=False, stim=False, exclude=[])
    sel_data = mne.pick_types(info, meg=signal_type, eeg=False, stim=False,
                              exclude=[])
    info_data = mne.pick_info(info, sel_data)
    info = mne.pick_info(info, sel)
    tstep = 1. / info['sfreq']

    # To simulate sources, we also need a source space. It can be obtained from
    # the forward solution of the sample subject.
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    fwd = mne.pick_channels_forward(fwd, include=info_data['ch_names'],
                                    ordered=True)
    # Define the time course of the activity for each source of the region to
    # activate. Here we use just a step of ones, the amplitude will be added at
    # later stage
    source_time_series = np.ones(int(signal_len/tstep))

    # Define when the activity occurs using events. The first column is the
    # sample of the event, the second is not used, and the third is the event
    # id. Here the events occur every 200 samples.
    events = np.zeros((n_events, 3), dtype=int)
    # Events sample
    events[:, 0] = 100 + 200 * np.arange(n_events)
    events[:, 2] = 1  # All events have the sample id.

    # Create simulated source activity. Here we use a SourceSimulator whose
    # add_data method is key. It specified where (label), what
    # (source_time_series), and when (events) an event type will occur.
    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)

    np.random.seed = 42
    min_amplitude = 10  # nAm
    max_amplitude = 100  # nAm
    for idx, parcel in enumerate(parcels):
        # select the amplitude of the signal between 10 and 100 nAm
        amplitude = (np.random.rand() *
                     (max_amplitude-min_amplitude) +
                     min_amplitude) * 1e-9
        source_simulator.add_data(
            parcel,
            source_time_series * amplitude,
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
