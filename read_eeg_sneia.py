import mne
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id=event_id, reject=reject)

# baseline noise cov, not a lot of samples
noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk', rank=None,
                                   verbose='error')

# butterfly mode shows the differences most clearly
raw.plot(events=events, butterfly=True)
raw.plot(noise_cov=noise_cov, events=events, butterfly=True)

mne.preprocessing.ICA(n_components=0.95, method='fastica').fit(epochs).plot_sources(epochs)
    # Path: plot_eeg_sneia.py

