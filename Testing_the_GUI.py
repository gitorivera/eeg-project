# Testing the GU
#use mne, numpy
import mne


filepath='data/eeglab_data.set'

eeglab_raw = mne.io.read_raw_eeglab(filepath, preload=True)

eeglab_raw.plot()

