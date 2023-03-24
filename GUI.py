import mne
from tkinter import filedialog, messagebox
import tkinter as tk

# Funciones
def load_data():
    file_path = filedialog.askopenfilename()
    if file_path:
        global eeglab_raw
        eeglab_raw = mne.io.read_raw_eeglab(file_path, preload=True)
        messagebox.showinfo("Cargar datos", "Datos cargados correctamente")

def filter_data():
    l_freq = float(low_cutoff_entry.get())
    h_freq = float(high_cutoff_entry.get())
    global eeglab_raw_filtered
    eeglab_raw_filtered = eeglab_raw.copy().filter(l_freq, h_freq, fir_design='firwin')
    messagebox.showinfo("Filtrar datos", "Datos filtrados correctamente")

def reject_artifacts():
    reject_criteria = dict(eeg=float(reject_threshold_entry.get()) * 1e-6)
    flat_criteria = dict(eeg=float(flat_threshold_entry.get()) * 1e-6)
    events = mne.find_events(eeglab_raw_filtered)
    event_id = {'event_name': 1}  # Reemplazar 'event_name' con el nombre de tu evento
    tmin, tmax = -0.2, 0.5  # Intervalo temporal de las épocas (desde -200 ms hasta 500 ms)

    global epochs
    epochs = mne.Epochs(eeglab_raw_filtered, events, event_id, tmin, tmax, reject=reject_criteria, flat=flat_criteria)
    epochs.drop_bad()
    messagebox.showinfo("Rechazar artefactos", "Artefactos rechazados correctamente")

# GUI
root = tk.Tk()
root.title("Procesamiento de datos EEG")

# Cargar datos
load_button = tk.Button(root, text="Cargar datos", command=load_data)
load_button.grid(row=0, column=0, padx=5, pady=5)

# Filtrar datos
filter_label = tk.Label(root, text="Filtrar datos")
filter_label.grid(row=1, column=0)

low_cutoff_label = tk.Label(root, text="Frecuencia de corte inferior (Hz):")
low_cutoff_label.grid(row=2, column=0)
low_cutoff_entry = tk.Entry(root)
low_cutoff_entry.grid(row=2, column=1)

high_cutoff_label = tk.Label(root, text="Frecuencia de corte superior (Hz):")
high_cutoff_label.grid(row=3, column=0)
high_cutoff_entry = tk.Entry(root)
high_cutoff_entry.grid(row=3, column=1)

filter_button = tk.Button(root, text="Aplicar filtro", command=filter_data)
filter_button.grid(row=4, column=0, padx=5, pady=5)

# Rechazar artefactos
reject_label = tk.Label(root, text="Rechazar artefactos")
reject_label.grid(row=5, column=0)

reject_threshold_label = tk.Label(root, text="Umbral de rechazo (µV):")
reject_threshold_label.grid(row=6, column=0)
reject_threshold_entry = tk.Entry(root)
reject_threshold_entry.grid(row=6, column=1)

flat_threshold_label = tk.Label(root, text="Umbral de canales planos (µV):")
flat_threshold_label.grid(row=7, column=0)
flat_threshold_entry = tk.Entry(root)
flat_threshold_entry.grid(row=7, column=1)

reject_button = tk.Button(root, text="Rechazar artefactos", command=reject_artifacts)
reject_button.grid(row=8, column=0, padx=5, pady=5)

root.mainloop()