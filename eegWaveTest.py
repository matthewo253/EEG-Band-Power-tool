import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import signal
from mne.time_frequency import psd_array_multitaper
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch

eegDataColumns = ["AF3", "F7", "F3", "FC5"]

filename = input("Enter the name of the CSV file: ")

if os.path.exists(filename) and filename.endswith(".csv"):
    preExistData = pd.read_csv(filename)

    if(preExistData.empty):
        print("The csv file is empty")
        sys.exit(1)

    columns_found = True
    for i in range(len(eegDataColumns)):
        if eegDataColumns[i] not in preExistData.columns:
            columns_found = False
            break
    if(columns_found == False):
        print("CSV file is missing one or more of the Data columns")
        sys.exit(1)

    allEEGData = preExistData[eegDataColumns]

else:
    print("The file you have selected does not exist")
    sys.exit(1)
    

# Transposes the data 
eegData = allEEGData.to_numpy().T  
fs = 128

# computes the bandpower using one of the signal processing methods
def bandpower(  data: NDArray[np.float64], fs: float, method: str, band: tuple[float, float], relative: bool = True, **kwargs,)-> NDArray[np.float64]: 
    assert data.ndim == 2, (
        "The provided data must be a 2D array of shape (n_channels, n_samples)."
    )
    if method == "periodogram":
        freqs, psd = periodogram(data, fs, **kwargs)
    elif method == "welch":
        freqs, psd = welch(data, fs, **kwargs)
    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR", **kwargs)
    else:
        raise RuntimeError(f"The provided method '{method}' is not supported.")
    # compute the bandpower
    assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
    assert band[0] <= band[1], (
        "The 'band' argument must be defined as (low, high) (in Hz)."
    )
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bandpower = simpson(psd[:, idx_band], dx=freq_res)
    bandpower = bandpower / simpson(psd, dx=freq_res) if relative else bandpower
    return bandpower

# finds where to append the data in the csv for the def bandpower
methods = ("periodogram", "welch", "multitaper")

datapoints = {method: [] for method in methods}

for i in methods:
    bp = bandpower(eegData, fs, i, band=(0.5, 4))
    datapoints[i].append(bp)
    bp = bandpower(eegData, fs, i, band=(4, 8))
    datapoints[i].append(bp)
    bp = bandpower(eegData, fs, i, band=(8, 13))
    datapoints[i].append(bp)
    bp = bandpower(eegData, fs, i, band=(13, 30))
    datapoints[i].append(bp)
    bp = bandpower(eegData, fs, i, band=(30, 46))
    datapoints[i].append(bp)

print("Bandpower (Alpha Band, relative):")
for method, values in datapoints.items():
    print(f"{method}: {np.round(values[0], 4)}")


delta = [] # (0.5–4 Hz) sleep
theta = [] #(4–8 Hz) drowsiness
alpha = [] # (8–12 Hz) relaxed
beta = [] #(12–30 Hz) alert
gamma = [] #(30 - 46 Hz) perception

# finds the appropriate band power to put the data in the csv file

for method in methods:
    bp_vals = datapoints[method]
    delta.append(bp_vals[0])  
    theta.append(bp_vals[1])  
    alpha.append(bp_vals[2])  
    beta.append(bp_vals[3])   
    gamma.append(bp_vals[4])  

# flattens the arrays for the band power
delta_vals = np.hstack(delta)
theta_vals = np.hstack(theta)
alpha_vals = np.hstack(alpha)
beta_vals = np.hstack(beta)
gamma_vals = np.hstack(gamma)


# creates a string Array
classState = np.array(['delta', 'theta', 'alpha', 'beta', 'gamma'])

# creates new arrays to assign values to datapoints array and what bandpower it fits in
allBandPowerData = []
className = []

for i in datapoints:
    indexVal = 0
    for j in datapoints[i]:
        allBandPowerData.append(j)
        className.append(classState[indexVal % len(classState)])
        indexVal+=1

allBandPowerData = np.array(allBandPowerData)
className = np.array(className)

# trains the class state and datapoints
eegData_train, eegData_test, BandClass_train, BandClass_test = train_test_split(allBandPowerData, className, test_size=0.33, random_state=42)


eegDataReduce = PCA(n_components=2)
eegDataReduce.fit(eegData_train)
     
eegDataReduce_train = eegDataReduce.transform(eegData_train)

dataSegment = eegData[:, :len(eegData)]

# finds the frequency and the psd for each of the data columns
fs = 128  
signalAF3 = dataSegment[:, 0]  # AF3 channel
freqAF3, psdAF3 = signal.welch(signalAF3, fs, nperseg=256)

signalF7 = dataSegment[:, 1]  # F7 channel
freqF7, psdF7 = signal.welch(signalF7, fs, nperseg=256)

signalF3 = dataSegment[:, 2]  # F3 channel
freqF3, psdF3 = signal.welch(signalF3, fs, nperseg=256)

signalFC5 = dataSegment[:, 3]  # FC5 channel
freqFC5, psdFC5 = signal.welch(signalFC5, fs, nperseg=256)

# graph for the power of all the channels
plt.figure(figsize=(8, 4))
plt.semilogy(freqAF3, psdAF3)
plt.semilogy(freqF7, psdF7)
plt.semilogy(freqF3, psdF3)
plt.semilogy(freqFC5, psdFC5)
plt.title("Power Spectral Density - AF3")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (log scale)")
plt.grid(True)
plt.tight_layout()
plt.show()

# graphs all the alpha powers of each of the bandpowers
plt.figure(figsize=(8, 5))
plt.boxplot(
    [delta_vals, theta_vals, alpha_vals, beta_vals, gamma_vals],
    labels=["delta", "theta", "alpha", "beta", "gamma"]
)
plt.title("Alpha Power by Mental State")
plt.xlabel("Mental State")
plt.ylabel("Alpha Band Power")
plt.tight_layout()
plt.show()

colors = {
    "delta": "blue",
    "theta": "cyan",
    "alpha": "green",
    "beta": "orange",
    "gamma": "red"
}


# Projects the pca of each of the Band Powers
plt.figure(figsize=(8, 6))

for label in classState:
    if label in BandClass_train:
        x_vals = []
        y_vals = []
        for i in range(len(BandClass_train)):
            if BandClass_train[i] == label:
                x_vals.append(eegDataReduce_train[i, 0])
                y_vals.append(eegDataReduce_train[i, 1])
        plt.scatter(x_vals, y_vals, label=label, alpha=0.7, color=colors[label])


plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Shows the Mean band power fo each of the methods

bands = ["delta (0.5–4Hz)", "theta (4–8Hz)", "alpha (8–13Hz)", "beta (13–30Hz)", "gamma (30–46Hz)"]

heatmap_data = np.empty((len(methods), 5))  

for i in range(len(methods)):
    method = methods[i]
    for j in range(5):
        heatmap_data[i, j] = np.mean(datapoints[method][j])

plt.figure(figsize=(8, 4))
im = plt.imshow(heatmap_data, cmap="viridis", aspect="auto")

plt.xticks(np.arange(len(bands)), bands, rotation=30)
plt.yticks(np.arange(len(methods)), methods)

for i in range(len(methods)):
    for j in range(len(bands)):
        if heatmap_data[i, j] < heatmap_data.max() / 2:
            text_color = "white"
        else:
            text_color = "black"
        plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color=text_color)


plt.colorbar(im, label="Mean Power")
plt.title("Mean Band Power Across Methods")
plt.xlabel("Frequency Band")
plt.ylabel("Method")
plt.tight_layout()
plt.show()