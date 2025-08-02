EEG Band Power Tool

This code analyzes EEG wave by using signal processing, it processes the signals from the frontal lobe of the brain. It uses data from the a csv file across multiple EEG waves
The results are visualized in plots using plots such as power spectral density, box plots, PCA projections, and heatmaps. 

Input:
Uses a .csv file
Required Columns: AF3, F7, F3, FC5

Features:
Signal processing methods: periodograms, welch, multitaper

Frequency Bands:
Delta: 0.5 - 4 Hz(sleep)
Theta: 4 - 8 Hz(drowsy)
Alpha: 8 - 13 Hz(relaxed)
Beta: 13 - 30 Hz(alert)
Gamma 30 - 46 Hz(perception)

Calculations done in the code:
Relative bandpower for each frequencies band and method
Principl component analysis for dimensional reduction
Mean power across frequency bands and signal processing method

Visualizations:
Power Spectral Density plot
  Displays psd for all 4 required columns using Welch's method
Box Plot of Bandpowers
  Shows the distribution of the bandpowers for the frequency bands
2d PCA projections
  Projects the band power values into a 2d visualization to show the seperation between bands
Heatmaps of mean signal processing
  Compares the average band power using the signal processing methods and the frequency bands

Dependencies used:
numpy, pandas, matplotlib, scikit-learn, scipy, mne

Notes:
Sampling frequency is hardcoded to 128 Hz
Code can be adapted to use more channels, change sampling rate, or add different band definitions.
