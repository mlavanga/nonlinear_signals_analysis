# nonlinear_signals_analysis
A short introduction how to recognise a nonlinear signals and extract the associated summary statistics

## Installation

```
python3 -mvenv env
. env/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install jupyterlab
pip3 install matplotlib
pip3 install nolds
pip3 install mne
pip3 install fathon
pip3 install -e .

```



## 1. The strange attractor: the Henon Map

Explain the duality of strange and chaotic attractor and how the fractal dimension play a role. Compute features for the Henon Map

## 2. The strange attractor in the continous time: the Lorenz Attractor

Compute features for the Lorenz Attractor. Explain the role of the spectrum.

## 3. The strange attractor in the continous time: the Logistic Map 

Compute features for the Logistic Map. The concept of Poincare Plot.

## 4. The linear attractor: how to obtain an integer dimension

Compute features for a linear attractor and obtain a phase space with a linear dimension.

## 5. The white noise, the fractional gaussian noise and Brownian motion

Compute Fractal dimension for a gaussian noise. Compute Hurst and beta for white noise, monofractal and Brownian motion. Explain link between Brownian motion and stange attractor

## 6. Sleep stage classification with MNE

Compite spectral and NLD on EEG by MNE for sleep-stages classification
