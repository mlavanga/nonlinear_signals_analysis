# Nonlinear signals analysis
A short introduction in nonlinear dynamics (NLD) and nonlinear signal analysis (NSA) to determine whether a biomedical signal is generated by strange attractor and if it is characterized by fractality or self-similarity. By the end of this tutorial, you should master the following concepts:

1. Fractal dimension and strange attractors
2. Chaotic attractors and Lyapunov exponents
3. Correlation dimension and Sample Entropy
4. Hurst Exponent
5. Dynamical Systems and maps vs time series analysis

## Acknowledgements

This introduction would not be possible without the NOLDs, FATHON and MNE-python libraries. 

Refer to the original pages of the two packages if you find the code and the concepts in this repo useful for your future work. 

The NOLDs package is a rather comprehensive toolbox to make estimate nonlinear dynamics properties, while the FATHON is a specific toolbox to estimate the Hurst exponent via the detrend fluctuation analysis. The MNE is more generic toolbox for EEG and MEG analysis in Python. 

1. https://github.com/CSchoel/nolds
2. https://github.com/stfbnc/fathon
3. https://mne.tools/stable/index.html

Whenever you use these packages, please cite the library webpage and the appropriate references

1. https://zenodo.org/record/3814723#.YVnDq2Yza3I
2. https://joss.theoj.org/papers/10.21105/joss.01828

## 1. Introduction to Nonlinear dynamics: the Henon Map

This tutorial introduces all the key concepts on NLD and NSA, such as the difference between strange and chaotic attractors, fractal or geometrical dimension of the attractor basin, correlation dimension, sample entropy and the Hurst exponent. The concept of dynamical system/map is also introduced and how the Takens's theorem can extend NLD to any signal or observable. 

## 2. Another strange attractor: the Logistic Map 

By considering the Logistic Map as strange attractor, this tutorial gives a very similar introduction, by also taking a glimpse to Poincare Plots.

## 3. The linear attractor: Validation of the NLD features

To validate the concepts shown in 1. and 2., this tutorial shows what happens when you remove the nonlinearities in a strange attractor such as the Henon map. 

## 4. The fractality of the fractional gaussian noise and Brownian motion

This tutorial just gives a more broader introduction to the Hurst exponent and the relationship with the spectrum beyond strange attractors. 

## 5. Sleep stage classification with MNE

The last tutorials took inspiration from the Sleep stages classification of the MNE python libraries (https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html) to perform the same task with NSA.

## Contact Information

If you have any questions, suggestions or corrections, you can find my contact information by browsing my personal website 

https://mlavanga.github.io/

This introduction has specifically been designed for biomedical signal processing and it targets master students or any student with a background in data processing and system theory background. 

## Installation

## 1. Git-repo

1. Clone this repo in your laptop

   1. Open the terminal and run the command 

   2. ```
      git clone https://github.com/mlavanga/nonlinear_signals_analysis.git
      ```

2. You can always download this folder from the website

## 2A. Linux and MacOS

1. Open the terminal and verify to have python3 by running

```
python3
```

The version should be < 3.9 if possible

2. Change directory to the nonlinear_signal_analysis folder. Run the following commands. 

```
rm -rf env
python3 -mvenv env
. env/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install jupyterlab
pip3 install matplotlib
pip3 install nolds
pip3 install mne
pip3 install fathon
```

3. Test the environment. Open a new tab on the terminal and run the following commands

```
. env/bin/activate
jupyter-lab
```

4. Copy the paste the locahost link in your favourite browser or use the deafult opening session
5. Test the tutorials (.ipynb file) by pressing Shift + Enter with your keyboard
6. Enjoy

## 2B. Windows

1. Create an Anaconda environment

2. Download an anaconda from the website

   1. https://www.anaconda.com/products/individual-d

3. Download the zip folder from the github website and unzip it. We suggest to place it in the documents folder

   1. https://github.com/mlavanga/nonlinear_signals_analysis.git

4. Open the Anaconda prompt

5. Move to the folder of the project by running the command (modify accordingly)

6. ```
   cd C:\Users\<Username>\Documents\nonlinear_signal_analysis
   ```

7. Run the following commands in the terminal

8. ```
   conda create -n env python=3.6
   conda activate env
   pip3 install --upgrade pip
   pip3 install wheel
   pip3 install jupyterlab
   pip3 install matplotlib
   pip3 install nolds
   pip3 install mne
   pip3 install fathon
   ```

9. Run the following command:

10. ```
    conda deactivate env
    conda activate env
    jupyter-lab
    ```

11. Copy the paste the locahost link in your favourite browser or use the deafult opening session

12. Test the tutorials (.ipynb file) by pressing Shift + Enter with your keyboard

13. Enjoy

## 2C. WINDOWS (if you are kind of GEEKY)

1. Windows Linux Subsystem (WSL): You can repeat the step above in the new WSL terminal if installed in windows --> See https://docs.microsoft.com/en-us/windows/wsl/install-win10#simplified-installation-for-windows-insiders
2. Enjoy again

