# NPSR (NeurIPS 2023) (--- Under construction ---)
## Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

Official PyTorch implementation for **N**ominality Score Conditioned Time Series Anomaly Detection by **P**oint/**S**equential **R**econstruction (NPSR).

A major difficulty for time series anomaly detection arises from modeling time-dependent relationships to find **contextual anomalies** while maintaining detection accuracy for **point anomalies**. In this paper, we propose NPSR, an algorithm that utilizes point-based and sequence-based reconstruction models. The point-based model quantifies point anomalies, and the sequence-based model quantifies both point and contextual anomalies. We formulate the observed time point $\textbf{x}^0_t$ is a two-stage deviated value from a nominal time point $\textbf{x}^*_t$.

$$
  \textbf{x}^0_t = \textbf{x}^*_t + \Delta\textbf{x}^c_t + \Delta\textbf{x}^p_t
$$

Under this formulation, we link the reconstruction errors with the deviations (anomalies) and introduce a nominality score $N(\cdot)$. We derive an induced anomaly score $\hat{A}(\cdot)$ by further integrating $N(\cdot)$ and the original anomaly score $A(\cdot)$. $\hat{A}(\cdot)$ is **theoretically proven** to be superior over $A(\cdot)$ under certain conditions.

<p align="center">
  <img src="imgs/models_and_scheme.png" width="900px"/>
</p>
<p align="center">
  Figure 1. (a) Performer-based autoencoder $M_{pt}$, (b) Performer-based stacked encoder $M_{seq}$, and (c) main scheme for NPSR.
</p>

## Main Results
We evaluate the performance of NPSR against 14 baselines over 7 datasets using the best F1 score ($\mathrm{F}1^\*$).

**Note**: Due to reliability concerns, we didn't use the point-adjusted best F1 score ($\mathrm{F}1^\*_{\mathrm{PA}}$) as the main metric. (See Appendix D)

<p align="left">
  Table 1. Best F1 score ($\mathrm{F1^*}$) results on several datasets, with bold text denoting the highest and underlined text denoting the second highest value. The deep learning methods are sorted with older methods at the top and newer ones at the bottom.
</p>
<p align="center">
  <img src="imgs/main_table.png" width="600px"/>
</p>

## Setup

### Prerequisites

### Getting Started

## Datasets



------------------------------------------------------------------------
Installation (to install pytorch cf. https://pytorch.org/get-started/locally/):
conda create -n npsr python=3.11
conda activate npsr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

------------------------------------------------------------------------
[Training]
usage: python main.py config.txt

[Testing]
The algorithm will do an evaluation every epoch

------------------------------------------------------------------------
'config.txt' contains all the settings 
raw dataset files should be put under ./datasets/[NAME]

After training, it is possible to use 'parse_results.ipynb' to visualize the training results.

This code will be put on github afterwards.


customized datasets how to implement


# Steps for initial clean-up step for the raw data using 'make.py'
After putting the files/dir in the correct folder (folder name has to match dataset name), execute: "python make_pk.py".
A .pk file with the same file name as the dataset will appear in the same folder.
The .pk file can be loaded by the main program and further preprocessed.


## SWaT dataset
You can get the SWaT and WADI dataset by filling out the form at:
https://docs.google.com/forms/d/1GOLYXa7TX0KlayqugUOOPMvbcwSQiGNMOjHuNqKcieA/viewform?edit_requested=true

This work uses the data from 'SWaT.A1 & A2_Dec 2015'.
Required raw files: 'SWaT_Dataset_Attack_v0.csv', 'SWaT_Dataset_Normal_v1.csv' (please convert them from .xlsx files first)

## WADI dataset
You can get the SWaT and WADI dataset by filling out the form at:
https://docs.google.com/forms/d/1GOLYXa7TX0KlayqugUOOPMvbcwSQiGNMOjHuNqKcieA/viewform?edit_requested=true

This work uses the 2017 year data.
Required raw files: 'WADI_14days.csv', 'WADI_attackdata.csv'

## PSM dataset
Dataset downloadable at:
https://github.com/eBay/RANSynCoders/tree/main/data

Required raw files: 'train.csv', 'test.csv', 'test_label.csv'

## MSL dataset
You can get the public datasets (SMAP and MSL) using:
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

Required raw files/dir: 'train' (dir), 'test' (dir), both containing the .npy files for each entity, and 'labeled_anomalies.csv'
No need to delete the SMAP data

## SMAP dataset
You can get the public datasets (SMAP and MSL) using:
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

Required raw files/dir: 'train' (dir), 'test' (dir), both containing the .npy files for each entity, and 'labeled_anomalies.csv'
No need to delete the MSL data

## SMD dataset
Dataset downloadable at:
https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

Required raw file/dir: 'train' (dir), 'test' (dir), 'test_label' (dir), all containing .txt files for each entity.




