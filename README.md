## Implementation code for "Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction"

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
