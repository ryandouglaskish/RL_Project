# RL_Project


YOUTUBE VIDEO: https://youtu.be/yo82idPFMbk

**Create virtual enviornment**
`$ python -m venv venv`
`$ source venv/bin/activate`
`$ pip install -r requirements.txt`

**Download data**
`$ mkdir -p data/raw/`
`kaggle datasets download -d prasoonkottarathil/btcinusd`

**Process Data**
`$ python process_data.py`
`$ python eda.py`

**Run Experiment**
Set experiment parameters in trainTorch.py.

Run
`$ python trainTorch.py`
