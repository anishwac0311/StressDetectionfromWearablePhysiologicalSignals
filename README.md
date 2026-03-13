# 210 Project

This project uses the `WESAD` dataset for stress and affect detection with two main workflows:

- feature-based modelling from extracted window-level CSVs
- raw time-series modelling from the original subject `.pkl` files

The repository also includes several standalone EDA scripts that generate plots from either the merged feature table or the raw WESAD signals.

## Project Layout

- `WESAD/data_wrangling.py`: extracts per-subject windowed features from the raw WESAD `.pkl` files
- `WESAD/readme_parser.py`: parses subject metadata and merges it with extracted features
- `WESAD/feature_extraction.ipynb`: notebook version of the feature-prep flow
- `WESAD/end_to_end_modelling.ipynb`: tabular and raw-sequence modelling notebook
- `WESAD/improved_raw_models.ipynb`: improved deep learning models on raw sequences
- `eda_m14.py`: general EDA on the merged feature table
- `eda_m14_window_timeseries.py`: per-subject window trace plots and outlier export
- `eda_m14_correlation.py`: overall and per-label correlation analysis
- `eda_m14_comparative.py`: bivariate plots and feature ranking
- `plot_wesad_raw_timeseries.py`: raw wrist-signal plots directly from WESAD `.pkl` files

## Data Location

The code expects the dataset in these locations:

- raw WESAD subject folders: `WESAD/data/WESAD/S2`, `WESAD/data/WESAD/S3`, ...
- raw subject files: `WESAD/data/WESAD/S2/S2.pkl`, etc.
- generated feature files: `WESAD/data/subject_feats/*.csv`
- combined feature table: `WESAD/data/may14_feats4.csv`
- merged feature + metadata table: `WESAD/data/m14_merged.csv`

If you are starting from the original WESAD download, place the extracted subject folders inside `WESAD/data/WESAD/`.

## Environment Setup

From the project root:

```powershell
.\venv\Scripts\Activate.ps1
```

If you are creating your own environment, install the libraries used by the scripts and notebooks, including:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `jupyter`
- `torch`
- `xgboost` (optional in `end_to_end_modelling.ipynb`; the notebook skips XGBoost if not installed)
- `cvxEDA`

## Prepare The Dataset

Run the following steps in order.

### 1. Download and unzip WESAD

Download the dataset from either of these sources:

- [UCI WESAD page](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
- [Original WESAD dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)

The original dataset is distributed as a large zip archive containing subject folders such as `S2`, `S3`, ..., `S17`.

After downloading:

1. Extract the archive.
2. Locate the folder that contains the subject directories like `S2`, `S3`, `S4`, etc.
3. Place that extracted folder's contents inside `WESAD/data/WESAD/`.

The final structure should look like this:

```text
project root/
├── WESAD/
│   ├── data/
│   │   ├── WESAD/
│   │   │   ├── S2/
│   │   │   │   ├── S2.pkl
│   │   │   │   ├── S2_readme.txt
│   │   │   │   └── S2_quest.csv
│   │   │   ├── S3/
│   │   │   └── ...
```

Do not place the files one folder too deep. For example, this is wrong:

- `WESAD/data/WESAD/WESAD/S2/S2.pkl`

This is correct:

- `WESAD/data/WESAD/S2/S2.pkl`

### 2. Extract window-level features from raw WESAD files

Change into the `WESAD` folder and run:

```powershell
cd .\WESAD
python .\data_wrangling.py
```

This creates:

- per-subject feature files in `WESAD/data/subject_feats/`
- combined sensor-only feature table at `WESAD/data/may14_feats4.csv`

### 3. Parse subject metadata and build the merged dataset

`readme_parser.py` defines the parser class, but it does not run automatically by itself. Use:

```powershell
python -c "from readme_parser import rparser; rparser()"
```

This creates:

- `WESAD/data/WESAD/readmes.csv`
- `WESAD/data/m14_merged.csv`

`m14_merged.csv` is the main input for the feature-based EDA scripts.

### 4. Optional notebook-based feature prep

You can also open `WESAD/feature_extraction.ipynb` and run the cells. It effectively mirrors the same flow:

- run feature extraction
- instantiate `rparser()`
- load `data/m14_merged.csv`

## Run The Notebooks

The notebooks use paths like `Path('data')`, so it is safest to launch Jupyter from inside the `WESAD` folder:

```powershell
cd .\WESAD
jupyter notebook
```

Then run the notebooks below.

### `feature_extraction.ipynb`

Use this when you want to regenerate and inspect the processed feature dataset.

Expected output/input:

- generates or depends on `data/may14_feats4.csv`
- generates or depends on `data/m14_merged.csv`

### `end_to_end_modelling.ipynb`

This notebook contains two modelling tracks:

- tabular classification using `data/may14_feats4.csv` and `data/m14_merged.csv`
- raw time-series models using the original `.pkl` files in `data/WESAD/`

It runs:

- sklearn baselines such as Logistic Regression and Random Forest
- optional XGBoost baselines
- PyTorch MLP models
- GRU, LSTM, and ResNet-style raw sequence models

### `improved_raw_models.ipynb`

This notebook focuses on improved raw-sequence modelling with:

- 60-second overlapping windows
- per-subject normalization
- CNN-LSTM, CNN-GRU, attention, and hybrid models

It requires the raw subject `.pkl` files under `WESAD/data/WESAD/`.

## Run The EDA Scripts

Run the EDA scripts from the project root unless you intentionally want outputs written somewhere else.

```powershell
# only needed if you are currently inside .\WESAD
cd ..
```

If you are already at the project root, run the scripts directly.

### 1. General merged-feature EDA

```powershell
python .\eda_m14.py
```

Input:

- `WESAD/data/m14_merged.csv`

Output folder:

- `eda_outputs_m14/`

Generates:

- label counts
- windows-per-subject plots
- feature boxplots
- a correlation heatmap
- PCA plots by label and subject

### 2. Subject window trace EDA

```powershell
python .\eda_m14_window_timeseries.py
```

Input:

- `WESAD/data/m14_merged.csv`

Output folder:

- `eda_outputs_m14_timeseries/`

Notes:

- edit `SUBJECT_ID` inside the script to choose a different subject
- exports subject label timelines, feature traces, and top outlier windows

### 3. Correlation-focused EDA

```powershell
python .\eda_m14_correlation.py
```

Input:

- `WESAD/data/m14_merged.csv`

Output folder:

- `eda_outputs_m14_correlation/`

Generates:

- overall Spearman and Pearson heatmaps
- top correlated feature pairs
- per-label Spearman heatmaps

### 4. Comparative feature EDA

```powershell
python .\eda_m14_comparative.py
```

Input:

- `WESAD/data/m14_merged.csv`

Output folder:

- `eda_outputs_m14_comparative/`

Generates:

- bivariate scatter plots
- per-label correlation plots
- mutual information feature ranking
- Kruskal-Wallis effect-size ranking

### 5. Raw signal visualization from original WESAD files

```powershell
python .\plot_wesad_raw_timeseries.py
```

Input:

- raw `.pkl` files in `WESAD/data/WESAD/`

Output folder:

- `eda_outputs_wesad_raw/`

Notes:

- edit `SUBJECT_ID` to switch subjects
- edit `START_MIN` and `END_MIN` to zoom into a specific time range
- plots raw wrist `EDA`, `BVP`, `TEMP`, and accelerometer magnitude with label overlays

## Recommended Run Order

If you are starting from raw data:

1. Put the original WESAD subject folders into `WESAD/data/WESAD/`.
2. Run `WESAD/data_wrangling.py`.
3. Run `python -c "from readme_parser import rparser; rparser()"` inside `WESAD`.
4. Run the EDA scripts that depend on `WESAD/data/m14_merged.csv`.
5. Open `WESAD/end_to_end_modelling.ipynb` or `WESAD/improved_raw_models.ipynb`.

If the processed CSVs already exist, you can skip straight to the EDA scripts and notebooks.
