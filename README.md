# Chlorine Study

This repository contains research, data processing pipelines, and analysis tools for studying chlorine detection and measurement systems.

## What Is In This Repo

- `src/`  
  Core modules for data processing, model utilities, and analysis pipelines.

- `scripts/`  
  Utility scripts for data preparation, experimentation, and batch processing.

- `notebooks/`  
  Jupyter notebooks for exploratory analysis and visualization.

- `data/`  
  Datasets and data artifacts used in the study.

- `models/`  
  Model checkpoints, trained artifacts, and evaluation results.

- `chlorine-vision-study/`  
  Companion repository with the smartphone-based chlorine detection app and training pipeline.

## Getting Started

### Setup

1. Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Configure VS Code to use the venv interpreter by opening `.vscode/settings.json`.

### Running Scripts

Scripts are located in the `scripts/` directory:

```bash
python scripts/your_script.py
```

### Running Notebooks

Launch Jupyter:

```bash
jupyter notebook notebooks/
```

## Project Structure

See `STRUCTURE.md` for detailed folder and file organization.

## Documentation

- `STRUCTURE.md` provides a comprehensive breakdown of the repository layout.
- `IMPORT_FIXES_SUMMARY.md` documents import path corrections and module organization.

## Dataset Workflow Idea

For new datasets, start by adding a stable `sample_id` to each record so images can be tracked through the pipeline. A typical first step is:

```bash
python -m scripts.add_sample_id "dataset/path"
```

Then convert the folder of images into JSON, enrich or normalize the records as needed, and continue into downstream processing steps like feature extraction, augmentation, and evaluation.

## Notes

- All Python code should run within the activated virtual environment.
- Ensure dependencies in `requirements.txt` are kept up-to-date for reproducibility.
