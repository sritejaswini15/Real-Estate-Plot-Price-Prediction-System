# Legacy: the first version of the pipeline

This folder keeps the project's first Colab notebook scripts, byte-for-byte unchanged, for reference. The `plotify` package replaces them. The scraper is still in use and has moved to [`scraper/`](../scraper).

| File | What it did | Replaced by |
|---|---|---|
| `EDA.py` | Read the data from a Google Sheet, cleaned prices, plotted histograms and correlations, and saved `cleaned_data.csv` | `plotify/cleaning.py`, `python -m plotify eda` |
| `Model_Evaluation.py` | Linear regression on one-hot encoded columns, one random 80/20 split | `python -m plotify train` |
| `Price_Prediction.py` | Saved the model and asked for plot details with `input()` | `python -m plotify predict`, the web app and the API |
| `cleaned_data.csv` | Output of `EDA.py` (60 rows) | `data/processed/plots.csv` (94 rows) |
| `README.original.md` | The first README | `README.md` |
| `reproduce_original.py` | Re-runs these scripts on the raw data, to compare with the new pipeline | — |
