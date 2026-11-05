# Kenya County Flood Risk Model

Moringa School Data Science capstone (DSF-PT15, Group 6): the machine learning engine for the Kenya County Flood Risk Early Warning project.

The model predicts whether a Kenyan county will flood **next month**, using the conditions observed this month: rainfall, soil moisture, vegetation, ocean climate signals and the terrain of the county. There is one row for each county and month (47 counties). The target is `flood_next_month` (1 = the county floods next month, 0 = it does not). Floods are rare (5% to 9% of the rows), so the models are judged with ROC-AUC, PR-AUC, recall and precision for floods and not with accuracy.

The base model is logistic regression. It is compared with the persistence baseline, the simple rule "a county that flooded this month will flood next month". Random Forest and XGBoost will be added later as comparison models, on the same data and with the same measures.

## Data sources

The modelling table was built in the capstone project folder (`Moringa Capstone Project`) from these sources:

| Data | Source |
|---|---|
| Rainfall (CHIRPS, per county) | WFP via the Humanitarian Data Exchange |
| Vegetation (NDVI, per county) | WFP via the Humanitarian Data Exchange |
| Oceanic Nino Index (ONI) and Dipole Mode Index (DMI) | NOAA Physical Sciences Laboratory |
| Soil moisture and runoff | ERA5-Land monthly (Copernicus / ECMWF) |
| River discharge | GloFAS v4.0 (Copernicus / ECMWF) |
| Slope | SRTM 30 m (NASA / USGS) |
| Land cover | ESA WorldCover 2021 |
| Rivers | HydroSHEDS Free Flowing Rivers (WWF) |
| Surface water | JRC Global Surface Water |
| Population | KNBS 2019 Kenya Population and Housing Census |
| Flood labels | DesInventar, EM-DAT (CRED / UCLouvain), Global Flood Database, Dartmouth Flood Observatory |

A county-month counts as a flood month if at least one of the four flood sources reports a flood. `flood_next_month` is that label moved back by one month. All data belongs to its providers.

## Folder structure

```
ML Engine/
|-- data/
|   |-- model_table.csv       created by notebook 01 (12,690 rows, 30 columns)
|   |-- train.csv             created by notebook 01 (9,870 rows)
|   `-- test.csv              created by notebook 01 (2,820 rows)
|-- notebooks/
|   |-- 01_preprocessing.ipynb
|   `-- LogisticRegression_Base.ipynb
|-- README.md
`-- .gitignore
```

The three CSV files are not in the repository (see "Data licence"). They are created when notebook 01 is run.

## Train/test split

The split is by time and not random, which is different from a usual random split:

| Set | Period | Rows | Flood rows |
|---|---|---|---|
| Train | July 2002 to December 2019 | 9,870 | 882 (8.9%) |
| Test | January 2020 to December 2024 | 2,820 | 144 (5.1%) |

- **Why by time:** the model has to forecast next month from this month, so it must be tested on months that come after the months it learned from. A random split would put months from the same years in both sets, so the model would be tested on months right next to the ones it trained on and the results would look better than they would be in real use.
- **Why July 2002:** the NDVI data starts in July 2002, so January 2000 to June 2002 has missing values and is dropped.
- **Why 2024:** the labels for 2025 are incomplete, so all of 2025 is dropped.
- **No leakage from scaling:** nothing is scaled or imputed in notebook 01. Scaling is done inside the pipeline of each model notebook, so it is learned from the training rows only.
- **`county_flood_rate` is recomputed:** the version shipped with the full table was made from 2000 to 2019 labels, so it had already seen the training answers. Notebook 01 replaces it with the flood rate of each county from the training rows only (the mean of `flood_this_month` for each county in `train`), and puts the same value on train and test.

`data/model_table.csv` keeps the shipped `county_flood_rate`, so it must not be used for modelling. The models use `train.csv` and `test.csv`.

## The 24 features

Chosen in the EDA notebook of the capstone project. The target (`flood_next_month`) and the baseline (`flood_this_month`) are never features.

| Group | Features |
|---|---|
| Rainfall | `rain_mm` (rain this month), `rain_lag1` and `rain_lag2` (rain 1 and 2 months earlier), `rain_pct_normal` (rain compared with normal for that county and month, 1.0 = normal), `neighbour_rain` (mean rain of the neighbouring counties) |
| Vegetation | `ndvi_pct_normal` (NDVI compared with normal) |
| Ocean climate | `oni` (El Nino index) and `dmi` (Indian Ocean Dipole), both the same for every county |
| Season | `month_sin` and `month_cos` (month of the year as a circle) |
| Soil and water | `soil_moisture_l1` (top soil layer), `runoff_rank` and `discharge_rank` (percentile of this month's runoff and river discharge for that county and month) |
| Terrain and land cover | `pct_flat` (share of the county with slope under 2 degrees), `area_km2`, `lc_tree`, `lc_grass`, `lc_crop`, `lc_bare` (land cover shares) |
| Rivers and surface water | `pct_within_1km_river`, `pct_water` (permanent water), `pct_occasional_water` |
| People | `pop_density` (2019 census population per km2) |
| County history | `county_flood_rate` (flood rate of the county in the training rows) |

The terrain, land cover, surface water and population features do not change over time: each county has the same value in every month.

## The baseline to beat

The persistence baseline predicts a flood next month when the county flooded this month (`flood_this_month`, used only for comparison). Its ROC-AUC is 0.705 on the training set and 0.722 on the test set. A model is only useful if it beats this baseline.

Result of the base model on the test set (from `LogisticRegression_Base.ipynb`):

| Method | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Persistence baseline | 0.722 | 0.250 | 0.472 | 0.472 |




## Notebook order

Run the notebooks from top to bottom, in this order. Each notebook needs the files created by the one before it.

1. `01_preprocessing.ipynb` (Data preparation): keeps 30 columns and the rows from July 2002 to December 2024, splits by time, fixes `county_flood_rate`, and writes `data/model_table.csv`, `data/train.csv` and `data/test.csv`.
2. `LogisticRegression_Base.ipynb` (Modelling + Evaluation): the base model (`StandardScaler` and logistic regression with balanced class weights), evaluation on the test set and comparison with the persistence baseline. Reads `data/train.csv` and `data/test.csv`.

Random Forest and XGBoost notebooks will be added later.

Paths inside the notebooks are relative to the `notebooks/` folder (for example `../data/train.csv`), so open the notebooks from that folder.

## How to run

1. Install the requirements below.
2. Notebook 01 reads the full modelling table from `../../Moringa Capstone Project/data/private/model_table_full.csv`, so the capstone project folder must sit next to this repository, with that file in it. The file is team-only.
3. Open Jupyter from the `notebooks/` folder and choose any Python kernel that has the packages installed.
4. Run `01_preprocessing` and then `LogisticRegression_Base` with "Run All".

To run everything from the command line, from the `notebooks/` folder:

```
jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace LogisticRegression_Base.ipynb
```

## Requirements

- Python 3.11
- pandas, matplotlib, scikit-learn (the notebooks were run with pandas 3.1 and scikit-learn 1.9)
- jupyter (to run the notebooks)
- xgboost (for the later comparison model)

Install with `pip install pandas matplotlib scikit-learn jupyter xgboost`.

## Data licence

The flood labels include data from EM-DAT (CRED / UCLouvain). The [EM-DAT terms of use](https://doc.emdat.be/docs/legal/terms-of-use/) do not allow its data, or databases derived from it, to be redistributed. The full modelling table and every file made from it (`model_table.csv`, `train.csv`, `test.csv`) are therefore **team-only**.

- `.gitignore` contains `data/*.csv`, so no CSV file is ever committed. Do not force-add them.
- Do not share these files outside the team, and do not publish them.
- The notebooks print only shapes, counts and rates, never rows or events from the flood records.
- To rebuild the full table, follow the "EM-DAT data is not included" steps in the README of the capstone project.
