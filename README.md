# Kenya County Flood Risk Model

Moringa School Data Science capstone (DSF-PT15, Group 6): the machine learning engine for the Kenya County Flood Risk Early Warning project.

The model predicts whether a Kenyan county will flood **next month**, using the conditions observed this month: rainfall, soil moisture, vegetation, ocean climate signals, the terrain of the county and whether the county flooded this month. There is one row for each county and month (47 counties). The target is `flood_next_month` (1 = the county floods next month, 0 = it does not). Floods are rare (5% to 9% of the rows), so accuracy is not used. The models are judged with PR-AUC, ROC-AUC, and recall, precision and F1 for floods.

Four models were built and compared on the same data and the same cross-validation folds: logistic regression, decision tree, random forest and XGBoost. They are also compared with two simple rules that need no model (the persistence and calendar baselines).

**Final model: XGBoost, with a threshold of 0.343** (chosen for 80% recall in cross-validation). On the 2020 to 2024 test set it has a recall of 0.979 and a precision of 0.120, with 1,172 warnings in 2,820 months. The random forest is a close second. See "Results" and "Which model we go with".

The dashboard uses two alert levels from the same model: **Watch** (threshold 0.343, about 80% of floods caught in cross-validation, "be ready") and **Warning** (threshold 0.437, about 70% caught, a bigger share of real alerts, "act now"). See "Watch and Warning levels" and "Dashboard app".

## The story in plain English

1. **The question.** At the end of each month, can we say which Kenyan counties are likely to have a flood next month? A flood early warning is only useful if it comes *before* the flood.
2. **The model.** We gave a model (XGBoost) the conditions of each county in each month: rainfall, vegetation, soil, the ocean climate signals El Niño (ONI) and the Indian Ocean Dipole (DMI), the terrain, and the flood history, together with whether the county really flooded the month after. It learned the pattern from 2002 to 2019. We kept 2020 to 2024 aside and only used those years to test it.
3. **Two alert levels.** The model gives each county a risk score. A **Watch** ("be ready") is a risk above 0.34, which catches about 80% of the floods in our cross-validation. A **Warning** ("act now") is a risk above 0.44, which catches about 70% of them. The price of catching most floods is many false alarms.
4. **From a backtest to a real forecast.** In the backtest we know the flood records, and the strongest input of the model is "did this county already flood this month?". In real life nobody publishes flood records at the end of the month, so a live forecast has to say "we do not know" (we use 0). That makes the live forecast less accurate than the backtest, and we measured by how much (see "Live accuracy").
5. **Does it keep working without that input?** We trained a second model without it (`XGBoost_no_flood_input.ipynb`). It found more floods, but only by raising many more alerts, and it ranked the counties no better, so we kept the first model.
6. **A forecast every month.** A GitHub Action downloads the newest public data every month, makes the forecast and saves it, and the dashboard shows it. The dashboard also replays the live forecast on 2020 to 2024 ("Would it have warned us?"), and lets you change rainfall and El Niño to see how the model reacts ("What if?").

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
|-- .github/workflows/
|   `-- monthly_forecast.yml      the monthly update (see "Monthly update with GitHub Actions")
|-- .streamlit/config.toml        the theme, for Streamlit Community Cloud (a copy of app/.streamlit/config.toml)
|-- data/
|   |-- model_table.csv       input of notebook 01 (12,690 rows, 30 columns)
|   |-- train.csv             created by notebook 01 (9,870 rows)
|   |-- test.csv              created by notebook 01 (2,820 rows)
|   `-- live/                 downloads of src/update_forecast.py (git-ignored)
|-- models/                   created by the model notebooks
|   |-- <model>.joblib            the fitted model
|   |-- <model>_settings.json     its settings, threshold and cross-validation scores
|   `-- xgboost_no_flood.joblib   the model without flood_this_month (not used by the app, see "Which model we go with")
|-- notebooks/
|   |-- 01_preprocessing.ipynb
|   |-- LogisticRegression.ipynb
|   |-- DecisionTree.ipynb
|   |-- RandomForest.ipynb
|   |-- XGBoost.ipynb
|   |-- XGBoost_no_flood_input.ipynb
|   |-- 05_model_comparison.ipynb
|   `-- 06_app_predictions.ipynb
|-- src/                      the live forecast (see "Live forecast")
|   |-- update_forecast.py
|   |-- county_adjacency.csv      which counties touch (copy of the training reference file)
|   |-- county_static.csv         static features, flood rate and training ranges of the 47 counties
|   `-- soil_climatology.csv      mean soil moisture of each county and month (the stand-in)
|-- app/                      the Streamlit dashboard (see "Dashboard app")
|   |-- app.py
|   |-- requirements.txt          the packages of the app only (used by Streamlit Community Cloud)
|   |-- .streamlit/config.toml    the theme
|   |-- forecast/
|   |   |-- latest_forecast.csv   made by src/update_forecast.py
|   |   `-- archive/<forecast month>.csv   one file for every forecast month, for scoring later
|   `-- data/                     created by notebook 06
|       |-- app_predictions.csv
|       `-- kenya_counties_simple.geojson
|-- README.md
|-- requirements.txt
`-- .gitignore
```

`<model>` is `logistic_regression`, `decision_tree`, `random_forest` or `xgboost`. The training CSV files are team-only, and a few files may only go to a private repository (see "Data licence").

## Data and train/test split

`data/model_table.csv` is a reduced version of the full table (the full table had 14,664 rows and 70 columns). It was reduced on purpose to 30 columns (4 keys, the target, `flood_this_month` and 24 features) and to July 2002 to December 2024. Notebook 01 only **reads** it and never writes to it. It checks that the columns and months match, then splits the rows by time:

| Set | Period | Rows | Flood rows |
|---|---|---|---|
| Train | July 2002 to December 2019 | 9,870 | 882 (8.9%) |
| Test | January 2020 to December 2024 | 2,820 | 144 (5.1%) |

- **Why by time:** the model forecasts next month from this month, so it must be tested on months that come after the months it learned from.
- **Why July 2002:** the NDVI data starts in July 2002. **Why 2024:** the labels for 2025 are incomplete, so no 2025 data is used.
- **`county_flood_rate`:** the version in the full table had seen the training answers. Notebook 01 replaces it with the flood rate of each county from the training rows only, with the same value on train and test.
- **Scaling:** nothing is scaled or imputed in notebook 01. Scaling is done inside the pipeline of the logistic regression.

## The 25 features

The 24 features chosen in the EDA notebook of the capstone project, plus `flood_this_month`. The target (`flood_next_month`) and the keys (`pcode`, `county`, `year`, `month`) are never features.

| Group | Features |
|---|---|
| Current flood | `flood_this_month` (did the county flood this month?). It is known at the end of the month we predict from, so it is a lag of the target, like `rain_lag1`. It is also the persistence baseline. |
| Rainfall | `rain_mm`, `rain_lag1`, `rain_lag2`, `rain_pct_normal`, `neighbour_rain` (mean rain of the neighbouring counties) |
| Vegetation | `ndvi_pct_normal` |
| Ocean climate | `oni` (El Nino index) and `dmi` (Indian Ocean Dipole), the same for every county |
| Season | `month_sin` and `month_cos` (month of the year as a circle) |
| Soil and water | `soil_moisture_l1`, `runoff_rank`, `discharge_rank` |
| Terrain and land cover | `pct_flat`, `area_km2`, `lc_tree`, `lc_grass`, `lc_crop`, `lc_bare` |
| Rivers and surface water | `pct_within_1km_river`, `pct_water`, `pct_occasional_water` |
| People | `pop_density` |
| County history | `county_flood_rate` |

The terrain, land cover, surface water and population features do not change over time. Many floods last several months, so the models may partly "predict" floods that are already happening. This is why the test results also score **new floods** (the county was not flooding this month) and **continuing floods** (it was) separately.

## How the models are built

Every model notebook does the same two jobs, using the training set only. The test set is used once, at the end.

1. **Choose the settings by PR-AUC.** `RandomizedSearchCV` scored with PR-AUC (`average_precision`) on 5 folds that keep whole years together (`GroupKFold` by year). All the settings are searched together from plain lists, and the class weight is one of the settings. PR-AUC measures how well a model ranks the flood months above the others, and does not depend on a threshold.
2. **Choose the threshold for 80% recall.** The out-of-fold probabilities of the best model are used to find the highest threshold where recall is at least 0.80. Recall is still the main goal, because a missed flood is the most dangerous mistake, but it is set by the threshold and not by weakening the model. The highest-F1 threshold was also checked: the models then only matched the persistence baseline (recall 0.345 to 0.418 against 0.463), so 80% recall was kept.

On the test set a month gets a warning when `predict_proba >= threshold`.

Two simple rules are used as baselines:

- **Persistence:** a flood next month if there is a flood this month.
- **Calendar:** a warning for the calendar months whose flood rate in the training years is above the overall rate.

`05_model_comparison.ipynb` also trains a small **four-feature logistic regression** (`flood_this_month`, `county_flood_rate`, `month_sin`, `month_cos`) to test whether the rain, soil and river features add information.

## Results

**Final settings and thresholds** (from the notebooks, saved in `models/`):

| Model | Final settings | Threshold |
|---|---|---|
| Logistic regression | scaled, `solver='liblinear'`, `C=0.004`, l2 penalty, no class weight | 0.125 |
| Decision tree | `max_depth=8`, `min_samples_leaf=130`, `ccp_alpha=0.0001`, `class_weight='balanced'` | 0.297 |
| Random forest | 300 trees, `max_depth=4`, `min_samples_leaf=20`, `max_features=0.3`, no class weight | 0.064 |
| **XGBoost (final)** | `max_depth=3`, `n_estimators=400`, `learning_rate=0.01`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=50`, `reg_lambda=5`, `scale_pos_weight=10.19` | **0.343** |

**Cross-validation on the training years** (5 year folds):

| Model | PR-AUC mean (std) | Out-of-fold recall | Out-of-fold precision | Months flagged |
|---|---|---|---|---|
| Logistic regression | 0.439 (0.068) | 0.800 | 0.166 | 0.430 |
| Decision tree | 0.398 (0.076) | 0.805 | 0.173 | 0.416 |
| Random forest | 0.465 (0.073) | 0.800 | 0.171 | 0.419 |
| **XGBoost** | **0.476 (0.080)** | 0.800 | 0.171 | 0.418 |

**Test set** (2020 to 2024, 144 floods in 2,820 months, each model at its own threshold):

| Method | Recall | Precision | F1 | ROC-AUC | PR-AUC | Warnings | New floods found (of 76) |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.875 | 0.125 | 0.219 | 0.870 | 0.388 | 1,009 | 58 |
| Decision tree | 0.938 | 0.105 | 0.189 | 0.828 | 0.291 | 1,281 | 67 |
| Random forest | 0.979 | 0.115 | 0.206 | 0.892 | 0.405 | 1,223 | 73 |
| **XGBoost (final)** | **0.979** | **0.120** | **0.214** | **0.904** | **0.374** | **1,172** | **73** |
| Persistence baseline | 0.472 | 0.472 | 0.472 | 0.722 | 0.250 | 144 | 0 |
| Calendar baseline | 0.847 | 0.087 | 0.157 | 0.762 | 0.141 | 1,410 | 66 |
| Four-feature baseline | 0.910 | 0.109 | 0.195 | 0.866 | 0.371 | 1,202 | 63 |

All four models find all 68 continuing floods, because they warn in every month where the county is already flooding. The new floods are the real test.

## Watch and Warning levels

The XGBoost notebook chooses two thresholds from the out-of-fold probabilities of the training years (the test set is not used):

| Level | Rule | Out-of-fold recall | Out-of-fold precision | Months flagged | Test recall | Test precision | Test alerts |
|---|---|---|---|---|---|---|---|
| Watch ("be ready") | threshold 0.343, the highest with recall of at least 0.80 | 0.800 | 0.171 | 0.418 | 0.979 | 0.120 | 1,172 |
| Warning ("act now") | threshold 0.437, the highest with recall of at least 0.70 | 0.701 | 0.235 | 0.266 | 0.903 | 0.161 | 806 |

Every Warning county is also above the Watch line. Both thresholds are saved in `models/xgboost_settings.json` (`watch_threshold` and `warning_threshold`; `threshold` is the same as the watch threshold).

## Which model we go with

**XGBoost.** The rule was fixed before the test set was used: the final model has the highest mean cross-validation PR-AUC.

- **The random forest is a close second.** The gap in cross-validation PR-AUC is 0.012, which is within the year-to-year std (0.073 and 0.080). On the test set the random forest has the higher PR-AUC (0.405 against 0.374), they have the same recall and the same new floods found, and XGBoost has the higher ROC-AUC and fewer warnings. The test set does not separate them. The random forest would also be a reasonable choice.
- **Against persistence:** the models find far more floods (recall 0.979 against 0.472), mainly new floods that persistence cannot find, but with a much lower precision (0.120 against 0.472). Most warnings are false alarms.
- **Against the four-feature baseline:** XGBoost is better in cross-validation PR-AUC (0.476 against 0.342), test recall and new floods found, and almost equal in test PR-AUC (0.374 against 0.371). So the rain, soil and river features add some information, not a lot.
- **Test recall is higher than 80%.** All 5 test years have a recall of 0.949 or more, while 9 of the 18 training years are below 0.80 in cross-validation. XGBoost flags the same share of months on the test set (0.416) as in cross-validation (0.418). The tables cannot say why. The cross-validation recall of 0.80 is the more careful number to plan with.

### The model without `flood_this_month`

`notebooks/XGBoost_no_flood_input.ipynb` trains XGBoost again without `flood_this_month` and without five features that the first model never used (19 features; the 24-feature version gives the same results), with the same settings, split and thresholds method, and a cross-validation that drops the training rows whose next-month target falls in a validation year. Test results (2020 to 2024):

| Model | Watch recall | Warning recall | PR-AUC | Watch / Warning alerts |
|---|---|---|---|---|
| 1. current model, real `flood_this_month` | 0.979 | 0.903 | 0.374 | 1,172 / 806 |
| 2. current model, `flood_this_month` = 0 | 0.882 | 0.757 | 0.159 | 1,130 / 744 |
| 3. current model, = 0 and the ERA5/GloFAS stand-ins (the live inputs) | 0.882 | 0.778 | 0.165 | 1,155 / 777 |
| 4. new model, 19 features | 0.944 | 0.875 | 0.139 | 1,399 / 1,103 |
| calendar baseline (month of the year only) | 0.847 | n/a | 0.141 | 1,410 |

The new model finds more floods, but it raises 24% more Watch alerts and 48% more Warning alerts, and at the same number of alerts the recall is about the same (94.4% against 92.4% at the Watch alerts). Its PR-AUC is lower (0.139 against 0.159) and about the same as the calendar baseline. The rule written before the comparison was that the new model must be at least as good on the Watch recall, the Warning recall and the PR-AUC. It was not, so **the current model stays in the live forecast and the app**. The notebook also shows that shuffling the ONI slightly improves the PR-AUC on the test years, which is a warning about how far to trust the ONI dependence (see "Limits of the live forecast").

## Limits

- `GroupKFold` trains on years both before and after the held-out years, so cross-validation is not a pure forecast of the future.
- `county_flood_rate` is calculated from all the training years, so the cross-validation scores are slightly optimistic.
- No 2025 data is used: the data stops in December 2024.
- Feature selection: the 24 features were chosen in the EDA notebooks using only the training rows (July 2002 to December 2019), so the test years were not used to choose them.
- The test set has only 144 floods in 5 years (3 of them in 2022), so small differences between models are not reliable.
- The best `learning_rate` (0.01) and `min_child_weight` (50) of XGBoost are at the edge of the lists that were searched.
- The threshold is a decision for the people who use the warnings. A new threshold must be chosen with cross-validation, not with the test set.

## Notebook order

Run the notebooks from top to bottom, in this order.

1. `01_preprocessing.ipynb`: reads `../data/model_table.csv`, checks the columns and months, splits by time, fixes `county_flood_rate`, and writes `data/train.csv` and `data/test.csv`.
2. `LogisticRegression.ipynb`: scaling, search, threshold, test set, baselines. Saves the model to `models/`.
3. `DecisionTree.ipynb`: same steps, plus a look inside the tree.
4. `RandomForest.ipynb`: same steps, plus the feature importance.
5. `XGBoost.ipynb`: same steps, plus the feature importance.
6. `XGBoost_no_flood_input.ipynb`: XGBoost without `flood_this_month` (see "Which model we go with"). Saves `models/xgboost_no_flood.joblib`; the other model files are not touched.
7. `05_model_comparison.ipynb`: loads the four models, settings and thresholds from `models/`, chooses the final model by CV PR-AUC, and compares them on the test set, with the baselines and year by year.
8. `06_app_predictions.ipynb`: loads the final XGBoost model, predicts the risk of next month for every county from January 2020 to December 2024 (with the real flood records, and again with the inputs of the live forecast), and writes the two files of the dashboard to `app/data/`. It needs the county shapes file of the data pipeline (`../../El Nino Predictor/data/reference/kenya_counties_full.geojson`), which it only reads.

Paths inside the notebooks are relative to the `notebooks/` folder (for example `../data/train.csv`), so open the notebooks from that folder.

## How to run

1. Install the requirements: `pip install -r requirements.txt` (Python 3.11).
2. Put `model_table.csv` in `data/` (team-only file, see "Data licence").
3. Open Jupyter from the `notebooks/` folder and run the notebooks in the order above with "Run All".

To load a saved model later, use the same package versions as in `requirements.txt`:

```python
import joblib, json
model = joblib.load('models/xgboost.joblib')
with open('models/xgboost_settings.json') as f:
    settings = json.load(f)
warning = model.predict_proba(X)[:, 1] >= settings['threshold']
```

To run everything from the command line, from the `notebooks/` folder:

```
jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace LogisticRegression.ipynb
jupyter nbconvert --to notebook --execute --inplace DecisionTree.ipynb
jupyter nbconvert --to notebook --execute --inplace RandomForest.ipynb
jupyter nbconvert --to notebook --execute --inplace XGBoost.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_model_comparison.ipynb
```

## Dashboard app

`app/app.py` is a Streamlit dashboard with three pages (a switch at the top). A page whose file is missing is left out of the switch.

1. **Latest forecast** (made by `src/update_forecast.py`): the forecast for next month. A banner says that this is a live forecast and how much less accurate it is than the backtest, with a pill for every stand-in input (flood records assumed none, soil moisture climatology, runoff and discharge ranks neutral, rainfall preliminary). It shows a strip of four numbers (Warning, Watch, no alert, highest risk), a **top 10 list of the highest-risk counties**, the map shaded by risk with amber and red outlines for Watch and Warning, and a county card (risk, a bar with the two lines, the inputs of the forecast). Because almost every county can be on Warning in a strong El Niño, the page then says to read the ranking. An optional "manual override" lets you enter floods you know about; the risk of those counties is recalculated with a flood recorded, and they are marked as a manual entry, not data.
2. **Would it have warned us? (2020–2024)**: the live forecast replayed on the 60 months that the model was not trained on, with the same inputs the live forecast has (flood records unknown). A plain summary at the top: floods recorded, how many had a Watch or Warning the month before, missed floods, and the average number of alerts per month. A month box and a map where the **colour is the alert level**, a dot on the floods that were caught and a cross on the floods that were missed (a coloured county without a mark is a false alarm). The county card shows its risk over 12 months with the flood months marked, and a list shows the floods of the chosen month. This page uses the flood records, so it can only be in a private deployment (see "Data licence").
3. **What if? (scenario explorer)**: pick a county, and move sliders for the rainfall of the month, the rainfall of the month before, ONI and DMI. Every slider starts at the live value of the county and stops at the limits of the training years (the range is shown), because the model has not seen anything outside it. The "% of normal" rainfall is worked out again from the rainfall slider, and every other input stays fixed (the page says so). It shows the risk, the level, the change from the live forecast, and how the risk of the county changes with ONI. A second control moves ONI for all 47 counties and redraws the map. This is the response of the model, not a forecast. It loads `models/xgboost.joblib`.

The app does not train anything. It reads `app/forecast/latest_forecast.csv` (page 1 and 3, made by `src/update_forecast.py`), `app/data/app_predictions.csv` and `app/data/kenya_counties_simple.geojson` (page 2 and the map shapes, made by `notebooks/06_app_predictions.ipynb`), and the model for page 3.

**How to run** (after `pip install -r requirements.txt`), from the `ML Engine` folder:

```
cd app
streamlit run app.py
```

Streamlit reads the theme from the `.streamlit/config.toml` of the folder it is started from. There is a copy in `app/.streamlit/` (for `cd app`) and one in the main folder (for `streamlit run app/app.py`, and for Streamlit Community Cloud). Keep the two copies the same.

The backtest months are January 2020 to December 2024. 2025 is left out because its flood records are almost empty. The last month (Dec 2024 data → forecast Jan 2025) is marked because its flood records may not be complete. The risk of the app can differ slightly from the test predictions of `XGBoost.ipynb`: 2 of the 2,820 rows have a different level, because the csv file of the full table keeps fewer digits than the training files (see notebook 06).

## Live forecast

`src/update_forecast.py` makes a real forecast for the month after the newest month that has all the data, and writes `app/forecast/latest_forecast.csv` (one row per county: risk, level, every input of the model, the training ranges of the sliders, the data month and forecast month, a flag for each stand-in, and the newest month of each source) and an archive copy `app/forecast/archive/<forecast month>.csv`. The files have no flood records.

```
python src/update_forecast.py --download         # download the newest data, then forecast
python src/update_forecast.py                    # forecast from the files already in data/live
python src/update_forecast.py --parity           # check the feature steps against data/model_table_full.csv
python src/update_forecast.py --make-reference   # rebuild src/county_static.csv and src/soil_climatology.csv (needs data/train.csv)
```

- **Downloads** (public, into `data/live/`, which is git-ignored): WFP rainfall (CHIRPS) and NDVI from HDX, NOAA PSL ONI and DMI, and the NOAA CPC ONI file, which has the newest months. A failed download is tried 3 times; if it still fails, an older copy of the file is used with a warning, and with no older copy the script stops with an error.
- **Which month:** the forecast is made for the newest month that all the sources have (for example, if NOAA has not published the ONI of October yet, the forecast uses September data and forecasts October). The data month and forecast month are in every row of the csv, and `newest_rainfall`, `newest_ndvi`, `newest_oni` and `newest_dmi` show which source was the limit. If a new run would give an older month than the file that is already there, nothing is changed. If the numbers are the same as before, nothing is changed either (so the repository has no empty commits).
- **Features:** the same steps as the training pipeline (monthly rainfall, lags, % of normal against 2000–2019, neighbour rainfall, NDVI % of normal, ONI shifted one month, DMI, month sine and cosine). `--parity` rebuilds 2000–2025 from the raw files of the training pipeline (which must be next to this folder) and compares every feature with `data/model_table_full.csv`; the largest difference is 5e-8 (csv rounding).
- **Stand-ins:** the model needs four inputs with no live data. `flood_this_month` is 0 (flood records are not known yet), `soil_moisture_l1` is the training mean of the county and month (`src/soil_climatology.csv`), and `runoff_rank` and `discharge_rank` are 0.5. The app shows a pill for each.
- **What it needs:** `models/xgboost.joblib`, `models/xgboost_settings.json` and the three csv files in `src/`. It does not need `data/train.csv`. The static features, the flood rate of each county and the training ranges are in `src/county_static.csv`; run `--make-reference` again only if the model or the training data change.

## Live accuracy

The live forecast does not know the flood records, so it is less accurate than the backtest of `XGBoost.ipynb`. On the 2020 to 2024 test set:

| Inputs | Watch recall | Warning recall | Watch precision | Warning precision |
|---|---|---|---|---|
| Backtest: real flood records (not possible live) | 0.979 | 0.903 | 0.120 | 0.161 |
| **Live: records unknown, stand-ins for soil, runoff and river discharge** | **0.882** | **0.778** | 0.110 | 0.144 |

The dashboard page "Would it have warned us?" shows this replay: of 144 floods recorded in 60 months, 127 (88%) had a Watch or Warning the month before, 112 (78%) had a Warning, and 17 were missed. To do that the forecast raised 19 alerts a month on average (13 of them Warnings) among 47 counties, and 1,027 of the 1,154 alerts (89%) were followed by no recorded flood. Planning with the cross-validation recall (80% and 70%) is more careful than reading the test numbers.

## Limits of the live forecast

- **Flood records arrive late and are incomplete.** The newest months of any record are the least complete, so some alerts counted as false alarms may be floods that were never recorded. The forecasts cannot be scored until the records of the forecast month exist, which can take months. The monthly job keeps an archive for that.
- **It depends a lot on El Niño.** ONI, DMI and the month of the year carry 34% of the gain of the model (and the flood records of the data month, which are unknown live, carry another 43%), and the three of them are the same for every county. In a strong El Niño the risk is high almost everywhere: for November 2026, 46 of 47 counties are on Warning. The alert level then hardly separates the counties, so use the risk score and the top 10 list. The "What if?" page shows how the risk of a county changes with ONI; it is a step, not a smooth line.
- **ONI is near the top of what the model has seen.** The live ONI (2.46) is close to the largest value of the training years (2.75), and the training years have only 4 months (November 2015 to February 2016, one El Niño) with an ONI of 2.4 or more. The model is extrapolating from a single event, and in the model without `flood_this_month`, shuffling the ONI slightly improved the PR-AUC on the test years.
- **County level only.** The forecast is for a whole county. It cannot say where in the county, and the counties differ a lot in size.
- **Stand-ins.** Soil moisture, runoff and river discharge have no live data yet (an Earth Engine and a GloFAS pipeline would be needed). They carry 1.2% of the gain of the model, and replacing them with the stand-ins did not lower the test recall, so this is the smallest of the limits.
- **Flood records of the data month are not known.** This is the biggest cost: 10 to 12 points of recall (0.979 to 0.882 at Watch, 0.903 to 0.778 at Warning) (see "Live accuracy"). The manual override of the app is the only way to add them.
- **Preliminary rainfall.** The newest CHIRPS months are preliminary; the run of the 20th replaces them with the final values when WFP has published them. The new DMI file differs from the one of the training by up to 0.07, with no visible effect.
- **One model, one month ahead.** There is no forecast for two or three months ahead.

## Monthly update with GitHub Actions

`.github/workflows/monthly_forecast.yml` makes the forecast every month without anybody running anything.

- **When:** at 06:00 UTC on the 10th of every month, and again on the 20th (to replace preliminary rainfall with the final values). It can also be started by hand: **Actions** tab → "Monthly flood forecast" → **Run workflow**.
- **What it does:** it runs on Ubuntu with Python 3.11, installs `requirements.txt`, runs `python src/update_forecast.py --download` (downloads the public data, makes the forecast, writes `app/forecast/latest_forecast.csv` and `app/forecast/archive/<forecast month>.csv`), and commits **only those files**, and only if they changed.
- **If a source has not published the newest month yet:** it does not fail. It forecasts the newest month that all the sources allow and writes the data month and forecast month in the csv. The run of the 20th (or the next month) catches up. The archive keeps one file for every forecast month.
- **If a download fails** (a website is down), the script stops with an error, the run turns red, GitHub sends an email, and the old forecast stays in place. Run the workflow again later.
- **What it needs in the repository:** `models/xgboost.joblib` and `models/xgboost_settings.json`, the three csv files in `src/`, and `requirements.txt` (see "Data licence" for the files that are allowed in a private repository). In the repository settings, **Settings → Actions → General → Workflow permissions** must allow the workflow to write (the workflow asks for `contents: write`). If the `main` branch is protected, the bot needs to be allowed to push to it. Scheduled workflows only run on the default branch.
- **GitHub pauses scheduled workflows** in a repository with no activity for 60 days. The monthly commits count as activity, but if the job is ever off for a long time, enable it again in the Actions tab.
- **Test it first by hand:** push the repository, then run the workflow once from the Actions tab and look at the commit it makes.

The same steps were tested on this computer, in a clean copy with only the files of the repository (no `data/train.csv`, no `data/live`), with a local repository as the remote. The cases tested: a first run (it commits the forecast), a second run with nothing new (nothing is committed), a newest month that is not published (October rainfall and NDVI removed: it forecasts October from September data, then November when the data arrive, and the archive keeps both), a source that goes back to an older month (the newer forecast is kept), a source that is behind (DMI without October: the forecast uses September data and `newest_dmi` says so), and a failed download with and without an older copy. The Ubuntu run itself (the installation of the packages and the push to GitHub) could only be tested by running the workflow once on GitHub.

## Deployment on Streamlit Community Cloud (private)

The plan: the repository stays **private** and the app is **invite-only** (viewers are added by email). Nothing here has been deployed or pushed yet. Do these steps in order.

**Before the first push**

1. Make sure the GitHub repository is **private** (GitHub → the repository → Settings → General → Danger Zone → visibility).
2. Check what is tracked: `git ls-files data models app`. The list must **not** contain `data/train.csv`, `data/test.csv`, `data/model_table.csv`, `data/model_table_full.csv`, or any model other than `models/xgboost.joblib` and `models/xgboost_settings.json`. If a training csv is listed, it is already in the repository (the `.gitignore` does not remove files that were committed before); take it out with `git rm --cached` and remember that it stays in the old commits (see "Data licence").
3. Check what will be added: `git status`. The new files should be the ones in the tree above: the workflow, `src/`, `app/`, `.streamlit/`, the two model files, and `requirements.txt`.
4. Commit and push to `main`.

**Start the monthly update**

5. On GitHub: **Actions** tab → "Monthly flood forecast" → **Run workflow**. It should finish green and add a commit "Update flood forecast for ...". (If the Actions tab asks to enable workflows, enable them.)

**Deploy the app**

6. Go to <https://share.streamlit.io> and sign in with GitHub. When it asks, allow access to your **private** repositories.
7. Click **Create app** and choose to deploy from an existing repository. Set: repository = this repository, branch = `main`, main file path = `app/app.py`.
8. Open **Advanced settings** and choose **Python 3.11**. No secrets are needed.
9. Click **Deploy**. The first start installs `app/requirements.txt` (Streamlit Cloud uses the requirements file next to the main file), which takes a few minutes.
10. Make it invite-only: in the app's **Settings → Sharing**, set the app to private ("only specific people can view this app") and add the email addresses of the viewers. They sign in with that email. Check the current limits of the free plan for private apps on the Streamlit Community Cloud documentation.
11. Open the link in a private window, signed out, and check that the app asks for a login.

**After that**

- Every time the monthly job commits a new forecast, Streamlit Community Cloud redeploys the app from the new commit by itself, so the dashboard follows the repository.
- If the model or the training data change: rerun `python src/update_forecast.py --make-reference`, rerun notebook 06, and commit the changed files in `src/`, `models/` and `app/data/`.

**A public deployment** is not possible with this version. The page "Would it have warned us?" uses the flood records of the forecast month (`app/data/app_predictions.csv`), which include EM-DAT data, and the terms of EM-DAT do not allow redistributing it or databases derived from it. A public version would need that page rebuilt from the public flood labels only (DesInventar, the Global Flood Database and the Dartmouth Flood Observatory), and the safest way is to retrain the model on those labels as well, because the current model and the flood rate of each county were made with labels that include EM-DAT. The two other pages (the live forecast and the scenario explorer) show only the model output.

## Next steps

- **Multi-horizon forecasts:** a model for two and three months ahead, and for "any flood in the next three months" (one model for each horizon, trained directly on that target), compared with the calendar and persistence baselines.
- **Finer locations:** sub-county or river-basin forecasts, because a county is a large area.
- **Score the archived forecasts** as the flood records of each forecast month arrive, so that the live accuracy is measured on real forecasts and not on a replay.
- **A flood-report source that is available in real time**, so that `flood_this_month` does not have to be 0. This is the biggest gain available, 10 to 12 points of recall.
- **Live soil moisture, runoff and river discharge** (Earth Engine and GloFAS), to remove the other stand-ins.
- **A public version built from public labels** (see above).
- A new search for the settings of the model without `flood_this_month`, which was not done because the settings were kept on purpose.

## Requirements

Python 3.11 and the exact package versions in `requirements.txt` (the notebooks were run with Python 3.11.18). Jupyter is also needed to run the notebooks. Models saved with other versions of scikit-learn or XGBoost may not load. `app/requirements.txt` has only the packages that the app needs (it also loads the model for the scenario page), and is the file that Streamlit Community Cloud installs. The GitHub Action installs `requirements.txt`.

## Data licence

The flood labels include data from EM-DAT (CRED / UCLouvain). The [EM-DAT terms of use](https://doc.emdat.be/docs/legal/terms-of-use/) do not allow its data, or databases derived from it, to be redistributed. The full modelling table and every file made from it (`model_table.csv`, `train.csv`, `test.csv`) are therefore **team-only**: they are in `.gitignore`, and must not be committed or shared outside the team.

- Do not commit or publish these files, and do not share them outside the team. If they were committed earlier, they are still in the history of the repository.
- The notebooks print only shapes, counts and rates, never rows or events from the flood records.
- To rebuild the full table, follow the "EM-DAT data is not included" steps in the README of the capstone project.

**What may go to a private repository.** The monthly job and the invite-only app need a few files that are derived from the labels, so `.gitignore` lets exactly these through. **They may only be pushed to a private repository, and the app must stay invite-only:**

- `models/xgboost.joblib` and `models/xgboost_settings.json`: the model was trained on labels that include EM-DAT data.
- `app/data/app_predictions.csv`: it has the flood recorded in every county and month (`flood_recorded`).
- `src/county_static.csv`: it has `county_flood_rate`, the flood rate of each county in the training years (47 numbers).
- `app/data/kenya_counties_simple.geojson`, `src/county_adjacency.csv`, `src/soil_climatology.csv` and the forecast files in `app/forecast/` are not derived from the flood records.
