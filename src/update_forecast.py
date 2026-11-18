"""Live forecast for the flood early-warning model.

What it does:
  1. (--download) downloads the newest WFP rainfall and NDVI files and the NOAA ONI and DMI files into data/live/
  2. rebuilds the features of the model from those files, with the same steps as the training pipeline
  3. predicts the risk of a flood in the month AFTER the newest month that has all the data
  4. applies the Watch and Warning thresholds of models/xgboost_settings.json
  5. writes app/forecast/latest_forecast.csv for the dashboard, and a copy in app/forecast/archive/<forecast month>.csv

If a source has not published the newest month yet, the forecast is made for the newest month that ALL the sources have.
The data month and the forecast month are written in every row of the csv.

Stand-ins (the model needs these 4 inputs, but there is no live data for them yet):
  flood_this_month = 0 (flood records are not known yet)
  soil_moisture_l1 = the training mean of the county and calendar month (src/soil_climatology.csv)
  runoff_rank and discharge_rank = 0.5 (the middle rank)
Every row of the output says which stand-ins were used.

The forecast only needs files of the repository (models/, src/*.csv) and the downloads, not data/train.csv.

Run it from the ML Engine folder:
  python src/update_forecast.py --download     (download, then forecast)
  python src/update_forecast.py                (forecast from the files already in data/live)
  python src/update_forecast.py --parity       (check the feature steps against the training table)
  python src/update_forecast.py --make-reference   (rebuild src/county_static.csv and src/soil_climatology.csv from data/train.csv)
"""
import argparse
import io
import json
import os
import re
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(ROOT, 'data', 'live')
OUT_FOLDER = os.path.join(ROOT, 'app', 'forecast')
ARCHIVE_FOLDER = os.path.join(OUT_FOLDER, 'archive')
ADJACENCY = os.path.join(ROOT, 'src', 'county_adjacency.csv')
# the per-county numbers that the model needs and that are the same every month (made by --make-reference)
STATIC_FILE = os.path.join(ROOT, 'src', 'county_static.csv')
SOIL_FILE = os.path.join(ROOT, 'src', 'soil_climatology.csv')
STATIC_COLUMNS = ['pct_flat', 'area_km2', 'lc_tree', 'lc_grass', 'lc_crop', 'lc_bare', 'pct_within_1km_river', 'pct_water',
                  'pct_occasional_water', 'pop_density', 'county_flood_rate']
# the range of the training years: rain of each county, and ONI and DMI (the same for all counties). The scenario page of the app uses them for its sliders
RANGE_COLUMNS = ['rain_min', 'rain_max', 'oni_min', 'oni_max', 'dmi_min', 'dmi_max']
# the raw files of the training pipeline (only read, and only by --parity)
TRAINING_RAW = os.path.join(os.path.dirname(ROOT), 'El Nino Predictor', 'data', 'raw')

BASE_END = 2019  # "normal" = the mean of the training years
BASE_START = 2000
# the features that come from live data (the other features are static or stand-ins)
LIVE_FEATURES = ['rain_mm', 'rain_lag1', 'rain_lag2', 'rain_pct_normal', 'neighbour_rain', 'ndvi_pct_normal', 'oni', 'dmi']
HEADERS = {'User-Agent': 'flood-early-warning-student-project'}


# ---------- downloads ----------

def fetch(url, params=None):
    # one web request, tried 3 times (the servers are sometimes slow or busy)
    for attempt in (1, 2, 3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=300)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            print(f'  attempt {attempt} failed for {url}: {error}')
            if attempt == 3:
                raise
            time.sleep(5 * attempt)


def hdx_file_url(package, file_name):
    # the download link of a file of a HDX dataset
    info = fetch('https://data.humdata.org/api/3/action/package_show', params={'id': package}).json()
    for resource in info['result']['resources']:
        if resource['name'] == file_name:
            return resource['url']
    raise RuntimeError(f'{file_name} not found in the HDX dataset {package}')


def download(get_url, file_name):
    # saves one file in data/live (written as .part first, so a broken download does not replace a good file).
    # get_url is a function, so that a failure while finding the link is handled in the same way as a failed download
    os.makedirs(LIVE, exist_ok=True)
    path = os.path.join(LIVE, file_name)
    try:
        url = get_url()
        content = fetch(url).content
    except Exception as error:
        # an older copy of the file is better than no forecast, but a missing file is an error
        if os.path.exists(path):
            print(f'WARNING: could not download {file_name} ({error}); using the older copy of the file')
            return
        raise
    with open(path + '.part', 'wb') as f:
        f.write(content)
    os.replace(path + '.part', path)
    print(f'downloaded {file_name}: {len(content) / 1e6:.1f} MB from {url}')


def download_all():
    download(lambda: hdx_file_url('ken-rainfall-subnational', 'ken-rainfall-subnat-full.csv'), 'ken-rainfall-subnat-full.csv')
    download(lambda: hdx_file_url('ken-ndvi-subnational', 'ken-ndvi-subnat-full.csv'), 'ken-ndvi-subnat-full.csv')
    # the PSL csv is the same file the model was trained with, but it is a few months behind: the CPC text file has the newest months
    download(lambda: 'https://psl.noaa.gov/data/correlation/oni.csv', 'oni_psl.csv')
    download(lambda: 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt', 'oni_cpc.txt')
    download(lambda: 'https://psl.noaa.gov/data/timeseries/month/data/dmi.ersst.long.data', 'dmi_psl.data')


# ---------- reading the raw files (same steps as the training pipeline) ----------

def read_dekads(path, value_col):
    # one row for each county and dekad: the n_pixels-weighted mean of the level 2 areas of the county
    # (level 1 rows are the old provinces, they are dropped)
    columns = ['date', 'adm_level', 'PCODE', 'n_pixels', value_col]
    header = list(pd.read_csv(path, nrows=0).columns)
    if 'version' in header:
        columns.append('version')
    d = pd.read_csv(path, dtype={'PCODE': str}, usecols=columns)
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['adm_level'] == 2].copy()
    d['pcode'] = d['PCODE'].str[:5]
    d['weighted'] = d[value_col] * d['n_pixels']
    g = d.groupby(['pcode', 'date']).agg(weighted=('weighted', 'sum'), n=('n_pixels', 'sum'))
    out = (g['weighted'] / g['n']).rename(value_col).reset_index()
    prelim = []
    if 'version' in header:
        # months where any dekad is not final (CHIRPS preliminary)
        p = d[d['version'] != 'final']
        prelim = sorted(set(zip(p['date'].dt.year, p['date'].dt.month)))
    return out, prelim


def dekads_to_months(d, value_col, how):
    # month = sum or mean of its 3 dekads. A month with fewer than 3 dekads is not finished and is left out
    d = d.assign(year=d['date'].dt.year, month=d['date'].dt.month)
    g = d.groupby(['pcode', 'year', 'month'])[value_col]
    out = g.agg(how).rename(value_col).to_frame()
    out['n'] = g.size()
    out = out[out['n'] == 3].drop(columns='n').reset_index()
    return out


def add_lag(df, col, k):
    # value of col k months earlier, inside each county (the months must follow each other)
    df = df.sort_values(['pcode', 'year', 'month']).reset_index(drop=True)
    t = df['year'] * 12 + df['month'] - 1
    assert (t.groupby(df['pcode']).diff().dropna() == 1).all(), 'months are not contiguous, the lags would be wrong'
    df[f'{col}_lag{k}'] = df.groupby('pcode')[col].shift(k)
    return df


def read_oni(psl_path, cpc_path):
    # ONI by month. The published value of month t-1 becomes the feature of month t (as in the training pipeline,
    # because the published value of month t would use the data of month t+1)
    raw = pd.read_csv(psl_path)
    raw.columns = ['date', 'oni']
    raw['date'] = pd.to_datetime(raw['date'])
    published = raw.set_index('date')['oni'].replace(-9999, np.nan)
    if cpc_path is not None:
        # fill the months that the PSL file does not have yet from the CPC file (3-month seasons, named by the middle month)
        middle = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
        cpc = pd.read_csv(cpc_path, sep=r'\s+')
        cpc['date'] = pd.to_datetime(dict(year=cpc['YR'], month=cpc['SEAS'].map(middle), day=1))
        published = published.combine_first(cpc.set_index('date')['ANOM'])
    last = max(published.index[-1], published.dropna().index[-1] + pd.DateOffset(months=1))
    published = published.reindex(pd.date_range(published.index[0], last, freq='MS'))
    out = pd.DataFrame({'oni': published.shift(1)})
    out['year'], out['month'] = out.index.year, out.index.month
    return out.reset_index(drop=True)


def read_dmi(path):
    # DMI by month. Two layouts: the PSL file (year and 12 monthly values) or the file of the training pipeline
    # (year, month, wtio, setio, dmi)
    rows = []
    for line in open(path, encoding='latin-1'):
        p = line.split()
        if len(p) == 13 and re.fullmatch(r'\d{4}', p[0]):
            rows += [(int(p[0]), k, float(v)) for k, v in enumerate(p[1:], start=1)]
        elif len(p) == 5 and re.fullmatch(r'\d{4}', p[0]) and re.fullmatch(r'\d{1,2}', p[1]):
            rows.append((int(p[0]), int(p[1]), float(p[4])))
    out = pd.DataFrame(rows, columns=['year', 'month', 'dmi'])
    out['dmi'] = out['dmi'].where(out['dmi'] > -90)
    return out


# ---------- the features ----------

def build_features(rain_path, ndvi_path, oni_path, oni_cpc_path, dmi_path):
    # rainfall: monthly total, lags, % of normal
    dekads, prelim = read_dekads(rain_path, 'rfh')
    rain = dekads_to_months(dekads, 'rfh', 'sum').rename(columns={'rfh': 'rain_mm'})
    rain = add_lag(rain, 'rain_mm', 1)
    rain = add_lag(rain, 'rain_mm', 2)
    rain = rain.rename(columns={'rain_mm_lag1': 'rain_lag1', 'rain_mm_lag2': 'rain_lag2'})
    base =rain[rain['year'].between(BASE_START, BASE_END)]
    normal = base.groupby(['pcode', 'month'])['rain_mm'].mean().rename('normal')
    rain = rain.join(normal, on=['pcode', 'month'])
    rain['rain_pct_normal'] = rain['rain_mm'] / rain['normal'].where(rain['normal'] > 0)
    rain = rain.rename(columns={'normal': 'rain_normal_mm'})

    # rainfall of the neighbour counties (mean of the counties that touch the county)
    wide = rain.pivot(index=['year', 'month'], columns='pcode', values='rain_mm').sort_index()
    adjacency = pd.read_csv(ADJACENCY)
    a = pd.crosstab(adjacency['pcode'], adjacency['neighbour_pcode']).reindex(index=wide.columns, columns=wide.columns, fill_value=0)
    a = a.div(a.sum(axis=1), axis=0)
    neighbour = pd.DataFrame(wide.to_numpy() @ a.to_numpy().T, index=wide.index, columns=wide.columns)
    neighbour = neighbour.stack().rename('neighbour_rain').reset_index()

    # vegetation: monthly mean, % of normal (normal = the training years)
    ndvi_dekads, _ = read_dekads(ndvi_path, 'vim')
    ndvi = dekads_to_months(ndvi_dekads, 'vim', 'mean').rename(columns={'vim': 'ndvi'})
    normal = ndvi[ndvi['year'] <= BASE_END].groupby(['pcode', 'month'])['ndvi'].mean().rename('normal')
    ndvi = ndvi.join(normal, on=['pcode', 'month'])
    ndvi['ndvi_pct_normal'] = ndvi['ndvi'] / ndvi['normal'].where(ndvi['normal'] > 0)

    # put everything together (a month stays only if its rainfall is complete)
    feat = rain.merge(neighbour, on=['pcode', 'year', 'month'], how='left')
    feat = feat.merge(ndvi[['pcode', 'year', 'month', 'ndvi_pct_normal']], on=['pcode', 'year', 'month'], how='left')
    feat = feat.merge(read_oni(oni_path, oni_cpc_path), on=['year', 'month'], how='left')
    feat = feat.merge(read_dmi(dmi_path), on=['year', 'month'], how='left')
    feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
    return feat, prelim


# ---------- parity test against the training table ----------

def compare(feat, table, label):
    # largest difference of each feature between the rebuilt features and the training table
    cols = LIVE_FEATURES + ['month_sin', 'month_cos']
    m = feat.merge(table[['pcode', 'year', 'month'] + cols], on=['pcode', 'year', 'month'], suffixes=('', '_table'))
    print(f'  {label}: {len(m)} county-months, {m["year"].min()}-{m["month"].min():02d} .. {m["year"].max()}-{m["month"].max():02d}')
    worst = 0.0
    for c in cols:
        both = m[c].notna() & m[c + '_table'].notna()
        # the table has no NDVI before 2002-07 and the files can differ in which months are blank
        only_one = int((m[c].notna() != m[c + '_table'].notna()).sum())
        diff = (m.loc[both, c] - m.loc[both, c + '_table']).abs()
        print(f'    {c:16s} rows {int(both.sum()):6d}   max diff {diff.max():.2e}   mean diff {diff.mean():.2e}   blank in only one: {only_one}')
        worst = max(worst, float(diff.max()))
    return worst


def run_parity():
    table = pd.read_csv(os.path.join(ROOT, 'data', 'model_table_full.csv'), dtype={'pcode': str})
    raw = {'rain': os.path.join(TRAINING_RAW, 'rainfall', 'ken-rainfall-subnat-full.csv'),
           'ndvi': os.path.join(TRAINING_RAW, 'ndvi', 'ken-ndvi-subnat-full.csv'),
           'oni': os.path.join(TRAINING_RAW, 'climate_indices', 'oni.csv'),
           'dmi': os.path.join(TRAINING_RAW, 'climate_indices', 'dmi_ersstv6.txt')}
    ok = True
    if all(os.path.exists(p) for p in raw.values()):
        print('PARITY 1: the rebuilt features against data/model_table_full.csv, using the raw files of the training pipeline')
        feat, _ = build_features(raw['rain'], raw['ndvi'], raw['oni'], None, raw['dmi'])
        worst = compare(feat[feat['year'] <= 2025], table, 'all months 2000-2025')
        worst_recent = compare(feat[feat['year'].between(2020, 2025)], table, 'months 2020-2025')
        ok = worst < 1e-6
        print('  RESULT:', 'PASS' if ok else 'FAIL', f'(largest difference {worst:.2e}, limit 1e-6)')
    else:
        print('PARITY 1 skipped: the raw files of the training pipeline are not on this computer')
        worst_recent = None

    live = [os.path.join(LIVE, f) for f in ['ken-rainfall-subnat-full.csv', 'ken-ndvi-subnat-full.csv', 'oni_psl.csv', 'dmi_psl.data']]
    if all(os.path.exists(p) for p in live):
        print('PARITY 2: the freshly downloaded files against the same table (how much did the sources change since training?)')
        feat, _ = build_features(live[0], live[1], live[2], os.path.join(LIVE, 'oni_cpc.txt'), live[3])
        compare(feat[feat['year'].between(2020, 2025)], table, 'months 2020-2025')
    return ok


# ---------- the forecast ----------

def make_reference():
    # the per-county numbers that do not change from month to month, taken from the training set:
    # the static features (and the flood rate of each county) and the mean soil moisture of each county and calendar month.
    # they are saved in src/ so that the forecast can run (also on GitHub) without data/train.csv, which is team-only
    train = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'), dtype={'pcode': str})
    static = train.groupby('pcode')[['county'] + STATIC_COLUMNS].first().reset_index()
    ranges = train.groupby('pcode')['rain_mm'].agg(rain_min='min', rain_max='max')
    static = static.join(ranges, on='pcode')
    for climate in ['oni', 'dmi']:
        static[f'{climate}_min'] = train[climate].min()
        static[f'{climate}_max'] = train[climate].max()
    soil = train.groupby(['pcode', 'month'])['soil_moisture_l1'].mean().rename('soil_moisture_l1').reset_index()
    assert len(static) == 47 and len(soil) == 47 * 12
    # the static features must be the same in every row of a county, or .first() would hide a problem
    assert (train.groupby('pcode')[STATIC_COLUMNS].nunique(dropna=False) == 1).all().all()
    static.to_csv(STATIC_FILE, index=False)
    soil.to_csv(SOIL_FILE, index=False)
    print(f'wrote {STATIC_FILE} ({len(static)} counties) and {SOIL_FILE} ({len(soil)} rows)')


def newest_month(rows, column=None, how_many=47):
    # the newest month (as text) that has a value for the column in at least how_many rows of the month, for the summary of the sources
    rows = rows.dropna(subset=[column]) if column is not None else rows
    counts = rows.groupby(['year', 'month']).size()
    counts = counts[counts >= how_many]
    if len(counts) == 0:
        return 'none'
    year, month = counts.index.max()
    return f'{year}-{month:02d}'


def save_forecast(out):
    # writes the latest forecast and the archive copy. Returns True if a file changed.
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    latest_path = os.path.join(OUT_FOLDER, 'latest_forecast.csv')
    archive_path = os.path.join(ARCHIVE_FOLDER, f"{out['forecast_year'].iloc[0]}-{out['forecast_month'].iloc[0]:02d}.csv")
    new_text = out.to_csv(index=False)
    changed = False

    if os.path.exists(latest_path):
        old = pd.read_csv(latest_path, dtype={'pcode': str})
        new = pd.read_csv(io.StringIO(new_text), dtype={'pcode': str})
        old_month = old['data_year'].iloc[0] * 12 + old['data_month'].iloc[0]
        new_month = new['data_year'].iloc[0] * 12 + new['data_month'].iloc[0]
        if new_month < old_month:
            # a source went back to an older month: keep the newer forecast that we have
            print('the new forecast is for an older month than latest_forecast.csv, so the files are not changed')
            return False
        # the same numbers as before: nothing to write (the date it was generated is not a change)
        if 'generated_on' in old.columns and old.drop(columns='generated_on').equals(new.drop(columns='generated_on')):
            print('the forecast is the same as latest_forecast.csv, no change')
            new_text = None
    if new_text is not None:
        with open(latest_path, 'w', newline='') as f:
            f.write(new_text)
        changed = True
    # the archive copy of this forecast month (written if it is missing or if the forecast changed)
    if changed or not os.path.exists(archive_path):
        archive_text = new_text if new_text is not None else open(latest_path).read()
        with open(archive_path, 'w', newline='') as f:
            f.write(archive_text)
        changed = True
    return changed


def make_forecast():
    rain_path, ndvi_path = os.path.join(LIVE, 'ken-rainfall-subnat-full.csv'), os.path.join(LIVE, 'ken-ndvi-subnat-full.csv')
    oni_cpc = os.path.join(LIVE, 'oni_cpc.txt')
    feat, prelim = build_features(rain_path, ndvi_path, os.path.join(LIVE, 'oni_psl.csv'), oni_cpc if os.path.exists(oni_cpc) else None,
                                  os.path.join(LIVE, 'dmi_psl.data'))

    # the newest month that each source has, and the newest month that has every live feature for all 47 counties
    sources = {'rainfall': newest_month(feat, 'rain_mm'), 'ndvi': newest_month(feat, 'ndvi_pct_normal'),
               'oni': newest_month(feat, 'oni', 1), 'dmi': newest_month(feat, 'dmi', 1)}
    complete = feat.dropna(subset=LIVE_FEATURES).groupby(['year', 'month']).size()
    complete = complete[complete == 47]
    if len(complete) == 0:
        raise SystemExit('no month has all the data for the 47 counties, so there is no forecast. Newest month of each source: ' + str(sources))
    data_year, data_month = complete.index.max()
    forecast_year, forecast_month = (data_year, data_month + 1) if data_month < 12 else (data_year + 1, 1)
    print('newest month of each source:', sources)
    print(f'newest month with all the data: {data_year}-{data_month:02d}  ->  forecast for {forecast_year}-{forecast_month:02d}')
    latest = feat[(feat['year'] == data_year) & (feat['month'] == data_month)].copy()

    # the static features and the soil moisture stand-in, from the reference files
    static = pd.read_csv(STATIC_FILE, dtype={'pcode': str}).set_index('pcode')
    soil = pd.read_csv(SOIL_FILE, dtype={'pcode': str}).set_index(['pcode', 'month'])['soil_moisture_l1']
    names = static['county']
    latest = latest.join(static[STATIC_COLUMNS + RANGE_COLUMNS], on='pcode').join(soil, on=['pcode', 'month'])
    latest['runoff_rank'] = 0.5
    latest['discharge_rank'] = 0.5
    latest['flood_this_month'] = 0

    # the model, and the risk with flood_this_month = 0 and = 1 (the dashboard uses the second one for a manual entry)
    model = joblib.load(os.path.join(ROOT, 'models', 'xgboost.joblib'))
    settings = json.load(open(os.path.join(ROOT, 'models', 'xgboost_settings.json')))
    watch, warning = settings['watch_threshold'], settings['warning_threshold']
    features = list(model.feature_names_in_)
    assert not latest[features].isna().any().any(), 'a feature is missing: ' + str(latest[features].isna().sum()[lambda s: s > 0].to_dict())
    latest['risk'] = model.predict_proba(latest[features])[:, 1]
    latest['risk_if_flood'] = model.predict_proba(latest.assign(flood_this_month=1)[features])[:, 1]
    latest['level'] = np.where(latest['risk'] >= warning, 'Warning', np.where(latest['risk'] >= watch, 'Watch', 'No alert'))

    # the rainfall of the last 3 months is preliminary if one of them is marked so in the file
    months = [(data_year, data_month)]
    for back in (1, 2):
        t = data_year * 12 + data_month - 1 - back
        months.append((t // 12, t % 12 + 1))
    rain_prelim = any(m in set(prelim) for m in months)

    out = pd.DataFrame({'pcode': latest['pcode'], 'county': latest['pcode'].map(names),
                        'data_year': data_year, 'data_month': data_month, 'forecast_year': forecast_year, 'forecast_month': forecast_month,
                        'risk': latest['risk'].round(4), 'risk_if_flood': latest['risk_if_flood'].round(4), 'level': latest['level'],
                        'watch_threshold': watch, 'warning_threshold': warning,
                        'rain_mm': latest['rain_mm'].round(4), 'rain_pct_normal': latest['rain_pct_normal'].round(6),
                        'ndvi_pct_normal': latest['ndvi_pct_normal'].round(6), 'oni': latest['oni'], 'dmi': latest['dmi'],
                        'standin_flood_records': True, 'standin_soil_moisture': True, 'standin_runoff_rank': True,
                        'standin_discharge_rank': True, 'rain_prelim': rain_prelim,
                        'newest_rainfall': sources['rainfall'], 'newest_ndvi': sources['ndvi'], 'newest_oni': sources['oni'], 'newest_dmi': sources['dmi'],
                        'generated_on': date.today().isoformat()})
    # every input of the model, so that the scenario page of the app can recalculate the risk, and the training ranges for its sliders
    inputs = ['rain_lag1', 'rain_lag2', 'neighbour_rain', 'month_sin', 'month_cos', 'soil_moisture_l1', 'runoff_rank', 'discharge_rank',
              'flood_this_month', 'rain_normal_mm'] + STATIC_COLUMNS + RANGE_COLUMNS
    out = out.join(latest[inputs])
    out = out.sort_values('risk', ascending=False).reset_index(drop=True)
    changed = save_forecast(out)

    print(f'{"wrote" if changed else "kept"} app/forecast/latest_forecast.csv and the archive copy: {len(out)} counties')
    print('alerts:', out['level'].value_counts().to_dict(), '| highest risk:', out.iloc[0]['county'], out.iloc[0]['risk'])
    print(f'oni {out.iloc[0]["oni"]}, dmi {out.iloc[0]["dmi"]}, rainfall preliminary: {rain_prelim}')
    print('stand-ins: flood_this_month = 0, soil_moisture_l1 = training mean of the month, runoff_rank = discharge_rank = 0.5')

    # a short summary for the page of the GitHub Actions run
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_file:
        with open(summary_file, 'a') as f:
            f.write(f'### Flood forecast for {forecast_year}-{forecast_month:02d} (from {data_year}-{data_month:02d} data)\n\n')
            f.write(f'- changed: **{changed}**\n- alerts: {out["level"].value_counts().to_dict()}\n')
            f.write(f'- newest month of each source: {sources}\n- rainfall preliminary: {rain_prelim}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='live forecast for the flood early-warning model')
    parser.add_argument('--download', action='store_true', help='download the newest rainfall, NDVI, ONI and DMI files first')
    parser.add_argument('--parity', action='store_true', help='only check the feature steps against data/model_table_full.csv')
    parser.add_argument('--make-reference', action='store_true', help='only rebuild src/county_static.csv and src/soil_climatology.csv from data/train.csv')
    args = parser.parse_args()
    if args.make_reference:
        make_reference()
        raise SystemExit(0)
    if args.download:
        download_all()
    if args.parity:
        raise SystemExit(0 if run_parity() else 1)
    make_forecast()
