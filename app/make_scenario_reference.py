"""Makes app/data/scenario_reference.csv, a small table that the scenario page needs when the user changes the data month.

Run it once from the ML Engine folder:   python app/make_scenario_reference.py

Why it exists: the deployed app does not have data/train.csv (it holds flood labels and is not published).
When the user picks another month on the scenario page, the model needs for that county and month:
  - the average soil moisture (the stand-in the live forecast uses),
  - the "normal" rainfall (to work out rain as a share of normal),
  - the lowest and highest rainfall seen in the training years (the slider limits).
This file only holds rainfall and soil numbers. It has NO flood records.
"""
import os
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(os.path.dirname(here), 'data', 'train.csv'), dtype={'pcode': str})

# the "normal" rainfall of a county and month is rain_mm / rain_pct_normal (it is the same number in every year)
train['rain_normal_mm'] = train['rain_mm'] / train['rain_pct_normal']

grouped = train.groupby(['pcode', 'month'])
reference = pd.DataFrame({
    'soil_moisture_l1': grouped['soil_moisture_l1'].mean(),
    'rain_normal_mm': grouped['rain_normal_mm'].mean(),
    'rain_min': grouped['rain_mm'].min(),
    'rain_max': grouped['rain_mm'].max(),
}).reset_index()

# the range of the neighbour rain is the same for every county (the model has one range for it)
reference['neighbour_rain_min'] = train['neighbour_rain'].min()
reference['neighbour_rain_max'] = train['neighbour_rain'].max()

out_file = os.path.join(here, 'data', 'scenario_reference.csv')
reference.to_csv(out_file, index=False)
print('wrote', out_file, reference.shape)
