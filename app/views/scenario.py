"""Page 3: Try a scenario. Change the conditions and see how the model reacts.

Two modes share one idea: take the LIVE inputs of a county, change some of them, and score the result with the real model (xgboost.joblib, the one
with the 'flood this month' input, the same as the live forecast).
  Simple   : a few big buttons, the answer in words only.
  Advanced : every input the model reads, the risk score and the ONI curve.
Before anything is changed, both modes give exactly the live forecast of the county.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from common import (eyebrow, badge, level_bar, get_forecast, get_reference, load_model, model_features, score_rows, level_of,
                    month_text, risk_line, LEVEL_INFO, MONTH_NAMES, WATCH, WARNING, HIGH)

forecast = get_forecast()
reference = get_reference()
if forecast is None or reference is None:
    st.error('The scenario page needs the forecast file and app/data/scenario_reference.csv (made by app/make_scenario_reference.py).')
    st.stop()
model = load_model()
features = model_features(model)

first = forecast.iloc[0]
live_month = int(first['data_month'])           # the data month of the live forecast (October for a November forecast)
live_oni = float(first['oni'])
counties = sorted(forecast['county'])
levels_order = ['No alert', 'Watch', 'Warning']  # Low, Medium, High


# =====================================================================
# helpers
# =====================================================================
def clip(value, low, high):
    return float(min(max(value, low), high))


def live_inputs(base):
    # the 25 live inputs of a county, exactly as the forecast file has them
    return {name: float(base[name]) for name in features}


def reference_row(pcode, month):
    # rain and soil numbers of a county in a calendar month (training years)
    return reference[(reference['pcode'] == pcode) & (reference['month'] == month)].iloc[0]


def score_one(inputs):
    # the risk score of one set of inputs
    return float(score_rows(model, pd.DataFrame([inputs]))[0])


def month_inputs(base, month):
    # the four numbers that follow from the month: month_sin, month_cos, soil moisture and the "normal" rain.
    # For the live month the forecast file's own numbers are used, so nothing changes by accident.
    if month == live_month:
        return {'month_sin': float(base['month_sin']), 'month_cos': float(base['month_cos']),
                'soil_moisture_l1': float(base['soil_moisture_l1']), 'rain_normal_mm': float(base['rain_normal_mm'])}
    ref = reference_row(base['pcode'], month)
    return {'month_sin': float(np.sin(2 * np.pi * month / 12)), 'month_cos': float(np.cos(2 * np.pi * month / 12)),
            'soil_moisture_l1': float(ref['soil_moisture_l1']), 'rain_normal_mm': float(ref['rain_normal_mm'])}


def risk_name(level):
    # 'high risk', 'medium risk' or 'low risk'
    return LEVEL_INFO[level]['name'].lower() + ' risk'


def compare_sentence(county, today_level, new_level):
    # one sentence that compares the scenario with today
    if today_level == new_level:
        return f'{county} stays {risk_name(new_level)}, the same as today.'
    went_down = levels_order.index(new_level) < levels_order.index(today_level)
    word = 'drop' if went_down else 'rise'
    return f'{county} would {word} from {LEVEL_INFO[today_level]["name"].lower()} to {LEVEL_INFO[new_level]["name"].lower()} risk.'


FLOOD_NOTE = ('<div class="callout warn"><b class="label">Flooding now</b>A flood this month is the strongest sign of another one next month, '
              'so the risk is high almost whatever else you change.</div>')


# =====================================================================
# title and mode switch
# =====================================================================
st.html(f"""
{eyebrow('TRY A SCENARIO · STARTING FROM ' + month_text(first['data_year'], first['data_month']).upper() + ' DATA')}
<h1 class="h1">Try a scenario</h1>
<p class="lead">Change the conditions and see how the model reacts. This shows how the forecast responds. It is not a prediction of what will happen.</p>
""")

mode = st.segmented_control('Mode', ['Simple', 'Advanced'], default='Simple', key='scenario_mode')
if mode is None:
    mode = 'Simple'

# the county is shared by both modes. The select box of each mode writes the choice here.
top_county = forecast.sort_values('risk', ascending=False).iloc[0]['county']
st.session_state.setdefault('scenario_county', top_county)


def remember_county(widget_key):
    st.session_state['scenario_county'] = st.session_state[widget_key]


def county_box(widget_key):
    st.selectbox('County', counties, index=counties.index(st.session_state['scenario_county']), key=widget_key,
                 on_change=remember_county, args=(widget_key,))
    return forecast[forecast['county'] == st.session_state[widget_key]].iloc[0]


# =====================================================================
# SIMPLE MODE
# =====================================================================
if mode == 'Simple':
    like_now = ('Strong El Niño (like now)' if live_oni > 1.0 else 'Weak El Niño (like now)' if live_oni > 0.5
                else 'La Niña (like now)' if live_oni < -0.5 else 'Normal (like now)')
    oni_choices = {'La Niña (cooler)': -1.0, 'Normal': 0.0, like_now: live_oni}
    # when the live value is "normal", the second and third button would be the same, so the second one is dropped
    if like_now == 'Normal (like now)':
        oni_choices = {'La Niña (cooler)': -1.0, like_now: live_oni}
    rain_choices = {'Less (half)': 0.5, 'Usual (today)': 1.0, 'Much more (double)': 2.0}

    controls, result = st.columns([1, 1])
    with controls:
        with st.container(key='card_simple_controls'):
            base = county_box('scn_county_simple')
            st.markdown('**Is this county flooding right now?**')
            flooding = st.segmented_control('Flooding now', ['No', 'Yes'], default='No', key='scn_flood', label_visibility='collapsed')
            st.markdown('**El Niño (Pacific Ocean)**')
            oni_pick = st.segmented_control('El Niño', list(oni_choices), default=like_now, key='scn_oni', label_visibility='collapsed')
            st.markdown('**Rain across the region this month**')
            rain_pick = st.segmented_control('Rain', list(rain_choices), default='Usual (today)', key='scn_rain', label_visibility='collapsed')
            st.html('<div class="note">Everything else stays at today\'s values.</div>')

    # a button group gives None when nothing is selected, so we fall back to today's choice
    flooding = flooding or 'No'
    oni_value = oni_choices.get(oni_pick, live_oni)
    factor = rain_choices.get(rain_pick, 1.0)

    # the inputs of the scenario: live values, with the three choices applied
    inputs = live_inputs(base)
    inputs['flood_this_month'] = 1.0 if flooding == 'Yes' else 0.0
    inputs['oni'] = oni_value
    if factor != 1.0:
        # the rain of the county and the rain of its neighbours are both multiplied, then kept inside the training range
        ref = reference_row(base['pcode'], live_month)
        rain = clip(inputs['rain_mm'] * factor, float(ref['rain_min']), float(ref['rain_max']))
        inputs['rain_mm'] = rain
        inputs['rain_pct_normal'] = rain / float(base['rain_normal_mm'])
        inputs['neighbour_rain'] = clip(inputs['neighbour_rain'] * factor, float(ref['neighbour_rain_min']), float(ref['neighbour_rain_max']))

    new_risk = score_one(inputs)
    new_level = level_of(new_risk)
    today_level = level_of(float(base['risk']))
    st.session_state['scenario_last_risk'] = new_risk   # not shown in Simple mode, kept so that the checks can read it

    with result:
        with st.container(key='card_scn_result'):
            st.html(f"""
<div class="mono">Result</div>
<div class="county-row"><span class="county-name">{base['county']}</span>{badge(new_level)}</div>
<div class="result-sentence">{compare_sentence(base['county'], today_level, new_level)}</div>
{level_bar(new_level)}
<div class="note">Today: {LEVEL_INFO[today_level]['name'].lower()} risk ({LEVEL_INFO[today_level]['action']}).</div>
""")
        if flooding == 'Yes':
            st.html(FLOOD_NOTE)

# =====================================================================
# ADVANCED MODE
# =====================================================================
else:
    follow_key = 'adv_follow'
    st.session_state.setdefault(follow_key, True)

    controls, result = st.columns([3, 2])
    with controls:
        # ---------- where and when ----------
        with st.container(key='card_adv_where'):
            st.html('<div class="h3">Where and when</div>')
            base = county_box('scn_county_adv')
            st.selectbox('Data month → forecast month', list(range(1, 13)), index=live_month - 1, key='adv_month',
                         format_func=lambda m: f'{MONTH_NAMES[m - 1]} → {MONTH_NAMES[m % 12]}')
            month = int(st.session_state['adv_month'])
            st.html('<div class="note">The model reads the month the data is from (the data month), and forecasts the next one. Changing it also changes the soil moisture (average of that month) and the normal rain, and the rain sliders start at the normal rain of that month.</div>')

        fixed = month_inputs(base, month)
        ref = reference_row(base['pcode'], month)
        scope = f"{base['pcode']}_{month}"   # the sliders of another county or month start again from their own live values

        # slider limits and starting values (the live values, moved inside the limits if they are outside)
        rain_low, rain_high = float(np.ceil(ref['rain_min'])), float(np.floor(ref['rain_max']))
        lag_low, lag_high = float(np.ceil(base['rain_min'])), float(np.floor(base['rain_max']))
        nb_low, nb_high = float(np.ceil(ref['neighbour_rain_min'])), float(np.floor(ref['neighbour_rain_max']))
        oni_low, oni_high = float(base['oni_min']), float(base['oni_max'])
        dmi_low, dmi_high = float(base['dmi_min']), float(base['dmi_max'])
        if month == live_month:
            # the live month: every slider starts at the live value
            start_rain, start_lag1, start_lag2, start_nb = base['rain_mm'], base['rain_lag1'], base['rain_lag2'], base['neighbour_rain']
        else:
            # another month: the live rain would be odd for that time of the year, so the rain sliders start at the NORMAL rain of that month
            # (this month and the two months before it). The neighbour rain is the live value scaled by the same ratio.
            normal_one_back = float(reference_row(base['pcode'], (month - 2) % 12 + 1)['rain_normal_mm'])
            normal_two_back = float(reference_row(base['pcode'], (month - 3) % 12 + 1)['rain_normal_mm'])
            start_rain, start_lag1, start_lag2 = fixed['rain_normal_mm'], normal_one_back, normal_two_back
            start_nb = base['neighbour_rain'] * fixed['rain_normal_mm'] / float(base['rain_normal_mm'])
        start = {
            'rain': clip(start_rain, rain_low, rain_high),
            'lag1': clip(start_lag1, lag_low, lag_high),
            'lag2': clip(start_lag2, lag_low, lag_high),
            'nb': clip(start_nb, nb_low, nb_high),
            'oni': clip(base['oni'], oni_low, oni_high),
            'dmi': clip(base['dmi'], dmi_low, dmi_high),
        }
        keys = {name: f'adv_{name}_{scope}' for name in start}
        flood_key = f'adv_flood_{scope}'
        for name, value in start.items():
            st.session_state.setdefault(keys[name], value)

        def sync_neighbours():
            # while "neighbours follow" is on, the neighbour rain moves by the same number of mm as the rain of this county
            if st.session_state[follow_key]:
                change = st.session_state[keys['rain']] - start['rain']
                st.session_state[keys['nb']] = clip(start['nb'] + change, nb_low, nb_high)

        def reset_all():
            for name in keys:
                st.session_state.pop(keys[name], None)
            st.session_state.pop(flood_key, None)
            st.session_state[follow_key] = True

        def caption(text, column, today):
            st.html(f'<div class="note">{text} <span class="mono" style="text-transform:none;letter-spacing:0.02em">{column}</span> · today: {today}</div>')

        # ---------- flooding ----------
        with st.container(key='card_adv_flood'):
            st.html('<div class="h3">Flooding</div>')
            flooding = st.segmented_control('Is this county flooding right now?', ['No', 'Yes'], default='No', key=flood_key)
            flooding = flooding or 'No'
            caption('A flood recorded in the data month.', 'flood_this_month', 'No (the live forecast assumes none)')

        # ---------- rain ----------
        with st.container(key='card_adv_rain'):
            st.html('<div class="h3">Rain</div>')
            st.slider('Rain this month (mm)', rain_low, rain_high, step=1.0, key=keys['rain'], on_change=sync_neighbours)
            caption(f'Total rain of the data month, training range {rain_low:.0f} to {rain_high:.0f} mm.', 'rain_mm', f"{base['rain_mm']:.0f} mm")
            rain_now = st.session_state[keys['rain']]
            share = rain_now / fixed['rain_normal_mm']
            st.html(f'<div class="note">Rain as a share of normal (worked out, no slider): <b class="ink">{share:.0%}</b> <span class="mono" style="text-transform:none;letter-spacing:0.02em">rain_pct_normal</span></div>')
            st.slider('Rain last month (mm)', lag_low, lag_high, step=1.0, key=keys['lag1'])
            caption('Rain one month before the data month.', 'rain_lag1', f"{base['rain_lag1']:.0f} mm")
            st.slider('Rain two months ago (mm)', lag_low, lag_high, step=1.0, key=keys['lag2'])
            caption('Rain two months before the data month.', 'rain_lag2', f"{base['rain_lag2']:.0f} mm")
            st.html('<div class="note"><i>In this model, past months\' rain mostly tells it the season, so more rain last month can lower the risk.</i></div>')
            st.toggle('Neighbouring counties get the same change in rain', key=follow_key, on_change=sync_neighbours)
            st.slider('Rain in neighbouring counties (mm)', nb_low, nb_high, step=1.0, key=keys['nb'], disabled=st.session_state[follow_key])
            caption('Average rain of the counties that touch this one.', 'neighbour_rain', f"{base['neighbour_rain']:.0f} mm")

        # ---------- ocean ----------
        with st.container(key='card_adv_ocean'):
            st.html('<div class="h3">Ocean</div>')
            st.slider('Pacific Ocean index (ONI)', oni_low, oni_high, step=0.01, key=keys['oni'])
            caption('El Niño (above 0) or La Niña (below 0). It is the published value of the month before the data month.', 'oni', f"{base['oni']:.2f}")
            st.slider('Indian Ocean index (DMI)', dmi_low, dmi_high, step=0.01, key=keys['dmi'])
            caption('Indian Ocean Dipole, data month.', 'dmi', f"{base['dmi']:.3f}")

        st.button('Reset to today\'s values', on_click=reset_all, key='adv_reset')

        # ---------- fixed inputs ----------
        score = pd.Series(model.get_booster().get_score(importance_type='total_gain'))
        no_effect = [name for name in features if name not in score.index]
        trait_names = {'pct_flat': 'Flat share of the county', 'area_km2': 'Area (km²)', 'lc_tree': 'Tree cover', 'lc_grass': 'Grassland',
                       'lc_crop': 'Cropland', 'lc_bare': 'Bare ground', 'pct_within_1km_river': 'Share within 1 km of a river',
                       'pct_water': 'Permanent water share', 'pct_occasional_water': 'Occasional water share', 'pop_density': 'Population density',
                       'county_flood_rate': 'Share of months with a flood in 2002–2019'}
        fixed_rows = [
            ('Vegetation (NDVI, % of normal)', 'ndvi_pct_normal', f"{base['ndvi_pct_normal']:.0%}", 'A user cannot know it, so it stays at today\'s value.'),
            ('Soil moisture', 'soil_moisture_l1', f"{fixed['soil_moisture_l1']:.3f}", 'Stand-in: the average of this county and month. It follows the month.'),
            ('Runoff rank', 'runoff_rank', f"{base['runoff_rank']:.2f}", 'Stand-in: the middle rank (no live data).'),
            ('River discharge rank', 'discharge_rank', f"{base['discharge_rank']:.2f}", 'Stand-in: the middle rank (no live data).'),
        ]
        for column, label in trait_names.items():
            if column not in no_effect:
                fixed_rows.append((label, column, f"{base[column]:.4g}", 'Fixed trait of the county.'))
        rows_html = ''.join(f'<div class="kv-row"><span class="k">{label}<br><span class="why">{why} <span class="mono" style="text-transform:none;letter-spacing:0.02em">{column}</span></span></span><span class="v">{value}</span></div>'
                            for label, column, value, why in fixed_rows)
        no_effect_text = ', '.join(trait_names.get(name, name) for name in no_effect)
        with st.container(key='card_adv_fixed'):
            st.html(f'<div class="h3">Fixed inputs (not sliders)</div>{rows_html}<div class="note" style="margin-top:10px">No effect on the result (the model never uses them): {no_effect_text}.</div>')

    # ---------- the scenario ----------
    inputs = live_inputs(base)
    inputs['flood_this_month'] = 1.0 if flooding == 'Yes' else 0.0
    inputs['rain_mm'] = st.session_state[keys['rain']]
    inputs['rain_pct_normal'] = st.session_state[keys['rain']] / fixed['rain_normal_mm']
    inputs['rain_lag1'] = st.session_state[keys['lag1']]
    inputs['rain_lag2'] = st.session_state[keys['lag2']]
    inputs['neighbour_rain'] = st.session_state[keys['nb']]
    inputs['oni'] = st.session_state[keys['oni']]
    inputs['dmi'] = st.session_state[keys['dmi']]
    inputs['month_sin'] = fixed['month_sin']
    inputs['month_cos'] = fixed['month_cos']
    inputs['soil_moisture_l1'] = fixed['soil_moisture_l1']
    new_risk = score_one(inputs)
    new_level = level_of(new_risk)
    st.session_state['scenario_last_risk'] = new_risk
    today_risk = float(base['risk'])
    today_level = level_of(today_risk)

    with result:
        with st.container(key='card_scn_result'):
            st.html(f"""
<div class="mono">Result</div>
<div class="county-row"><span class="county-name">{base['county']}</span>{badge(new_level)}</div>
<div class="result-sentence">{compare_sentence(base['county'], today_level, new_level)}</div>
{level_bar(new_level)}
<div class="kv-row"><span class="k">Risk score (relative, not a probability)</span><span class="v">{new_risk:.3f}</span></div>
<div class="kv-row"><span class="k">Today's risk score</span><span class="v">{today_risk:.3f} ({LEVEL_INFO[today_level]['name']})</span></div>
<div class="kv-row"><span class="k">Watch / Warning lines</span><span class="v">{WATCH:.4f} / {WARNING:.4f}</span></div>
""")
            # how the risk of this county changes with the ONI, the other inputs as set
            st.html('<div class="mono" style="margin-top:12px">Risk score of this county as ONI changes (other inputs as set)</div>')
            grid = np.linspace(oni_low, oni_high, 56)
            curve = pd.DataFrame([inputs] * len(grid))
            curve['oni'] = grid
            curve_risk = score_rows(model, curve)
            figure = risk_line(grid, curve_risk, '.1f', height=230)
            figure.update_traces(mode='lines', hovertemplate='ONI %{x:.2f}: %{y:.2f}<extra></extra>')
            figure.add_trace(go.Scatter(x=[inputs['oni']], y=[new_risk], mode='markers', hoverinfo='skip', marker=dict(size=12, color=HIGH, line=dict(color='white', width=2))))
            st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})
            st.html('<div class="note">The curve is a set of steps, not a smooth line. The model learned it from few El Niño months (mostly 2015/16), so read the shape loosely. See the expert details below.</div>')
        if flooding == 'Yes':
            st.html(FLOOD_NOTE)

# =====================================================================
# details for experts (a normal section at the bottom)
# =====================================================================
flood_low, flood_high = float(forecast['risk_if_flood'].min()), float(forecast['risk_if_flood'].max())
st.html('<div class="h2" style="margin-top:34px">Details for experts</div>')
st.html(f"""
<div class="card">
<ul class="todo" style="margin-top:0">
<li><b>Model.</b> The page scores the real model <code>models/xgboost.joblib</code> (XGBoost, 25 inputs, <b>with</b> the <code>flood_this_month</code> input), the same file as the live forecast, with the inputs in the exact training column order. The thresholds come from <code>models/xgboost_settings.json</code>: Watch {WATCH:.4f}, Warning {WARNING:.4f}.</li>
<li><b>Starting point.</b> Every scenario starts from the live row of the county (data month {month_text(first['data_year'], first['data_month'])}). At untouched inputs the score equals the live forecast.</li>
<li><b>Derived inputs.</b> <code>rain_pct_normal</code> = rain / normal rain of the county and month. <code>month_sin</code> and <code>month_cos</code> come from the data month with numpy. <code>soil_moisture_l1</code> follows the month (county average of the training years). In Simple mode the regional rain buttons multiply both <code>rain_mm</code> and <code>neighbour_rain</code>.</li>
<li><b>Fixed inputs.</b> NDVI, soil moisture, runoff rank and discharge rank (stand-ins: 0.5) and the county traits stay at their live values. Five inputs have zero importance in the model and are hidden.</li>
<li><b>Training range.</b> The sliders stay inside the 2002–2019 training range (rain: the range of the chosen county and month), because the model has not seen anything outside it.</li>
<li><b>Flood = Yes.</b> It pushes every county to about {flood_low:.2f}–{flood_high:.2f} (the <code>risk_if_flood</code> column), so all counties are High and the ONI slider has almost no effect. Historically, a county with a flood this month had a flood the next month in less than half of the cases, so this score overstates it.</li>
<li><b>ONI is a step function.</b> The model learned it from a few splits (about 0.14, 0.33, 1.01 and 1.36). ONI ≥ 1.5 comes from only 11 months, mostly the 2015/16 El Niño. Between about 0.3 and 1.0 the score is <i>lower</i> than at 0, which mirrors the quiet neutral years in the data. That is why the Simple mode has no "moderate" button. Past rain months act as season markers, so more rain last month can lower the risk.</li>
<li><b>Not a probability.</b> The risk score is relative: the model is tuned to catch floods (class weight about 10), so its scores run far above the real flood frequency.</li>
</ul>
</div>
""")
