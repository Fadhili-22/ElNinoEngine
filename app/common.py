"""Shared helpers of the flood app: file paths, loading the data, colours, the page style (CSS), small html blocks and the maps.

Every page in app/views/ imports from here, so a colour, a level name or a threshold is defined in ONE place only.
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 1. Files
# =====================================================================
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(APP_FOLDER, 'data')
MODELS_FOLDER = os.path.join(os.path.dirname(APP_FOLDER), 'models')

FORECAST_FILE = os.path.join(APP_FOLDER, 'forecast', 'latest_forecast.csv')
BACKTEST_FILE = os.path.join(DATA_FOLDER, 'app_predictions.csv')
REFERENCE_FILE = os.path.join(DATA_FOLDER, 'scenario_reference.csv')
SHAPES_FILE = os.path.join(DATA_FOLDER, 'kenya_counties_simple.geojson')
MODEL_FILE = os.path.join(MODELS_FOLDER, 'xgboost.joblib')
SETTINGS_FILE = os.path.join(MODELS_FOLDER, 'xgboost_settings.json')


# =====================================================================
# 2. Thresholds: read from the settings file of the model, never typed here
# =====================================================================
def read_settings():
    # the settings json written when the model was trained (thresholds, cross-validation results)
    with open(SETTINGS_FILE) as settings_file:
        return json.load(settings_file)


SETTINGS = read_settings()
WATCH = float(SETTINGS['watch_threshold'])      # at or above this line: "Medium" (Watch, be ready)
WARNING = float(SETTINGS['warning_threshold'])  # at or above this line: "High" (Warning, act now)


def level_of(risk):
    # the internal level name ('Warning', 'Watch' or 'No alert') of a risk score
    if risk >= WARNING:
        return 'Warning'
    if risk >= WATCH:
        return 'Watch'
    return 'No alert'


# =====================================================================
# 3. Colours and names
# =====================================================================
INK = '#1F2328'
BODY = '#5B6470'
LINE = '#E2E5E9'
GREEN = '#0B5D3B'

# the three risk colours, the same on every map and badge
HIGH = '#B42318'
HIGH_DARK = '#8F1D14'    # the highest scores inside High
HIGH_LIGHT = '#D9574A'   # the lowest scores inside High
MEDIUM = '#F2C46D'
LOW = '#E7EAEE'

# the colours of the backtest map
OUTCOME_COLORS = {'Warned in time': '#0B5D3B', 'Missed': '#D92D20', 'Warned but no flood': '#F2C46D', 'Quiet': '#E7EAEE'}

# the model and the data use 'Warning', 'Watch' and 'No alert'. The public pages say High, Medium and Low.
LEVEL_INFO = {
    'Warning': {'name': 'High', 'badge': 'HIGH RISK', 'action': 'Act now', 'css': 'high', 'color': HIGH},
    'Watch': {'name': 'Medium', 'badge': 'MEDIUM RISK', 'action': 'Be ready', 'css': 'medium', 'color': MEDIUM},
    'No alert': {'name': 'Low', 'badge': 'LOW RISK', 'action': 'No alert', 'css': 'low', 'color': LOW},
}

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
SANS = 'IBM Plex Sans, sans-serif'
MONO = 'IBM Plex Mono, monospace'


def month_text(year, month, short=False):
    # 'November 2026' (or 'Nov 2026' when short)
    name = MONTH_NAMES[int(month) - 1]
    return f'{name[:3] if short else name} {int(year)}'


# =====================================================================
# 4. Loading the data (cached, so a page change does not read the files again)
# =====================================================================
@st.cache_data
def load_forecast(modified):
    # the live forecast made by src/update_forecast.py. 'modified' is the time the file was saved, so a new file is read again.
    return pd.read_csv(FORECAST_FILE, dtype={'pcode': str})


@st.cache_data
def load_backtest(modified):
    # the 2020-2024 replay made by notebook 06. For every county and month we say what happened:
    #   Warned in time = a flood was recorded and there was a Watch or Warning the month before
    #   Missed = a flood was recorded and there was no alert
    #   Warned but no flood = an alert and no flood
    #   Quiet = no alert and no flood
    preds = pd.read_csv(BACKTEST_FILE, dtype={'pcode': str})
    preds['forecast_date'] = pd.to_datetime(dict(year=preds['forecast_year'], month=preds['forecast_month'], day=1))
    flood = preds['flood_recorded'] == 1
    alert = preds['level_live'] != 'No alert'
    preds['outcome'] = 'Quiet'
    preds.loc[alert & ~flood, 'outcome'] = 'Warned but no flood'
    preds.loc[alert & flood, 'outcome'] = 'Warned in time'
    preds.loc[~alert & flood, 'outcome'] = 'Missed'
    return preds


@st.cache_data
def load_reference(modified):
    # rainfall and soil numbers per county and month (made by app/make_scenario_reference.py), used when the scenario month changes
    return pd.read_csv(REFERENCE_FILE, dtype={'pcode': str})


@st.cache_data
def load_shapes():
    # the simplified county shapes, one point inside each county, and the west, south, east, north edges of Kenya
    with open(SHAPES_FILE) as shapes_file:
        geo = json.load(shapes_file)
    shapes = gpd.read_file(SHAPES_FILE)
    points = shapes.geometry.representative_point()
    dots = pd.DataFrame({'pcode': shapes['pcode'], 'lon': points.x, 'lat': points.y})
    bounds = [float(b) for b in shapes.total_bounds]
    return geo, dots, bounds


@st.cache_resource
def load_model():
    # the XGBoost model WITH the 'flood this month' input (the same one the live forecast uses)
    import joblib
    return joblib.load(MODEL_FILE)


def get_forecast():
    if not os.path.exists(FORECAST_FILE):
        return None
    return load_forecast(os.path.getmtime(FORECAST_FILE))


def get_backtest():
    if not os.path.exists(BACKTEST_FILE):
        return None
    return load_backtest(os.path.getmtime(BACKTEST_FILE))


def get_reference():
    if not os.path.exists(REFERENCE_FILE):
        return None
    return load_reference(os.path.getmtime(REFERENCE_FILE))


def forecast_with_levels(forecast, manual_counties=()):
    # the live forecast sorted from the highest risk to the lowest. Counties in manual_counties get their
    # "risk_if_flood" instead (the manual override of the live page). The level is worked out from the thresholds of the settings file.
    data = forecast.copy()
    data['manual'] = data['county'].isin(list(manual_counties))
    data['risk'] = data['risk'].where(~data['manual'], data['risk_if_flood'])
    data['level'] = [level_of(r) for r in data['risk']]
    data = data.sort_values('risk', ascending=False).reset_index(drop=True)
    data['rank'] = data.index + 1
    return data


def backtest_numbers(preds):
    # the headline numbers of the backtest, counted from the file (nothing typed by hand)
    flood = preds['flood_recorded'] == 1
    alert = preds['level_live'] != 'No alert'
    numbers = {
        'floods': int(flood.sum()),
        'warned': int((flood & alert).sum()),
        'warned_warning': int((flood & (preds['level_live'] == 'Warning')).sum()),
        'alerts': int(alert.sum()),
        'false_alarms': int((alert & ~flood).sum()),
    }
    numbers['missed'] = numbers['floods'] - numbers['warned']
    # the same count when the flood record of the data month is known (the 'level' column of the file)
    alert_known = preds['level'] != 'No alert'
    numbers['warned_known'] = int((flood & alert_known).sum())
    numbers['warned_warning_known'] = int((flood & (preds['level'] == 'Warning')).sum())
    return numbers


# =====================================================================
# 5. The model
# =====================================================================
def model_features(model):
    # the 25 inputs of the model in the exact order of the training table
    return list(model.feature_names_in_)


def score_rows(model, rows):
    # risk scores of a table of inputs (one row per county or scenario). The columns are put in the training order first.
    return model.predict_proba(rows[model_features(model)].astype(float))[:, 1]


# =====================================================================
# 6. The page style (one css block for all pages)
# =====================================================================
CSS = """
<style>
:root { --ink: #1F2328; --body: #5B6470; --line: #E2E5E9; --page: #F4F5F7; --green: #0B5D3B; --green2: #0E8A5F; --tint: #EEF6F2;
        --high: #B42318; --medium: #F2C46D; --low: #E7EAEE; --mono: 'IBM Plex Mono', ui-monospace, monospace; }

/* page */
.stApp { background: var(--page); color: var(--ink); font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 1.2rem 1.5rem 3rem; }
[data-testid="stDecoration"], [data-testid="stDeployButton"], #MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { background: #FFFFFF; border-bottom: 1px solid var(--line); }
a { color: var(--green); }

/* text blocks */
.eyebrow { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green); margin: 4px 0 12px; }
.h1 { font-size: clamp(2rem, 5.2vw, 3.2rem); line-height: 1.08; font-weight: 700; letter-spacing: -0.03em; color: var(--ink); margin: 0 0 14px; }
.h1 em { font-style: normal; color: var(--green2); }
.h2 { font-size: 1.55rem; line-height: 1.2; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); margin: 26px 0 10px; }
.h3 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); margin: 0 0 6px; }
.lead { font-size: 1.1rem; line-height: 1.55; color: var(--body); max-width: 46rem; margin: 0 0 18px; }
.note { font-size: 0.9rem; line-height: 1.5; color: var(--body); }
.mono { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: var(--body); }
.ink { color: var(--ink); }
.section-gap { height: 18px; }
[data-testid="stWidgetLabel"] p { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--body); }

/* white cards: html cards (class card) and containers with a key that starts with card_ */
.card, div[class*="st-key-card_"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 20px 22px; box-shadow: none; }
.card.tint, div[class*="st-key-card_tint"] { background: var(--tint); border-color: #CFE5DA; }
.card + .card { margin-top: 12px; }

/* buttons and links that look like buttons (44px high at least) */
.stButton button, [data-testid="stPageLink"] a, [data-testid="stBaseLinkButton-secondary"], [data-testid="stBaseLinkButton-primary"] {
    min-height: 44px; border-radius: 10px; font-weight: 600; border: 1px solid var(--line); background: #FFFFFF; color: var(--ink); }
.stButton button[kind="primary"], div[class*="st-key-primary"] [data-testid="stPageLink"] a { background: var(--green); border-color: var(--green); color: #FFFFFF; }
.stButton button:hover, [data-testid="stPageLink"] a:hover { border-color: var(--green); color: var(--green); }
.stButton button[kind="primary"]:hover, div[class*="st-key-primary"] [data-testid="stPageLink"] a:hover { background: #084A2F; color: #FFFFFF; }
div[class*="st-key-primary"] [data-testid="stPageLink"] a p { color: #FFFFFF; }
[data-testid="stButtonGroup"] button { min-height: 44px; }

/* select boxes */
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; min-height: 48px; box-shadow: none; }

/* badges and the three-part level bar */
.badge { display: inline-block; font-family: var(--mono); font-size: 12px; font-weight: 600; letter-spacing: 0.1em; padding: 5px 12px; border-radius: 999px; vertical-align: middle; }
.badge-high { background: var(--high); color: #FFFFFF; }
.badge-medium { background: var(--medium); color: var(--ink); }
.badge-low { background: var(--low); color: var(--ink); }
.badge-manual { background: #FFFFFF; color: var(--ink); border: 1px solid #AAB1BB; text-transform: none; letter-spacing: 0.02em; font-weight: 500; }
.badge-ok { background: #0B5D3B; color: #FFFFFF; }
.badge-missed { background: #D92D20; color: #FFFFFF; }
.badge-warned-nf { background: #F2C46D; color: var(--ink); }
.badge-quiet { background: #E7EAEE; color: var(--ink); }
.levelbar { display: flex; gap: 4px; margin: 12px 0 8px; }
.seg { flex: 1; text-align: center; padding: 9px 4px; font-size: 0.85rem; border-radius: 8px; color: var(--ink); }
.seg-low { background: var(--low); }
.seg-medium { background: var(--medium); }
.seg-high { background: var(--high); color: #FFFFFF; }
.seg.current { outline: 3px solid var(--ink); outline-offset: 2px; font-weight: 700; }

/* the big result of a county */
.county-name { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.15; color: var(--ink); margin-right: 10px; }
.county-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin: 8px 0 10px; }
.result-sentence { font-size: 1.15rem; font-weight: 600; color: var(--ink); margin: 6px 0 4px; }
.todo { margin: 10px 0 0; padding-left: 1.2rem; color: var(--ink); line-height: 1.65; }

/* lists */
.rank-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); }
.rank-row:last-child { border-bottom: none; }
.rank-row .rank { font-family: var(--mono); font-size: 12px; color: var(--body); width: 24px; }
.rank-row .name { flex: 0 0 120px; color: var(--ink); font-weight: 600; }
.rank-row .bar { flex: 1; height: 8px; background: #EEF0F3; border-radius: 4px; min-width: 40px; }
.rank-row .bar i { display: block; height: 100%; border-radius: 4px; }
.kv-row { display: flex; justify-content: space-between; gap: 14px; padding: 9px 0; border-bottom: 1px solid var(--line); }
.kv-row:last-child { border-bottom: none; }
.kv-row .k { color: var(--ink); }
.kv-row .v { font-family: var(--mono); font-size: 13px; color: var(--ink); text-align: right; }
.kv-row .why { color: var(--body); font-size: 0.85rem; }

/* the stat strip: the gap shows the border colour between the cells */
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; margin: 14px 0; }
.strip.three { grid-template-columns: repeat(3, 1fr); }
.cell { background: #FFFFFF; padding: 16px 20px; }
.cell .mono { display: block; margin-bottom: 6px; }
.cell .num { font-size: 1.9rem; line-height: 1.1; font-weight: 700; letter-spacing: -0.02em; color: var(--ink); }
.cell .num.big { font-size: 2.6rem; }
.cell .sub { font-size: 0.85rem; color: var(--body); margin-top: 6px; }

/* callouts */
.callout { background: #FFFFFF; border: 1px solid var(--line); border-left: 5px solid var(--green); border-radius: 10px; padding: 14px 18px; color: var(--ink); line-height: 1.55; margin: 10px 0; }
.callout b.label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green); margin-right: 8px; }
.callout.warn { border-left-color: #B7791F; }
.callout.warn b.label { color: #8A5A00; }

/* pills for the stand-ins */
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.pill { display: inline-block; font-size: 0.82rem; padding: 4px 10px; border-radius: 999px; background: #FFFFFF; border: 1px solid #CFD4DB; color: var(--ink); }

/* map legend */
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; margin-top: 8px; font-size: 0.88rem; color: var(--ink); }
.legend .item { display: inline-flex; align-items: center; gap: 8px; }
.swatch { display: inline-block; width: 22px; height: 14px; border-radius: 3px; }
.swatch.low { background: var(--low); border: 1px solid #CFD4DB; }
.swatch.medium { background: var(--medium); }
.swatch.high { background: linear-gradient(90deg, #D9574A, #8F1D14); }
.swatch.pick { background: #FFFFFF; border: 3px solid var(--ink); }
.swatch.o-ok { background: #0B5D3B; }
.swatch.o-missed { background: #D92D20; }
.swatch.o-warned { background: #F2C46D; }
.swatch.o-quiet { background: #E7EAEE; border: 1px solid #CFD4DB; }
.dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; background: #111827; border: 1.5px solid #FFFFFF; box-shadow: 0 0 0 1px var(--line); }

/* the menu bar at the top (built in app.py): name on the left, links on the right, the active page in light green */
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
.st-key-topbar { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 6px 12px; gap: 2px 4px; margin-bottom: 22px; flex-wrap: wrap; }
.st-key-topbar .brand { font-weight: 700; letter-spacing: -0.02em; color: var(--ink); white-space: nowrap; padding-right: 12px; }
.st-key-topbar .brand-mark { color: var(--green2); }
.st-key-topbar > [data-testid="stElementContainer"]:first-child { margin-right: auto; }
.st-key-topbar [data-testid="stPageLink"] a { border: none; background: transparent; padding: 0 14px; }
.st-key-topbar [data-testid="stPageLink"] a[aria-current="page"] { background: var(--tint); }
.st-key-topbar [data-testid="stPageLink"] a[aria-current="page"] p { color: var(--green); }

/* the hero of the home page */
.hero { padding: 10px 0 6px; }
.way-num { font-family: var(--mono); font-size: 13px; color: var(--green); font-weight: 600; letter-spacing: 0.1em; }
.start-badge { display: inline-block; font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 0.1em; background: var(--green); color: #FFFFFF; padding: 3px 10px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }
.footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--body); font-size: 0.88rem; }

/* phone width */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
    .strip, .strip.three { grid-template-columns: repeat(2, 1fr); }
    .cell .num.big { font-size: 2.1rem; }
    .card, div[class*="st-key-card_"] { padding: 16px; }
    .county-name { font-size: 1.7rem; }
    .rank-row .name { flex: 0 0 96px; }
    /* the map frame is 680px high, but Kenya is only about 440px high at this width, so the empty parts are cut */
    .st-key-card_map > [data-testid="stElementContainer"]:first-child { margin-bottom: -200px; }
    .st-key-card_map [data-testid="stFullScreenFrame"] { margin-top: -120px; }
}
</style>
"""


def inject_css():
    st.html(CSS)


# =====================================================================
# 7. Small html blocks
# =====================================================================
def eyebrow(text):
    return f'<div class="eyebrow">── {text}</div>'


def badge(level):
    # the coloured label of a level: HIGH RISK, MEDIUM RISK or LOW RISK
    info = LEVEL_INFO[level]
    return f'<span class="badge badge-{info["css"]}">{info["badge"]}</span>'


def level_bar(level):
    # the Low / Medium / High bar with the current level outlined
    parts = ''
    for name in ['No alert', 'Watch', 'Warning']:
        info = LEVEL_INFO[name]
        current = ' current' if name == level else ''
        parts += f'<div class="seg seg-{info["css"]}{current}">{info["name"]}</div>'
    return f'<div class="levelbar" role="img" aria-label="Risk level: {LEVEL_INFO[level]["name"]}">{parts}</div>'


def level_sentence(level, month_word):
    # one plain sentence for a level
    if level == 'Warning':
        return f'Act now: be ready for flooding in {month_word}.'
    if level == 'Watch':
        return f'Be ready: flooding is possible in {month_word}.'
    return f'No alert for {month_word}. Keep an eye on the weather updates.'


def legend_risk(extra=''):
    return f"""<div class="legend">
<span class="item"><span class="swatch low"></span>Low</span>
<span class="item"><span class="swatch medium"></span>Medium</span>
<span class="item"><span class="swatch high"></span>High (darker = higher)</span>
<span class="item"><span class="swatch pick"></span>Your county</span>{extra}
</div>"""


def rank_rows(data, count=10):
    # the highest-risk counties with a RELATIVE bar (no numbers): the bar compares each county with the lowest and highest county of the list,
    # so the bars still differ when all counties are close together
    shown = data.head(count)
    low, high = shown['risk'].min(), shown['risk'].max()
    span = max(high - low, 1e-9)
    rows = ''
    for row in shown.itertuples():
        width = 30 + 70 * (row.risk - low) / span
        color = LEVEL_INFO[row.level]['color']
        rows += (f'<div class="rank-row"><span class="rank">{row.rank:02d}</span><span class="name">{row.county}</span>'
                 f'<span class="bar"><i style="width:{width:.0f}%;background:{color}"></i></span>{badge(row.level)}</div>')
    return rows


# =====================================================================
# 8. Maps and charts
# =====================================================================
def map_clicked(map_key, pick_key):
    # a click on a county of the map chooses that county (the choice is written into the select box of the page)
    points = st.session_state[map_key]['selection']['points']
    if len(points) > 0 and points[0].get('location') is not None:
        geo, dots, bounds = load_shapes()
        names = {feature['properties']['pcode']: feature['properties']['county'] for feature in geo['features']}
        st.session_state[pick_key] = names[points[0]['location']]


def _finish_map(fig, geo, bounds, picked_rows, marked_rows):
    # shared last steps of both maps: an outline around the chosen county, dots for marked counties, and the frame of Kenya
    outline_traces = []
    if len(picked_rows) > 0:
        outline_traces = [(picked_rows, 'white', 7), (picked_rows, INK, 3.5)]
    for rows, line_color, line_width in outline_traces:
        fig.add_trace(go.Choropleth(geojson=geo, locations=rows['pcode'], featureidkey='properties.pcode', z=[1] * len(rows),
                                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], showscale=False, showlegend=False,
                                    marker_line_color=line_color, marker_line_width=line_width, hoverinfo='skip'))
    if len(marked_rows) > 0:
        _, dots_position, _ = load_shapes()
        marked = marked_rows.merge(dots_position, on='pcode')
        fig.add_trace(go.Scattergeo(lon=marked['lon'], lat=marked['lat'], mode='markers', showlegend=False, hoverinfo='skip',
                                    marker=dict(size=10, symbol='circle', color='#111827', line=dict(color='white', width=1.5))))
    # the frame is set from the bounds of the shapes (fitbounds left half of the frame empty with these shapes)
    fig.update_geos(fitbounds=False, lonaxis_range=[bounds[0] - 0.1, bounds[2] + 0.1], lataxis_range=[bounds[1] - 0.1, bounds[3] + 0.1],
                    visible=False, showframe=False, bgcolor='rgba(0,0,0,0)')
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family=SANS, color=BODY), hoverlabel=dict(bgcolor='white', bordercolor=LINE, font=dict(family=SANS, color=INK)))
    return fig


def make_risk_map(data, picked, show_manual=True):
    # map of the three risk levels. Medium and Low have one colour each. High goes from light red (lowest High score)
    # to dark red (highest score), so the counties inside High can still be compared. data needs: pcode, county, risk, level, rank, manual.
    geo, dots_position, bounds = load_shapes()
    high = data['level'] == 'Warning'
    top_risk = data['risk'].max()
    span = max(top_risk - WARNING, 1e-9)
    # z: 0 = Low, 1 = Medium, 2 to 3 = High (2 = lightest red, 3 = darkest red)
    z = pd.Series(0.0, index=data.index)
    z[data['level'] == 'Watch'] = 1.0
    z[high] = 2.0 + ((data.loc[high, 'risk'] - WARNING) / span).clip(0, 1)
    colorscale = [[0, LOW], [1 / 6, LOW], [1 / 6, MEDIUM], [0.5, MEDIUM], [0.5, HIGH_LIGHT], [2 / 3, HIGH_LIGHT], [1, HIGH_DARK]]
    names = data['level'].map(lambda name: LEVEL_INFO[name]['name'] + ' risk')
    custom = np.column_stack([data['county'], names, data['rank']])
    fig = go.Figure(go.Choropleth(geojson=geo, locations=data['pcode'], featureidkey='properties.pcode', z=z, zmin=0, zmax=3,
                                  colorscale=colorscale, marker_line_color='white', marker_line_width=0.8, showscale=False, showlegend=False,
                                  customdata=custom, hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]} · rank %{customdata[2]} of ' + str(len(data)) + '<extra></extra>',
                                  selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    marked = data[data['manual']] if show_manual else data.iloc[0:0]
    return _finish_map(fig, geo, bounds, data[data['county'] == picked], marked)


def make_outcome_map(data, picked):
    # map of the backtest month: one colour per county, by what happened. data needs: pcode, county, outcome, risk_live.
    geo, dots_position, bounds = load_shapes()
    order = list(OUTCOME_COLORS)
    z = data['outcome'].map({name: i for i, name in enumerate(order)})
    # a discrete colour scale: every outcome owns one quarter of the scale
    colorscale = []
    for i, name in enumerate(order):
        colorscale += [[i / 4, OUTCOME_COLORS[name]], [(i + 1) / 4, OUTCOME_COLORS[name]]]
    fig = go.Figure(go.Choropleth(geojson=geo, locations=data['pcode'], featureidkey='properties.pcode', z=z, zmin=0, zmax=3.9999,
                                  colorscale=colorscale, marker_line_color='white', marker_line_width=0.8, showscale=False, showlegend=False,
                                  customdata=np.column_stack([data['county'], data['outcome']]),
                                  hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
                                  selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    return _finish_map(fig, geo, bounds, data[data['county'] == picked], data.iloc[0:0])


def risk_line(x, y, x_format, height=240):
    # line chart of a risk score with the Watch and Warning lines (technical, for the expert parts)
    line = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name='Risk score', line=dict(color=GREEN, width=2.5),
                                marker=dict(size=6, color=GREEN, line=dict(color='white', width=1.5))))
    line.add_hline(y=WATCH, line_dash='dot', line_color='#D99A1E', line_width=2.5, annotation_text='WATCH', annotation_position='bottom left',
                   annotation_font=dict(family=MONO, size=10, color=INK))
    line.add_hline(y=WARNING, line_dash='dot', line_color=HIGH, line_width=2.5, annotation_text='WARNING', annotation_position='top left',
                   annotation_font=dict(family=MONO, size=10, color=INK))
    line.update_yaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], gridcolor='#EEF0F3', zeroline=False, tickfont=dict(family=MONO, size=10))
    line.update_xaxes(tickformat=x_format, showgrid=False, tickfont=dict(family=MONO, size=10), linecolor=LINE)
    line.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(family=SANS, color=BODY), hoverlabel=dict(bgcolor='white', bordercolor=LINE, font=dict(family=SANS, color=INK)))
    return line
