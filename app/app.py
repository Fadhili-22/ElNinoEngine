import os
import json
from functools import partial
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Kenya flood early warning', layout='wide')

# the two thresholds of the backtest (from models/xgboost_settings.json). The live forecast file has its own copy of them.
WATCH = 0.3425
WARNING = 0.4374

# colours of the three levels (the amber and red are also the outlines on the map)
RED = '#d92d20'
AMBER = '#f59e0b'
LEVEL_COLORS = {'Warning': RED, 'Watch': AMBER, 'No alert': '#5b7db1'}
PILL_CLASS = {'Warning': 'pill-warning', 'Watch': 'pill-watch', 'No alert': 'pill-none',
              'Caught': 'pill-caught', 'Missed': 'pill-missed', 'False alarm': 'pill-false', 'No flood': 'pill-none'}
# light to dark blue, used for the risk of every county
BLUES = [[0, '#e8f0fb'], [0.25, '#b7cdee'], [0.5, '#6c97d6'], [0.75, '#2f5fb0'], [1, '#12327a']]
# fill colours of the map "Would it have warned us?" (the colour is the alert level)
FILL_NONE = '#e3e8ef'
FILL_WATCH = '#fbbf4a'
FILL_WARNING = '#e5483d'
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DEFAULT_MONTH = 'Apr 2024 data → forecast May 2024'
# the three pages
LIVE_VIEW = 'Latest forecast'
BACK_VIEW = 'Would it have warned us? (2020–2024)'
WHATIF_VIEW = 'What if? (scenario explorer)'
# font of the charts (loaded by the theme in .streamlit/config.toml)
SANS = 'IBM Plex Sans, sans-serif'
MONO = 'IBM Plex Mono, monospace'

app_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(app_folder, 'data')
forecast_file = os.path.join(app_folder, 'forecast', 'latest_forecast.csv')
backtest_file = os.path.join(data_folder, 'app_predictions.csv')
model_file = os.path.join(os.path.dirname(app_folder), 'models', 'xgboost.joblib')

# the one block of css (data-testid and key classes only, no generated class names)
CSS = """
<style>
:root { --ink: #1f2328; --body: #5b6470; --line: #e2e5e9; --page: #f4f5f7; --red: #d92d20; --amber: #f59e0b;
        --mono: 'IBM Plex Mono', ui-monospace, monospace; }

/* page */
.stApp { background: var(--page); }
[data-testid="stMainBlockContainer"] { max-width: 1500px; padding: 2.2rem 2.4rem 3rem; }
h1, h2, h3, h4 { color: var(--ink); }

/* hide deploy button, menu, header and footer */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stDeployButton"], #MainMenu, footer { display: none !important; }

/* small uppercase mono labels */
.mono, [data-testid="stWidgetLabel"] p {
    font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--body); }

/* header block */
.hdr-eyebrow { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: var(--body); margin-bottom: 10px; }
.hdr-title { font-size: 2.6rem; line-height: 1.1; font-weight: 700; color: var(--ink); margin: 0 0 10px 0; letter-spacing: -0.02em; }
.hdr-desc { font-size: 1.02rem; color: var(--body); margin: 0 0 14px 0; }
.hdr-month { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink); }
.hdr-month span { color: var(--body); }

/* the month box and the view switch */
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #fff; border: 1px solid var(--line); border-radius: 10px; box-shadow: none; }

/* notice with a thick left border */
.callout { background: #fff; border: 1px solid var(--line); border-left: 5px solid var(--amber); border-radius: 8px; padding: 12px 16px; color: var(--ink); font-size: 0.95rem; margin-bottom: 12px; }
.callout b { font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-right: 10px; color: #92400e; }
.callout.info { border-left-color: #1f4fa8; }
.callout.info b { color: #1f4fa8; }
.callout .standins { margin-top: 10px; }

/* the joined stat strip: the grid gap shows the border colour between the cells */
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: 14px; }
.cell { background: #fff; padding: 16px 20px 14px; }
.cell .mono { display: block; margin-bottom: 6px; }
.cell .num { font-size: 2.6rem; line-height: 1; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.cell .num.warn { color: var(--red); }
.cell .num.watch { color: #d97706; }
.cell .num.good { color: #1f4fa8; }
.cell .sub { font-size: 0.82rem; color: var(--body); margin-top: 6px; }

/* alert pills */
.pill { display: inline-block; font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px; }
.pill-warning { background: #fdecea; color: #b42318; }
.pill-watch { background: #fef3c7; color: #92400e; }
.pill-none { background: #eef0f3; color: var(--body); }
.pill-caught { background: #dbe7fb; color: #1f4fa8; }
.pill-missed { background: #1f2328; color: #fff; }
.pill-false { background: #fff; color: var(--body); border: 1px dashed #aab1bb; }
/* pills of the stand-ins (the inputs that are not real data) and of a manual entry */
.standins { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.pill-standin { background: #fff; color: var(--body); border: 1px solid #cfd4db; text-transform: none; letter-spacing: 0.02em; font-weight: 500; cursor: help; }
.pill-manual { background: #fff7e0; color: #92400e; border: 1px solid #f3d38a; text-transform: none; letter-spacing: 0.02em; }

/* white cards (containers with a key get an st-key-... class) */
.st-key-map_card, .st-key-detail_card, .st-key-top_card, .st-key-floods_card, .st-key-whatif_card, [data-testid="stExpander"] {
    background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; box-shadow: none; }
[data-testid="stExpander"] { padding: 4px 8px; }
[data-testid="stExpander"] details { border: none; }

/* map legend */
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: center; margin-top: 6px; font-size: 0.85rem; }
.legend .item { display: inline-flex; align-items: center; gap: 8px; }
.swatch { display: inline-block; width: 22px; height: 14px; border-radius: 3px; background: #fff; }
.ramp { display: inline-block; width: 110px; height: 10px; border-radius: 3px; background: linear-gradient(90deg, #e8f0fb, #b7cdee 25%, #6c97d6 50%, #2f5fb0 75%, #12327a); }
.swatch.red { border: 3px solid var(--red); }
.swatch.amber { border: 3px solid var(--amber); }
.swatch.pick { border: 3px solid var(--ink); }
.swatch.f-warning { background: #e5483d; }
.swatch.f-watch { background: #fbbf4a; }
.swatch.f-none { background: #e3e8ef; border: 1px solid #cfd4db; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #111827; border: 1.5px solid #fff; box-shadow: 0 0 0 1px var(--line); }
.cross { font-weight: 800; color: #111827; font-size: 15px; line-height: 10px; }

/* county card */
.county-name { font-size: 2rem; font-weight: 700; color: var(--ink); line-height: 1.15; letter-spacing: -0.02em; margin-right: 12px; }
.county-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 12px 0 14px; }
.county-row .county-name { margin-right: 4px; }
.risk-row { display: flex; align-items: baseline; gap: 14px; }
.risk-num { font-size: 3.4rem; font-weight: 700; line-height: 1; color: var(--ink); letter-spacing: -0.03em; }
.section-label { margin: 14px 0 4px; }
.what { border-top: 1px solid var(--line); margin-top: 10px; padding-top: 12px; color: var(--ink); }
.what .mono { margin-right: 10px; }
.kv-row { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--line); }
.kv-row:last-of-type { border-bottom: none; }
.kv-row .k { color: var(--body); }
.kv-row .v { font-family: var(--mono); font-size: 13px; color: var(--ink); text-align: right; }
.small-note { font-size: 0.82rem; color: var(--body); margin-top: 4px; }

/* lists of counties (riskiest, and the floods of a month) */
.rank-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); }
.rank-row:last-child { border-bottom: none; }
.rank-row .rank { font-family: var(--mono); font-size: 12px; color: var(--body); width: 22px; }
.rank-row .name { flex: 0 0 120px; color: var(--ink); font-weight: 600; }
.rank-row .bar { flex: 1; height: 6px; background: #eef0f3; border-radius: 3px; min-width: 40px; }
.rank-row .bar i { display: block; height: 100%; border-radius: 3px; background: #2f5fb0; }
.rank-row .score { font-family: var(--mono); font-size: 13px; color: var(--ink); width: 38px; text-align: right; }

/* phone width */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
    .hdr-title { font-size: 1.9rem; }
    .strip { grid-template-columns: repeat(2, 1fr); }
    .cell .num { font-size: 2.1rem; }
    .st-key-map_card, .st-key-detail_card, .st-key-top_card, .st-key-floods_card, .st-key-whatif_card { padding: 14px; }
    /* the map frame is 680px high, but Kenya is only about 440px high at this width, so the empty parts are cut */
    .st-key-map_card > [data-testid="stElementContainer"]:first-child { margin-bottom: -200px; }
    .st-key-map_card [data-testid="stFullScreenFrame"] { margin-top: -120px; }
    .county-name { font-size: 1.6rem; }
    .risk-num { font-size: 2.8rem; }
    .rank-row .name { flex: 0 0 90px; }
}
</style>
"""


@st.cache_data
def load_backtest(modified):
    # the backtest file made by notebook 06 (pcode as text). The modified time makes the cache reload a new file.
    preds = pd.read_csv(backtest_file, dtype={'pcode': str})
    preds['forecast_date'] = pd.to_datetime(dict(year=preds['forecast_year'], month=preds['forecast_month'], day=1))
    # the page shows what the LIVE forecast would have said (no flood records of the data month). An older file without those columns uses the other ones.
    preds['live_columns'] = 'risk_live' in preds.columns
    if 'risk_live' not in preds.columns:
        preds['risk_live'] = preds['risk']
        preds['level_live'] = preds['level']
    # what happened, for every county and month: a flood that had an alert before it (caught) or not (missed), an alert and no flood (false alarm)
    flood = preds['flood_recorded'] == 1
    alert = preds['level_live'] != 'No alert'
    preds['outcome'] = 'No flood'
    preds.loc[alert & ~flood, 'outcome'] = 'False alarm'
    preds.loc[alert & flood, 'outcome'] = 'Caught'
    preds.loc[~alert & flood, 'outcome'] = 'Missed'
    return preds


@st.cache_data
def load_forecast(modified):
    # the live forecast made by src/update_forecast.py
    return pd.read_csv(forecast_file, dtype={'pcode': str})


@st.cache_data
def load_shapes():
    # the simplified county shapes, and one point inside each county for the flood dots
    with open(os.path.join(data_folder, 'kenya_counties_simple.geojson')) as shapes_file:
        geo = json.load(shapes_file)
    shapes = gpd.read_file(os.path.join(data_folder, 'kenya_counties_simple.geojson'))
    points = shapes.geometry.representative_point()
    dots = pd.DataFrame({'pcode': shapes['pcode'], 'lon': points.x, 'lat': points.y})
    # west, south, east, north of Kenya (for the map frame)
    bounds = [float(b) for b in shapes.total_bounds]
    return geo, dots, bounds


@st.cache_resource
def load_model():
    # the XGBoost model, only needed by the scenario page (None if the file or the xgboost package is missing)
    try:
        import joblib
        return joblib.load(model_file)
    except Exception:
        return None


def pill(level):
    # the small coloured label of a level (or of an outcome)
    return f'<span class="pill {PILL_CLASS[level]}">{level}</span>'


def month_text(year, month):
    return f'{MONTH_NAMES[int(month) - 1]} {int(year)}'


def level_of(risk, watch, warning):
    return 'Warning' if risk >= warning else 'Watch' if risk >= watch else 'No alert'


def map_clicked(map_key, pick_key):
    # a click on a county of the map selects that county
    points = st.session_state[map_key]['selection']['points']
    if len(points) > 0 and points[0].get('location') is not None:
        names = {f['properties']['pcode']: f['properties']['county'] for f in geo['features']}
        st.session_state[pick_key] = names[points[0]['location']]


def make_map(data, mode, picked, dots=()):
    # the map of Kenya. mode 'risk': every county shaded by its risk, with an amber or red outline for Watch and Warning.
    # mode 'level': every county filled with the colour of its alert level.
    # dots: a list of (rows, symbol, size) with the counties to mark (rows need a pcode)
    if mode == 'risk':
        fig = go.Figure(go.Choropleth(geojson=geo, locations=data['pcode'], featureidkey='properties.pcode', z=data['risk'],
                                      zmin=0, zmax=1, colorscale=BLUES, marker_line_color='white', marker_line_width=0.8,
                                      customdata=data[['county', 'level']], showlegend=False, showscale=False,
                                      hovertemplate='<b>%{customdata[0]}</b><br>Risk %{z:.2f}<br>Alert: %{customdata[1]}<extra></extra>',
                                      selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
        # outlines (no fill): amber for Watch, red for Warning. A white line below each one keeps it clear on dark blue.
        outlines = [(data[data['level'] == 'Watch'], AMBER, 2.5), (data[data['level'] == 'Warning'], RED, 2.5)]
    else:
        level_number = data['level'].map({'No alert': 0, 'Watch': 1, 'Warning': 2})
        discrete = [[0, FILL_NONE], [1 / 3, FILL_NONE], [1 / 3, FILL_WATCH], [2 / 3, FILL_WATCH], [2 / 3, FILL_WARNING], [1, FILL_WARNING]]
        fig = go.Figure(go.Choropleth(geojson=geo, locations=data['pcode'], featureidkey='properties.pcode', z=level_number,
                                      zmin=0, zmax=2, colorscale=discrete, marker_line_color='white', marker_line_width=0.8,
                                      customdata=data[['county', 'level', 'risk', 'outcome']], showlegend=False, showscale=False,
                                      hovertemplate='<b>%{customdata[0]}</b><br>Alert: %{customdata[1]} (risk %{customdata[2]:.2f})<br>%{customdata[3]}<extra></extra>',
                                      selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
        outlines = []
    # the chosen county has a thick dark outline (with a white line below it)
    outlines.append((data[data['county'] == picked], '#1f2328', 3.5))
    for rows, color, width in outlines:
        if len(rows) == 0:
            continue
        for line_color, line_width in [('white', width + 3), (color, width)]:
            fig.add_trace(go.Choropleth(geojson=geo, locations=rows['pcode'], featureidkey='properties.pcode', z=[1] * len(rows),
                                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], showscale=False, showlegend=False,
                                        marker_line_color=line_color, marker_line_width=line_width, hoverinfo='skip'))
    for rows, symbol, size in dots:
        marked = rows.merge(dots_position, on='pcode')
        fig.add_trace(go.Scattergeo(lon=marked['lon'], lat=marked['lat'], mode='markers', showlegend=False,
                                    marker=dict(size=size, symbol=symbol, color='#111827', line=dict(color='white', width=1.5)), hoverinfo='skip'))
    # fitted to Kenya, no frame, no axes, transparent background
    # (fitbounds='locations' left half of the frame empty with these shapes, so the frame is set from the shapes' own bounds)
    fig.update_geos(fitbounds=False, lonaxis_range=[bounds[0] - 0.1, bounds[2] + 0.1], lataxis_range=[bounds[1] - 0.1, bounds[3] + 0.1],
                    visible=False, showframe=False, bgcolor='rgba(0,0,0,0)')
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family=SANS, color='#5b6470'), hoverlabel=dict(bgcolor='white', bordercolor='#e2e5e9', font=dict(family=SANS, color='#1f2328')))
    return fig


def threshold_bar(risk, level, watch, warning):
    # short bar with the watch and warning lines
    bar = go.Figure()
    bar.add_trace(go.Bar(x=[1], y=[''], orientation='h', marker_color='#eef0f3', width=0.55, hoverinfo='skip'))
    bar.add_trace(go.Bar(x=[risk], y=[''], orientation='h', marker_color=LEVEL_COLORS[level], width=0.55, hoverinfo='skip'))
    bar.add_vline(x=watch, line_dash='dot', line_color=AMBER, line_width=2.5)
    bar.add_vline(x=warning, line_dash='dot', line_color=RED, line_width=2.5)
    bar.add_annotation(x=watch, y=1, yref='paper', yshift=12, text=f'WATCH {watch:.2f}', showarrow=False, xanchor='right', xshift=4,
                       font=dict(family=MONO, size=10, color='#92400e'))
    bar.add_annotation(x=warning, y=1, yref='paper', yshift=12, text=f'WARNING {warning:.2f}', showarrow=False, xanchor='left', xshift=-4,
                       font=dict(family=MONO, size=10, color='#b42318'))
    bar.update_xaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], showgrid=False, zeroline=False, tickfont=dict(family=MONO, size=10))
    bar.update_yaxes(visible=False)
    bar.update_layout(barmode='overlay', height=112, margin=dict(l=8, r=8, t=26, b=24), showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family=SANS, color='#5b6470'))
    return bar


def risk_line(x, y, watch, warning, x_format, height=240):
    # line chart of a risk with the Watch and Warning lines
    line = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name='Risk', line=dict(color='#1f4fa8', width=2.5),
                                marker=dict(size=6, color='#1f4fa8', line=dict(color='white', width=1.5))))
    line.add_hline(y=watch, line_dash='dot', line_color=AMBER, line_width=2, annotation_text='WATCH', annotation_position='bottom left',
                   annotation_font=dict(family=MONO, size=10, color='#92400e'))
    line.add_hline(y=warning, line_dash='dot', line_color=RED, line_width=2, annotation_text='WARNING', annotation_position='top left',
                   annotation_font=dict(family=MONO, size=10, color='#b42318'))
    line.update_yaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], gridcolor='#eef0f3', zeroline=False, tickfont=dict(family=MONO, size=10))
    line.update_xaxes(tickformat=x_format, showgrid=False, tickfont=dict(family=MONO, size=10), linecolor='#e2e5e9')
    line.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(family=SANS, color='#5b6470'), hoverlabel=dict(bgcolor='white', bordercolor='#e2e5e9', font=dict(family=SANS, color='#1f2328')))
    return line


# which pages can be shown (a page needs its file)
preds = load_backtest(os.path.getmtime(backtest_file)) if os.path.exists(backtest_file) else None
forecast = load_forecast(os.path.getmtime(forecast_file)) if os.path.exists(forecast_file) else None
model = load_model() if (forecast is not None and os.path.exists(model_file)) else None
geo, dots_position, bounds = load_shapes()
views = ([LIVE_VIEW] if forecast is not None else []) + ([BACK_VIEW] if preds is not None else []) + ([WHATIF_VIEW] if model is not None else [])
if len(views) == 0:
    st.error('No data to show. Run `python src/update_forecast.py` for the live forecast, or notebook 06 for the backtest.')
    st.stop()

# the page comes first, the rest depends on it
st.session_state.setdefault('view', views[0])
if st.session_state['view'] not in views:
    st.session_state['view'] = views[0]
view = st.session_state['view']

# the backtest months, newest first, with the label of the month box
if preds is not None:
    months = preds[['year', 'month', 'forecast_year', 'forecast_month', 'records_complete']].drop_duplicates()
    months = months.sort_values(['year', 'month'], ascending=False).reset_index(drop=True)
    months['label'] = (months['month'].apply(lambda m: MONTH_NAMES[m - 1]) + ' ' + months['year'].astype(str) + ' data → forecast '
                       + months['forecast_month'].apply(lambda m: MONTH_NAMES[m - 1]) + ' ' + months['forecast_year'].astype(str))


def riskiest_county(label):
    # the county with the highest risk in the chosen backtest month
    month = months[months['label'] == label].iloc[0]
    rows = preds[(preds['year'] == month['year']) & (preds['month'] == month['month'])]
    return rows.sort_values('risk_live', ascending=False).iloc[0]['county']


def view_changed():
    # a new page starts with its riskiest county (it is chosen again below)
    st.session_state.pop('county_pick', None)


def month_changed():
    # a new month starts with its riskiest county
    st.session_state['county_pick'] = riskiest_county(st.session_state['month_pick'])


# the text of the header
if view == BACK_VIEW:
    st.session_state.setdefault('month_pick', DEFAULT_MONTH if DEFAULT_MONTH in list(months['label']) else months['label'][0])
    label = st.session_state['month_pick']
    month = months[months['label'] == label].iloc[0]
    forecast_text = month_text(month['forecast_year'], month['forecast_month'])
    data_text = month_text(month['year'], month['month'])
    header_line = f'Forecast for {forecast_text} <span>· based on {data_text} data</span>'
    description = 'Would the forecast have warned us? The live forecast, replayed on the months it never saw (2020 to 2024).'
elif view == LIVE_VIEW:
    first = forecast.iloc[0]
    forecast_text = month_text(first['forecast_year'], first['forecast_month'])
    data_text = month_text(first['data_year'], first['data_month'])
    header_line = f"Forecast for {forecast_text} <span>· based on {data_text} data · updated {first['generated_on']}</span>"
    description = 'County flood risk for next month, from an XGBoost model with a Watch and a Warning level.'
else:
    first = forecast.iloc[0]
    data_text = month_text(first['data_year'], first['data_month'])
    header_line = f'Starting from the live values of {data_text} <span>· a what-if, not a forecast</span>'
    description = 'Change the rainfall and the ocean indices and see how the model reacts.'

st.html(CSS)

# header: eyebrow, title, description, and the page switch (with the month box of the backtest)
head_left, head_right = st.columns([3, 2], vertical_alignment='bottom')
with head_left:
    st.html(f"""
<div class="hdr-eyebrow">── KENYA · COUNTY FLOOD EARLY WARNING</div>
<h1 class="hdr-title">Kenya flood early warning</h1>
<p class="hdr-desc">{description}</p>
<div class="hdr-month">{header_line}</div>
""")
with head_right:
    if len(views) > 1:
        st.radio('Page', views, key='view', horizontal=True, on_change=view_changed)
    if view == BACK_VIEW:
        st.selectbox('Month', months['label'], key='month_pick', on_change=month_changed)

if len(views) == 1 and view == LIVE_VIEW:
    st.html('<div class="callout"><b>Note</b>The 2020–2024 backtest is not part of this copy of the app. Its file contains flood records, so it is not published.</div>')

# =====================================================================
# page 1: the latest forecast
# =====================================================================
if view == LIVE_VIEW:
    watch, warning = float(first['watch_threshold']), float(first['warning_threshold'])
    # the stand-ins of the forecast, as small pills (hover for the explanation)
    pills = ''
    if first['standin_flood_records']:
        pills += '<span class="pill pill-standin" title="Flood records of the data month are not available yet. The model was told that no county had a flood. You can enter floods you know about below.">Flood records: assumed none</span>'
    if first['standin_soil_moisture']:
        pills += '<span class="pill pill-standin" title="No new soil moisture data yet. The average of this calendar month in the training years is used.">Soil moisture: climatology</span>'
    if first['standin_runoff_rank']:
        pills += '<span class="pill pill-standin" title="No new runoff data yet. The middle rank (0.5) is used.">Runoff rank: neutral (0.5)</span>'
    if first['standin_discharge_rank']:
        pills += '<span class="pill pill-standin" title="No new river discharge data yet. The middle rank (0.5) is used.">River discharge rank: neutral (0.5)</span>'
    if first['rain_prelim']:
        pills += '<span class="pill pill-standin" title="The latest CHIRPS rainfall is preliminary and can change a little when the final values come out.">Rainfall: preliminary</span>'

    st.html(f"""
<div class="callout"><b>Live forecast</b>This is a real forecast, so no flood records are known yet and some inputs are stand-ins. It is less accurate than the 2020–2024 backtest: with the flood records unknown,
the model catches about 88% of the floods at Watch and 78% at Warning (98% and 90% when the records are known). Many alerts are false alarms.
<div class="standins">{pills}</div></div>
""")

    # optional manual entry: the risk is taken from the column of the forecast file that was calculated with a flood recorded
    with st.expander('Optional: enter flood records you know about (manual override)'):
        st.markdown(f'The model uses "a flood was recorded this month" as an input, and no data source reports it live. If you know of floods in {data_text}, '
                    'choose the counties here and their risk is recalculated with a flood recorded. This is **your entry, not data**. It is not saved '
                    'and it is marked on the map and in the county card.')
        manual = st.multiselect(f'Counties with a flood recorded in {data_text} (manual entry)', sorted(forecast['county']), key='override')

    data = forecast.copy()
    data['manual'] = data['county'].isin(manual)
    data['risk'] = data['risk'].where(~data['manual'], data['risk_if_flood'])
    data['level'] = [level_of(r, watch, warning) for r in data['risk']]
    data = data.sort_values('risk', ascending=False).reset_index(drop=True)
    if 'county_pick' not in st.session_state:
        st.session_state['county_pick'] = data.iloc[0]['county']

    n_warning = int((data['level'] == 'Warning').sum())
    n_watch = int((data['level'] == 'Watch').sum())
    n_none = int((data['level'] == 'No alert').sum())
    top = data.iloc[0]

    # when almost every county is on Warning, the alert level does not separate the counties, so the page says what to use instead
    if n_warning >= 0.6 * len(data):
        st.html(f"""
<div class="callout info"><b>Read the ranking</b>{n_warning} of {len(data)} counties are on Warning this month. The model reacts mostly to the very strong El Niño (ONI {top['oni']:.2f})
and to the time of the year, which are the same for every county, so the alert level hardly separates the counties. Use the risk score and the top 10 list to see where the risk is highest.</div>
""")

    st.html(f"""
<div class="strip">
<div class="cell"><span class="mono">Warning</span><div class="num warn">{n_warning}</div><div class="sub">counties, act now</div></div>
<div class="cell"><span class="mono">Watch</span><div class="num watch">{n_watch}</div><div class="sub">counties, be ready</div></div>
<div class="cell"><span class="mono">No alert</span><div class="num">{n_none}</div><div class="sub">counties</div></div>
<div class="cell"><span class="mono">Highest risk</span><div class="num">{top["risk"]:.2f}</div><div class="sub">{top["county"]}</div></div>
</div>
""")

    left, right = st.columns([3, 2])
    with left:
        with st.container(key='map_card'):
            flooded = data[data['manual']]
            fig = make_map(data, 'risk', st.session_state['county_pick'], dots=[(flooded, 'circle', 9)])
            st.plotly_chart(fig, width='stretch', on_select=partial(map_clicked, 'map_live', 'county_pick'), selection_mode='points', key='map_live',
                            config={'displayModeBar': False, 'scrollZoom': False})
            st.html("""
<div class="legend">
<span class="item"><span class="mono">Risk</span>0<span class="ramp"></span>1</span>
<span class="item"><span class="swatch red"></span>Warning</span>
<span class="item"><span class="swatch amber"></span>Watch</span>
<span class="item"><span class="swatch pick"></span>Chosen county</span>
<span class="item"><span class="dot"></span>Flood entered by hand</span>
<span class="item mono">Click a county to choose it</span>
</div>
""")

    with right:
        # the ten counties with the highest risk
        with st.container(key='top_card'):
            top10 = data.head(10)
            rows_html = ''
            for rank, row in enumerate(top10.itertuples(), start=1):
                rows_html += (f'<div class="rank-row"><span class="rank">{rank:02d}</span><span class="name">{row.county}</span>'
                              f'<span class="bar"><i style="width:{row.risk * 100:.0f}%"></i></span><span class="score">{row.risk:.2f}</span>{pill(row.level)}</div>')
            st.html(f'<div class="mono" style="margin-bottom:6px">Highest-risk counties · top 10</div>{rows_html}')

        with st.container(key='detail_card'):
            st.selectbox('County', sorted(data['county']), key='county_pick')
            county = data[data['county'] == st.session_state['county_pick']].iloc[0]
            rank_of_county = int(data.index[data['county'] == county['county']][0]) + 1
            manual_pill = '<span class="pill pill-manual">Manual entry</span>' if county['manual'] else ''
            st.html(f"""
<div class="county-row"><span class="county-name">{county['county']}</span>{pill(county['level'])}{manual_pill}</div>
<div class="risk-row"><span class="risk-num">{county['risk']:.2f}</span><span class="mono">Risk score · rank {rank_of_county} of {len(data)}</span></div>
""")
            st.plotly_chart(threshold_bar(county['risk'], county['level'], watch, warning), width='stretch', config={'displayModeBar': False})
            # the inputs of this county for the data month (there is no history of live forecasts yet)
            st.html(f"""
<div class="mono section-label">Inputs of this forecast · {data_text} data</div>
<div class="kv-row"><span class="k">Rainfall of the month</span><span class="v">{county['rain_mm']:.0f} mm · {county['rain_pct_normal']:.0%} of normal</span></div>
<div class="kv-row"><span class="k">Vegetation (NDVI)</span><span class="v">{county['ndvi_pct_normal']:.0%} of normal</span></div>
<div class="kv-row"><span class="k">Pacific ocean index (ONI)</span><span class="v">{county['oni']:.2f}</span></div>
<div class="kv-row"><span class="k">Indian Ocean index (DMI)</span><span class="v">{county['dmi']:.2f}</span></div>
<div class="small-note">The two ocean indices are the same for every county, and together with the time of the year they carry most of the forecast.</div>
""")
            if county['manual']:
                what = '<b>flood entered by hand</b> (your entry, not data)'
            else:
                what = '<b>not known yet</b> (the forecast assumes no flood)'
            st.html(f'<div class="what"><span class="mono">Flood records</span>{what}</div>')

# =====================================================================
# page 2: would it have warned us? (2020-2024)
# =====================================================================
elif view == BACK_VIEW:
    if 'county_pick' not in st.session_state:
        st.session_state['county_pick'] = riskiest_county(label)

    # the plain summary of all 60 months
    flood = preds['flood_recorded'] == 1
    alert = preds['level_live'] != 'No alert'
    n_months = len(months)
    n_floods = int(flood.sum())
    n_caught = int((flood & alert).sum())
    n_caught_warning = int((flood & (preds['level_live'] == 'Warning')).sum())
    n_missed = n_floods - n_caught
    n_alerts = int(alert.sum())
    n_false = int((alert & ~flood).sum())
    avg_alerts = n_alerts / n_months
    avg_warnings = (preds['level_live'] == 'Warning').sum() / n_months
    n_counties = preds['pcode'].nunique()

    st.html(f"""
<div class="callout info"><b>How to read this</b>This replays the live forecast on {n_months} months that the model was not trained on: same model, same Watch and Warning lines,
and the same inputs the live forecast has (the flood records of the data month are not known, the soil and river inputs are stand-ins).
For each month, the forecast is compared with the floods that were recorded the <b style="font-family:inherit;letter-spacing:0;text-transform:none;font-size:inherit;margin:0;color:inherit">next</b> month.
With the flood records known, which is not possible live, the model would catch 98% at Watch and 90% at Warning.</div>
""")
    st.html(f"""
<div class="strip">
<div class="cell"><span class="mono">Floods recorded</span><div class="num">{n_floods}</div><div class="sub">county-months with a flood, {n_months} months</div></div>
<div class="cell"><span class="mono">Warned the month before</span><div class="num good">{n_caught}</div><div class="sub">{n_caught / n_floods:.0%} had a Watch or Warning · {n_caught_warning} ({n_caught_warning / n_floods:.0%}) had a Warning</div></div>
<div class="cell"><span class="mono">Missed floods</span><div class="num warn">{n_missed}</div><div class="sub">{n_missed / n_floods:.0%} had no alert before them</div></div>
<div class="cell"><span class="mono">Alerts per month</span><div class="num watch">{avg_alerts:.0f}</div><div class="sub">of {n_counties} counties, {avg_warnings:.0f} of them Warnings</div></div>
</div>
<p class="small-note" style="margin-top:-4px">The price of catching {n_caught / n_floods:.0%} of the floods: {n_false:,} of the {n_alerts:,} alerts ({n_false / n_alerts:.0%}) were followed by no recorded flood. The flood records of the last months may be incomplete, so some false alarms may be floods that were not recorded.</p>
""")
    if not preds['live_columns'].iloc[0]:
        st.html('<div class="callout"><b>Note</b>This copy of the backtest file was made with the real flood records of the data month, so the numbers above are higher than what the live forecast would have given. Run notebook 06 again to make the live version.</div>')

    # the month that is chosen above
    data = preds[(preds['year'] == month['year']) & (preds['month'] == month['month'])].copy()
    data['risk'] = data['risk_live']
    data['level'] = data['level_live']
    data = data.sort_values('risk', ascending=False).reset_index(drop=True)
    m_floods = int((data['outcome'].isin(['Caught', 'Missed'])).sum())
    m_caught = int((data['outcome'] == 'Caught').sum())
    m_missed = int((data['outcome'] == 'Missed').sum())
    m_false = int((data['outcome'] == 'False alarm').sum())
    st.html(f"""<div class="hdr-month" style="margin:6px 0 10px">{forecast_text}: <span>{m_floods} floods recorded · {m_caught} warned the month before · {m_missed} missed · {m_false} false alarms</span></div>""")
    if not month['records_complete']:
        st.html('<div class="callout"><b>Note</b>Flood records for this month may not be complete yet.</div>')

    left, right = st.columns([3, 2])
    with left:
        with st.container(key='map_card'):
            caught_rows = data[data['outcome'] == 'Caught']
            missed_rows = data[data['outcome'] == 'Missed']
            fig = make_map(data, 'level', st.session_state['county_pick'], dots=[(caught_rows, 'circle', 10), (missed_rows, 'x', 11)])
            st.plotly_chart(fig, width='stretch', on_select=partial(map_clicked, 'map_back', 'county_pick'), selection_mode='points', key='map_back',
                            config={'displayModeBar': False, 'scrollZoom': False})
            st.html("""
<div class="legend">
<span class="item"><span class="swatch f-warning"></span>Warning</span>
<span class="item"><span class="swatch f-watch"></span>Watch</span>
<span class="item"><span class="swatch f-none"></span>No alert</span>
<span class="item"><span class="dot"></span>Flood caught (alert before it)</span>
<span class="item"><span class="cross">✕</span>Flood missed (no alert)</span>
<span class="item"><span class="swatch f-warning" style="width:10px;height:10px"></span>no mark = false alarm</span>
<span class="item"><span class="swatch pick"></span>Chosen county</span>
<span class="item mono">Click a county to choose it</span>
</div>
""")

    with right:
        with st.container(key='detail_card'):
            st.selectbox('County', sorted(data['county']), key='county_pick')
            county = data[data['county'] == st.session_state['county_pick']].iloc[0]
            st.html(f"""
<div class="county-row"><span class="county-name">{county['county']}</span>{pill(county['level'])}{pill(county['outcome'])}</div>
<div class="risk-row"><span class="risk-num">{county['risk']:.2f}</span><span class="mono">Risk score · forecast for {forecast_text}</span></div>
""")
            st.plotly_chart(threshold_bar(county['risk'], county['level'], WATCH, WARNING), width='stretch', config={'displayModeBar': False})

            # risk of this county over the last 12 forecast months, with a dot where a flood was recorded
            st.html('<div class="mono section-label">Risk over the last 12 months · dot = flood recorded</div>')
            history = preds[(preds['pcode'] == county['pcode']) & (preds['forecast_date'] <= pd.Timestamp(year=int(month['forecast_year']), month=int(month['forecast_month']), day=1))]
            history = history.sort_values('forecast_date').tail(12)
            line = risk_line(history['forecast_date'], history['risk_live'], WATCH, WARNING, '%b %y')
            line.update_traces(hovertemplate='%{x|%b %Y}: %{y:.2f}<extra></extra>')
            floods_in_history = history[history['flood_recorded'] == 1]
            line.add_trace(go.Scatter(x=floods_in_history['forecast_date'], y=floods_in_history['risk_live'], mode='markers', hoverinfo='skip',
                                      marker=dict(size=13, color='rgba(0,0,0,0)', line=dict(color='#1f2328', width=2.5))))
            st.plotly_chart(line, width='stretch', config={'displayModeBar': False})
            explain = {'Caught': 'a flood was recorded, and there was a Watch or Warning the month before',
                       'Missed': 'a flood was recorded, but there was no alert the month before',
                       'False alarm': 'there was an alert, but no flood was recorded',
                       'No flood': 'no alert and no flood was recorded'}[county['outcome']]
            st.html(f'<div class="what"><span class="mono">What happened</span><b>{county["outcome"]}</b>: {explain}</div>')

        # the floods of the month
        with st.container(key='floods_card'):
            flooded = data[data['outcome'].isin(['Caught', 'Missed'])]
            if len(flooded) == 0:
                body = '<div class="small-note">No flood was recorded in this month.</div>'
            else:
                body = ''.join(f'<div class="rank-row"><span class="name" style="flex:1">{row.county}</span><span class="score" style="width:auto">risk {row.risk:.2f}</span>{pill(row.level)}{pill(row.outcome)}</div>'
                               for row in flooded.itertuples())
            st.html(f'<div class="mono" style="margin-bottom:6px">Floods recorded in {forecast_text}</div>{body}')

# =====================================================================
# page 3: what if? (scenario explorer)
# =====================================================================
else:
    features = list(model.feature_names_in_)
    watch, warning = float(first['watch_threshold']), float(first['warning_threshold'])
    st.html(f"""
<div class="callout info"><b>Not a forecast</b>This page shows how the <b style="font-family:inherit;letter-spacing:0;text-transform:none;font-size:inherit;margin:0;color:inherit">model</b> responds when you change an input.
Everything starts from the live values of {data_text}. The sliders stay inside the range of the training years (2002 to 2019), because the model has not seen anything outside it.
All the other inputs stay fixed at their live values: rainfall two months ago, the rain of the neighbouring counties, vegetation, the month of the year, the soil and river stand-ins, the county traits, and "no flood this month".
Changing the rainfall of the month also changes the "% of normal" input, which is worked out from it.</div>
""")

    # ---------- one county ----------
    st.session_state.setdefault('wi_county', forecast.sort_values('risk', ascending=False).iloc[0]['county'])
    with st.container(key='whatif_card'):
        st.html('<div class="mono" style="margin-bottom:8px">1 · Change the inputs of one county</div>')
        st.selectbox('County', sorted(forecast['county']), key='wi_county')
        pick = st.session_state['wi_county']
        base = forecast[forecast['county'] == pick].iloc[0]

        def clipped(value, low, high):
            # the live value, moved inside the training range if it is outside
            return float(min(max(value, low), high))

        slider_keys = [f'wi_rain_{pick}', f'wi_lag_{pick}', f'wi_oni_{pick}', f'wi_dmi_{pick}']

        def reset_sliders():
            for slider_key in slider_keys:
                st.session_state.pop(slider_key, None)

        controls, result = st.columns([1, 1])
        with controls:
            # whole millimetres, moved inside the training range
            rain_low, rain_high = float(np.ceil(base['rain_min'])), float(np.floor(base['rain_max']))
            rain = st.slider('Rainfall of the month (mm)', rain_low, rain_high, clipped(base['rain_mm'], rain_low, rain_high), step=1.0, key=slider_keys[0])
            st.caption(f'Training range: {rain_low:.0f} to {rain_high:.0f} mm · live value: {base["rain_mm"]:.0f} mm')
            lag = st.slider('Rainfall of the month before (mm)', rain_low, rain_high, clipped(base['rain_lag1'], rain_low, rain_high), step=1.0, key=slider_keys[1])
            st.caption(f'Training range: {rain_low:.0f} to {rain_high:.0f} mm · live value: {base["rain_lag1"]:.0f} mm')
            oni_low, oni_high = float(base['oni_min']), float(base['oni_max'])
            oni = st.slider('Pacific ocean index (ONI)', oni_low, oni_high, clipped(base['oni'], oni_low, oni_high), step=0.01, key=slider_keys[2])
            st.caption(f'Training range: {oni_low:.2f} to {oni_high:.2f} · live value: {base["oni"]:.2f}')
            dmi_low, dmi_high = float(base['dmi_min']), float(base['dmi_max'])
            dmi = st.slider('Indian Ocean index (DMI)', dmi_low, dmi_high, clipped(base['dmi'], dmi_low, dmi_high), step=0.01, key=slider_keys[3])
            st.caption(f'Training range: {dmi_low:.2f} to {dmi_high:.2f} · live value: {base["dmi"]:.2f}')
            st.button('Reset to the live values', on_click=reset_sliders, key=f'wi_reset_{pick}')

        # the live inputs of the county, with the four changed ones (the % of normal is worked out again)
        scenario = {name: float(base[name]) for name in features}
        scenario['rain_mm'] = rain
        scenario['rain_lag1'] = lag
        scenario['rain_pct_normal'] = rain / float(base['rain_normal_mm']) if base['rain_normal_mm'] > 0 else np.nan
        scenario['oni'] = oni
        scenario['dmi'] = dmi
        scenario_risk = float(model.predict_proba(pd.DataFrame([scenario])[features])[:, 1][0])
        scenario_level = level_of(scenario_risk, watch, warning)
        live_risk = float(base['risk'])
        change = scenario_risk - live_risk
        change_text = 'no change' if abs(change) < 0.005 else f'{change:+.2f}'
        with result:
            st.html(f"""
<div class="county-row"><span class="county-name">{pick}</span>{pill(scenario_level)}</div>
<div class="risk-row"><span class="risk-num">{scenario_risk:.2f}</span><span class="mono">Model risk with these inputs</span></div>
<div class="small-note">Live forecast: {live_risk:.2f} ({base['level']}) · {change_text}</div>
""")
            st.plotly_chart(threshold_bar(scenario_risk, scenario_level, watch, warning), width='stretch', config={'displayModeBar': False})
            # how the risk of this county changes with El Nino strength, the other sliders as set
            st.html('<div class="mono section-label">Risk of this county as ONI changes (other sliders as set)</div>')
            grid = np.linspace(oni_low, oni_high, 41)
            curve = pd.DataFrame([scenario] * len(grid))
            curve['oni'] = grid
            curve_risk = model.predict_proba(curve[features])[:, 1]
            curve_line = risk_line(grid, curve_risk, watch, warning, '.1f', height=220)
            curve_line.update_traces(mode='lines', hovertemplate='ONI %{x:.2f}: %{y:.2f}<extra></extra>')
            curve_line.add_trace(go.Scatter(x=[oni], y=[scenario_risk], mode='markers', hoverinfo='skip', marker=dict(size=12, color=RED, line=dict(color='white', width=2))))
            st.plotly_chart(curve_line, width='stretch', config={'displayModeBar': False})

    # ---------- the whole country ----------
    st.html('<div style="height:14px"></div>')
    all_low, all_high = float(forecast['oni_min'].iloc[0]), float(forecast['oni_max'].iloc[0])
    live_oni = float(first['oni'])

    def reset_national():
        st.session_state.pop('wi_oni_all', None)

    # the control comes first (on a phone it is above the map), the map is on its right
    controls_column, map_column = st.columns([2, 3])
    with controls_column:
        with st.container(key='detail_card'):
            st.html('<div class="mono" style="margin-bottom:8px">2 · Change El Niño strength for the whole country</div>')
            oni_all = st.slider('Pacific ocean index (ONI), all counties', all_low, all_high, clipped(live_oni, all_low, all_high), step=0.01, key='wi_oni_all')
            st.caption(f'Training range: {all_low:.2f} to {all_high:.2f} · live value: {live_oni:.2f}')
            st.button('Reset to the live value', on_click=reset_national, key='wi_reset_all')
            # every county with its live inputs, and only the ONI changed
            everyone = forecast.copy()
            everyone['oni'] = oni_all
            everyone['risk'] = model.predict_proba(everyone[features])[:, 1]
            everyone['level'] = [level_of(r, watch, warning) for r in everyone['risk']]
            live_warning = int((forecast['level'] == 'Warning').sum())
            now_warning = int((everyone['level'] == 'Warning').sum())
            now_watch = int((everyone['level'] == 'Watch').sum())
            now_none = int((everyone['level'] == 'No alert').sum())
            st.html(f"""
<div class="strip" style="margin-top:12px">
<div class="cell"><span class="mono">Warning</span><div class="num warn">{now_warning}</div><div class="sub">live: {live_warning}</div></div>
<div class="cell"><span class="mono">Watch</span><div class="num watch">{now_watch}</div><div class="sub">counties</div></div>
<div class="cell"><span class="mono">No alert</span><div class="num">{now_none}</div><div class="sub">counties</div></div>
<div class="cell"><span class="mono">Average risk</span><div class="num">{everyone["risk"].mean():.2f}</div><div class="sub">live: {forecast["risk"].mean():.2f}</div></div>
</div>
<div class="small-note">Only the ONI changes, every county keeps its other live inputs. This is the response of the model, not a forecast of what ONI will do.</div>
""")
    with map_column:
        with st.container(key='map_card'):
            fig = make_map(everyone, 'risk', st.session_state['wi_county'])
            st.plotly_chart(fig, width='stretch', on_select=partial(map_clicked, 'map_whatif', 'wi_county'), selection_mode='points', key='map_whatif',
                            config={'displayModeBar': False, 'scrollZoom': False})
            st.html("""
<div class="legend">
<span class="item"><span class="mono">Risk</span>0<span class="ramp"></span>1</span>
<span class="item"><span class="swatch red"></span>Warning</span>
<span class="item"><span class="swatch amber"></span>Watch</span>
<span class="item"><span class="swatch pick"></span>County of section 1</span>
<span class="item mono">Click a county to choose it</span>
</div>
""")

# =====================================================================
# how this works
# =====================================================================
with st.expander('How this works'):
    st.markdown("""
- The forecast comes from an **XGBoost** model trained on the years 2002 to 2019, with cross-validation that keeps whole years together.
- **Watch** ("be ready") catches about **80%** of the floods and **Warning** ("act now") about **70%**, measured in cross-validation. A Warning county is always also above the Watch line.
- The **risk score** is the model's probability of a recorded flood in the county next month. The alert level comes from two lines on that score.
- Many alerts are **false alarms**. This is the price of missing fewer floods.
- The forecast is for the **whole county**, not for a specific place in it.
- Data: CHIRPS rainfall, NDVI vegetation, ERA5-Land soil moisture and runoff, GloFAS river discharge, the ONI and DMI ocean climate indices, and flood records from four sources (DesInventar, EM-DAT, Global Flood Database, Dartmouth Flood Observatory).
""")
    if view == LIVE_VIEW:
        st.markdown("""
**About the latest forecast**
- It is made by `src/update_forecast.py` from the newest rainfall and NDVI (WFP), and the newest ONI and DMI (NOAA). A GitHub Action runs it every month. The steps are the same as in the training, and they were checked against the training table.
- If a source has not published the newest month yet, the forecast is made for the newest month that all of them have.
- Four inputs have no live data yet and are **stand-ins** (the pills above the map): flood records (assumed none), soil moisture (the average of the month), and the runoff and river discharge ranks (the middle rank).
- The ocean indices, the county and the month of the year carry most of the forecast, and the first two are the same all year for all counties. In a strong El Niño with a positive Indian Ocean Dipole the risk is high almost everywhere, so the counties differ little.
- The dot on the map is only used for floods you enter by hand.
""")
    elif view == BACK_VIEW:
        st.markdown("""
**About this page**
- It is the live forecast, replayed on the months January 2020 to December 2024, which the model was not trained on.
- A **caught** flood is a county-month with a recorded flood that had a Watch or a Warning the month before. A **missed** flood had no alert. A **false alarm** is an alert that was followed by no recorded flood.
- The flood records come from four sources and are less complete for the most recent months, so some alerts counted as false alarms may be floods that were not recorded.
""")
    else:
        st.markdown("""
**About the scenario page**
- Each slider changes one input of the model, and the risk is calculated again with the same model and the same two lines. Nothing is retrained.
- The model only knows the ranges of the training years, so the sliders stop at those limits. Inside them, the answer is what the model learned, not a law of nature.
- The two ocean indices are the same for every county, so moving ONI moves the whole country together.
""")
