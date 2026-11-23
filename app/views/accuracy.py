"""Page 4: How accurate is it? The 2020-2024 backtest. Technical language is fine here.

The backtest file (made by notebook 06) holds, for every county and month of 2020-2024, the risk score the live forecast WOULD have given
(columns risk_live / level_live) and the flood that was recorded the next month. Every number on this page is counted from that file.
"""
from functools import partial
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from common import (eyebrow, make_outcome_map, map_clicked, risk_line, get_backtest, backtest_numbers, month_text,
                    WATCH, WARNING, SETTINGS, INK)

preds = get_backtest()
if preds is None:
    st.html(f'{eyebrow("BACKTEST · 2020–2024")}<h1 class="h1">How accurate is it?</h1>')
    st.info('The 2020–2024 backtest is not part of this copy of the app. Its file contains flood records, so it is not published.')
    st.stop()

numbers = backtest_numbers(preds)

# the 60 months, oldest first. The label is the FORECAST month ("May 2024" = the forecast made from April 2024 data).
months = preds[['forecast_year', 'forecast_month', 'year', 'month', 'records_complete']].drop_duplicates()
months = months.sort_values(['forecast_year', 'forecast_month']).reset_index(drop=True)
months['label'] = [month_text(y, m, short=True) for y, m in zip(months['forecast_year'], months['forecast_month'])]
labels = list(months['label'])
default_label = 'May 2024' if 'May 2024' in labels else labels[-1]
st.session_state.setdefault('bt_month', default_label)

BADGE_CLASS = {'Warned in time': 'ok', 'Missed': 'missed', 'Warned but no flood': 'warned-nf', 'Quiet': 'quiet'}


def outcome_badge(outcome):
    return f'<span class="badge badge-{BADGE_CLASS[outcome]}">{outcome.upper()}</span>'


def step_month(change):
    # the previous / next buttons move the slider by one month
    position = labels.index(st.session_state['bt_month']) + change
    st.session_state['bt_month'] = labels[min(max(position, 0), len(labels) - 1)]


# ---------------------------------------------------------------------
# 1. title
# ---------------------------------------------------------------------
st.html(f"""
{eyebrow('BACKTEST · 2020–2024')}
<h1 class="h1">Would it have warned us? We replayed five years it had never seen.</h1>
<p class="lead">Same model, same alert lines and the same inputs the live forecast gets, run month by month on 2020–2024 and compared with the floods recorded the next month.</p>
""")

# ---------------------------------------------------------------------
# 2. three headline numbers
# ---------------------------------------------------------------------
share_warned = numbers['warned'] / numbers['floods']
share_missed = numbers['missed'] / numbers['floods']
share_false = numbers['false_alarms'] / numbers['alerts']
st.html(f"""
<div class="strip three">
<div class="cell"><span class="mono">Floods warned in time</span><div class="num big">{share_warned:.0%}</div><div class="sub">{numbers['warned']} of {numbers['floods']} floods had a Medium or High alert the month before</div></div>
<div class="cell"><span class="mono">Floods missed</span><div class="num big">{share_missed:.0%}</div><div class="sub">{numbers['missed']} floods had no alert</div></div>
<div class="cell"><span class="mono">Alerts that were false alarms</span><div class="num big">{share_false:.0%}</div><div class="sub">{numbers['false_alarms']:,} of {numbers['alerts']:,} alerts were followed by no recorded flood</div></div>
</div>
<p class="note" style="margin-top:-4px">Recent flood records may be incomplete, so some false alarms may be floods that were never recorded.</p>
""")

# ---------------------------------------------------------------------
# 3. month by month
# ---------------------------------------------------------------------
st.html('<div class="h2">Month by month</div>')
st.select_slider('Forecast month', options=labels, key='bt_month')
with st.container(horizontal=True):
    st.button('← Previous month', on_click=step_month, args=(-1,), key='bt_prev')
    st.button('Next month →', on_click=step_month, args=(1,), key='bt_next')

chosen = months[months['label'] == st.session_state['bt_month']].iloc[0]
month_data = preds[(preds['forecast_year'] == chosen['forecast_year']) & (preds['forecast_month'] == chosen['forecast_month'])].copy()
month_data = month_data.sort_values('risk_live', ascending=False).reset_index(drop=True)
chosen_name = month_text(chosen['forecast_year'], chosen['forecast_month'])
counts = month_data['outcome'].value_counts()
n_floods = int(counts.get('Warned in time', 0) + counts.get('Missed', 0))
st.html(f"""<div class="mono" style="margin:10px 0 6px;color:#1F2328">{chosen_name}: {n_floods} floods · {int(counts.get('Warned in time', 0))} warned in time · {int(counts.get('Missed', 0))} missed · {int(counts.get('Warned but no flood', 0))} false alarms</div>""")
if not chosen['records_complete']:
    st.html('<div class="callout warn"><b class="label">Note</b>Flood records for this month may not be complete yet.</div>')

# the county of the panel. First visit: the county with the highest score of the chosen month.
st.session_state.setdefault('bt_county', month_data.iloc[0]['county'])

# ---------------------------------------------------------------------
# 4 and 5. map by outcome, and the panel of one county
# ---------------------------------------------------------------------
map_column, panel_column = st.columns([3, 2])
with map_column:
    with st.container(key='card_map'):
        figure = make_outcome_map(month_data, st.session_state['bt_county'])
        st.plotly_chart(figure, width='stretch', on_select=partial(map_clicked, 'bt_map', 'bt_county'), selection_mode='points', key='bt_map',
                        config={'displayModeBar': False, 'scrollZoom': False})
        st.html("""<div class="legend">
<span class="item"><span class="swatch o-ok"></span>Warned in time</span>
<span class="item"><span class="swatch o-missed"></span>Missed</span>
<span class="item"><span class="swatch o-warned"></span>Warned but no flood</span>
<span class="item"><span class="swatch o-quiet"></span>Quiet</span>
</div><div class="note" style="margin-top:6px">Click a county to see its history.</div>""")

with panel_column:
    with st.container(key='card_panel'):
        st.selectbox('County', sorted(month_data['county']), key='bt_county')
        row = month_data[month_data['county'] == st.session_state['bt_county']].iloc[0]
        sentence = {
            'Warned in time': f'A flood was recorded in {row["county"]} in {chosen_name}, and there was a Medium or High alert the month before.',
            'Missed': f'A flood was recorded in {row["county"]} in {chosen_name}, but there was no alert the month before.',
            'Warned but no flood': f'There was an alert for {row["county"]} before {chosen_name}, but no flood was recorded.',
            'Quiet': f'No alert for {row["county"]} and no flood was recorded in {chosen_name}.',
        }[row['outcome']]
        st.html(f"""
<div class="county-row"><span class="county-name">{row['county']}</span>{outcome_badge(row['outcome'])}</div>
<div class="result-sentence" style="font-weight:500">{sentence}</div>
<div class="mono" style="margin-top:14px">Risk score over the last 12 months · ring = flood recorded</div>
""")
        # the risk score of this county in the 12 forecast months up to the chosen one
        history = preds[(preds['pcode'] == row['pcode']) & (preds['forecast_date'] <= pd.Timestamp(year=int(chosen['forecast_year']), month=int(chosen['forecast_month']), day=1))]
        history = history.sort_values('forecast_date').tail(12)
        line = risk_line(history['forecast_date'], history['risk_live'], '%b %y')
        line.update_traces(hovertemplate='%{x|%b %Y}: %{y:.2f}<extra></extra>')
        flood_months = history[history['flood_recorded'] == 1]
        line.add_trace(go.Scatter(x=flood_months['forecast_date'], y=flood_months['risk_live'], mode='markers', hoverinfo='skip',
                                  marker=dict(size=13, color='rgba(0,0,0,0)', line=dict(color=INK, width=2.5))))
        st.plotly_chart(line, width='stretch', config={'displayModeBar': False})

# ---------------------------------------------------------------------
# 6. all floods of the month (an expander is fine here)
# ---------------------------------------------------------------------
with st.expander(f'All floods in {chosen_name}'):
    flooded = month_data[month_data['outcome'].isin(['Warned in time', 'Missed'])]
    if len(flooded) == 0:
        st.write('No flood was recorded in this month.')
    else:
        rows_html = ''.join(f'<div class="rank-row"><span class="name" style="flex:1">{r.county}</span>{outcome_badge(r.outcome)}</div>' for r in flooded.itertuples())
        st.html(rows_html)

# ---------------------------------------------------------------------
# 7. how this works (a normal section, not a dropdown)
# ---------------------------------------------------------------------
cv_watch = float(SETTINGS['oof_recall'])
cv_warning = float(SETTINGS['warning_oof_recall'])
known_watch = numbers['warned_known'] / numbers['floods']
known_warning = numbers['warned_warning_known'] / numbers['floods']
warning_share = numbers['warned_warning'] / numbers['floods']
st.html(f"""
<div class="h2" style="margin-top:34px">How this works</div>
<div class="card">
<ul class="todo" style="margin-top:0">
<li><b>Model.</b> XGBoost (25 inputs, including <code>flood_this_month</code>), trained on 2002–2019 and tested on 2020–2024, which it never saw. Two alert lines on its risk score: <b>Watch</b> (Medium, "be ready") at {WATCH:.4f} and <b>Warning</b> (High, "act now") at {WARNING:.4f}. They come from <code>models/xgboost_settings.json</code>.</li>
<li><b>The replay recipe.</b> For every month the model gets the inputs a live forecast would have: rainfall, vegetation and the two ocean indices of the data month, with <b>no flood records</b> (the flood of the data month is set to "no flood"), soil moisture set to the average of that month, and the runoff and river discharge ranks set to 0.5. It is then compared with the floods recorded the following month.</li>
<li><b>Three settings, three accuracy figures.</b> Do not mix them:
<ul>
<li><b>Live recipe, as on this page:</b> Watch catches {share_warned:.0%} of the floods ({numbers['warned']} of {numbers['floods']}) and Warning {warning_share:.0%} ({numbers['warned_warning']}).</li>
<li><b>With the real flood-this-month value</b> (records known, not possible live): {known_watch:.0%} at Watch and {known_warning:.0%} at Warning on the same months.</li>
<li><b>Cross-validation on 2002–2019, with the real flood-this-month value:</b> the lines were chosen to catch {cv_watch:.0%} (Watch) and {cv_warning:.0%} (Warning) of the floods. No cross-validation figure exists for the live recipe.</li>
</ul></li>
<li><b>False alarms.</b> {numbers['false_alarms']:,} of the {numbers['alerts']:,} alerts ({share_false:.0%}) were followed by no recorded flood. This is the price of catching most floods. The records of the last months are less complete, so some false alarms may be floods that were not recorded.</li>
<li><b>One strong El Niño.</b> In the test years the model put most counties at High during the 2023–24 El Niño, while few floods were recorded. The model learned strong El Niños mostly from one event (2015/16), so it is least certain in exactly this situation.</li>
<li><b>Data.</b> CHIRPS rainfall, NDVI vegetation, ERA5-Land soil moisture and runoff, GloFAS river discharge, the ONI and DMI ocean indices, and flood records from four sources (DesInventar, EM-DAT, Global Flood Database, Dartmouth Flood Observatory).</li>
<li><b>The risk score is relative, not a probability.</b> The model is tuned to catch floods, so its scores run far above the real flood frequency. Compare counties with each other and with the two lines. Do not read 0.77 as "77%".</li>
<li>The forecast covers a whole county, not a specific place in it.</li>
</ul>
</div>
""")
