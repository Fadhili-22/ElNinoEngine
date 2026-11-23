"""Page 2: Live forecast. For everyone: plain words, no raw scores, no jargon.
The technical details (scores, thresholds, inputs, stand-ins, manual override) are in the last section of the page.
"""
from functools import partial
import streamlit as st
from common import (eyebrow, badge, level_bar, level_sentence, legend_risk, rank_rows, make_risk_map, map_clicked, get_forecast, get_backtest,
                    backtest_numbers, forecast_with_levels, month_text, MONTH_NAMES, LEVEL_INFO, WATCH, WARNING)

forecast = get_forecast()
if forecast is None:
    st.error('No forecast file found. Run `python src/update_forecast.py` first.')
    st.stop()

first = forecast.iloc[0]
forecast_month = month_text(first['forecast_year'], first['forecast_month'])
forecast_month_word = MONTH_NAMES[int(first['forecast_month']) - 1]
data_month_word = MONTH_NAMES[int(first['data_month']) - 1]

# ---------------------------------------------------------------------
# the counties the visitor entered by hand (expert section at the bottom). The widget is drawn at the end of the page,
# but its value is already in session_state when the page runs again, so the map and the card can use it here.
# ---------------------------------------------------------------------
manual_counties = st.session_state.get('live_override', [])
data = forecast_with_levels(forecast, manual_counties)
n_counties = len(data)
n_high = int((data['level'] == 'Warning').sum())

# the county of the select box (the map click writes into the same key). The first visit shows the highest-risk county.
st.session_state.setdefault('live_county', data.iloc[0]['county'])


# ---------------------------------------------------------------------
# small helpers of this page
# ---------------------------------------------------------------------
def rain_words(share_of_normal):
    # the rain of the month in plain words
    if share_of_normal < 0.7:
        return 'much drier than usual'
    if share_of_normal < 0.9:
        return 'a little drier than usual'
    if share_of_normal <= 1.1:
        return 'normal'
    if share_of_normal <= 1.3:
        return 'a little wetter than usual'
    return 'much wetter than usual'


def why_paragraphs(county_row):
    # the "Why?" text, written from the data of this month (ocean index, number of high-risk counties, rain of the county)
    oni = float(first['oni'])
    paragraphs = []
    if oni > 1.0:
        tail = (f'which is why {n_high} of {n_counties} counties are high risk this month.' if n_high >= 0.6 * n_counties
                else f'and {n_high} of {n_counties} counties are high risk this month.')
        paragraphs.append('A strong El Niño is under way. The Pacific Ocean is much warmer than usual, which in Kenya tends to bring heavier short rains '
                          f'from October to December. It affects the whole country at once, {tail}')
        if oni >= 1.5:
            paragraphs.append('El Niño this strong has only happened once in the years the model learned from (2015/16), so treat this as a strong caution, not a certainty.')
    elif oni > 0.5:
        paragraphs.append('A weak El Niño is under way: the Pacific Ocean is a little warmer than usual. Its effect on the forecast is small, '
                          'so the time of the year and the history of each county matter more this month.')
    elif oni >= -0.5:
        paragraphs.append('The Pacific Ocean is close to normal, so there is no strong El Niño or La Niña pushing the forecast. '
                          'This month the forecast mostly reflects the time of the year and the flood history of each county.')
    else:
        paragraphs.append('A La Niña is under way: the Pacific Ocean is cooler than usual, which in Kenya tends to bring drier short rains. '
                          'It affects the whole country at once, so the time of the year and the history of each county matter more.')
    if oni <= 1.0:
        rate = float(county_row['county_flood_rate'])
        paragraphs.append(f'In the years the model learned from (2002 to 2019), a flood was recorded in {county_row["county"]} in about '
                          f'{round(rate * 100)} of every 100 months.')
    share = float(county_row['rain_pct_normal'])
    paragraphs.append(f'Rainfall in {county_row["county"]} last month ({data_month_word}) was {rain_words(share)} ({share:.0%} of usual).')
    return paragraphs


# ---------------------------------------------------------------------
# 1. title
# ---------------------------------------------------------------------
st.html(f"""
{eyebrow('LIVE FORECAST · ' + forecast_month.upper())}
<h1 class="h1">Is my county at risk in {forecast_month_word}?</h1>
""")

# ---------------------------------------------------------------------
# 2. choose the county
# ---------------------------------------------------------------------
st.selectbox('Choose your county', sorted(data['county']), key='live_county')
county = data[data['county'] == st.session_state['live_county']].iloc[0]
level = county['level']
info = LEVEL_INFO[level]

# ---------------------------------------------------------------------
# 3. result card and what you can do
# ---------------------------------------------------------------------
manual_badge = '<span class="badge badge-manual">Flood entered by hand</span>' if county['manual'] else ''
if level == 'No alert':
    todo = '<p class="note" style="font-size:1rem;margin-top:12px"><b class="ink">No action needed now.</b> Keep an eye on weather updates.</p>'
else:
    todo = """<div class="h3" style="margin-top:14px">What you can do</div>
<ul class="todo">
<li>Follow Kenya Meteorological Department updates.</li>
<li>Clear drains and gutters near your home.</li>
<li>Know how you would leave if water rises, and where you would go.</li>
<li>Keep your phone charged and your papers in a waterproof bag.</li>
</ul>"""

with st.container(key='card_result'):
    st.html(f"""
<div class="county-row"><span class="county-name">{county['county']}</span>{badge(level)}{manual_badge}</div>
<div class="result-sentence">{level_sentence(level, forecast_month_word)}</div>
{level_bar(level)}
<div class="note">Ranked {int(county['rank'])} of {n_counties} counties (1 = highest risk)</div>
{todo}
""")

# ---------------------------------------------------------------------
# 4. why is it high (or medium, or low)?
# ---------------------------------------------------------------------
title = {'Warning': 'Why so high?', 'Watch': 'Why medium?', 'No alert': 'Why low?'}[level]
paragraphs = ''.join(f'<p style="margin:0 0 10px;line-height:1.6">{text}</p>' for text in why_paragraphs(county))
st.html(f'<div class="card" style="margin-top:12px"><div class="h3">{title}</div>{paragraphs}</div>')

# ---------------------------------------------------------------------
# 5. map and top 10
# ---------------------------------------------------------------------
st.html('<div class="section-gap"></div>')
map_column, top_column = st.columns([3, 2])
with map_column:
    with st.container(key='card_map'):
        figure = make_risk_map(data, st.session_state['live_county'])
        st.plotly_chart(figure, width='stretch', on_select=partial(map_clicked, 'live_map', 'live_county'), selection_mode='points',
                        key='live_map', config={'displayModeBar': False, 'scrollZoom': False})
        extra = '<span class="item"><span class="dot"></span>Flood entered by hand</span>' if len(manual_counties) > 0 else ''
        st.html(legend_risk(extra) + '<div class="note" style="margin-top:6px">Click a county to choose it.</div>')
with top_column:
    with st.container(key='card_top10'):
        st.html(f'<div class="mono" style="margin-bottom:6px">Highest risk this month</div>{rank_rows(data, 10)}')

# ---------------------------------------------------------------------
# 6. one caveat line
# ---------------------------------------------------------------------
st.html('<div class="section-gap"></div>')
st.html('<div class="callout warn"><b class="label">Good to know</b>This is an early warning, not a promise. Most past warnings were not followed by a flood, '
        'and the forecast covers a whole county, not one street.</div>')
st.page_link('views/accuracy.py', label='How accurate is it? →')

# ---------------------------------------------------------------------
# 7. details for experts (a normal section, not a dropdown)
# ---------------------------------------------------------------------
st.html('<div class="h2" style="margin-top:34px">Details for experts</div>')

# the numbers of the chosen county
st.html(f"""
<div class="card">
<div class="h3">{county['county']}: risk score and inputs</div>
<div class="kv-row"><span class="k">Risk score (relative, not a probability)</span><span class="v">{county['risk']:.3f}</span></div>
<div class="kv-row"><span class="k">Rank</span><span class="v">{int(county['rank'])} of {n_counties}</span></div>
<div class="kv-row"><span class="k">Watch line (Medium) / Warning line (High)</span><span class="v">{WATCH:.4f} / {WARNING:.4f}</span></div>
<div class="kv-row"><span class="k">Rainfall of {data_month_word}</span><span class="v">{county['rain_mm']:.0f} mm · {county['rain_pct_normal']:.0%} of normal</span></div>
<div class="kv-row"><span class="k">Vegetation (NDVI)</span><span class="v">{county['ndvi_pct_normal']:.0%} of normal</span></div>
<div class="kv-row"><span class="k">Pacific Ocean index (ONI)</span><span class="v">{county['oni']:.2f}</span></div>
<div class="kv-row"><span class="k">Indian Ocean index (DMI)</span><span class="v">{county['dmi']:.2f}</span></div>
<p class="note" style="margin:10px 0 0">The risk score is not a probability: the model is tuned to catch floods, so its scores run far above the real flood frequency. Use it only to compare counties and to place them against the two lines.
The two ocean indices are the same for every county. On this month's data the ocean index (ONI) is the largest single driver (about 40% of the effect).</p>
</div>
""")

# the stand-ins: inputs of the model that have no live data yet
standins = []
if first['standin_flood_records']:
    standins.append('Flood records of the data month: assumed none (no source reports them live)')
if first['standin_soil_moisture']:
    standins.append('Soil moisture: the average of this calendar month in the training years (climatology)')
if first['standin_runoff_rank']:
    standins.append('Runoff rank: 0.5 (neutral)')
if first['standin_discharge_rank']:
    standins.append('River discharge rank: 0.5 (neutral)')
if first['rain_prelim']:
    standins.append('Rainfall: preliminary CHIRPS values, they can change a little when the final values come out')
pills = ''.join(f'<span class="pill">{text}</span>' for text in standins)
st.html(f'<div class="card"><div class="h3">Stand-ins in this forecast</div><div class="note">These inputs are not real data for {data_month_word}.</div><div class="pills">{pills}</div></div>')

# the live accuracy (counted from the backtest file)
preds = get_backtest()
if preds is not None:
    numbers = backtest_numbers(preds)
    watch_share = numbers['warned'] / numbers['floods']
    warning_share = numbers['warned_warning'] / numbers['floods']
    known_watch = numbers['warned_known'] / numbers['floods']
    known_warning = numbers['warned_warning_known'] / numbers['floods']
    st.html(f"""
<div class="card"><div class="h3">Accuracy of a live forecast like this one</div>
<p style="margin:0 0 8px;line-height:1.6">With the flood records of the data month unknown (as now), the replay of 2020–2024 gave a Watch or Warning before <b>{watch_share:.0%}</b> of the floods ({numbers['warned']} of {numbers['floods']}) at the Watch line
and before <b>{warning_share:.0%}</b> ({numbers['warned_warning']} of {numbers['floods']}) at the Warning line. With the flood records known, which is not possible live, it would be {known_watch:.0%} and {known_warning:.0%}.
Most alerts were false alarms. See the accuracy page.</p></div>
""")

# the manual override
with st.container(key='card_override'):
    st.html('<div class="h3">Manual override: floods you know about</div>')
    st.markdown(f'The model uses "a flood was recorded this month" as an input, and no data source reports it live, so the forecast assumes none. '
                f'If you know of floods in {data_month_word}, choose the counties here and their risk score is recalculated with a flood recorded '
                '(the `risk_if_flood` column of the forecast file). This is **your entry, not data**. It is not saved. Counties you enter are marked on the map and on their card.')
    st.multiselect(f'Counties with a flood recorded in {data_month_word} (manual entry)', sorted(forecast['county']), key='live_override')

# all counties as a table
table = data[['rank', 'county', 'risk', 'level']].rename(columns={'rank': 'Rank', 'county': 'County', 'risk': 'Risk score', 'level': 'Level'})
table['Level'] = table['Level'].map(lambda name: LEVEL_INFO[name]['name'])
st.html('<div class="h3" style="margin-top:16px">All counties</div>')
st.dataframe(table, hide_index=True, width='stretch', height=320)
