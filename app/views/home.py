"""Page 1: Home. The landing page: what the app is, a live preview and links to the other pages.

All numbers on this page are read from the forecast file and the backtest file. Nothing is typed by hand.
"""
import streamlit as st
from common import eyebrow, get_forecast, get_backtest, forecast_with_levels, backtest_numbers, month_text, MONTH_NAMES

forecast = get_forecast()
if forecast is None:
    st.error('No forecast file found. Run `python src/update_forecast.py` first.')
    st.stop()

first = forecast.iloc[0]
data = forecast_with_levels(forecast)
forecast_month = month_text(first['forecast_year'], first['forecast_month'])
forecast_month_word = MONTH_NAMES[int(first['forecast_month']) - 1]
n_counties = len(data)
n_high = int((data['level'] == 'Warning').sum())
top_three = ', '.join(data['county'].head(3))

# ---------------------------------------------------------------------
# the numbers of the backtest (None when the backtest file is not part of this copy of the app)
# ---------------------------------------------------------------------
preds = get_backtest()
if preds is not None:
    numbers = backtest_numbers(preds)
    share_warned = numbers['warned'] / numbers['floods']
    # "about 9 in 10" is the share, rounded to the nearest tenth
    about_in_ten = f'About {round(share_warned * 10)} in 10'
else:
    about_in_ten = 'See the accuracy page'

# ---------------------------------------------------------------------
# hero: eyebrow, headline, lead and the two buttons
# ---------------------------------------------------------------------
st.html(f"""
<div class="hero">
{eyebrow('KENYA · ALL 47 COUNTIES')}
<h1 class="h1">Know your county's flood risk <em>for the month ahead.</em></h1>
<p class="lead">A free forecast, updated in the first week of every month, built from rainfall, vegetation and ocean conditions.
Pick your county and see, in plain words, whether to be ready.</p>
</div>
""")

button_one, button_two, _ = st.columns([1, 1.2, 2])
with button_one:
    with st.container(key='primary_check'):
        st.page_link('views/live.py', label='Check my county', use_container_width=True)
with button_two:
    st.page_link('views/accuracy.py', label='How accurate is it?', use_container_width=True)

st.html('<div class="section-gap"></div>')

# ---------------------------------------------------------------------
# live preview card
# ---------------------------------------------------------------------
# the El Nino sentence is only added when the ocean index says it is true (ONI above 1.0 is a strong El Nino)
reason = ', mainly because of a strong El Niño' if first['oni'] > 1.0 else ''
st.html(f"""
<div class="card">
<div class="mono">Live preview · {forecast_month}</div>
<div class="h2" style="margin-top:8px">{n_high} of {n_counties} counties at high risk in {forecast_month_word}{reason}</div>
<div class="note" style="font-size:1rem">Highest risk right now: <b class="ink">{top_three}</b></div>
</div>
""")
with st.container(key='plain_map'):
    st.page_link('views/live.py', label='See the map →')

# ---------------------------------------------------------------------
# stat strip
# ---------------------------------------------------------------------
st.html(f"""
<div class="strip">
<div class="cell"><span class="mono">Forecast for</span><div class="num">{forecast_month}</div></div>
<div class="cell"><span class="mono">Covers</span><div class="num">All {n_counties} counties</div></div>
<div class="cell"><span class="mono">Past floods warned</span><div class="num">{about_in_ten}</div><div class="sub">floods, in the 2020–2024 replay</div></div>
<div class="cell"><span class="mono">Updated</span><div class="num">Early each month</div><div class="sub">last: {first['generated_on']}</div></div>
</div>
""")

# ---------------------------------------------------------------------
# three ways to use it
# ---------------------------------------------------------------------
st.html('<div class="h2">Three ways to use it</div>')
way_one, way_two, way_three = st.columns(3)
with way_one:
    with st.container(key='card_tint_way1'):
        st.html('<div class="way-num">01 <span class="start-badge">START HERE</span></div><div class="h3" style="margin-top:10px">Live forecast</div>'
                f'<div class="note">See whether your county is at risk in {forecast_month_word}, and what you can do about it.</div>')
        st.page_link('views/live.py', label='Open the live forecast →')
with way_two:
    with st.container(key='card_way2'):
        st.html('<div class="way-num">02</div><div class="h3" style="margin-top:10px">Try a scenario</div>'
                '<div class="note">Change the rain, the ocean conditions or a flood this month, and see how the forecast reacts.</div>')
        st.page_link('views/scenario.py', label='Try a scenario →')
with way_three:
    with st.container(key='card_way3'):
        st.html('<div class="way-num">03</div><div class="h3" style="margin-top:10px">How accurate is it?</div>'
                '<div class="note">We replayed five years the model had never seen and counted what it warned and what it missed.</div>')
        st.page_link('views/accuracy.py', label='See the replay →')

# ---------------------------------------------------------------------
# read this first
# ---------------------------------------------------------------------
read_first = ('This is an early warning, not a promise. '
              + ('It warned before ' + about_in_ten.lower() + ' past floods, ' if preds is not None else '')
              + 'but most of its warnings were not followed by a flood. It covers a whole county, not your street. '
              'Always follow official alerts from the Kenya Meteorological Department and your county government.')
st.html(f'<div class="callout warn" style="margin-top:18px"><b class="label">Read this first</b>{read_first}</div>')

# ---------------------------------------------------------------------
# footer
# ---------------------------------------------------------------------
st.html(f"""<div class="footer">Moringa School · Data Science capstone · Group 6 · Forecast updated {first['generated_on']}</div>""")
