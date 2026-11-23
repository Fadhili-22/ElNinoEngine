"""Entry point of the flood app. Start it with:  streamlit run app.py   (or app/run_app.ps1)

This file does only three things:
  1. page settings and the shared style,
  2. list the four pages,
  3. show the menu at the top and run the page that was chosen.
The pages themselves are in the folder views/. Shared helpers are in common.py.
"""
import streamlit as st

st.set_page_config(page_title='Kenya flood early warning', page_icon='🌊', layout='wide')

from common import inject_css  # noqa: E402  (after set_page_config, which must be the first streamlit call)

inject_css()

# the four pages. The first one is the landing page. url_path is the end of the web address of the page.
pages = [
    st.Page('views/home.py', title='Home', default=True),
    st.Page('views/live.py', title='Live forecast', url_path='live'),
    st.Page('views/scenario.py', title='Try a scenario', url_path='scenario'),
    st.Page('views/accuracy.py', title='How accurate is it?', url_path='accuracy'),
]

# The menu bar is built by us (see below) so that its links wrap on a phone. The built-in top menu of streamlit
# folds into a sidebar button on narrow screens. position='hidden' keeps the page routing but draws no menu.
current_page = st.navigation(pages, position='hidden')

# the menu bar: the name on the left, the four links on the right. A horizontal container wraps its links when the screen is narrow.
with st.container(horizontal=True, vertical_alignment='center', key='topbar'):
    st.html('<span class="brand"><span class="brand-mark">◆</span> Kenya Flood Early Warning</span>')
    for page in pages:
        st.page_link(page, label=page.title)

current_page.run()
