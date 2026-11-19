import os
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.docstore.document import Document

# --- PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Kenya Flood Early Warning", page_icon="🌊", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* --- SIDEBAR FIX --- */
    [data-testid="stSidebar"] {
        background-color: #0F4C81;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #0F4C81 0%, #1E88E5 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
        color: white !important;
    }
    .hero-subtitle {
        font-size: 18px;
        opacity: 0.9;
        color: white !important;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #0F4C81;
    }
    div[data-testid="metric-container"] label {
        color: #6c757d !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0F4C81 !important;
    }
    
    /* Risk Cards */
    .risk-card-high {
        padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #FF4B4B 0%, #D32F2F 100%);
        color: white; text-align: center; font-weight: bold; font-size: 28px; box-shadow: 0 6px 20px rgba(255,75,75,0.3);
    }
    .risk-card-mod {
        padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #FFA500 0%, #F57C00 100%);
        color: white; text-align: center; font-weight: bold; font-size: 28px; box-shadow: 0 6px 20px rgba(255,165,0,0.3);
    }
    .risk-card-low {
        padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%);
        color: white; text-align: center; font-weight: bold; font-size: 28px; box-shadow: 0 6px 20px rgba(76,175,80,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# --- TRANSLATION DICTIONARY ---
TEXTS = {
    "English": {
        "title": "🌊 Kenya Flood Early Warning System",
        "subtitle": "Predict next-month flood risk and access AI-powered safety protocols.",
        "tab1": "📊 Historical Insights",
        "tab2": "🚨 Risk Predictor",
        "tab3": "🤖 AI Assistant",
        "header1": "📍 Historical Flooding Across Kenya",
        "header2": "Check Flood Risk for a County",
        "header3": "🤖 AI Safety & Preparedness Assistant",
        "risk_extreme": "🚨 EXTREME RISK",
        "risk_high": "⚠️ HIGH RISK",
        "risk_mod": "⚠️ MODERATE RISK",
        "risk_low": "✅ LOW RISK",
        "risk_action_extreme": "Evacuate low-lying areas immediately.",
        "risk_action_high": "Prepare to evacuate. Stock supplies.",
        "risk_action_mod": "Monitor closely. Clear drainage.",
        "risk_action_low": "No immediate action required.",
        "btn_predict": "🔍 Predict Flood Risk",
        "select_county": "Select County",
        "flood_now": "Currently experiencing flooding?",
        "flood_last": "Was flooding last month?",
        "rain": "How heavy is the rain this month?",
        "chat_placeholder": "Ask a question...",
        "hist_chart_title": "Top 15 Most Flood-Prone Counties",
        "hist_info_header": "🚨 High-Risk Areas",
        "metric_total": "Total Flood Records (2002-2024)",
        "metric_counties": "Counties Impacted",
        "rain_levels": ["No rain", "Light rain", "Moderate rain", "Heavy rain", "Extreme rain"],
        "flooding_now_msg": "🚨 Since the county is currently flooding, weather conditions are assumed critical. The model will prioritize immediate flood status.",
        "overview_title": "ℹ️ What does this app do? (Click to expand)",
        "overview_intro": "Welcome to the **Kenya Flood Early Warning System**. This tool uses artificial intelligence and 20 years of historical weather data to predict where floods are most likely to happen next month.",
        "overview_how": "**How to use this app:**",
        "overview_tab1": "📊 **Historical Insights:** See which counties have flooded the most in the past. Understand *when* floods usually happen (seasonality) and *how severe* they can be.",
        "overview_tab2": "🚨 **Risk Predictor:** Select a county, answer two simple questions about the current weather and flood status, and get an immediate risk assessment with safety advice.",
        "overview_tab3": "🤖 **AI Assistant:** Chat with our AI bot. You can ask it questions like *'Which counties are the most dangerous?'* or *'How do I purify water?'* It knows all the official safety protocols.",
        "overview_note": "*Note: This is a decision-support tool. Always follow official guidance from the Kenya Meteorological Department (KMD) and the National Disaster Management Authority (NDMA).*",
        "kpi_roc": "Model Accuracy (ROC-AUC)",
        "kpi_pr": "Rare Event Detection (PR-AUC)",
        "kpi_counties": "Counties Monitored",
        "kpi_records": "Historical Records",
        "chart_season": "Seasonality: Flood Incidents by Month",
        "chart_severity": "Severity: Floods vs. Deaths",
        "top5_header": "**Top 5 Most Affected**",
        "data_source": "Data sources: DesInventar and EM-DAT.",
        "clear_chat": "🗑️ Clear Chat",
        "weather_info": "💡 The model automatically factors in ocean climate indices (ONI & DMI) based on the latest available data for this county.",
        "action_header": "**What to do:**",
        "action_high": "Move to higher ground immediately. Contact local authorities. Do not attempt to cross flooded roads.",
        "action_mod": "Prepare an emergency kit. Monitor local radio for updates. Clear drainage around your home.",
        "action_low": "No immediate action required. Stay informed by checking local weather reports.",
        "context_higher": "📈 This risk level is **significantly higher** than the historical average for {}.",
        "context_above": "📊 This risk level is slightly above the historical average for {}.",
        "context_below": "📉 This risk level is below the historical average for {}.",
        "footer": "This is a decision-support tool. Always follow official KMD and NDMA guidance.",
        "initial_chat": "Habari! 👋 I can predict flood risks and provide safety protocols. Try asking:\n- *Which counties are historically the most dangerous?*\n- *What should I do if the risk is HIGH?*\n- *How do I purify water during a flood?*"
    },
    "Kiswahili": {
        "title": "🌊 Mfumo wa Tahadhari ya Mafuriko Kenya",
        "subtitle": "Tabiri hatari ya mafuriko mwezi ujao na upate ushauri wa usalama kupitia AI.",
        "tab1": "📊 Historia ya Mafuriko",
        "tab2": "🚨 Kipimo cha Hatari",
        "tab3": "🤖 Msaidizi wa AI",
        "header1": "📍 Historia ya Mafuriko Kote Kenya",
        "header2": "Angalia Hatari ya Mafuriko kwa Kaunti",
        "header3": "🤖 Msaidizi wa Usalama na Maandalizi ya AI",
        "risk_extreme": "🚨 HATARI KUBWA",
        "risk_high": "⚠️ HATARI KUBWA",
        "risk_mod": "⚠️ HATARI YA WASTANI",
        "risk_low": "✅ HATARI NDOGO",
        "risk_action_extreme": "Hamisha watu mara moja kutoka maeneo ya chini.",
        "risk_action_high": "Jiandae kuhamia. Hifadhi vifaa muhimu.",
        "risk_action_mod": "Fuatilia kwa karibu. Safisha mifereji.",
        "risk_action_low": "Hakuna hatua ya haraka inayohitajika.",
        "btn_predict": "🔍 Tabiri Hatari",
        "select_county": "Chagua Kaunti",
        "flood_now": "Je, kuna mafuriko sasa hivi?",
        "flood_last": "Je, kulikuwa na mafuriko mwezi uliopita?",
        "rain": "Je, mvua ni kubwa kiasi gani mwezi huu?",
        "chat_placeholder": "Uliza swali...",
        "hist_chart_title": "Kaunti 15 Zilizoathiriwa Zaidi na Mafuriko",
        "hist_info_header": "🚨 Maeneo ya Hatari Kubwa",
        "metric_total": "Jumla ya Matukio ya Mafuriko (2002-2024)",
        "metric_counties": "Kaunti Zilizoathiriwa",
        "rain_levels": ["Hakuna mvua", "Mvua kidogo", "Mvua ya wastani", "Mvua kubwa", "Mvua iliyokithiri"],
        "flooding_now_msg": "🚨 Kwa kuwa kaunti inakumbana na mafuriko sasa hivi, hali ya hewa inachukuliwa kuwa mbaya. Mfumo utazingatia hali ya mafuriko ya sasa.",
        "overview_title": "ℹ️ Programu hii inafanya nini? (Bofya ili kupanua)",
        "overview_intro": "Karibu kwenye **Mfumo wa Tahadhari ya Mafuriko Kenya**. Zana hii inatumia akili bandia na data ya hali ya hewa ya miaka 20 kupitisha mahali ambapo mafuriko yana uwezekano mkubwa wa kutokea mwezi ujao.",
        "overview_how": "**Jinsi ya kutumia programu hii:**",
        "overview_tab1": "📊 **Historia ya Mafuriko:** Angalia kaunti zipi zimekumbwa na mafuriko mengi zaidi huko nyuma. Elewa *lini* mafuriko hufanyika (msimu) na *jinsi yanavyoweza kuwa mabaya*.",
        "overview_tab2": "🚨 **Kipimo cha Hatari:** Chagua kaunti, jibu maswali mawili rahisi kuhusu hali ya hewa ya sasa na hali ya mafuriko, na upate tathmini ya haraka ya hatari pamoja na ushauri wa usalama.",
        "overview_tab3": "🤖 **Msaidizi wa AI:** Zungumza na bot yetu ya AI. Unaweza kuiuliza maswali kama *'Ni kaunti zipi zina hatari kubwa kihistoria?'* au *'Nitaje maji salama vipi?'* Inajua itifaki zote rasmi za usalama.",
        "overview_note": "*Kumbuka: Hii ni zana ya usaidizi wa maamuzi. Daima fuata mwongozo rasmi kutoka Idara ya Hali ya Hewa Kenya (KMD) na Mamlaka ya Kitaifa ya Usimamizi wa Maafa (NDMA).*",
        "kpi_roc": "Usahihi wa Mfumo (ROC-AUC)",
        "kpi_pr": "Ugunduzi wa Matukio Nadra (PR-AUC)",
        "kpi_counties": "Kaunti Zinazofuatiliwa",
        "kpi_records": "Kumbukumbu za Kihistoria",
        "chart_season": "Msimu: Matukio ya Mafuriko kwa Mwezi",
        "chart_severity": "Ukali: Mafuriko dhidi ya Vifo",
        "top5_header": "**Kaunti 5 Zilizoathiriwa Zaidi**",
        "data_source": "Vyanzo vya data: DesInventar na EM-DAT.",
        "clear_chat": "🗑️ Futa Mazungumzo",
        "weather_info": "💡 Mfumo huzingatia kiotomatiki viashiria vya hali ya hewa ya bahari (ONI & DMI) kulingana na data ya hivi karibuni inayopatikana kwa kaunti hii.",
        "action_header": "**Nini cha kufanya:**",
        "action_high": "Hamisha mara moja kwenda mahali palipo juu. Wasiliana na mamlaka za mitaa. Usijaribu kuvuka barabara zilizofurika.",
        "action_mod": "Andaa vifaa vya dharura. Fuatilia redio ya ndani kwa taarifa mpya. Safisha mifereji karibu na nyumba yako.",
        "action_low": "Hakuna hatua ya haraka inayohitajika. Endelea kufahamishwa kwa kuangalia ripoti za hali ya hewa za ndani.",
        "context_higher": "📈 Kiwango hiki cha hatari ni **kikubwa sana** kuliko wastani wa kihistoria wa {}.",
        "context_above": "📊 Kiwango hiki cha hatari kiko juu kidogo ya wastani wa kihistoria wa {}.",
        "context_below": "📉 Kiwango hiki cha hatari kiko chini ya wastani wa kihistoria wa {}.",
        "footer": "Hii ni zana ya usaidizi wa maamuzi. Daima fuata mwongozo rasmi wa KMD na NDMA.",
        "initial_chat": "Habari! 👋 Naweza kutabiri hatari ya mafuriko na kutoa itifaki za usalama. Jaribu kuuliza:\n- *Ni kaunti zipi zina hatari kubwa kihistoria?*\n- *Nifanye nini ikiwa hatari ni KUBWA?*\n- *Nitaje maji salama vipi wakati wa mafuriko?*"
    }
}

# --- CACHED LOADERS ---
@st.cache_resource(show_spinner="Loading model...")
def load_model_artifacts():
    model = joblib.load("artifacts/xgb_model.pkl")
    with open("artifacts/feature_columns.json") as f:
        features = json.load(f)
    with open("artifacts/county_rates.json") as f:
        county_rates = json.load(f)
    latest_df = pd.read_csv("artifacts/latest_features.csv")
    return model, features, county_rates, latest_df

@st.cache_data(show_spinner="Loading historical flood data...")
def load_flood_history():
    try:
        county_df = pd.read_csv("county_flood_history.csv")
        monthly_df = pd.read_csv("monthly_flood_history.csv")
        return county_df, monthly_df
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

try:
    model, feature_cols, county_rates, latest_df = load_model_artifacts()
    flood_history, monthly_history = load_flood_history()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# ==========================================
# 🤖 AI AGENT TOOLS & SETUP
# ==========================================

@tool
def predict_flood_probability(county_name: str) -> str:
    """Useful for predicting the flood risk for a specific Kenyan county.
    Input should be the name of the county (e.g., 'Nairobi', 'Kisumu', 'Mombasa')."""
    try:
        search_name = str(county_name).strip().lower()
        df_counties = latest_df["county"].astype(str).str.lower().str.strip()
        row = latest_df[df_counties == search_name]
        
        if row.empty:
            row = latest_df[df_counties.str.contains(search_name)]
            
        if row.empty:
            return f"I couldn't find data for '{county_name}' in the dataset. Please try another county."
            
        county_row = row.iloc[0]
        
        # Bulletproof Safety Net
        input_dict = {}
        for col in feature_cols:
            if col in county_row and pd.notna(county_row[col]):
                input_dict[col] = county_row[col]
            else:
                input_dict[col] = 0 

        X_input = pd.DataFrame([input_dict], columns=feature_cols)
        prob = float(model.predict_proba(X_input)[0, 1])
        
        baseline = county_rates.get(county_row['county'], 0.05)
        current_flood = input_dict.get('flood_this_month', 0)
        
        return (
            f"The XGBoost model predicts a {prob*100:.1f}% chance of flooding next month in {county_row['county']}. "
            f"For context, the historical baseline for this county is {baseline*100:.1f}%. "
            f"The model currently assumes the county is {'currently flooding' if current_flood == 1 else 'not currently flooding'}."
        )
    except Exception as e:
        return f"Tool Error: {str(e)}"

@st.cache_resource(show_spinner="Building AI Agent...")
def build_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY not set in environment."
    
    with open("flood_protocols.txt", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents([Document(page_content=text)])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(search_kwargs={"k": 3}),
        "flood_protocols_search",
        "Search for official Kenya flood safety protocols, health advice, and emergency contacts."
    )
    
    tools = [retriever_tool, predict_flood_probability]
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="openai/gpt-oss-20b")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced Flood Safety Assistant for Kenya. 
        You have two helpful capabilities:
        1. Predicting flood risk for a specific county (e.g., 'Nairobi', 'Kisumu').
        2. Searching for safety advice, health protocols, and emergency contacts.

        CRITICAL INSTRUCTIONS:
        - You are strictly forbidden from saying "I don't know" or refusing to answer. 
        - If a user asks about a specific county, you MUST use your prediction capability. 
        - Never guess the prediction yourself; always use your capability.
        - NEVER output code, function names, or Python syntax. Speak in natural, plain language only.
        - Do not mention that you are using a "tool" or "database". Just give the answer as if you already knew it.

        Respond in this language: {language}."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor, None

agent_executor, agent_error = build_agent()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌊 Kenya Flood EWS")
    language = st.radio("Language / Lugha", ["English", "Kiswahili"])
    t = TEXTS[language]
    
    st.divider()
    st.markdown("**Model:** XGBoost")
    st.markdown("**ROC-AUC:** 0.896")
    st.markdown("**PR-AUC:** 0.392 (Baseline: 0.05)")
    st.caption("📈 Our PR-AUC is nearly 8x higher than a random guess, proving the model is learning real patterns.")
    st.divider()
    st.caption(t["footer"])

# --- MAIN HEADER (Hero Section) ---
st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">{t["title"]}</div>
        <div class="hero-subtitle">{t["subtitle"]}</div>
    </div>
""", unsafe_allow_html=True)

# --- APP OVERVIEW (ONBOARDING) ---
with st.expander(t["overview_title"], expanded=False):
    st.markdown(f"""
    {t["overview_intro"]}
    
    {t["overview_how"]}
    * {t["overview_tab1"]}
    * {t["overview_tab2"]}
    * {t["overview_tab3"]}
    
    {t["overview_note"]}
    """)
    st.write("") 

# --- KPI DASHBOARD ---
col1, col2, col3, col4 = st.columns(4)
col1.metric(t["kpi_roc"], "0.896", "Top 10%")
col2.metric(t["kpi_pr"], "0.392", "8x Baseline")
col3.metric(t["kpi_counties"], "47", "Nationwide")
col4.metric(t["kpi_records"], f"{flood_history['total_floods'].sum() if not flood_history.empty else 0:,}", "Since 2002")
st.write("---")

tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

# ======================== TAB 1: HISTORICAL INSIGHTS ========================
with tab1:
    st.header(t["header1"])
    
    if not flood_history.empty:
        # --- ROW 1: Where do floods happen? ---
        col_map, col_info = st.columns([2, 1])
        
        with col_map:
            with st.container(border=True):
                top_counties = flood_history.sort_values('total_floods', ascending=False).head(15)
                fig = px.bar(
                    top_counties, x='county', y='total_floods', 
                    color='total_floods', color_continuous_scale='Reds',
                    range_color=[0, top_counties['total_floods'].max()],
                    title=t["hist_chart_title"],
                    labels={'total_floods': 'Total Recorded Floods', 'county': 'County'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=-45, 
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with col_info:
            with st.container(border=True):
                st.markdown(f"### {t['hist_info_header']}")
                
                # DYNAMIC TEXT GENERATION: Matches the top 3 bars
                top_3 = flood_history.sort_values('total_floods', ascending=False).head(3)['county'].tolist()
                dynamic_text = f"The counties on the left have recorded the highest number of flood incidents over the last two decades. The top three most affected counties are **{top_3[0]}**, **{top_3[1]}**, and **{top_3[2]}**. Flooding is often driven by rivers bursting their banks and poor urban drainage."
                
                st.markdown(dynamic_text)
                st.divider()
                st.markdown(t["top5_header"])
                top_5 = flood_history.sort_values('total_floods', ascending=False).head(5)
                st.dataframe(
                    top_5[['county', 'total_floods']].rename(columns={'county': 'County', 'total_floods': 'Floods'}),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption(t["data_source"])
        
        st.write("---")
        
        # --- ROW 2: When do floods happen & Severity ---
        col_season, col_severity = st.columns([1, 1])
        
        with col_season:
            with st.container(border=True):
                if not monthly_history.empty:
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    monthly_history['Month Name'] = monthly_history['month'].apply(lambda x: month_names[x-1])
                    
                    fig_month = px.bar(
                        monthly_history, x='Month Name', y='total_floods',
                        color='total_floods', 
                        color_continuous_scale=["#90CAF9", "#1565C0", "#0D47A1"], # <--- Custom Scale
                        range_color=[0, monthly_history['total_floods'].max()],
                        title=t["chart_season"],
                        labels={'total_floods': 'Total Floods', 'Month Name': 'Month'}
                    )
                    fig_month.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin={"r":0,"t":40,"l":0,"b":0}
                    )
                    st.plotly_chart(fig_month, use_container_width=True)
                    
        with col_severity:
            with st.container(border=True):
                fig_sev = px.scatter(
                    flood_history, x='total_floods', y='total_deaths',
                    hover_name='county', size='total_deaths', color='total_deaths',
                    color_continuous_scale=["#FFAB91", "#E53935", "#B71C1C"], # <--- Custom Scale
                    range_color=[0, flood_history['total_deaths'].max()],
                    title=t["chart_severity"],
                    labels={'total_floods': 'Total Floods', 'total_deaths': 'Total Deaths'}
                )
                fig_sev.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig_sev, use_container_width=True)

    else:
        st.warning("⚠️ Historical flood history data not found. Please ensure the CSV files are in the app folder.")

# ======================== TAB 2: RISK PREDICTOR ========================
with tab2:
    st.header(t["header2"])
    
    with st.container(border=True):
        col_a, col_b = st.columns([1, 1])
        available_counties = sorted(county_rates.keys())
        
        with col_a:
            county = st.selectbox(t["select_county"], available_counties, index=0)
            row = latest_df[latest_df["county"] == county]
            if row.empty:
                st.error(f"No feature snapshot available for {county}.")
                st.stop()
            county_row = row.iloc[0]

            is_flooding = st.checkbox(t["flood_now"], value=bool(county_row.get("flood_this_month", 0)))
            was_flooding = st.checkbox(t["flood_last"], value=bool(county_row.get("flood_last_month", 0)))

        with col_b:
            st.markdown("**Current Weather Conditions**")
            
            # CONDITIONAL UI: Only ask about rain if they are NOT currently flooding
            if not is_flooding:
                rain_choice = st.selectbox(
                    t["rain"],
                    t["rain_levels"]
                )
                rain_map = {
                    "No rain": 0, "Hakuna mvua": 0,
                    "Light rain": 50, "Mvua kidogo": 50,
                    "Moderate rain": 120, "Mvua ya wastani": 120,
                    "Heavy rain": 250, "Mvua kubwa": 250,
                    "Extreme rain": 400, "Mvua iliyokithiri": 400
                }
                rain_val = rain_map[rain_choice]
            else:
                st.info(t["flooding_now_msg"])
                rain_val = 250 # Default to Heavy Rain internally just to feed the model

            # HIDDEN COMPLEX VARIABLES
            st.info(t["weather_info"])

        st.write("")
        if st.button(t["btn_predict"], type="primary", use_container_width=True):
            input_dict = county_row.to_dict()
            
            for col in feature_cols:
                if col not in input_dict or pd.isna(input_dict[col]):
                    input_dict[col] = 0

            # Overwrite with human inputs
            input_dict["flood_this_month"] = int(is_flooding)
            input_dict["flood_last_month"] = int(was_flooding)
            input_dict["rain_mm"] = rain_val
            
            # Sync Neighbour Rain
            input_dict["neighbour_rain"] = rain_val 

            X_input = pd.DataFrame([input_dict])[feature_cols]
            prob = float(model.predict_proba(X_input)[0, 1])

            st.write("---")

            # --- RESULTS DISPLAY (Action-Oriented) ---
            if prob >= 0.60:
                st.markdown(f'<div class="risk-card-high">{t["risk_high"]}<br><span style="font-size:18px;">{t["risk_action_high"]}</span></div>', unsafe_allow_html=True)
                st.error(f"{t['action_header']} {t['action_high']}")
            elif prob >= 0.30:
                st.markdown(f'<div class="risk-card-mod">{t["risk_mod"]}<br><span style="font-size:18px;">{t["risk_action_mod"]}</span></div>', unsafe_allow_html=True)
                st.warning(f"{t['action_header']} {t['action_mod']}")
            else:
                st.markdown(f'<div class="risk-card-low">{t["risk_low"]}<br><span style="font-size:18px;">{t["risk_action_low"]}</span></div>', unsafe_allow_html=True)
                st.success(f"{t['action_header']} {t['action_low']}")

            st.write("")
            
            # Context without exact percentage
            baseline = county_rates.get(county, 0.05)
            if prob > baseline * 2:
                st.info(t["context_higher"].format(county))
            elif prob > baseline:
                st.info(t["context_above"].format(county))
            else:
                st.info(t["context_below"].format(county))
                
            st.caption(t["footer"])

# ======================== TAB 3: AI ASSISTANT (AGENT) ========================
with tab3:
    # Header with a Clear Chat button on the right
    col_header, col_clear = st.columns([4, 1])
    with col_header:
        st.header(t["header3"])
    with col_clear:
        st.write("") # Spacer to align the button nicely
        if st.button(t["clear_chat"], use_container_width=True):
            # Reset the messages list to just the initial greeting
            st.session_state.messages = [{"role": "assistant", "content": t["initial_chat"]}]
            st.rerun()

    if agent_executor is None:
        st.warning(f"⚠️ AI Agent is offline: {agent_error}")
    else:
        # Initialize chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": t["initial_chat"]}]

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if user_input := st.chat_input(t["chat_placeholder"]):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking and predicting..." if language == "English" else "Inafikiri na kutabiri..."):
                    try:
                        result = agent_executor.invoke({
                            "input": user_input,
                            "language": language
                        })
                        answer = result["output"]
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                st.markdown(answer)

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})