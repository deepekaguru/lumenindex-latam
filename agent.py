# ══════════════════════════════════════════════════════════════════════════════
# LUMENINDEX AI AGENT — LSF AI Gateway + Neon PostgreSQL
# Natural language queries over LATAM development data
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import requests
from sqlalchemy import create_engine, text
import pandas as pd
import re
import os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LumenIndex AI Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F4F0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1C2B1E;
    }
    [data-testid="stSidebar"] * {
        color: #D8F3DC !important;
    }

    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Example question buttons */
    .example-btn {
        background-color: #FFFFFF;
        border: 1.5px solid #2D6A4F;
        border-radius: 20px;
        padding: 8px 16px;
        color: #2D6A4F;
        font-size: 0.82rem;
        cursor: pointer;
        margin: 4px;
        display: inline-block;
        transition: all 0.2s;
    }
    .example-btn:hover {
        background-color: #2D6A4F;
        color: white;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #E8F5E9;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }

    /* Stats bar */
    .stats-bar {
        background: white;
        border-radius: 10px;
        padding: 12px 20px;
        display: flex;
        gap: 32px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 4px solid #2D6A4F;
    }
    .stat-item {
        text-align: center;
    }
    .stat-num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2D6A4F;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6B7280;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* Stbutton override for example questions */
    div[data-testid="column"] .stButton button {
        background-color: #FFFFFF;
        border: 1.5px solid #2D6A4F;
        border-radius: 20px;
        color: #2D6A4F;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 6px 12px;
        width: 100%;
        transition: all 0.2s;
    }
    div[data-testid="column"] .stButton button:hover {
        background-color: #2D6A4F;
        color: white;
        border-color: #2D6A4F;
    }

    /* Clear button */
    .stButton button[kind="secondary"] {
        background-color: transparent;
        border: 1px solid #E5E7EB;
        color: #6B7280;
        border-radius: 8px;
        font-size: 0.8rem;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        border-radius: 24px;
        border: 2px solid #2D6A4F;
    }
</style>
""", unsafe_allow_html=True)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
try:
    LSF_TOKEN = st.secrets["LSF_TOKEN"]
    DATABASE_URL = st.secrets["DATABASE_URL"]
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()
except Exception as e:
    st.error(f"Secrets error: {e}")
    st.stop()

LSF_BASE_URL = "https://livingstonesglobal.online/designconnect/v1/chat/completions"
MODEL = "gpt-4o"

# ── DATABASE SCHEMA ───────────────────────────────────────────────────────────
DB_SCHEMA = """
Tables in PostgreSQL database:

1. lumenindex_combined (PRIMARY TABLE)
   - country VARCHAR: 17 LATAM countries
   - year INTEGER: 1990 to 2020
   - female_lfp_rate NUMERIC: female labor force participation rate (%)
   - agri_research_spending NUMERIC: agricultural research spending
   - fte_researchers NUMERIC: full time equivalent researchers
   - fte_researchers_phd_pct NUMERIC: % of researchers with PhD
   - gdp_per_capita NUMERIC: GDP per capita in USD
   - unemployment_rate NUMERIC: unemployment rate (%)
   - electricity_access NUMERIC: electricity access (%)
   - internet_access NUMERIC: internet access (%)
   - life_expectancy NUMERIC: life expectancy at birth (years)
   - poverty_headcount NUMERIC: poverty headcount ratio (%)
   - rural_pop_pct NUMERIC: rural population (%)

2. female_labor_force_latam: country, year, lfp_rate
3. chile_development_indicators: country, indicator_name, indicator_code, year, value
4. asti_agricultural_research: country, year, indicator, value
5. latam_poverty: country, year, poverty_headcount

Countries: Argentina, Bolivia, Brazil, Chile, Colombia, Costa Rica,
Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras,
Mexico, Nicaragua, Panama, Paraguay, Peru, Uruguay.
"""

SQL_SYSTEM_PROMPT = f"""You are a PostgreSQL expert. Write a SQL query to answer the user's question.
Return ONLY the raw SQL query. No explanation, no markdown, no code blocks, just the SQL.

{DB_SCHEMA}

Rules:
- Use lumenindex_combined as primary table
- Filter NULLs with IS NOT NULL where needed
- Use ROUND(value::numeric, 2) for numbers
- Always ORDER BY results
- LIMIT 20 rows max
- Return ONLY raw SQL, nothing else"""

INSIGHT_SYSTEM_PROMPT = """You are the LumenIndex AI Agent for Living Stones Foundation.
You help non-technical stakeholders understand rural development data for Latin America.

Rules:
- Answer in clear, plain English only
- Never mention SQL, queries, tables, databases, or technical terms
- Be warm, insightful, and connect findings to rural development impact
- Structure your answer with a clear main finding, then supporting details
- Keep answers concise — 2-3 short paragraphs maximum
- End with one actionable insight for Living Stones Foundation"""

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def clean_response(text: str) -> str:
    text = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```SQL.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'SELECT\s+.*?;', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def call_lsf(messages: list, system: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LSF_TOKEN}"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 2000,
        "temperature": 0.1
    }
    response = requests.post(LSF_BASE_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

def execute_query(sql: str) -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

def extract_sql(sql_response: str) -> str:
    sql = sql_response.strip()
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*', '', sql)
    sql = sql.strip()
    if 'SELECT' in sql.upper():
        select_idx = sql.upper().find('SELECT')
        sql = sql[select_idx:]
    return sql.strip()

def query_agent(user_question: str, conversation_history: list) -> tuple:
    sql_messages = [{"role": "user", "content": f"Write SQL to answer: {user_question}"}]
    sql_response = call_lsf(sql_messages, SQL_SYSTEM_PROMPT)
    sql_query = extract_sql(sql_response)

    df_result = None
    response_text = ""

    if sql_query and 'SELECT' in sql_query.upper():
        df_result = execute_query(sql_query)

        if not df_result.empty and 'Error' not in df_result.columns:
            data_str = df_result.to_string(index=False)
            insight_messages = conversation_history + [
                {"role": "user", "content": f"""Question: {user_question}

Data:
{data_str}

Answer in plain English with insights about rural development in Latin America. No SQL or technical terms."""}
            ]
            response_text = call_lsf(insight_messages, INSIGHT_SYSTEM_PROMPT)
            response_text = clean_response(response_text)
        elif 'Error' in df_result.columns:
            general_messages = conversation_history + [{"role": "user", "content": user_question}]
            response_text = call_lsf(general_messages, INSIGHT_SYSTEM_PROMPT)
            response_text = clean_response(response_text)
            df_result = None
        else:
            response_text = "No data found for that query. Try asking about a different country or time period."
    else:
        general_messages = conversation_history + [{"role": "user", "content": user_question}]
        response_text = call_lsf(general_messages, INSIGHT_SYSTEM_PROMPT)
        response_text = clean_response(response_text)

    return response_text, df_result

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:2.5rem;'>🌱</div>
        <div style='font-size:1.1rem; font-weight:700; color:#3DD68C;'>LumenIndex</div>
        <div style='font-size:0.75rem; color:#B7E4C7; margin-top:4px;'>AI Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**📊 Data Coverage**")
    st.markdown("- 🌍 17 LATAM countries")
    st.markdown("- 📅 1990 to 2020")
    st.markdown("- 📈 13 development indicators")
    st.markdown("- 🗄️ Neon PostgreSQL cloud")

    st.markdown("---")

    st.markdown("**💡 Topics You Can Ask About**")
    topics = [
        "🏦 GDP & Economic Growth",
        "👩 Female Labor Force",
        "🌾 Agricultural Research",
        "💡 Electricity Access",
        "🌐 Internet Penetration",
        "❤️ Life Expectancy",
        "📉 Poverty Headcount",
        "🏘️ Rural Population",
        "📚 PhD Researchers",
        "💼 Unemployment Rate",
    ]
    for topic in topics:
        st.markdown(f"<div style='font-size:0.8rem; padding:3px 0; color:#B7E4C7;'>{topic}</div>",
                   unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**🔗 Links**")
    st.markdown("[📊 Main Dashboard](https://lumenindex-latam01.streamlit.app/)")
    st.markdown("[📁 GitHub Repo](https://github.com/deepekaguru/lumenindex-latam)")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#52B788; text-align:center;'>
        Living Stones Foundation<br>
        Applied Data & Digital Innovation Lab<br>
        Built by Deepeka Gurunathan · 2026
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #1B4332, #2D6A4F, #40916C);
     padding: 28px 32px; border-radius: 16px; margin-bottom: 20px;
     box-shadow: 0 4px 20px rgba(45,106,79,0.2);'>
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <div>
            <h1 style='color: white; margin: 0; font-size: 1.9rem; font-weight:700;'>LumenIndex AI Agent</h1>
            <p style='color: #B7E4C7; margin: 4px 0 0; font-size: 0.9rem;'>
                Ask questions about LATAM rural development data in plain English
            </p>
        </div>
    </div>
    <div style='display:flex; gap:24px; margin-top:16px;'>
        <div style='background:rgba(255,255,255,0.15); border-radius:8px; padding:8px 16px; text-align:center;'>
            <div style='color:white; font-weight:700; font-size:1.2rem;'>17</div>
            <div style='color:#B7E4C7; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;'>Countries</div>
        </div>
        <div style='background:rgba(255,255,255,0.15); border-radius:8px; padding:8px 16px; text-align:center;'>
            <div style='color:white; font-weight:700; font-size:1.2rem;'>13</div>
            <div style='color:#B7E4C7; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;'>Indicators</div>
        </div>
        <div style='background:rgba(255,255,255,0.15); border-radius:8px; padding:8px 16px; text-align:center;'>
            <div style='color:white; font-weight:700; font-size:1.2rem;'>30+</div>
            <div style='color:#B7E4C7; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;'>Years of Data</div>
        </div>
        <div style='background:rgba(255,255,255,0.15); border-radius:8px; padding:8px 16px; text-align:center;'>
            <div style='color:white; font-weight:700; font-size:1.2rem;'>GPT-4o</div>
            <div style='color:#B7E4C7; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;'>Powered By</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Example questions
st.markdown('<p class="section-label">💡 Try asking one of these</p>', unsafe_allow_html=True)

example_questions = [
    "Which country has the highest female labor force participation?",
    "How has Chile's GDP changed from 2000 to 2020?",
    "Which 5 countries have the lowest poverty headcount?",
    "Compare internet access across all LATAM countries in 2015",
    "Which countries improved the most in female LFP since 1990?",
    "What is the average GDP per capita across all LATAM countries?",
]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(example_questions[0], use_container_width=True):
        st.session_state.example_q = example_questions[0]
    if st.button(example_questions[3], use_container_width=True):
        st.session_state.example_q = example_questions[3]
with col2:
    if st.button(example_questions[1], use_container_width=True):
        st.session_state.example_q = example_questions[1]
    if st.button(example_questions[4], use_container_width=True):
        st.session_state.example_q = example_questions[4]
with col3:
    if st.button(example_questions[2], use_container_width=True):
        st.session_state.example_q = example_questions[2]
    if st.button(example_questions[5], use_container_width=True):
        st.session_state.example_q = example_questions[5]

st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Welcome message when chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div style='background: white; border-radius: 12px; padding: 20px 24px;
         border-left: 4px solid #2D6A4F; margin-bottom: 16px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
            <span style='font-size:1.5rem;'>🤖</span>
            <span style='font-weight:600; color:#1B4332; font-size:1rem;'>LumenIndex AI Agent</span>
        </div>
        <p style='color:#374151; margin:0; font-size:0.9rem; line-height:1.6;'>
            Hello! I'm the LumenIndex AI Agent, here to help you explore rural development data 
            across Latin America. I can answer questions about GDP, poverty, female workforce 
            participation, agricultural research, internet access, and more — across 
            17 countries from 1990 to 2020.
        </p>
        <p style='color:#374151; margin:8px 0 0; font-size:0.9rem; line-height:1.6;'>
            Just type your question below or click one of the example questions above to get started! 🌱
        </p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "dataframe" in message and message["dataframe"] is not None:
            st.dataframe(message["dataframe"], use_container_width=True, hide_index=True)

# Chat input
question = st.chat_input("Ask anything about LATAM development data...")

# Handle example question buttons
if "example_q" in st.session_state:
    question = st.session_state.example_q
    del st.session_state.example_q

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                response_text, df_result = query_agent(
                    question,
                    st.session_state.conversation_history
                )
                st.markdown(response_text)
                if df_result is not None and not df_result.empty and 'Error' not in df_result.columns:
                    st.dataframe(df_result, use_container_width=True, hide_index=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "dataframe": df_result,
                })
                st.session_state.conversation_history.append({"role": "user", "content": question})
                st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
                if len(st.session_state.conversation_history) > 20:
                    st.session_state.conversation_history = st.session_state.conversation_history[-20:]
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.session_state.messages:
    col_clear, col_space = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9CA3AF; font-size:0.72rem; padding:8px 0;'>
    🌱 LumenIndex AI Agent · Living Stones Foundation · Applied Data & Digital Innovation Lab (LATAM)<br>
    Powered by GPT-4o + Neon PostgreSQL · Built by Deepeka Gurunathan · 2026
</div>
""", unsafe_allow_html=True)
