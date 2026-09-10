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
    page_icon="🤖",
    layout="wide"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

# ── DATABASE SCHEMA CONTEXT ───────────────────────────────────────────────────
DB_SCHEMA = """
You have access to a PostgreSQL database with the following tables:

1. lumenindex_combined (main table)
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

2. female_labor_force_latam
   - country, year, lfp_rate

3. chile_development_indicators
   - country (Chile only), indicator_name, indicator_code, year, value

4. asti_agricultural_research
   - country, year, indicator, value

5. latam_poverty
   - country, year, poverty_headcount

Countries: Argentina, Bolivia, Brazil, Chile, Colombia, Costa Rica,
Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras,
Mexico, Nicaragua, Panama, Paraguay, Peru, Uruguay.

Chile is the only High tier country (LumenIndex score 73.9).
All others are Medium tier (scores 38-51).
"""

SQL_SYSTEM_PROMPT = f"""You are a PostgreSQL expert. Given a question about LATAM development data, 
write ONLY a valid PostgreSQL SQL query. Return ONLY the SQL query, nothing else — no explanation, 
no markdown, no code blocks, just raw SQL.

Database schema:
{DB_SCHEMA}

Rules:
- Use lumenindex_combined as primary table
- Filter NULL values with IS NOT NULL
- Round numbers with ROUND(value::numeric, 2)
- Always ORDER BY results
- LIMIT 20 rows maximum
- Return ONLY the SQL query, no other text"""

INSIGHT_SYSTEM_PROMPT = """You are the LumenIndex AI Agent for Living Stones Foundation.
You help non-technical stakeholders understand rural development data for Latin America.

Your responses should be:
- Clear and in plain English
- Warm and insightful
- Connected to rural development and social impact
- Free of any technical jargon, SQL, or database references

Never mention SQL, queries, tables, or databases. Just answer the question directly."""

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def clean_response(text: str) -> str:
    """Remove any SQL or code blocks from response text"""
    text = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```SQL.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'SELECT\s+.*?;', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def call_lsf(messages: list, system: str) -> str:
    """Call LSF AI Gateway"""
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

# ── DATABASE FUNCTIONS ────────────────────────────────────────────────────────
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

# ── MAIN AGENT FUNCTION ───────────────────────────────────────────────────────
def query_agent(user_question: str, conversation_history: list) -> tuple:
    """
    Step 1: Get SQL from GPT-4o (SQL-only call)
    Step 2: Execute SQL against Neon
    Step 3: Get plain English interpretation (no SQL)
    """

    # Step 1: Get SQL only
    sql_messages = [{"role": "user", "content": f"Write a SQL query to answer: {user_question}"}]
    sql_response = call_lsf(sql_messages, SQL_SYSTEM_PROMPT)

    # Clean up SQL response
    sql_query = sql_response.strip()
    sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'```\s*', '', sql_query)
    sql_query = sql_query.strip()

    df_result = None
    response_text = ""

    # Step 2: Execute SQL
    if sql_query.upper().startswith("SELECT"):
        df_result = execute_query(sql_query)

        if not df_result.empty and 'Error' not in df_result.columns:
            data_str = df_result.to_string(index=False)

            # Step 3: Get plain English interpretation
            insight_messages = conversation_history + [
                {"role": "user", "content": f"""Question: {user_question}

Data results:
{data_str}

Please answer the question in plain English with key insights about what this means for rural development in Latin America. Do not mention SQL, queries, or databases."""}
            ]
            response_text = call_lsf(insight_messages, INSIGHT_SYSTEM_PROMPT)
            response_text = clean_response(response_text)
        else:
            response_text = "I couldn't find data to answer that question. Could you try rephrasing it?"
    else:
        # No SQL needed — general question
        general_messages = conversation_history + [
            {"role": "user", "content": user_question}
        ]
        response_text = call_lsf(general_messages, INSIGHT_SYSTEM_PROMPT)
        response_text = clean_response(response_text)

    return response_text, df_result

# ── STREAMLIT UI ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #2D6A4F, #52B788);
     padding: 24px 28px; border-radius: 16px; margin-bottom: 24px;
     box-shadow: 0 2px 12px rgba(45,106,79,0.15);'>
    <h1 style='color: white; margin: 0; font-size: 1.8rem;'>🤖 LumenIndex AI Agent</h1>
    <p style='color: #D8F3DC; margin: 6px 0 0; font-size: 0.9rem;'>
        Ask questions about LATAM rural development data in plain English
    </p>
    <p style='color: #B7E4C7; margin: 4px 0 0; font-size: 0.75rem;'>
        Powered by GPT-4o · Connected to Neon PostgreSQL · 17 LATAM Countries · 1990–2020
    </p>
</div>
""", unsafe_allow_html=True)

# Example questions
st.markdown("**💡 Try asking:**")
col1, col2, col3 = st.columns(3)

example_questions = [
    "Which country has the highest female labor force participation?",
    "How has Chile's GDP changed from 2000 to 2020?",
    "Which 5 countries have the lowest poverty headcount?",
    "Compare internet access across all LATAM countries in 2015",
    "Which countries improved the most in female LFP since 1990?",
    "What is the average GDP per capita across all LATAM countries?",
]

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

                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )

                if len(st.session_state.conversation_history) > 20:
                    st.session_state.conversation_history = st.session_state.conversation_history[-20:]

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg
                })

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9CA3AF; font-size:0.72rem;'>
    🌱 LumenIndex AI Agent · Living Stones Foundation · Powered by GPT-4o + Neon PostgreSQL · Built by Deepeka Gurunathan · 2026
</div>
""", unsafe_allow_html=True)
