# ══════════════════════════════════════════════════════════════════════════════
# LUMENINDEX AI AGENT — Claude API + Neon PostgreSQL
# Natural language queries over LATAM development data
# Embed this into your existing app.py or run standalone
# ══════════════════════════════════════════════════════════════════════════════

# ── INSTALL REQUIRED PACKAGES ─────────────────────────────────────────────────
# pip install anthropic langchain langchain-community langchain-anthropic sqlalchemy psycopg2-binary streamlit

import streamlit as st
from anthropic import Anthropic
from sqlalchemy import create_engine, text
import pandas as pd

# ── PAGE CONFIG (remove if embedding into existing app.py) ────────────────────
st.set_page_config(
    page_title="LumenIndex AI Agent",
    page_icon="🤖",
    layout="wide"
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Store these in Streamlit Cloud Secrets:
# ANTHROPIC_API_KEY = "your-claude-api-key"
# DATABASE_URL = "postgresql://..."

try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    DATABASE_URL = st.secrets["DATABASE_URL"]
except:
    # Fallback for local testing — use .env or hardcode temporarily
    import os
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-claude-api-key-here")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:YOUR_PASSWORD@ep-curly-fire-ata7105s.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require")

# ── DATABASE SCHEMA CONTEXT ───────────────────────────────────────────────────
DB_SCHEMA = """
You have access to a PostgreSQL database called lumenindex with the following tables:

1. lumenindex_combined (main table — use this most often)
   - country VARCHAR: LATAM country name (17 countries)
   - year INTEGER: year from 1990 to 2020
   - female_lfp_rate NUMERIC: female labor force participation rate (%)
   - agri_research_spending NUMERIC: agricultural research spending
   - fte_researchers NUMERIC: full time equivalent researchers
   - fte_researchers_phd_pct NUMERIC: % of researchers with PhD
   - gdp_per_capita NUMERIC: GDP per capita in USD
   - unemployment_rate NUMERIC: unemployment rate (%)
   - electricity_access NUMERIC: electricity access (% of population)
   - internet_access NUMERIC: internet access (% of population)
   - life_expectancy NUMERIC: life expectancy at birth (years)
   - poverty_headcount NUMERIC: poverty headcount ratio (%)
   - rural_pop_pct NUMERIC: rural population (% of total)

2. female_labor_force_latam
   - country VARCHAR
   - year INTEGER (1990-2025)
   - lfp_rate NUMERIC: female labor force participation rate (%)

3. chile_development_indicators
   - country VARCHAR (Chile only)
   - indicator_name VARCHAR
   - indicator_code VARCHAR
   - year INTEGER (2000-2023)
   - value NUMERIC

4. asti_agricultural_research
   - country VARCHAR
   - year INTEGER
   - indicator VARCHAR
   - value NUMERIC

5. latam_poverty
   - country VARCHAR
   - year INTEGER
   - poverty_headcount NUMERIC

The 17 LATAM countries are: Argentina, Bolivia, Brazil, Chile, Colombia, 
Costa Rica, Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras, 
Mexico, Nicaragua, Panama, Paraguay, Peru, Uruguay.

Chile is the only High development tier country with a LumenIndex score of 73.9.
All other countries are Medium tier with scores between 38-51.
"""

SYSTEM_PROMPT = f"""You are the LumenIndex AI Agent, an expert data analyst for Living Stones Foundation's 
rural development index project covering Latin American countries.

You have access to a PostgreSQL database with development indicators for 17 LATAM countries from 1990 to 2020.

{DB_SCHEMA}

Your job is to:
1. Understand the user's question about LATAM development data
2. Write a PostgreSQL SQL query to answer it
3. Execute the query and interpret the results
4. Provide a clear, insightful answer with context

When writing SQL:
- Always use the lumenindex_combined table as the primary source unless specifically asked about another table
- Handle NULL values with COALESCE or IS NOT NULL filters where appropriate
- Round numeric results to 2 decimal places using ROUND(value::numeric, 2)
- Always ORDER BY results for readability
- Limit results to 20 rows maximum unless the user asks for all data

Response format:
1. Brief answer in plain English
2. Key insight or finding
3. The SQL query you used (in a code block)
4. Data table if relevant

Be conversational, insightful, and always connect findings to rural development and social impact context relevant to Living Stones Foundation's mission."""

# ── DATABASE QUERY FUNCTION ───────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

def execute_query(sql: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

def extract_sql(text: str) -> str:
    """Extract SQL query from Claude's response"""
    import re
    # Look for SQL in code blocks
    patterns = [
        r'```sql\n(.*?)\n```',
        r'```SQL\n(.*?)\n```',
        r'```\n(SELECT.*?)\n```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for SELECT statements
    select_match = re.search(r'(SELECT\s+.*?;)', text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return None

# ── CLAUDE AI AGENT ───────────────────────────────────────────────────────────
def query_agent(user_question: str, conversation_history: list) -> tuple:
    """
    Send question to Claude, extract SQL, execute it, return answer + data
    Returns: (response_text, dataframe_or_None)
    """
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Build messages with conversation history
    messages = conversation_history + [
        {"role": "user", "content": user_question}
    ]
    
    # First call — get SQL from Claude
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    
    response_text = response.content[0].text
    
    # Try to extract and execute SQL
    sql_query = extract_sql(response_text)
    df_result = None
    
    if sql_query:
        df_result = execute_query(sql_query)
        
        # If we got data, ask Claude to interpret it
        if not df_result.empty and 'Error' not in df_result.columns:
            data_str = df_result.to_string(index=False)
            
            # Second call — interpret the actual results
            interpretation_messages = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": f"The query returned this data:\n\n{data_str}\n\nPlease provide a clear, insightful interpretation of these results in the context of LATAM rural development."}
            ]
            
            interpretation = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=interpretation_messages
            )
            
            response_text = interpretation.content[0].text
    
    return response_text, df_result, sql_query

# ── STREAMLIT UI ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #2D6A4F, #52B788);
     padding: 24px 28px; border-radius: 16px; margin-bottom: 24px;'>
    <h1 style='color: white; margin: 0; font-size: 1.8rem;'>🤖 LumenIndex AI Agent</h1>
    <p style='color: #D8F3DC; margin: 6px 0 0; font-size: 0.9rem;'>
        Ask questions about LATAM rural development data in plain English
    </p>
    <p style='color: #B7E4C7; margin: 4px 0 0; font-size: 0.75rem;'>
        Powered by Claude AI · Connected to Neon PostgreSQL · 17 LATAM Countries · 1990–2020
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
    "What is the average LumenIndex score across all countries?",
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
        if "sql" in message and message["sql"]:
            with st.expander("🔍 SQL Query Used"):
                st.code(message["sql"], language="sql")

# Chat input
question = st.chat_input("Ask anything about LATAM development data...")

# Handle example question buttons
if "example_q" in st.session_state:
    question = st.session_state.example_q
    del st.session_state.example_q

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Querying database and analyzing..."):
            try:
                response_text, df_result, sql_query = query_agent(
                    question,
                    st.session_state.conversation_history
                )

                # Display response
                st.markdown(response_text)

                # Display data table if available
                if df_result is not None and not df_result.empty and 'Error' not in df_result.columns:
                    st.dataframe(df_result, use_container_width=True, hide_index=True)

                # Show SQL used
                if sql_query:
                    with st.expander("🔍 SQL Query Used"):
                        st.code(sql_query, language="sql")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "dataframe": df_result,
                    "sql": sql_query
                })

                # Update conversation history for context
                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )

                # Keep conversation history to last 10 exchanges
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
    🌱 LumenIndex AI Agent · Living Stones Foundation · Powered by Claude AI + Neon PostgreSQL · Built by Deepeka Gurunathan · 2026
</div>
""", unsafe_allow_html=True)