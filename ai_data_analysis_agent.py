import os
import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.agent.duckdb import DuckDbAgent

# 🌐 Streamlit UI
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("📊 AI Data Analyst Agent")

# 🔐 Sidebar: API key input
st.sidebar.header("🔐 API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
st.sidebar.markdown("🔑 [Don't have a key? Get it here](https://platform.openai.com/account/api-keys)")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()

# 📁 Preprocess uploaded file
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # Clean string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # Try parsing dates and numbers
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        # Save to temporary CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# 📤 File upload
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        st.success("✅ File successfully uploaded and processed.")
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)
        st.write("🧾 **Detected Columns:**", columns)

        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }

        model = {
            "id": "gpt-4",
            "api_key": openai_api_key,
            "provider": "openai"
        }

        tool = {
            "name": "pandas_tool",
            "type": "pandas",
            "description": "Tool to operate on pandas DataFrames"
        }

        duckdb_agent = DuckDbAgent(
            model=model,
            semantic_model=json.dumps(semantic_model),
            tools=[tool],
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql```, and give the final answer."
        )

        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None

        # 🧠 User question input
        user_query = st.text_area("💬 Ask a question about your data:")

        if st.button("🔍 Submit Query"):
            if not user_query.strip():
                st.warning("Please enter a valid query.")
            else:
                try:
                    with st.spinner('🔄 Processing your query...'):
                        response = duckdb_agent.run(user_query)
                        content = getattr(response, 'content', str(response))
                        st.markdown(content)
                except Exception as e:
                    st.error(f"❌ Error: {e}")