import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import hashlib  # For hashing the plot data
import time
import contextlib

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # Insert your openai api key here


st.set_page_config(page_title="AI Data Analyst Chatbot", layout="wide")
st.title("📊 AI Chatbot for Data Analysis")

# Initialize session state
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if 'plots' not in st.session_state:
    st.session_state.plots=[]
if 'analyzed_plots' not in st.session_state:
    st.session_state.analyzed_plots=[]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "Hello! I an AI Data Analyst. My job is to help you analyze and clean your datasets. Ask me anything you like!"}
    ]

# ---------- Cleaning & Validation ----------
def detect_and_clean_data(df):
    issues = []
    temp_df = df.copy()

    # Detect columns with more than 50% null values
    threshold = 0.5
    high_null_cols = temp_df.columns[temp_df.isnull().mean() > threshold]
    if len(high_null_cols) > 0:
        issues.append(f"Columns with >50% null values: {', '.join(high_null_cols)}")

    # Detect other null values
    null_columns = temp_df.columns[temp_df.isnull().any()]
    if len(null_columns) > 0:
        issues.append(f"Null values detected in: {', '.join(null_columns)}")

    # Detect datetime-looking columns
    for col in temp_df.select_dtypes(include=["object"]).columns:
        try:
            temp_df[col] = pd.to_datetime(temp_df[col], errors='raise')
            issues.append(f"Column '{col}' seems to contain datetime values.")
        except:
            continue

    return issues, df

def clean_data(df, issues):
    df_cleaned = df.copy()

    # Drop columns with >50% null values
    threshold = 0.5
    df_cleaned = df_cleaned.loc[:, df_cleaned.isnull().mean() <= threshold]

    # Drop remaining rows with null values
    df_cleaned = df_cleaned.dropna()

    # Attempt to parse object columns as datetime
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            try:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='ignore')
            except:
                pass

    return df_cleaned, issues


def permission_to_clean_data(file_name, df):
    issues, df = detect_and_clean_data(df)
    if issues:
        st.write(f"### Issues in `{file_name}`:")
        for issue in issues:
            st.write(issue)
        user_input = st.radio(f"Clean `{file_name}`?", ("No", "Yes"), key=f"radio_{file_name}")
        if user_input == "Yes":
            st.write("### Cleaning...")
            cleaned_df, issues = clean_data(df, issues)
            st.session_state.datasets[file_name] = cleaned_df
            st.write("### Cleaned Data:")
            st.dataframe(cleaned_df)
            st.download_button(
                label=f"Download Cleaned `{file_name}`",
                data=cleaned_df.to_csv(index=False),
                file_name=f"cleaned_{file_name}.csv",
                mime="text/csv",
                key=f"download_{file_name}"
            )
            return cleaned_df
        else:
            st.write(f"Skipped cleaning for `{file_name}`.")
            return df
    else:
        st.write(f"No issues found in `{file_name}`!")
        return df

# ---------- GPT Handling ----------
def generate_data_summary():
    summary = ""
    for name, df in st.session_state.datasets.items():
        summary += f"\n### Dataset: {name}\n- Shape: {df.shape}\n- Columns:\n{df.dtypes.to_string()}\n"
    return summary

def answer_user_query_with_gpt(query, df):
    openai.api_key = api_key
    st.session_state.chat_history.append({"role": "user", "content": query})
    system_msg = {
        "role": "system",
        "content": f"""You are a data analyst chatbot. Help users analyze data.
        Datasets: {generate_data_summary()}
        For visualizations, return Python code using matplotlib/seaborn. Assume all dataframes are imported and dataframe is named `df`. 
        AT ANY POINT, DO NOT ATTEMPT TO LOAD THE DATASETS AGAIN.
        WRAP CODE IN TRIPLE BACKTICKS WITH 'python'."""
    }
    try:
        reminder_message = {"role": "system", "content": "Remember to wrap code in triple backticks with 'python'!"}
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[reminder_message] + [system_msg] + st.session_state.chat_history
        )
        bot_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        return bot_reply
    except Exception as e:
        return f"Error: {e}"

# ---------- GPT-4 Turbo Analysis ----------
def analyze_plot_with_gpt(image_base64):
    openai.api_key = api_key
    prompt = "This is a plot generated from the dataset. Please analyze the visualization and provide observations, patterns, interpretation, and potential insights."

    image_message = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst who explains data visualizations clearly."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, image_message]}
            ],
            max_tokens=1000
        )
        gpt_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": "### 📈 GPT Analysis of the Plot:"})
        st.session_state.chat_history.append({"role": "assistant", "content": gpt_reply})
    except Exception as e:
        st.error(f"⚠️ Error analyzing plot with GPT Vision: {e}")

def analyze_nonplot_output_with_gpt(output_text):
    import openai
    openai.api_key = api_key

    prompt = f"This is a statistical summary or output derived from a dataset:\n\n{output_text}\n\nPlease analyze it and provide meaningful observations, patterns, and interpretations."

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst who explains tabular and textual data insights clearly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        gpt_reply = response.choices[0].message.content
        st.markdown("### 📈 GPT Analysis of the Results:")
        st.markdown(gpt_reply)
        # st.session_state.chat_history.append({"role": "assistant", "content": "### 📊 GPT Analysis of the Output:"})
        # st.session_state.chat_history.append({"role": "assistant", "content": gpt_reply})
    except Exception as e:
        st.error(f"⚠️ Error analyzing non-plot output with GPT: {e}")

def statistically_analyze_dataset_with_gpt(df=None, numerical_col=None):
    """
    Analyzes a dataset and sends a summary to GPT for insights. Like number of nulls, number of categorical and numeric columns, number of classes in categorical columns, statistics of numeric columns.
    """
    summary = {}
    if df is None:
        df = next(iter(st.session_state.datasets.values())) # Take the first dataset if df is not provided
    # Basic dataset info
    total_rows = df.shape[0]
    summary["total_entries"] = total_rows
    summary["column_summary"] = {}
    if numerical_col is None:
        numerical_col = df.select_dtypes(include=['number']).columns
        # Filter out columns that are actually categorical despite being numeric
        likely_categoricals = []
        for col in numerical_col:
            unique_vals = df[col].dropna().unique()
            nunique = len(unique_vals)
            unique_ratio = nunique / len(df)

            # Detect likely categorical columns
            if (
                nunique < 20  # label encoded
                or set(unique_vals).issubset({0, 1})  # one-hot encoded
            ):
                likely_categoricals.append(col)
        # Remove likely categorical columns from numerical columns
        numerical_col = [col for col in numerical_col if col not in likely_categoricals]

    # Analyze each column
    for col in df.columns:
        col_data = df[col]
        col_info = {
            "null_values": col_data.isnull().sum(),
            "non_null_values": col_data.count(),
            "dtype": str(col_data.dtype)
        }

        # Identify numerical and categorical columns
        if col in numerical_col:
            col_info["type"] = "numerical"
            col_info["statistics"] = col_data.describe().to_dict()
        else:
            col_info["type"] = "categorical"
            col_info["num_categories"] = col_data.nunique()
            col_info["top_categories"] = col_data.value_counts().head(5).to_dict()

        summary["column_summary"][col] = col_info

    # Format summary into a readable text for GPT
    summary_text = f"Your dataset contains {total_rows} entries with the following columns:\n\n"
    for col, info in summary["column_summary"].items():
        summary_text += f"- **{col}**: ({info['type']})\n"
        summary_text += f"  - Non-null values: {info['non_null_values']}\n"
        summary_text += f"  - Null values: {info['null_values']}\n"
        if info["type"] == "categorical":
            summary_text += f"  - Unique categories: {info['num_categories']}\n"
            summary_text += f"  - Top categories: {info['top_categories']}\n"
        else:
            summary_text += f"  - Statistics: {info['statistics']}\n"
        summary_text += "\n"

    # Send to GPT for analysis
    openai.api_key = api_key 
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert data analyst who provides insights on dataset structures."},
            {"role": "user", "content": f"Here is the dataset summary:\n\n{summary_text}\n\nAnalyze this and provide insights."}
        ],
        max_tokens=1000
    )

    return summary_text, response.choices[0].message.content

# ---------- Code Execution ----------
def execute_generated_code(bot_reply, df, retry_count=0, max_retries=3):
    if "```python" not in bot_reply:
        return
    try:
        code_block = bot_reply.split("```python")[1].split("```")[0]
        if 'plt.show()' in code_block:
            code_block = code_block.replace("plt.show()", "")

            fig, ax = plt.subplots(figsize=(5, 3))
            exec_globals = {
                "st": st,
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "io": io,
                "df": df,
                "fig": fig,
                "ax": ax
            }
            exec(code_block, exec_globals)

            # Save and analyze plot
            fig = plt.gcf()  # Get the current figure
            if fig.get_axes():  # Ensure the figure has content
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)  # Save as an image
                plt.close(fig) 
                plt.close("all") # <-- Prevent memory leak
                st.session_state.plots.append(buf)  # Store image in session state
                st.session_state.chat_history.append({"role": "assistant", "content": buf}) # Save the image to the chat history
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode("utf-8")
                analyze_plot_with_gpt(image_base64)
        else:
            exec_globals = {
                "st": st,
                "pd": pd,
                "df": df,
                "io": io
            }

            try:
                # Try eval first if it's a single expression
                output_text = str(eval(code_block, exec_globals))
            except Exception as e:
                # Smart handling for multi-line exec()
                code_lines = code_block.strip().splitlines()
                last_line = code_lines[-1].strip() if code_lines else ""

                try:
                    # Try compiling last line as an expression
                    compile(last_line, "<string>", "eval")
                    code_lines[-1] = f"_result = {last_line}"
                except SyntaxError:
                    # It's not an expression (e.g., already print or assignment)
                    pass

                updated_code_block = "\n".join(code_lines)

                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    exec(updated_code_block, exec_globals)

                result = exec_globals.get("_result")
                if result is not None:
                    output_text = str(result)
                else:
                    output_text = output_buffer.getvalue().strip()

            if output_text:
                st.markdown(f"```\n{output_text}\n```")
                analyze_nonplot_output_with_gpt(output_text)


    except Exception as e:
        error_message = str(e)
        st.error(f"⚠️ Error running code: {error_message}")
        if retry_count >= max_retries:
            st.error("❌ Reached maximum number of retries. Unable to fix the code.")
            return

        # Send code and error to GPT for fixing
        # Get the dataset summary
        raw_summary = generate_data_summary()

        # Rewrite the summary so GPT knows to use "df"
        dataset_summary = f"""The dataset is provided below. Regardless of its original name, it is available in the code as the variable `df`:\n{raw_summary}"""

        # Write the error prompt
        prompt = f"""The following Python code raised an error while generating a plot using a DataFrame named `df`. 
Please fix the error and return only the corrected code in a code block (no explanation needed) WRAP CODE IN TRIPLE BACKTICKS WITH 'python'.

### Dataset context:
{dataset_summary}

### Error:
{error_message}

### Code:
```python
{code_block}
```"""
        # Get corrected code from GPT
        openai.api_key = api_key 
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert Python programmer who fixes broken matplotlib/seaborn code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        corrected_code_input = response.choices[0].message.content

        # Try again with corrected code
        execute_generated_code(corrected_code_input, df, retry_count+1, max_retries=3)
        
def generate_plot_for_analysis_solely(code_input, df, retry_count=0, max_retries=3):
    if "```python" not in code_input:
        return
    try:
        code_block = code_input.split("```python")[1].split("```")[0]
        code_block = code_block.replace("plt.show()", "")

        fig, ax = plt.subplots(figsize=(5, 3))
        exec_globals = {
            "st": st,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "io": io,
            "df": df,
            "fig": fig,
            "ax": ax
        }
        exec(code_block, exec_globals)

        # Save and analyze plot
        fig = plt.gcf()  # Get the current figure
        if fig.get_axes():  # Ensure the figure has content
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)  # Save as an image
            plt.close(fig)
            plt.close("all")  # <-- Prevent memory leak
            # plt.close(fig)
            st.session_state.plots.append(buf)  # Store image in session state
            st.session_state.chat_history.append({"role": "assistant", "content": buf}) # Save the image to the chat history
            buf.seek(0)  # Reset buffer position
            # Compute a unique hash of the plot
            plot_hash = hashlib.md5(buf.read()).hexdigest()
            if plot_hash not in st.session_state.analyzed_plots:
                # Store hash and analyze plot
                st.session_state.analyzed_plots.append(plot_hash)
                buf.seek(0)  # Reset buffer position before using it again
                # Convert image to base64 and analyze
                image_base64 = base64.b64encode(buf.read()).decode("utf-8")
                analyze_plot_with_gpt(image_base64)
            # buf.close()

    except Exception as e:
        error_message = str(e)
        st.error(f"⚠️ Error running code: {error_message}")
        if retry_count >= max_retries:
            st.error("❌ Reached maximum number of retries. Unable to fix the code.")
            return

        # Send code and error to GPT for fixing
        # Get the dataset summary
        raw_summary = generate_data_summary()

        # Rewrite the summary so GPT knows to use "df"
        dataset_summary = f"""The dataset is provided below. Regardless of its original name, it is available in the code as the variable `df`:\n{raw_summary}"""

        # Write the error prompt
        prompt = f"""The following Python code raised an error while generating a plot using a DataFrame named `df`. 
Please fix the error and return only the corrected code in a code block (no explanation needed) WRAP CODE IN TRIPLE BACKTICKS WITH 'python'.

### Dataset context:
{dataset_summary}

### Error:
{error_message}

### Code:
```python
{code_block}
```"""

        # Get corrected code from GPT
        openai.api_key = api_key 
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert Python programmer who fixes broken matplotlib/seaborn code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        corrected_code_input = response.choices[0].message.content

        # Try again with corrected code
        generate_plot_for_analysis_solely(corrected_code_input, df, retry_count+1, max_retries=3)

def auto_plot_analysis(df):
    """Automatically generate Python code for histograms, boxplots, and pair plots for numerical columns."""
    numerical_cols = df.select_dtypes(include=['number']).columns
    likely_categoricals = []
    for col in numerical_cols:
        unique_vals = df[col].dropna().unique()
        nunique = len(unique_vals)

        if nunique < 20 or set(unique_vals).issubset({0, 1}):
            likely_categoricals.append(col)

    numerical_cols = [col for col in numerical_cols if col not in likely_categoricals]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    code_blocks = []
    heat_pair_blocks = []

    summary_text, statistical_analysis = statistically_analyze_dataset_with_gpt(df, numerical_cols)
    st.session_state.chat_history.append({"role": "assistant", "content": summary_text})
    st.session_state.chat_history.append({"role": "assistant", "content": statistical_analysis})

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": """We can investigate the distribution of each categorical column by plotting a value count plot. 
We can also investigate the distribution of each numerical column by plotting the histogram and boxplot. 
* If the categorical column has a lot of unique values, it may be skipped for readability. *"""
    })

    for col in categorical_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or (
            df[col].dtype == object and pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.8 * len(df)
        ):
            code_block = f"""
parsed_dates = pd.to_datetime(df['{col}'], errors='coerce')
monthly_series = parsed_dates.dt.to_period('M').astype(str)
yearly_series = parsed_dates.dt.to_period('Y').astype(str)

if monthly_series.nunique() <= 36:
    fig, ax = plt.subplots()
    sns.countplot(x=monthly_series, ax=ax)
    ax.set_title('Count Plot of {col} (by Month)')
    ax.tick_params(axis='x', rotation=45)
else:
    fig, ax = plt.subplots()
    sns.countplot(x=yearly_series, ax=ax)
    ax.set_title('Count Plot of {col} (by Year)')
    ax.tick_params(axis='x', rotation=45)
"""
            code_blocks.append(code_block)
        elif df[col].nunique(dropna=False) < 30:
            code_block = f"""
fig, ax = plt.subplots()
sns.countplot(x=df['{col}'], ax=ax)
ax.set_title('Count Plot of {col}')
ax.tick_params(axis='x', rotation=45)
"""
            code_blocks.append(code_block)

    for col in numerical_cols:
        code_blocks.append(f"""
fig, ax = plt.subplots()
sns.histplot(df['{col}'], bins=30, kde=True, ax=ax)
ax.set_title('Histogram with KDE of {col}')
""")
        code_blocks.append(f"""
fig, ax = plt.subplots()
sns.boxplot(x=df['{col}'], ax=ax)
ax.set_title('Boxplot of {col}')
""")

    for code in code_blocks:
        st.session_state.chat_history.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        generate_plot_for_analysis_solely(f"```python\n{code}\n```", df)

    if len(numerical_cols) > 1:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "We can also plot a heatmap and pairplot showing the correlation between each numerical column."
        })
        heat_pair_blocks.append(f"""
fig, ax = plt.subplots(figsize=(10,6))
corr_matrix = df[{list(numerical_cols)}].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
""")
        heat_pair_blocks.append(f"""
fig, ax = plt.subplots(figsize=(10,6))
sns.pairplot(df[{list(numerical_cols)}], kind='reg', diag_kind='kde', plot_kws={{'line_kws': {{'color': 'red'}}}})
""")
        for code in heat_pair_blocks:
            st.session_state.chat_history.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            generate_plot_for_analysis_solely(f"```python\n{code}\n```", df)


# ---------- App Body ----------
uploaded_files = st.file_uploader("Upload CSV/Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        st.session_state.datasets[f.name] = df
        st.write(f"**{f.name}**")
        st.dataframe(df.head())
        permission_to_clean_data(f.name, df)
        current_df=st.session_state.datasets[f.name]

    user_input = st.chat_input("Ask a question about your data. Try saying: 'Analyse my data'.")
    start_time = time.time()  # Start timer
    if user_input:
        start_time = time.time()  # Start timer
        # If the user requests analysis, automatically generate plots
        if user_input.strip().lower() == "analyse my data" or user_input.strip().lower() == "analyze my data":
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            bot_message="Sure! I will now analyze the dataset. This could take a moment..."
            st.markdown(bot_message)
            st.session_state.chat_history.append({"role": "assistant", "content": "Here is the analysis:"})
            auto_plot_analysis(current_df)

        else:
            bot_reply = answer_user_query_with_gpt(user_input, current_df)

    
    for i, msg in enumerate(st.session_state.chat_history):
        
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            if isinstance(msg["content"], io.BytesIO):  # If the content is an image (BytesIO)
                st.image(msg["content"], use_container_width=False, width=500)
            else:
                st.markdown(msg["content"])
            # If this is an assistant message and the next message is NOT an image, execute code
            if msg["role"] == "assistant" and (
                i + 1 < len(st.session_state.chat_history) and not isinstance(st.session_state.chat_history[i + 1]["content"], io.BytesIO)
            ):
                try:
                    execute_generated_code(msg["content"], current_df)
                except:
                    pass
            elif msg["role"] == "assistant" and (
                i == len(st.session_state.chat_history)-1 and not isinstance(st.session_state.chat_history[i]["content"], io.BytesIO) 
            ): # Accounting for the last message
                try:
                    execute_generated_code(msg["content"], current_df)
                except:
                    pass
    end_time = time.time()  # End timer
    print(f"Execution time: {end_time - start_time:.4f} seconds --------------------------------------------------")
