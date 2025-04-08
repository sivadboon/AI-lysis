# 📊 AI Chatbot for Data Analysis

This is an interactive AI-powered chatbot that helps users analyze and clean datasets using natural language commands. The app is built with **Streamlit** and integrates **OpenAI GPT** models for visualization and interpretation.

---

## 👥 Contributors

- 🎓 **Aloysius Ang** (`U2120520B`)
- 🎓 **Wang Shang An Davis** (`U2121998F`) — [GitHub](https://github.com/sivadboon)


## 🚀 Features

- Upload multiple CSV or Excel files.
- Automatically detect and clean missing values and datetime formats.
- Generate statistical summaries and visualizations (histograms, boxplots, correlation heatmaps, pair plots).
- Ask questions about your dataset using natural language.
- Get plot analysis and interpretations powered by GPT-4 Vision.

---

## 🛠️ Requirements

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/account/api-keys)
- The following Python packages:

### 1. Clone the Repository

```bash
git clone https://github.com/sivadboon/AI-lysis.git
cd AI-lysis
```

### 2. Set Up Environment

```bash
pip install -r requirements.txt
```

### 3. Add Your OpenAI API Key

#### Create a file called .env in the root of the project folder and add:

```bash
OPENAI_API_KEY=your-api-key
```

### 4. Run the Streamlit App

```bash
streamlit run app_streamlit_final.py
```

#### The app will open in your default browser at http://localhost:8501.
