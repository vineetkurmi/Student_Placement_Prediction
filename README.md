# 🎓 Student Placement Predictor

An educational Machine Learning project that predicts student placement outcomes using a Decision Tree classifier. It combines a FastAPI backend, an interactive web interface, Groq AI-generated explanations, and an improvement simulator.

## 🎯 Project Objective

The main learning goal of this project is to understand the **end-to-end Machine Learning workflow** — from data preprocessing and model training to prediction, API development, and AI integration.

This project explores how machine learning predictions can be served through a web application and how hypothetical changes in student performance can affect model predictions.

## ✨ Features

- **Placement Prediction:** Predicts whether a student is likely to be placed using a Decision Tree classifier.
- **Prediction Probabilities:** Displays the model's estimated probabilities for placement outcomes.
- **AI Mentor:** Uses Groq AI to generate explanations and suggestions based on prediction results.
- **Improvement Simulator:** Simulates changes in student performance parameters and compares the resulting predictions.
- **Interactive Web Interface:** Allows users to enter student details and view results.
- **FastAPI Backend:** Connects the machine learning model with the web interface.

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Backend:** FastAPI
- **AI Integration:** Groq API
- **Frontend:** HTML, CSS, JavaScript
- **Model Storage:** Joblib

## ⚙️ How It Works

1. Student performance data is used to train a Decision Tree classifier.
2. The trained model processes student inputs and predicts a placement outcome.
3. The application displays the prediction and its associated probability.
4. Groq AI generates additional explanations and suggestions.
5. The improvement simulator modifies selected input parameters and compares the model's predictions for the original and simulated profiles.

## 🚀 Getting Started

### Prerequisites

- Python installed
- `uv` package manager or `pip`
- A Groq API key for AI-powered explanations

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Student Placement Pridiction"
```

Replace the placeholder URL with your repository's actual URL.

### 2. Create and activate a virtual environment

Using `uv`:

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

If your project dependencies are listed in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_new_groq_api_key
```

Ensure your application loads the `.env` file, for example using `python-dotenv`, if that is how your project is configured.

**Security:** Never commit your `.env` file or expose your API key publicly. Add `.env` to `.gitignore`.

### 5. Run the application

If your FastAPI application is defined in `main.py` and the FastAPI instance is named `app`:

```bash
uv run uvicorn main:app --reload
```

Open the local URL shown in your terminal. If the application serves its frontend there, you can interact with the predictor through your browser.

## 📊 Model and Dataset Limitations

This project is intended for educational and experimental use.

- The dataset may contain bias, limited diversity, or patterns that do not represent real-world student placement outcomes.
- Predictions depend on the data used for training and may not generalize to other student populations.
- A Decision Tree may return identical probabilities for different student profiles when they reach the same leaf node.
- Prediction probabilities are model estimates, not guarantees of placement or measures of a student's true employability.
- The improvement simulator demonstrates hypothetical model outcomes; it does not establish that changing a particular parameter will cause real-world placement.
- AI-generated explanations may be incomplete or inaccurate.

**This application should not be used to make actual hiring, admission, or academic evaluation decisions.**

## 📚 Key Learning Outcomes

Through this project, I explored:

- Data preprocessing and feature handling
- Supervised learning and classification
- Decision Tree model training
- Prediction probabilities and model evaluation
- Building APIs with FastAPI
- Integrating a large language model API
- Connecting machine learning models to a web interface
- Understanding dataset bias and model limitations

## 🔮 Future Improvements

- Evaluate performance on unseen data using appropriate metrics.
- Investigate dataset bias and improve data quality.
- Add model explainability tools.
- Improve automated testing and input validation.
- Compare the model with other classification algorithms.
- Deploy the application for demonstration purposes.

## 👨‍💻 Project Status

**Educational project — under development.**

Built to strengthen practical skills in Machine Learning, backend development, and AI integration.

---

*Learning by building, experimenting, and improving.*
