\# 💰 EMIPredict AI — Smart EMI Prediction System



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://emipredict-ai.streamlit.app)

\[!\[Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)

\[!\[Machine Learning](https://img.shields.io/badge/ML%20Project-Streamlit%20App-orange)](https://streamlit.io/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



---



\## 🌟 Project Overview

\*\*EMIPredict AI\*\* is an intelligent machine learning system designed to:

\- 📊 Predict \*\*EMI Eligibility\*\* (Classification)

\- 💵 Estimate \*\*Maximum EMI Amount\*\* (Regression)

\- 🧠 Provide \*\*Data-Driven Financial Insights\*\*



It empowers financial institutions and individuals to make informed EMI-related decisions based on credit score, income, and spending behavior.



---



\## 🧩 Key Features

\- ✅ End-to-end ML pipeline: Data Cleaning → EDA → Feature Engineering → Modeling → Deployment  

\- 🔍 Exploratory Data Analysis with 15+ insightful charts  

\- 📈 Hypothesis Testing (T-test, Chi-square, ANOVA, Correlation)  

\- 🤖 Dual ML Models: Classification + Regression (XGBoost, Random Forest, Logistic/Linear Regression)  

\- 📂 Streamlit Web App for real-time predictions  

\- ⚡ Cloud-deployable and production-ready architecture  



---



\## 🧠 Tech Stack

| Category | Tools / Libraries |

|-----------|------------------|

| \*\*Language\*\* | Python 3.10 |

| \*\*ML / Data Science\*\* | Scikit-learn, XGBoost, Pandas, NumPy |

| \*\*Visualization\*\* | Matplotlib, Seaborn |

| \*\*App Framework\*\* | Streamlit |

| \*\*Deployment\*\* | Streamlit Cloud / Hugging Face Spaces |

| \*\*Model Persistence\*\* | Joblib |

| \*\*Version Control\*\* | GitHub |



---



\## 🏗️ Project Structure
EMIPredict_AI/
│
├── app.py # Streamlit web app
├── EMIPredict_AI.ipynb # Full training + EDA + modeling notebook
├── requirements.txt # Project dependencies
├── README.md # Project documentation
│
├── artifacts/
│ ├── models/
│ │ ├── XGBoost_classification.joblib
│ │ ├── XGBoost_regression.joblib
│ │ ├── scaler.joblib
│ │ └── feature_names.json
│ └── eda_charts/ # Generated EDA visuals
│
└── sample_data/
└── emi_prediction_dataset.csv


---

---

## 📊 Exploratory Data Analysis (EDA)
The notebook contains detailed EDA and insights:
- Income & Credit Score Distributions  
- Outlier Analysis (Boxplots)  
- Correlation Heatmap  
- Expense-to-Income and Debt-to-Income Ratios  
- Feature Relationships with EMI  

> All EDA charts are stored in `artifacts/eda_charts/`.

---

## 🧪 Model Performance Summary

| Task | Best Model | Metric | Performance |
|:------|-------------|---------|--------------|
| **Classification (Eligibility)** | XGBoost Classifier | Accuracy | **94.6%** |
| **Regression (Max EMI)** | XGBoost Regressor | R² Score | **0.97** |

✅ XGBoost models outperformed others, showing strong generalization and accuracy.

---

## 🚀 How to Run Locally

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/EMIPredict_AI.git
cd EMIPredict_AI
🔹 Step 2: Install Dependencies
pip install -r requirements.txt

🔹 Step 3: Run the Streamlit App
streamlit run app.py


Then open the displayed local URL in your browser (e.g., http://localhost:8501).

☁️ Cloud Deployment
Streamlit Cloud (Recommended)

Push this repo to GitHub

Go to https://share.streamlit.io

Connect your repo → Select app.py → Deploy

Hugging Face Spaces

Create a new Space → SDK: Streamlit

Upload all files

App auto-deploys and runs instantly

🧩 Input Features (Example)
Feature	Description
monthly_salary	Monthly income of applicant
credit_score	Credit score (300–900)
current_emi_amount	Existing EMI burden
other_monthly_expenses	Monthly living expenses
years_of_employment	Work experience in years
dependents	Number of dependents
...	Additional derived and engineered features
📈 Sample Output

Classification Prediction:
✅ EMI Eligible (Confidence: 92%)
Regression Prediction:
💵 Predicted Maximum EMI: ₹ 23,540.00

🧭 Future Improvements

🔁 Hyperparameter tuning using Optuna / GridSearchCV

📊 SHAP/LIME-based feature explainability

☁️ Containerized deployment (Docker + AWS / Render)

🧩 Add database integration for real-time user records

👩‍💻 Author

Mahima [Your Last Name]
💼 AI/ML Developer | Data Science Enthusiast
📧 Email: [your.email@example.com
]
🔗 GitHub: https://github.com/mahima-patel-ui

🔗 LinkedIn: https://www.linkedin.com/in/mahima-patel-051936272

📜 License

This project is licensed under the MIT License — feel free to use, modify, and share with credit.

🧠 “Data tells a story — this project transforms that story into intelligent financial decisions.”


