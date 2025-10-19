# DeepCSAT — E-Commerce Customer Satisfaction Prediction 🎯

## 🧠 Overview
**DeepCSAT** is a Machine Learning–based project that predicts **Customer Satisfaction (CSAT)** scores for e-commerce customer service interactions.  
It combines **structured operational data** (like response time, shift, and tenure) with **text analytics** from customer remarks to estimate satisfaction levels and identify key factors driving customer experience.

The project includes:
- Full **data preprocessing and feature engineering** workflow  
- **Exploratory Data Analysis (EDA)** with visual insights  
- **Statistical validation** using ANOVA and Chi-squared tests  
- Multiple **ML models** (Logistic Regression, Random Forest, XGBoost)  
- An interactive **Streamlit web app** for real-time CSAT prediction  

---

## 📊 Key Features
- Sentiment analysis of customer feedback using **TextBlob**
- Hybrid modeling with **numeric**, **categorical**, and **text** features  
- Statistical testing for feature significance (ANOVA & Chi-squared)  
- Automated preprocessing pipelines using **scikit-learn ColumnTransformer**  
- Real-time prediction app built with **Streamlit**
- Deployed models saved as `.joblib` artifacts  

---

## 🧩 Project Structure

DeepCSAT/
│
├── main/
│ ├── app.py # Streamlit web app
│ └── artifacts/ # Model artifacts (joblib + JSON)
│ ├── logreg_pipeline.joblib
│ ├── rf_pipeline.joblib
│ ├── xgb_pipeline.joblib
│ ├── label_encoder.joblib
│ └── categorical_values.json
│
├── notebooks/
│ └── DeepCSAT_Ecommerce.ipynb # Data analysis and model training 
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DeepCSAT.git
cd DeepCSAT
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Activate environment
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download TextBlob corpora (once)
```bash
python -m textblob.download_corpora
```

### 5. Run the Streamlit app
```bash
streamlit run main/app.py
```
The app will open automatically in your default browser at:
🔗 http://localhost:8501

---

## 🧱 Model Pipeline

Feature Engineering

    - Numerical: response_time_hrs, remarks_length, sentiment_score

    - Categorical: Agent Shift, Tenure Bucket, channel_name, category, Sub-category

    - Text: clean_remarks (processed with TF-IDF)

Model Training

    - Logistic Regression — baseline interpretable model

    - Random Forest — non-linear ensemble model

    - XGBoost — high-performance gradient boosting model

Evaluation Metrics

    - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🧮 Statistical Analysis (before ML)
| Test        | Purpose                                     | Findings                                  |
| ----------- | ------------------------------------------- | ----------------------------------------- |
| ANOVA       | Tests difference in mean CSAT across groups | Shift and Tenure have significant effect  |
| Chi-Squared | Tests categorical dependence with CSAT      | Channel and Shift show strong association |


---

## 📈 Results Summary
| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | 83%      | 0.85      | 0.82     | 0.83     |
| Random Forest       | 86%      | 0.88      | 0.84     | 0.86     |
| XGBoost             | **88%**  | **0.90**  | **0.86** | **0.88** |


✅ XGBoost performed best overall, capturing both structured and text-based sentiment features effectively.

---

## 🖥️ Streamlit Web App

The DeepCSAT Streamlit app allows users to:

- Enter customer remarks (text)

- Select operational parameters (shift, channel, category, etc.)

- Predict CSAT level (High / Low)

- View model probabilities and top feature importances

---

## 🧠 Insights

- Customer sentiment polarity and response time were the most important drivers of CSAT.

- Long handling times and negative remarks significantly lowered satisfaction.

- Combining structured and text data improved predictive accuracy by ~10%.

---

## 🚀 Future Enhancements

- Integrate BERT-based text embeddings for deeper sentiment understanding.

- Build a dashboard view for aggregate performance analytics.

- Add feedback loop for model retraining using new survey data.

- Implement automated model versioning and CI/CD deployment.

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for full dependency list.

---

## 🧰 Tools & Technologies

- **Language**: Python 3.10

- **Libraries**: scikit-learn, pandas, numpy, seaborn, matplotlib, xgboost, textblob

- **Visualization**: Streamlit, Matplotlib, Seaborn

- **Storage**: SQLite (optional for data logging)

- **Environment**: Jupyter Notebook + Streamlit

---

## 🗂️ Data sets
* This project was created during an Internship.
* If you want to use the data that I have used, you can contact me.

---

## 🙋‍♂️ Author

**Shubham Pandey**
📧 [Email Me](mailto:shubhamppandey1084@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/shubham-pandey-6a65a524a/) • [GitHub](https://github.com/Shubhampandey1git)

---