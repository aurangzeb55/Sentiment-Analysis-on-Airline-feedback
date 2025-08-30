# ✈️ Sentiment Analysis using PySpark

This project performs **Sentiment Analysis** on airline passenger feedback using **PySpark**.  
It builds a **Logistic Regression** classification pipeline with text preprocessing (tokenization, stopword removal, TF-IDF), model training, evaluation (Accuracy, F1-Score, Confusion Matrix), hyperparameter tuning, and visualization of insights.

---

## 📂 Project Structure
  pyspark-sentiment/
│── src/
│ └── sentiment_pyspark.py # Main script
│── data/
│ └── BA_Airline.csv # Dataset (not included if too large)
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .gitignore # Ignore unnecessary files

---

## ⚙️ Setup Instructions

1. Clone Repository
   ```bash
   git clone https://github.com/<your-username>/pyspark-sentiment.git
   cd pyspark-sentiment
   ```

2. Creat Visual Environment:
     ```
     python -m venv .venv
     ```
# Activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

4. Install Dependencies

    ```
    pip install -r requirements.txt
    ```

---

## 📊 Dataset

File: BA_Airline.csv
Columns Used:
feedback_text → Passenger review text
sentiment → Target variable (Positive / Negative)
Additional service-related columns (Food & Beverages, Inflight Entertainment, Wifi & Connectivity)

⚠️ If the dataset is large, use Git LFS or place it manually in the data/ folder.
Alternatively, provide only a sample dataset for testing.

---

## 🚀 Running the Project
Option 1: Run with Python
python src/sentiment_pyspark.py


Option 2: Run with Spark Submit
spark-submit --master local[*] src/sentiment_pyspark.py

---

## 🔑 Features

Preprocessing pipeline (Tokenizer → StopWordsRemover → CountVectorizer → TF-IDF → Label Encoding)
Logistic Regression Model
Train/Test Split
Model Evaluation: Accuracy, F1-Score
Hyperparameter Tuning with CrossValidator
Data Visualization:
Word frequency
Confusion matrix
Model performance over iterations
Word count distribution by sentiment
Scatter plots

---

## 📈 Example Visualizations

Top 10 Most Frequent Words
Confusion Matrix
Model Performance (Accuracy & F1)
Word Count Distribution by Sentiment

---

## 🛠️ Technologies Used

PySpark
Matplotlib
Seaborn
Pandas
Scikit-learn

---

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.

---

## 🙌 Author

Developed by Aurangzeb Sheikh ✨
If you find this useful, don’t forget to ⭐ star the repo!

---

## 🌐 Connect With Me
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aurangzeb55)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aurangzeb-sheikh-71ba6b2ba)
