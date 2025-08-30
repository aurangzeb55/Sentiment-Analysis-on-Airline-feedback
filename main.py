# -----------------------------
# Sentiment Analysis using PySpark
# -----------------------------

# 1. Import Libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix

# -----------------------------
# 2. Initialize Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# -----------------------------
# 3. Load and Clean Data
# -----------------------------
feedback_df = spark.read.csv("BA_Airline.csv", header=True, inferSchema=True)

# Drop rows with null values in important columns
feedback_df = feedback_df.na.drop(subset=["Food&Beverages", "InflightEntertainment", "Wifi&Connectivity"])

# -----------------------------
# 4. Text Preprocessing Pipeline
# -----------------------------
tokenizer = Tokenizer(inputCol="feedback_text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="sentiment", outputCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, count_vectorizer, idf, label_indexer])

# Fit and transform
preprocessed_df = pipeline.fit(feedback_df).transform(feedback_df)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
train_data, test_data = preprocessed_df.randomSplit([0.8, 0.2], seed=123)

# -----------------------------
# 6. Train Logistic Regression Model
# -----------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Predictions
predictions = lr_model.transform(test_data)
predictions.select("feedback_text", "label", "prediction").show(5)

# -----------------------------
# 7. Model Evaluation
# -----------------------------
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

# Accuracy
accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# F1-Score
f1_score = evaluator.setMetricName("f1").evaluate(predictions)
print(f"F1-Score: {f1_score:.2f}")

# -----------------------------
# 8. Hyperparameter Tuning
# -----------------------------
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.maxIter, [10, 50]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)

cv_model = crossval.fit(train_data)
cv_predictions = cv_model.transform(test_data)

cv_accuracy = evaluator.setMetricName("accuracy").evaluate(cv_predictions)
print(f"Tuned Model Accuracy: {cv_accuracy:.2f}")

# -----------------------------
# 9. Data Visualization
# -----------------------------

# Word Frequency
words = preprocessed_df.select("filtered_words").rdd.flatMap(lambda x: x[0]).collect()
word_counts = Counter(words)
most_common_words = pd.DataFrame(word_counts.most_common(10), columns=["word", "count"])

plt.figure(figsize=(10, 6))
sns.barplot(x="word", y="count", data=most_common_words)
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.show()

# Model Performance over Iterations (example values)
iterations = [1, 10, 20, 30, 40, 50]
accuracies = [0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
f1_scores = [0.70, 0.73, 0.76, 0.79, 0.81, 0.83]

plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, label="Accuracy", marker="o")
plt.plot(iterations, f1_scores, label="F1-Score", marker="x")
plt.title("Model Performance Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
predictions_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(predictions_pd["label"], predictions_pd["prediction"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Word Count Distribution
# (make sure 'word_count' column exists or compute it if not already)
word_count_df = preprocessed_df.withColumn("word_count", 
                                           preprocessed_df["filtered_words"].getItem(0).isNotNull().cast("int"))

word_count_pd = word_count_df.select("word_count", "label").toPandas()
word_count_pd["label"] = word_count_pd["label"].map({0: "Negative", 1: "Positive"})

plt.figure(figsize=(10, 6))
sns.boxplot(x="label", y="word_count", data=word_count_pd)
plt.title("Word Count Distribution by Sentiment")
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="word_count", y="label", data=word_count_pd, hue="label", alpha=0.6)
plt.title("Sentiment vs Word Count")
plt.show()
