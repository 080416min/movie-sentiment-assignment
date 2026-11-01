import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import os

os.makedirs("output", exist_ok=True)
df = pd.read_csv("data/movie_reviews.csv")
X, y = df["text"], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
with open("output/result.txt", "w", encoding="utf-8") as f:
f.write(f"准确率: {accuracy_score(y_test, y_pred):.4f}\n")
f.write(f"F1值: {f1_score(y_test, y_pred):.4f}\n")
f.write("混淆矩阵:\n")
f.write(str(confusion_matrix(y_test, y_pred)))
f.write("\n分类报告:\n")
f.write(classification_report(y_test, y_pred))
