"""
train.py — обучение модели, которая определяет уровень знания темы.

1. Считает признаки для каждого студента и каждой темы (logic.topic_features).
2. Делит СТУДЕНТОВ на обучение (240) и тест (60): один студент целиком
   либо в обучении, либо в тесте — иначе была бы утечка данных.
3. Обучает две модели — логистическую регрессию и дерево решений —
   и сравнивает их с простым правилом без ML.
4. Логистическая регрессия точнее, её сохраняем в model.joblib (её использует сайт).
   Дерево решений сохраняем картинкой results/tree.png: по нему видно,
   как признаки влияют на решение (удобно объяснять на защите).

Запуск: python train.py   (сначала python generate_data.py)
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from logic import FEATURES, topic_features

SEED = 42
os.makedirs("results", exist_ok=True)

topics = pd.read_csv("data/topics.csv")
questions = pd.read_csv("data/questions.csv")
answers = pd.read_csv("data/students.csv")
true_levels = pd.read_csv("data/true_levels.csv")


def build_dataset(student_ids):
    """Одна строка = (студент, тема): 4 признака + настоящий уровень."""
    parts = []
    for sid in student_ids:
        feats = topic_features(answers[answers["student_id"] == sid], questions, topics)
        feats["student_id"] = sid
        parts.append(feats.reset_index())
    data = pd.concat(parts, ignore_index=True)
    return data.merge(true_levels, on=["student_id", "topic_id"])


# Делим по студентам
students = np.array(sorted(answers["student_id"].unique()))
rng = np.random.default_rng(SEED)
rng.shuffle(students)
test_ids, train_ids = students[:60], students[60:]
pd.Series(sorted(test_ids), name="student_id").to_csv("data/test_students.csv", index=False)

train = build_dataset(train_ids)
test = build_dataset(test_ids)
X_train, y_train = train[FEATURES], train["true_level"]
X_test, y_test = test[FEATURES], test["true_level"]

tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(X_train, y_train)
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
rule = X_test["score"].apply(lambda s: 2 if s >= 0.7 else (1 if s >= 0.4 else 0))

lines = [f"Обучение: {len(train_ids)} студентов ({len(train)} строк), "
         f"тест: {len(test_ids)} студентов ({len(test)} строк)", ""]
for name, pred in [("Простое правило (без ML)", rule),
                   ("Decision Tree", tree.predict(X_test)),
                   ("Logistic Regression (осн.)", logreg.predict(X_test))]:
    lines.append(f"{name:28s} accuracy = {accuracy_score(y_test, pred):.3f}   "
                 f"F1 (macro) = {f1_score(y_test, pred, average='macro'):.3f}")
report = "\n".join(lines)
print(report)
with open("results/train_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

joblib.dump(logreg, "model.joblib")

plt.figure(figsize=(22, 10))
plot_tree(tree, feature_names=FEATURES, class_names=["0 не знает", "1 частично", "2 знает"],
          filled=True, max_depth=3, fontsize=9)
plt.tight_layout()
plt.savefig("results/tree.png", dpi=120)
print("\nСохранено: model.joblib, results/tree.png, results/train_report.txt")
