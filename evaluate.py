"""
evaluate.py — модельді тексеру.

1. Жасанды деректердің ТЕСТ бөлігінде (60 студент, модель оларды оқыту кезінде көрмеген):
   accuracy, F1, әр деңгей бойынша precision/recall, қателер матрицасы;
   модельді қарапайым ережемен (ML-сіз) салыстыру.
2. Қате мысалы: модель қай жерде қателескенін көрсетеді.
3. Нақты студенттер (бар болса): сайттан жүктелген answers_R01.csv, answers_R02.csv ...
   файлдары data/real/ папкасында болса — оларды data/real_answers.csv-ке біріктіреді
   және жүйе нәтижесін студенттің өзін-өзі бағалауымен салыстырады.

Нәтижелер: results/evaluation.txt, results/confusion_matrix.png
Іске қосу: python evaluate.py
"""
import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from logic import FEATURES, LEVEL_NAMES, predict_levels, rule_levels, topic_features

os.makedirs("results", exist_ok=True)
topics = pd.read_csv("data/topics.csv")
questions = pd.read_csv("data/questions.csv")
answers = pd.read_csv("data/students.csv")
true_levels = pd.read_csv("data/true_levels.csv")
test_ids = pd.read_csv("data/test_students.csv")["student_id"]
model = joblib.load("model.joblib")

out = []
def say(text=""):
    print(text)
    out.append(text)

# ---------- 1. Жасанды тест бөлігі ----------
rows = []
for sid in test_ids:
    feats = topic_features(answers[answers["student_id"] == sid], questions, topics)
    pred = predict_levels(feats, model)
    rule = rule_levels(feats)
    for t in feats.index:
        rows.append({"student_id": sid, "topic_id": t, "model": pred[t], "rule": rule[t],
                     **feats.loc[t, FEATURES].to_dict()})
res = pd.DataFrame(rows).merge(true_levels, on=["student_id", "topic_id"])

names = [LEVEL_NAMES[i] for i in (0, 1, 2)]
say("=== 1. Жасанды деректердің тест бөлігі (60 студент x 8 тақырып = 480) ===")
for col, title in [("rule", "Қарапайым ереже (ML-сіз)"), ("model", "Модель (Logistic Regression)")]:
    say(f"{title:32s} accuracy = {accuracy_score(res.true_level, res[col]):.3f}   "
        f"F1 (macro) = {f1_score(res.true_level, res[col], average='macro'):.3f}")
say()
say("Модель, әр деңгей бойынша:")
say(classification_report(res.true_level, res.model, target_names=names, digits=3))

missed = ((res.true_level == 0) & (res.model != 0)).sum()
say(f"Нақты олқылықтар (білмейді): {(res.true_level == 0).sum()}, "
    f"модель тапты: {((res.true_level == 0) & (res.model == 0)).sum()}, жіберіп алды: {missed}")
say()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in [(axes[0], "rule", "Қарапайым ереже"), (axes[1], "model", "Модель")]:
    ConfusionMatrixDisplay(confusion_matrix(res.true_level, res[col], labels=[0, 1, 2]),
                           display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} (accuracy {accuracy_score(res.true_level, res[col]):.2f})")
    ax.set_xlabel("Болжам")
    ax.set_ylabel("Нақты деңгей")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=120)

# ---------- 2. Қате мысалы ----------
say("=== 2. Қате мысалы ===")
err = res[(res.true_level == 0) & (res.model == 2)]
if err.empty:
    err = res[(res.true_level == 0) & (res.model == 1)]
if err.empty:
    err = res[res.true_level != res.model]
e = err.iloc[0]
say(f"Студент {e.student_id}, тақырып {e.topic_id}: нақты деңгей — {LEVEL_NAMES[e.true_level]}, "
    f"модель — {LEVEL_NAMES[e.model]}.")
say(f"Белгілер: дұрыс {e.score:.0%}, «болжадым» {e.guess_rate:.0%}, орташа уақыт {e.avg_time:.0f} с.")
if e.score >= 0.5:
    say("Себебі: студент тақырыпты білмесе де, сұрақтардың көбіне кездейсоқ дұрыс жауап берген. "
        "Тақырыпқа 4 сұрақ қана болғандықтан, мұндай сәттілік нәтижені бұрмалайды.")
else:
    say("Себебі: дұрыс жауап аз болса да, студент жауаптардың көбін «сенімдімін» деп белгілеген. "
        "Модель сенімділікке де сүйенеді, сондықтан өзіне артық сенетін студенттің деңгейін асырып бағалады.")
say()

# ---------- 3. Нақты студенттер ----------
say("=== 3. Нақты студенттер ===")
files = sorted(glob.glob("data/real/answers_*.csv"))
if files:
    pd.concat([pd.read_csv(f) for f in files]).to_csv("data/real_answers.csv", index=False)
if os.path.exists("data/real_answers.csv"):
    real = pd.read_csv("data/real_answers.csv")
    per_topic = real.groupby(["student_id", "topic_id"])[["self_level", "predicted_level"]].first()
    agree = (per_topic.self_level == per_topic.predicted_level).mean()
    close = ((per_topic.self_level - per_topic.predicted_level).abs() <= 1).mean()
    say(f"Студенттер саны: {real.student_id.nunique()}, тақырып бағалары: {len(per_topic)}")
    say(f"Жүйе өзін-өзі бағалаумен толық сәйкес: {agree:.0%}, бір деңгейге дейін айырмашылықпен: {close:.0%}")
    say("Ескерту: өзін-өзі бағалау да дәл емес, сондықтан бұл шамамен тексеру ғана.")
else:
    say("Әзірге жоқ. Сайттан жүктелген answers_R01.csv ... файлдарын data/real/ папкасына салып, "
        "evaluate.py-ды қайта іске қосыңыз.")

with open("results/evaluation.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\nСақталды: results/evaluation.txt, results/confusion_matrix.png")
