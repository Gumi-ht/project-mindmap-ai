"""
logic.py — общие функции, которые используют и модель (train.py), и сайт (app.py).

- topic_features()  — по ответам одного студента считает 4 признака для каждой темы;
- rule_levels()     — простое правило без ML (для сравнения с моделью);
- predict_levels()  — уровень темы по обученной модели;
- build_route()     — персональный маршрут: слабые темы в правильном порядке.
"""
import pandas as pd

LEVEL_NAMES = {0: "Білмейді", 1: "Жартылай біледі", 2: "Біледі"}
FEATURES = ["score", "guess_rate", "avg_time", "prereq_score"]


def prereq_list(value):
    """'T2;T3' -> ['T2', 'T3'], пустое значение -> []"""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return str(value).split(";")


def load_prereqs(topics):
    return {row.topic_id: prereq_list(row.prereq) for row in topics.itertuples()}


def topic_features(answers, questions, topics):
    """
    answers: таблица ответов ОДНОГО студента (q_id, correct, confident, time_sec).
    Возвращает таблицу: одна строка = одна тема, 4 признака:
      score        — доля правильных с учётом сложности (сложный вопрос весит больше);
      guess_rate   — доля ответов, где студент отметил «угадал»;
      avg_time     — среднее время ответа в секундах;
      prereq_score — score по базовой теме (если базовых несколько — самый низкий).
    """
    a = answers.merge(questions[["q_id", "topic_id", "difficulty"]], on="q_id", how="left",
                      suffixes=("_drop", ""))
    a = a.drop(columns=[c for c in a.columns if c.endswith("_drop")])
    a["weighted"] = a["correct"] * a["difficulty"]

    rows = []
    for t in topics["topic_id"]:
        part = a[a["topic_id"] == t]
        if len(part) == 0:
            rows.append({"topic_id": t, "score": 0.0, "guess_rate": 1.0, "avg_time": 60.0})
            continue
        rows.append({
            "topic_id": t,
            "score": part["weighted"].sum() / part["difficulty"].sum(),
            "guess_rate": 1 - part["confident"].mean(),
            "avg_time": part["time_sec"].mean(),
        })
    feats = pd.DataFrame(rows).set_index("topic_id")

    prereqs = load_prereqs(topics)
    feats["prereq_score"] = [
        min(feats.loc[p, "score"] for p in prereqs[t]) if prereqs[t] else 1.0
        for t in feats.index
    ]
    return feats


def rule_levels(feats):
    """Простое правило без ML: >=70% — знает, 40–70% — частично, <40% — не знает."""
    return {t: 2 if s >= 0.7 else (1 if s >= 0.4 else 0) for t, s in feats["score"].items()}


def predict_levels(feats, model):
    """Уровень каждой темы по обученной модели."""
    preds = model.predict(feats[FEATURES])
    return dict(zip(feats.index, [int(p) for p in preds]))


def build_route(levels, topics):
    """
    Маршрут: темы с уровнем 0 и 1, отсортированные так, чтобы базовая тема
    всегда шла раньше зависимой. Среди доступных — сначала «не знает», потом «частично».
    Возвращает список словарей: topic_id, name, level, reason, material, explanation.
    """
    prereqs = load_prereqs(topics)
    info = topics.set_index("topic_id")
    order = list(topics["topic_id"])
    weak = [t for t in order if levels.get(t, 0) < 2]

    route, done = [], set()
    while len(route) < len(weak):
        # доступны темы, у которых все слабые базовые темы уже в маршруте
        ready = [t for t in weak if t not in done
                 and all(p in done or p not in weak for p in prereqs[t])]
        ready.sort(key=lambda t: (levels[t], order.index(t)))
        t = ready[0]
        done.add(t)

        needed_for = [d for d in weak if t in prereqs[d]]
        if needed_for:
            reason = "Негізгі тақырып: " + ", ".join(needed_for) + " үшін қажет"
        else:
            reason = LEVEL_NAMES[levels[t]]
        route.append({
            "topic_id": t,
            "name": info.loc[t, "name"],
            "level": levels[t],
            "reason": reason,
            "material": info.loc[t, "material_link"],
            "explanation": info.loc[t, "explanation"],
        })
    return route
