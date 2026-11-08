"""
generate_data.py — жасанды (синтетикалық) студенттерді жасау.

Не делает скрипт:
1. Каждому студенту случайно задаёт настоящий уровень по каждой теме:
   0 — не знает, 1 — частично, 2 — знает.
   Правило: если базовая тема слабая, зависимая тема тоже чаще слабая.
2. По этому уровню «отвечает» на 32 тестовых вопроса:
   кто знает тему — чаще отвечает правильно, уверенно и быстро;
   кто не знает — иногда угадывает (шанс 25% при 4 вариантах).
3. Сохраняет:
   data/students.csv     — ответы (что видит модель);
   data/true_levels.csv  — настоящие уровни (правильные ответы для обучения и проверки).

Запуск: python generate_data.py
"""
import numpy as np
import pandas as pd

N_STUDENTS = 300
SEED = 42  # фиксируем случайность, чтобы результат повторялся

rng = np.random.default_rng(SEED)

topics = pd.read_csv("data/topics.csv")
questions = pd.read_csv("data/questions.csv")
questions = questions[questions["use"] == "test"]  # только 32 тестовых вопроса


def prereq_list(value):
    """'T2;T3' -> ['T2', 'T3'], пустое значение -> []"""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return str(value).split(";")


prereqs = {row.topic_id: prereq_list(row.prereq) for row in topics.itertuples()}

# Вероятность правильного ответа в зависимости от уровня
P_CORRECT = {0: 0.25, 1: 0.55, 2: 0.90}
# Вероятность, что студент отметит «уверен»
P_CONFIDENT = {0: 0.20, 1: 0.50, 2: 0.85}
# Среднее время ответа в секундах
MEAN_TIME = {0: 40, 1: 33, 2: 25}

answer_rows = []
level_rows = []

for s in range(1, N_STUDENTS + 1):
    student_id = f"S{s:03d}"
    # Общая «сила» студента: у сильных больше шанс знать темы
    skill = rng.uniform(0.15, 0.85)
    levels = {}

    # Темы в файле идут по порядку: базовые раньше зависимых
    for t in topics["topic_id"]:
        level = rng.choice([0, 1, 2], p=[(1 - skill) ** 2,
                                         2 * skill * (1 - skill),
                                         skill ** 2])
        # Если базовая тема не знает (0), зависимая чаще всего не выше «частично»
        base_levels = [levels[p] for p in prereqs[t]]
        if base_levels and min(base_levels) == 0 and rng.random() < 0.85:
            level = min(level, 1)
        levels[t] = int(level)
        level_rows.append({"student_id": student_id, "topic_id": t, "true_level": int(level)})

    for q in questions.itertuples():
        level = levels[q.topic_id]
        p = P_CORRECT[level]
        if level > 0:
            p -= 0.10 * (q.difficulty - 2)  # сложный вопрос немного труднее
        p = float(np.clip(p, 0.25, 0.97))
        correct = int(rng.random() < p)
        confident = int(rng.random() < P_CONFIDENT[level])
        time_sec = MEAN_TIME[level] * (0.8 + 0.2 * q.difficulty) * rng.lognormal(0, 0.5)
        answer_rows.append({
            "student_id": student_id,
            "q_id": q.q_id,
            "topic_id": q.topic_id,
            "correct": correct,
            "confident": confident,
            "time_sec": round(time_sec, 1),
        })

pd.DataFrame(answer_rows).to_csv("data/students.csv", index=False)
pd.DataFrame(level_rows).to_csv("data/true_levels.csv", index=False)
print(f"Готово: {N_STUDENTS} студентов, {len(answer_rows)} ответов")
