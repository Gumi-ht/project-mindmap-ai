"""
app.py — «Білім картасы» сайты (Streamlit).

Беттер:
  1. Тест            — 32 сұрақ, әр жауаптан кейін «сенімдімін / болжадым»;
  2. Менің картам    — тақырыптар майндмапы, түсі — деңгей;
  3. Оқу бағытым     — әлсіз тақырыптар дұрыс ретпен, материал және жаттығу сұрақтары;
  4. Мұғалім         — топтың жылу картасы (тест бөлігіндегі 60 жасанды студент);
  5. Демо            — қорғауға арналған 3 дайын мысал.

Іске қосу: streamlit run app.py
"""
import time

import joblib
import pandas as pd
import streamlit as st

from logic import LEVEL_NAMES, build_route, load_prereqs, predict_levels, topic_features

st.set_page_config(page_title="Білім картасы", page_icon="🧭", layout="wide")

COLORS = {0: "#f8d7da", 1: "#fff3cd", 2: "#d4edda"}      # қызыл / сары / жасыл
BORDERS = {0: "#c0392b", 1: "#d4a017", 2: "#27ae60"}
ICONS = {0: "🔴", 1: "🟡", 2: "🟢"}


# ---------- Деректерді жүктеу ----------
@st.cache_data
def load_data():
    topics = pd.read_csv("data/topics.csv")
    questions = pd.read_csv("data/questions.csv")
    return topics, questions


@st.cache_resource
def load_model():
    """model.joblib жүктейміз. Файл жоқ болса немесе ашылмаса — модельді деректерден қайта оқытамыз
    (train.py-дағыдай: Logistic Regression, тек оқыту бөлігіндегі студенттер)."""
    try:
        return joblib.load("model.joblib")
    except Exception:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from logic import FEATURES
        topics_, questions_ = load_data()
        answers = pd.read_csv("data/students.csv")
        true_levels = pd.read_csv("data/true_levels.csv")
        test_ids = set(pd.read_csv("data/test_students.csv")["student_id"])
        parts = []
        for sid, part in answers.groupby("student_id"):
            if sid in test_ids:
                continue
            f = topic_features(part, questions_, topics_).reset_index()
            f["student_id"] = sid
            parts.append(f)
        data = pd.concat(parts).merge(true_levels, on=["student_id", "topic_id"])
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
            data[FEATURES], data["true_level"])


topics, questions = load_data()
model = load_model()
test_q = questions[questions["use"] == "test"].sample(frac=1, random_state=7).reset_index(drop=True)
topic_name = dict(zip(topics["topic_id"], topics["name"]))


def levels_from_answers(answers_df):
    feats = topic_features(answers_df, questions, topics)
    return predict_levels(feats, model), feats


# ---------- Көрсету функциялары ----------
def show_map(levels):
    """Майндмап: Graphviz DOT, түйін түсі — деңгей."""
    prereqs = load_prereqs(topics)
    lines = ['digraph G {', 'rankdir=TB; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=2];']
    for t in topics["topic_id"]:
        lv = levels[t]
        label = f"{t}\\n{topic_name[t]}".replace('"', "'")
        lines.append(f'{t} [label="{label}", fillcolor="{COLORS[lv]}", color="{BORDERS[lv]}"];')
    for t, ps in prereqs.items():
        for p in ps:
            lines.append(f"{p} -> {t};")
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)
    st.caption("🟢 біледі   🟡 жартылай біледі   🔴 білмейді   · Нұсқа базалық тақырыптан тәуелді тақырыпқа қарай")


def show_table(levels, feats):
    df = pd.DataFrame({
        "Тақырып": [f"{t} — {topic_name[t]}" for t in feats.index],
        "Дұрыс (%)": (feats["score"] * 100).round(0).astype(int).values,
        "Болжадым (%)": (feats["guess_rate"] * 100).round(0).astype(int).values,
        "Деңгей": [f"{ICONS[levels[t]]} {LEVEL_NAMES[levels[t]]}" for t in feats.index],
    })
    st.dataframe(df, hide_index=True, use_container_width=True)


def show_route(levels, key_prefix="r"):
    route = build_route(levels, topics)
    if not route:
        st.success("Барлық тақырыптар жасыл — оқу бағыты қажет емес. Жарайсыз! 🎉")
        return
    st.write(f"Сізге **{len(route)} тақырыпты** осы ретпен оқу ұсынылады:")
    for i, step in enumerate(route, 1):
        with st.expander(f"{i}. {ICONS[step['level']]} {step['topic_id']} — {step['name']}", expanded=(i == 1)):
            st.markdown(f"**Неге бағытта:** {step['reason']}")
            st.markdown(step["explanation"])
            st.markdown(f"📖 Материал: {topics.set_index('topic_id').loc[step['topic_id'], 'material_lecture']}, "
                        f"[қосымша оқу]({step['material']})")
            st.markdown("**Өзіңізді тексеріңіз:**")
            practice = questions[(questions["topic_id"] == step["topic_id"]) & (questions["use"] == "practice")]
            for q in practice.itertuples():
                opts = {"a": q.a, "b": q.b, "c": q.c, "d": q.d}
                choice = st.radio(q.question, list(opts), format_func=lambda k, o=opts: f"{k}) {o[k]}",
                                  index=None, key=f"{key_prefix}_{q.q_id}")
                if choice:
                    if choice == q.correct:
                        st.success("Дұрыс!")
                    else:
                        st.error(f"Қате. Дұрыс жауап: {q.correct}) {opts[q.correct]}")


# ---------- Беттер ----------
page = st.sidebar.radio("Бөлім", ["📝 Тест", "🗺️ Менің картам", "🧭 Оқу бағытым", "👩‍🏫 Мұғалім", "🎬 Демо"])
st.sidebar.markdown("---")
st.sidebar.caption("Білім картасы · ICT1103 топтық жобасы")

if "answers" not in st.session_state:
    st.session_state.answers = []
    st.session_state.idx = 0
    st.session_state.started = False


# 1. ТЕСТ
if page == "📝 Тест":
    st.title("📝 Білім деңгейін тексеру")

    if not st.session_state.started:
        st.write("Тест ICT1103 курсының 8 тақырыбы бойынша 32 сұрақтан тұрады. "
                 "Әр жауаптан кейін «сенімдімін» немесе «болжадым» деп белгілеңіз — бұл олқылықтарды дәлірек табуға көмектеседі.")
        code = st.text_input("Студент коды (атыңызды жазбаңыз, мысалы R01)", value="R01")
        st.markdown("**Алдымен өзіңізді бағалаңыз:** әр тақырыпты қаншалықты білесіз?")
        self_eval = {}
        cols = st.columns(2)
        for i, t in enumerate(topics["topic_id"]):
            with cols[i % 2]:
                self_eval[t] = st.select_slider(f"{t} — {topic_name[t]}", options=[0, 1, 2], value=1,
                                                format_func=lambda v: LEVEL_NAMES[v], key=f"self_{t}")
        if st.button("Тестті бастау ▶️", type="primary"):
            st.session_state.update(started=True, idx=0, answers=[], code=code,
                                    self_eval=self_eval, q_start=time.time())
            st.rerun()

    elif st.session_state.idx < len(test_q):
        i = st.session_state.idx
        q = test_q.iloc[i]
        st.progress(i / len(test_q), text=f"Сұрақ {i + 1} / {len(test_q)}")
        st.subheader(q["question"])
        opts = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"]}
        choice = st.radio("Жауапты таңдаңыз:", list(opts), format_func=lambda k: f"{k}) {opts[k]}",
                          index=None, key=f"ans_{i}")
        conf = st.radio("Жауабыңызға сенімдісіз бе?", ["Сенімдімін", "Болжадым"], index=None,
                        horizontal=True, key=f"conf_{i}")
        if st.button("Келесі ➡️", type="primary", disabled=(choice is None or conf is None)):
            st.session_state.answers.append({
                "q_id": q["q_id"],
                "correct": int(choice == q["correct"]),
                "confident": int(conf == "Сенімдімін"),
                "time_sec": round(time.time() - st.session_state.q_start, 1),
            })
            st.session_state.idx += 1
            st.session_state.q_start = time.time()
            st.rerun()

    else:
        answers = pd.DataFrame(st.session_state.answers)
        levels, feats = levels_from_answers(answers)
        st.session_state.result = (levels, feats)
        st.success(f"Тест аяқталды! Дұрыс жауаптар: {answers['correct'].sum()} / {len(answers)}. "
                   "Нәтижені «Менің картам» және «Оқу бағытым» бөлімдерінен көріңіз.")
        show_map(levels)

        # 4-қатысушыға: нақты студенттерде тексеру үшін жауаптарды жүктеп алу
        out = answers.copy()
        out.insert(0, "student_id", st.session_state.code)
        out["topic_id"] = out["q_id"].map(dict(zip(questions["q_id"], questions["topic_id"])))
        out["self_level"] = out["topic_id"].map(st.session_state.self_eval)
        out["predicted_level"] = out["topic_id"].map(levels)
        st.download_button("⬇️ Жауаптарды жүктеп алу (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"answers_{st.session_state.code}.csv", mime="text/csv")
        if st.button("Тестті қайта тапсыру 🔄"):
            st.session_state.update(started=False, idx=0, answers=[])
            st.rerun()


# 2. МЕНІҢ КАРТАМ
elif page == "🗺️ Менің картам":
    st.title("🗺️ Менің білім картам")
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        levels, feats = st.session_state.result
        show_map(levels)
        show_table(levels, feats)


# 3. ОҚУ БАҒЫТЫМ
elif page == "🧭 Оқу бағытым":
    st.title("🧭 Менің оқу бағытым")
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        show_route(st.session_state.result[0])


# 4. МҰҒАЛІМ
elif page == "👩‍🏫 Мұғалім":
    st.title("👩‍🏫 Топтың нәтижесі")
    st.caption("Мысал: тест бөлігіндегі 60 жасанды студент (data/test_students.csv)")

    @st.cache_data
    def group_levels():
        answers = pd.read_csv("data/students.csv")
        ids = pd.read_csv("data/test_students.csv")["student_id"]
        rows = {}
        for sid in ids:
            lv, _ = levels_from_answers(answers[answers["student_id"] == sid])
            rows[sid] = lv
        return pd.DataFrame(rows).T

    grid = group_levels()
    weak = (grid == 0).sum().rename(lambda t: f"{t} {topic_name[t]}")
    st.subheader("Қанша студент тақырыпты білмейді")
    st.bar_chart(weak, horizontal=True, color="#c0392b")
    worst = weak.idxmax()
    st.warning(f"Ең қиын тақырып: **{worst}** — {weak.max()} студент білмейді. Оны қайта түсіндіру ұсынылады.")

    st.subheader("Жылу картасы: студент × тақырып")
    styled = grid.style.map(lambda v: f"background-color: {COLORS[v]}; color: {BORDERS[v]}; font-weight: 600")
    st.dataframe(styled, use_container_width=True, height=420)


# 5. ДЕМО
elif page == "🎬 Демо":
    st.title("🎬 Демо: үш студент")
    st.write("Қорғауға арналған дайын мысалдар (тест бөлігіндегі жасанды студенттер).")

    @st.cache_data
    def demo_students():
        answers = pd.read_csv("data/students.csv")
        ids = pd.read_csv("data/test_students.csv")["student_id"]
        stats = []
        for sid in ids:
            lv, f = levels_from_answers(answers[answers["student_id"] == sid])
            stats.append((sid, lv, f["score"].mean(), f["guess_rate"].mean()))
        strong = max(stats, key=lambda s: s[2])
        base_gap = next((s for s in stats if s[1]["T4"] == 0 and s[1]["T6"] < 2 and s[1]["T1"] == 2), stats[0])
        guesser = max((s for s in stats if s[0] not in (strong[0], base_gap[0])), key=lambda s: s[3])
        return {"💪 Күшті студент": strong[0],
                "🧱 Базалық тақырыпта олқылық": base_gap[0],
                "🎲 Көп болжаған студент": guesser[0]}

    demos = demo_students()
    choice = st.radio("Мысалды таңдаңыз:", list(demos), horizontal=True)
    sid = demos[choice]
    answers = pd.read_csv("data/students.csv")
    levels, feats = levels_from_answers(answers[answers["student_id"] == sid])
    st.session_state.result = (levels, feats)
    st.caption(f"Студент коды: {sid}")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_map(levels)
    with c2:
        show_table(levels, feats)
    st.subheader("Оқу бағыты")
    show_route(levels, key_prefix=f"demo_{sid}")
