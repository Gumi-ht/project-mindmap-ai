"""
app.py — «Білім картасы» сайты (Streamlit).

Беттер:
  1. Тест            — 32 сұрақ, әр жауаптан кейін «сенімдімін / болжадым»;
  2. Менің картам    — тақырыптар майндмапы, түсі — деңгей;
  3. Оқу бағытым     — әлсіз тақырыптар дұрыс ретпен, материал және жаттығу сұрақтары;
  4. Мұғалім         — топтың нәтижесі (тест бөлігіндегі 60 жасанды студент);
  5. Демо            — қорғауға арналған 3 дайын мысал.

Дизайн: өз CSS стильдері мен анимациялары (st.html арқылы).
Іске қосу: streamlit run app.py
"""
import html as h
import time

import joblib
import pandas as pd
import streamlit as st

from logic import FEATURES, LEVEL_NAMES, build_route, load_prereqs, predict_levels, topic_features

st.set_page_config(page_title="Білім картасы", layout="wide")

# Деңгей түстері: (мәтін/жиек түсі, фон түсі)
LV = {0: ("#b42318", "#fdecea"), 1: ("#b54708", "#fef3e2"), 2: ("#067647", "#e7f6ec")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&display=swap');
:root{--ink:#101828;--muted:#667085;--line:#e4e7ec;--bg:#f5f6f8;--accent:#0e7490;--accent2:#6366f1;}
html,body,.stApp,[class*="css"],button,input,label,p,div{font-family:'Golos Text',system-ui,sans-serif;}
.stApp{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1120px;}
h1,h2,h3{letter-spacing:-0.02em;color:var(--ink);}

/* бүйір мәзір */
[data-testid="stSidebar"]{background:#0f172a;}
[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
[data-testid="stSidebar"] div[role="radiogroup"] label{padding:9px 12px;border-radius:10px;margin:2px 0;transition:background .2s;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(255,255,255,.06);}
.brand{font-size:20px;font-weight:700;color:#fff !important;letter-spacing:-.02em;margin:4px 0 2px;}
.brand-sub{font-size:12.5px;color:#64748b !important;margin-bottom:18px;}

/* анимациялар */
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:translate(-50%,-50%) scale(.82)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes grow{from{width:0}}
@keyframes drift{0%,100%{transform:translate(0,0)}50%{transform:translate(40px,-24px)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(180,35,24,.35)}50%{box-shadow:0 0 0 8px rgba(180,35,24,0)}}

/* hero */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 34px;background:#0f172a;animation:rise .6s ease both;margin-bottom:6px;}
.hero h1{color:#fff;margin:0 0 6px;font-size:2rem;font-weight:700;position:relative;}
.hero p{color:#cbd5e1;margin:0;max-width:660px;position:relative;line-height:1.55;}
.hero .eyebrow{color:#67e8f9;font-size:12px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;position:relative;}
.blob{position:absolute;border-radius:50%;filter:blur(46px);opacity:.5;animation:drift 14s ease-in-out infinite;}

/* карточкалар */
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:22px 24px;animation:rise .5s ease both;}
.q-meta{font-size:12.5px;font-weight:600;color:var(--accent);letter-spacing:.06em;text-transform:uppercase;}
.q-text{font-size:21px;font-weight:600;color:var(--ink);line-height:1.4;margin-top:8px;}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0 14px;}
.stat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 18px;animation:rise .5s ease both;}
.stat .v{font-size:28px;font-weight:700;color:var(--ink);}
.stat .l{font-size:13px;color:var(--muted);}

/* майндмап */
.map-wrap{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:18px;padding:6px;animation:rise .5s ease both;}
.map{position:relative;height:430px;min-width:780px;}
.map svg{position:absolute;inset:0;width:100%;height:100%;}
.map path{fill:none;stroke:#cbd5e1;stroke-width:2;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1s ease forwards;}
.node{position:absolute;width:17.5%;opacity:0;animation:pop .55s cubic-bezier(.2,.8,.25,1.15) forwards;}
.node-in{padding:11px 13px;border-radius:14px;border:1.5px solid;transition:transform .2s ease, box-shadow .2s ease;cursor:default;}
.node-in:hover{transform:translateY(-3px) scale(1.03);box-shadow:0 10px 26px rgba(16,24,40,.12);}
.node .id{font-size:11px;font-weight:700;letter-spacing:.08em;opacity:.8;}
.node .nm{font-size:13px;font-weight:600;line-height:1.28;color:var(--ink);margin-top:3px;}
.node .bar{height:4px;border-radius:4px;background:rgba(16,24,40,.08);margin-top:9px;overflow:hidden;}
.node .bar i{display:block;height:100%;border-radius:4px;animation:grow 1.1s ease both;}
.node.weak .node-in{animation:pulse 2.4s ease-in-out 1.6s 2;}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 6px 4px;color:var(--muted);font-size:13px;align-items:center;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;}

/* кесте */
.tbl{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid var(--line);border-radius:18px;overflow:hidden;animation:rise .5s ease both;margin-top:14px;}
.tbl th{font-size:12px;font-weight:600;color:var(--muted);text-align:left;padding:12px 16px;background:#fafbfc;border-bottom:1px solid var(--line);letter-spacing:.04em;text-transform:uppercase;}
.tbl td{padding:12px 16px;border-bottom:1px solid #f1f2f4;font-size:14px;color:var(--ink);}
.tbl tr:last-child td{border-bottom:none;}
.pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap;}
.pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;}
.mini{height:6px;width:120px;border-radius:6px;background:#eef0f3;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px;}
.mini i{display:block;height:100%;border-radius:6px;animation:grow 1s ease both;}

/* оқу бағыты */
.route{position:relative;margin:8px 0 0 18px;border-left:2px dashed #d0d5dd;padding-left:30px;}
.step{position:relative;margin-bottom:14px;animation:rise .5s ease both;}
.step .num{position:absolute;left:-50px;top:20px;width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:#fff;font-size:15px;box-shadow:0 0 0 5px var(--bg);}
.step h4{margin:0 0 4px;font-size:17px;color:var(--ink);}
.step .why{font-size:13px;font-weight:600;margin-bottom:8px;}
.step p{margin:0 0 10px;color:#475467;line-height:1.55;font-size:14.5px;}
.step a{color:var(--accent);font-weight:600;text-decoration:none;}
.done{background:#e7f6ec;color:#067647;border-radius:16px;padding:18px 22px;font-weight:600;animation:rise .5s ease both;}

/* мұғалім */
.bars .row{display:grid;grid-template-columns:minmax(160px,300px) 1fr 44px;align-items:center;gap:14px;margin:9px 0;font-size:14px;color:var(--ink);}
.track{height:12px;background:#eef0f3;border-radius:6px;overflow:hidden;}
.fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#f97066,#b42318);animation:grow 1.1s ease both;}
.heat{display:grid;grid-template-columns:64px repeat(8,1fr);gap:4px;font-size:11.5px;color:var(--muted);}
.heat .c{height:16px;border-radius:5px;animation:rise .4s ease both;}
.heat .hd{text-align:center;font-weight:600;}

/* Streamlit виджеттері */
.stButton>button{border-radius:12px;padding:.6rem 1.3rem;font-weight:600;border:1px solid var(--line);transition:transform .15s, box-shadow .2s;}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(14,116,144,.18);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#0e7490,#6366f1);border:none;color:#fff;}
.stDownloadButton>button{border-radius:12px;font-weight:600;}
.main div[role="radiogroup"] label{background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:3px 0;transition:border-color .15s, background .15s;}
.main div[role="radiogroup"] label:hover{border-color:#0e7490;background:#f0fbfd;}
[data-testid="stProgress"] div[role="progressbar"] > div > div > div{background:linear-gradient(90deg,#0e7490,#6366f1);}
[data-testid="stExpander"]{background:#fff;border-radius:14px;border:1px solid var(--line);}
@media (max-width:700px){.stats{grid-template-columns:1fr}.hero{padding:22px}.hero h1{font-size:1.5rem}}
</style>
"""
st.html(CSS)


# ---------- Деректерді жүктеу ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/topics.csv"), pd.read_csv("data/questions.csv")


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
prereqs = load_prereqs(topics)


def levels_from_answers(answers_df):
    feats = topic_features(answers_df, questions, topics)
    return predict_levels(feats, model), feats


def pill(level):
    fg, bg = LV[level]
    return f'<span class="pill" style="color:{fg};background:{bg}">{LEVEL_NAMES[level]}</span>'


def hero(eyebrow, title, text):
    st.html(f"""
    <div class="hero">
      <div class="blob" style="width:260px;height:260px;background:#0e7490;right:-40px;top:-80px"></div>
      <div class="blob" style="width:220px;height:220px;background:#6366f1;right:180px;bottom:-120px;animation-delay:-5s"></div>
      <div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p>
    </div>""")


# ---------- Майндмап ----------
def layout():
    """Әр тақырыптың орны: баған = тереңдік (базалық тақырыптар сол жақта), бағандағы орны — ата-аналарына қарай."""
    depth = {}

    def d(t):
        if t not in depth:
            depth[t] = 0 if not prereqs[t] else 1 + max(d(p) for p in prereqs[t])
        return depth[t]

    for t in topics["topic_id"]:
        d(t)
    max_d = max(depth.values()) or 1
    pos = {}
    for col in range(max_d + 1):
        ids = [t for t in topics["topic_id"] if depth[t] == col]
        ids.sort(key=lambda t: sum(pos[p][1] for p in prereqs[t]) / len(prereqs[t]) if prereqs[t] else 0)
        n = len(ids)
        ys = [215] if n == 1 else [90 + i * (250 / (n - 1)) for i in range(n)]
        for t, y in zip(ids, ys):
            pos[t] = (90 + col * (820 / max_d), y)
    return pos, depth


def show_map(levels, feats):
    pos, depth = layout()
    paths = []
    for t, ps in prereqs.items():
        for p in ps:
            (x1, y1), (x2, y2) = pos[p], pos[t]
            xm = (x1 + x2) / 2
            paths.append(f'<path pathLength="1" vector-effect="non-scaling-stroke" '
                         f'style="animation-delay:{0.25 + depth[p] * 0.2:.2f}s" '
                         f'd="M{x1:.0f} {y1:.0f} C{xm:.0f} {y1:.0f},{xm:.0f} {y2:.0f},{x2:.0f} {y2:.0f}"/>')
    nodes = []
    for i, t in enumerate(topics["topic_id"]):
        x, y = pos[t]
        lv = levels[t]
        fg, bg = LV[lv]
        score = int(round(feats.loc[t, "score"] * 100))
        nodes.append(
            f'<div class="node{" weak" if lv == 0 else ""}" '
            f'style="left:{x / 10:.1f}%;top:{y / 430 * 100:.1f}%;animation-delay:{depth[t] * 0.2 + i * 0.04:.2f}s">'
            f'<div class="node-in" style="background:{bg};border-color:{fg}">'
            f'<div class="id" style="color:{fg}">{t} · {LEVEL_NAMES[lv]}</div>'
            f'<div class="nm">{h.escape(topic_name[t])}</div>'
            f'<div class="bar"><i style="width:{score}%;background:{fg};animation-delay:{0.6 + depth[t] * 0.2:.2f}s"></i></div>'
            f'</div></div>')
    legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                     for k in (2, 1, 0))
    st.html(f"""
    <div class="map-wrap"><div class="map">
      <svg viewBox="0 0 1000 430" preserveAspectRatio="none">{''.join(paths)}</svg>
      {''.join(nodes)}
    </div></div>
    <div class="legend">{legend}<span style="margin-left:auto">Жолақ — дұрыс жауаптар үлесі · сызық базалық тақырыптан күрделіге қарай</span></div>""")


def show_table(levels, feats):
    rows = []
    for i, t in enumerate(feats.index):
        s = int(round(feats.loc[t, "score"] * 100))
        g = int(round(feats.loc[t, "guess_rate"] * 100))
        fg = LV[levels[t]][0]
        rows.append(f'<tr style="animation:rise .4s ease both;animation-delay:{i * 0.05:.2f}s">'
                    f'<td><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</td>'
                    f'<td><span class="mini"><i style="width:{s}%;background:{fg}"></i></span>{s}%</td>'
                    f'<td>{g}%</td><td>{pill(levels[t])}</td></tr>')
    st.html(f'<table class="tbl"><tr><th>Тақырып</th><th>Дұрыс жауаптар</th><th>Болжадым</th>'
            f'<th>Деңгей</th></tr>{"".join(rows)}</table>')


def show_route(levels, key_prefix="r"):
    route = build_route(levels, topics)
    if not route:
        st.html('<div class="done">Барлық тақырыптар жасыл — оқу бағыты қажет емес. Жарайсыз!</div>')
        return
    lecture = topics.set_index("topic_id")["material_lecture"]
    steps = []
    for i, s in enumerate(route, 1):
        fg, bg = LV[s["level"]]
        steps.append(f"""
        <div class="step" style="animation-delay:{i * 0.12:.2f}s">
          <div class="num" style="background:{fg}">{i}</div>
          <div class="card" style="animation:none">
            <h4>{s['topic_id']} · {h.escape(s['name'])}</h4>
            <div class="why" style="color:{fg}">{h.escape(s['reason'])}</div>
            <p>{h.escape(s['explanation'])}</p>
            <div style="font-size:13.5px;color:#667085">{h.escape(lecture[s['topic_id']])} ·
              <a href="{h.escape(s['material'])}" target="_blank">қосымша оқу →</a></div>
          </div>
        </div>""")
    st.html(f'<p style="color:#475467;margin:4px 0 6px">Сізге <b>{len(route)} тақырыпты</b> осы ретпен оқу ұсынылады — '
            f'алдымен базалық тақырыптар.</p><div class="route">{"".join(steps)}</div>')

    st.markdown("#### Өзіңізді тексеріңіз")
    for s in route:
        with st.expander(f"{s['topic_id']} · {s['name']}"):
            practice = questions[(questions["topic_id"] == s["topic_id"]) & (questions["use"] == "practice")]
            for q in practice.itertuples():
                opts = {"a": q.a, "b": q.b, "c": q.c, "d": q.d}
                choice = st.radio(q.question, list(opts), format_func=lambda k, o=opts: f"{k})  {o[k]}",
                                  index=None, key=f"{key_prefix}_{q.q_id}")
                if choice:
                    if choice == q.correct:
                        st.success("Дұрыс")
                    else:
                        st.error(f"Қате. Дұрыс жауап: {q.correct}) {opts[q.correct]}")


def show_stats(levels, feats):
    counts = {k: sum(1 for v in levels.values() if v == k) for k in (2, 1, 0)}
    tiles = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.08:.2f}s"><div class="v" style="color:{LV[k][0]}">{counts[k]}</div>'
        f'<div class="l">{LEVEL_NAMES[k]}</div></div>' for i, k in enumerate((2, 1, 0)))
    st.html(f'<div class="stats">{tiles}</div>')


# ---------- Бүйір мәзір ----------
with st.sidebar:
    st.html('<div class="brand">Білім картасы</div><div class="brand-sub">ICT1103 · топтық жоба</div>')
    page = st.radio("Бөлім", ["Тест", "Менің картам", "Оқу бағытым", "Мұғалім", "Демо"],
                    label_visibility="collapsed")

if "answers" not in st.session_state:
    st.session_state.answers = []
    st.session_state.idx = 0
    st.session_state.started = False


# ---------- 1. Тест ----------
if page == "Тест":
    if not st.session_state.started:
        hero("Білім деңгейін тексеру", "Қай тақырыпты білесіз, қайсысын жоқ?",
             "ICT1103 курсының 8 тақырыбы бойынша 32 сұрақ. Әр жауаптан кейін «сенімдімін» немесе "
             "«болжадым» деп белгілеңіз — жүйе олқылықтарыңызды тауып, жеке оқу бағытын құрады.")
        st.write("")
        code = st.text_input("Студент коды — атыңызды жазбаңыз (мысалы, R01)", value="R01")
        st.markdown("**Алдымен өзіңізді бағалаңыз:** әр тақырыпты қаншалықты білесіз?")
        self_eval = {}
        cols = st.columns(2)
        for i, t in enumerate(topics["topic_id"]):
            with cols[i % 2]:
                self_eval[t] = st.select_slider(f"{t} · {topic_name[t]}", options=[0, 1, 2], value=1,
                                                format_func=lambda v: LEVEL_NAMES[v], key=f"self_{t}")
        if st.button("Тестті бастау", type="primary"):
            st.session_state.update(started=True, idx=0, answers=[], code=code,
                                    self_eval=self_eval, q_start=time.time())
            st.rerun()

    elif st.session_state.idx < len(test_q):
        i = st.session_state.idx
        q = test_q.iloc[i]
        st.progress(i / len(test_q))
        st.html(f'<div class="card"><div class="q-meta">Сұрақ {i + 1} / {len(test_q)} · {q["topic_id"]} · '
                f'{h.escape(topic_name[q["topic_id"]])}</div><div class="q-text">{h.escape(q["question"])}</div></div>')
        opts = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"]}
        choice = st.radio("Жауапты таңдаңыз", list(opts), format_func=lambda k: f"{k})  {opts[k]}",
                          index=None, key=f"ans_{i}")
        conf = st.radio("Жауабыңызға сенімдісіз бе?", ["Сенімдімін", "Болжадым"], index=None,
                        horizontal=True, key=f"conf_{i}")
        if st.button("Келесі сұрақ", type="primary", disabled=(choice is None or conf is None)):
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
        hero("Тест аяқталды", f"Дұрыс жауаптар: {answers['correct'].sum()} / {len(answers)}",
             "Төменде — сіздің білім картаңыз. Толық оқу бағыты «Оқу бағытым» бөлімінде.")
        st.write("")
        show_stats(levels, feats)
        show_map(levels, feats)

        # 4-қатысушыға: нақты студенттерде тексеру үшін жауаптарды жүктеп алу
        out = answers.copy()
        out.insert(0, "student_id", st.session_state.code)
        out["topic_id"] = out["q_id"].map(dict(zip(questions["q_id"], questions["topic_id"])))
        out["self_level"] = out["topic_id"].map(st.session_state.self_eval)
        out["predicted_level"] = out["topic_id"].map(levels)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Жауаптарды жүктеп алу (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name=f"answers_{st.session_state.code}.csv", mime="text/csv")
        with c2:
            if st.button("Тестті қайта тапсыру"):
                st.session_state.update(started=False, idx=0, answers=[])
                st.rerun()


# ---------- 2. Менің картам ----------
elif page == "Менің картам":
    hero("Нәтиже", "Менің білім картам", "Әр тақырыптың түсі — деңгейі, сызықтар — тақырыптар арасындағы байланыс.")
    st.write("")
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        levels, feats = st.session_state.result
        show_stats(levels, feats)
        show_map(levels, feats)
        show_table(levels, feats)


# ---------- 3. Оқу бағытым ----------
elif page == "Оқу бағытым":
    hero("Жеке жоспар", "Менің оқу бағытым",
         "Әлсіз тақырыптар дұрыс ретпен: алдымен басқа тақырыптарға негіз болатындары.")
    st.write("")
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        show_route(st.session_state.result[0])


# ---------- 4. Мұғалім ----------
elif page == "Мұғалім":
    hero("Мұғалім панелі", "Топтың нәтижесі",
         "Мысал ретінде — тест бөлігіндегі 60 жасанды студент (data/test_students.csv).")
    st.write("")

    @st.cache_data
    def group_levels():
        answers = pd.read_csv("data/students.csv")
        ids = pd.read_csv("data/test_students.csv")["student_id"]
        return pd.DataFrame({sid: levels_from_answers(answers[answers["student_id"] == sid])[0]
                             for sid in ids}).T

    grid = group_levels()
    weak = (grid == 0).sum()
    worst = weak.idxmax()
    total = len(grid)
    st.html(f'<div class="card" style="border-left:4px solid #b42318"><b>Ең қиын тақырып: {worst} · '
            f'{h.escape(topic_name[worst])}</b><br><span style="color:#667085">{weak.max()} студент ({weak.max() * 100 // total}%) '
            f'бұл тақырыпты білмейді — оны қайта түсіндіру ұсынылады.</span></div>')

    rows = "".join(
        f'<div class="row"><div><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</div>'
        f'<div class="track"><div class="fill" style="width:{weak[t] / total * 100:.0f}%;animation-delay:{i * 0.07:.2f}s"></div></div>'
        f'<div style="text-align:right;font-weight:600">{weak[t]}</div></div>'
        for i, t in enumerate(weak.sort_values(ascending=False).index))
    st.html(f'<div class="card" style="margin-top:12px"><h3 style="margin:0 0 6px;font-size:18px">'
            f'Қанша студент тақырыпты білмейді</h3><div class="bars">{rows}</div></div>')

    cells = ['<div></div>'] + [f'<div class="hd">{t}</div>' for t in grid.columns]
    for r, (sid, row) in enumerate(grid.head(20).iterrows()):
        cells.append(f'<div>{sid}</div>')
        for t in grid.columns:
            cells.append(f'<div class="c" title="{sid} · {t}: {LEVEL_NAMES[row[t]]}" '
                         f'style="background:{LV[row[t]][1]};border:1px solid {LV[row[t]][0]}33;'
                         f'animation-delay:{min(r * 0.015, 0.9):.2f}s"></div>')
    legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                     for k in (2, 1, 0))
    st.html(f'<div class="card" style="margin-top:12px"><h3 style="margin:0 0 12px;font-size:18px">'
            f'Жылу картасы: студент × тақырып <span style="font-weight:400;color:#667085;font-size:14px">(алғашқы 20 студент)</span></h3><div class="heat">{"".join(cells)}</div>'
            f'<div class="legend">{legend}</div></div>')


# ---------- 5. Демо ----------
elif page == "Демо":
    hero("Қорғауға арналған", "Үш студент — үш түрлі жағдай",
         "Тест бөлігіндегі жасанды студенттер: модель оларды оқыту кезінде көрмеген.")
    st.write("")

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
        return {"Күшті студент": strong[0],
                "Базалық тақырыпта олқылық": base_gap[0],
                "Көп болжаған студент": guesser[0]}

    demos = demo_students()
    choice = st.radio("Мысал", list(demos), horizontal=True, label_visibility="collapsed")
    sid = demos[choice]
    answers = pd.read_csv("data/students.csv")
    levels, feats = levels_from_answers(answers[answers["student_id"] == sid])
    st.session_state.result = (levels, feats)
    st.caption(f"Студент коды: {sid}")
    show_stats(levels, feats)
    show_map(levels, feats)
    show_table(levels, feats)
    st.markdown("### Оқу бағыты")
    show_route(levels, key_prefix=f"demo_{sid}")
