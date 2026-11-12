"""
app.py — «Білім картасы» сайты (Streamlit).

Беттер (жоғарғы мәзір):
  1. Тест            — 32 сұрақ, әр жауаптан кейін «сенімдімін / болжадым»;
  2. Менің картам    — тақырыптар майндмапы, түсі — деңгей;
  3. Оқу бағытым     — әлсіз тақырыптар дұрыс ретпен, материал және жаттығу сұрақтары;
  4. Мұғалім         — топтың нәтижесі (тест бөлігіндегі 60 жасанды студент);
  5. Демо            — қорғауға арналған 3 дайын мысал.

Дизайн: қараңғы тақырып, өз CSS стильдері мен анимациялары (st.html арқылы).
Іске қосу: streamlit run app.py
"""
import base64
import html as h
import time

import joblib
import pandas as pd
import streamlit as st

from logic import FEATURES, LEVEL_NAMES, build_route, load_prereqs, predict_levels, topic_features

st.set_page_config(page_title="Білім картасы", layout="wide", initial_sidebar_state="collapsed")

# Деңгей түстері: (негізгі түс, мөлдір фон)
LV = {0: ("#f87171", "rgba(248,113,113,.13)"),
      1: ("#fbbf24", "rgba(251,191,36,.13)"),
      2: ("#34d399", "rgba(52,211,153,.13)")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700;800&display=swap');
:root{color-scheme:dark;--ink:#eef1f8;--muted:#8b93a7;--line:rgba(255,255,255,.09);--glass:rgba(255,255,255,.045);
      --bg:#070b14;--c1:#22d3ee;--c2:#8b5cf6;--c3:#f472b6;}
html,body,.stApp,[class*="css"],button,input,label,p,div,span{font-family:'Golos Text',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none;}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:2.2rem;max-width:1140px;position:relative;z-index:1;}
h1,h2,h3,h4{letter-spacing:-.02em;color:var(--ink);}

/* ---------- жанды фон: жылжитын жарық дақтар + тор ---------- */
.stApp::before{content:"";position:fixed;inset:-25%;z-index:0;pointer-events:none;
  background:radial-gradient(38% 32% at 20% 25%,rgba(34,211,238,.20),transparent 70%),
             radial-gradient(34% 30% at 80% 20%,rgba(139,92,246,.22),transparent 70%),
             radial-gradient(30% 28% at 60% 85%,rgba(244,114,182,.14),transparent 70%);
  animation:aurora 22s ease-in-out infinite alternate;}
.stApp::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
  background-size:46px 46px;mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%);}

/* ---------- анимациялар ---------- */
@keyframes aurora{0%{transform:translate(0,0) rotate(0)}50%{transform:translate(4%,-3%) rotate(8deg)}100%{transform:translate(-4%,3%) rotate(-6deg)}}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes enter{from{opacity:0;transform:translateY(22px) scale(.985);filter:blur(6px)}to{opacity:1;transform:none;filter:none}}
@keyframes word{from{opacity:0;transform:translateY(.6em) rotate(4deg);filter:blur(8px)}to{opacity:1;transform:none;filter:none}}
@keyframes cardIn{from{opacity:0;transform:perspective(900px) translateX(60px) rotateY(-14deg);filter:blur(8px)}to{opacity:1;transform:none;filter:none}}
@keyframes slideIn{from{opacity:0;transform:translateX(-26px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:translate(-50%,-50%) scale(.6)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}
@keyframes popIn{0%{opacity:0;transform:scale(.4) rotate(-90deg)}70%{transform:scale(1.12) rotate(8deg)}100%{opacity:1;transform:none}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes flow{to{stroke-dashoffset:-1}}
@keyframes grow{from{width:0}}
@keyframes growY{from{transform:scaleY(0)}}
@keyframes ring{from{stroke-dashoffset:100}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes pulse{0%{opacity:.9;transform:scale(1)}100%{opacity:0;transform:scale(1.35)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(34,211,238,.55)}50%{box-shadow:0 0 18px 3px rgba(34,211,238,.55)}}
@keyframes shine{0%{transform:translateX(-120%) skewX(-20deg)}60%,100%{transform:translateX(220%) skewX(-20deg)}}
@keyframes shimmer{to{background-position:200% center}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes up{0%{transform:translateY(0);opacity:0}15%{opacity:.8}100%{transform:translateY(-180px);opacity:0}}
@keyframes wave{from{opacity:0;transform:scale(.3)}to{opacity:1;transform:none}}

/* әр бет ауысқанда элементтер кезекпен шығады */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
.block-container [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]{animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3){animation-delay:.05s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(4){animation-delay:.12s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(5){animation-delay:.19s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(6){animation-delay:.26s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(7){animation-delay:.33s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(n+8){animation-delay:.4s}

/* ---------- жоғарғы бренд пен мәзір ---------- */
.topbar{display:flex;align-items:center;gap:14px;margin-bottom:4px;}
.logo{width:42px;height:42px;flex:none;border-radius:13px;filter:drop-shadow(0 0 12px rgba(34,211,238,.45));}
.brand{font-size:21px;font-weight:800;letter-spacing:-.03em;
  background:linear-gradient(90deg,#fff,#a5f3fc,#c4b5fd,#fff);background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:shimmer 6s linear infinite;}
.brand-sub{font-size:12.5px;color:var(--muted);margin-top:1px;}

.st-key-nav div[role="radiogroup"],.st-key-demo div[role="radiogroup"]{
  display:flex;flex-wrap:nowrap;gap:4px;padding:5px;border-radius:999px;width:fit-content;max-width:100%;overflow-x:auto;
  background:rgba(255,255,255,.05);border:1px solid var(--line);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);scrollbar-width:none;}
.st-key-nav div[role="radiogroup"]::-webkit-scrollbar{display:none;}
.st-key-nav [role="radiogroup"] > *,.st-key-demo [role="radiogroup"] > *{margin:0 !important;}
.st-key-nav :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]),.st-key-demo :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]){margin:0 !important;padding:9px 18px;border-radius:999px;white-space:nowrap;cursor:pointer;
  transition:background .35s ease,box-shadow .35s ease,transform .2s ease;}
.st-key-nav :is(label[data-baseweb="radio"] > div:first-child,[data-testid="stRadioOption"] > div > div:not(:last-child)),.st-key-demo :is(label[data-baseweb="radio"] > div:first-child,[data-testid="stRadioOption"] > div > div:not(:last-child)){display:none !important;}
.st-key-nav :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]) *,.st-key-demo :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]) *{color:var(--muted) !important;font-weight:600;transition:color .3s;}
.st-key-nav :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]):hover,.st-key-demo :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]):hover{background:rgba(255,255,255,.06);}
.st-key-nav :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]),.st-key-demo :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]){background:linear-gradient(135deg,var(--c1),var(--c2));box-shadow:0 6px 24px -6px rgba(139,92,246,.7);}
.st-key-nav :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]) *,.st-key-demo :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]) *{color:#fff !important;}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:34px 38px;margin:6px 0 8px;isolation:isolate;
  background:linear-gradient(135deg,rgba(15,23,42,.85),rgba(30,27,75,.75));border:1px solid rgba(255,255,255,.1);}
.hero::before{content:"";position:absolute;inset:-60%;z-index:-2;
  background:conic-gradient(from 0deg,transparent 0 70%,rgba(34,211,238,.35),rgba(139,92,246,.35),transparent);animation:spin 9s linear infinite;}
.hero::after{content:"";position:absolute;inset:1px;border-radius:25px;z-index:-1;background:linear-gradient(135deg,#0b1222,#151236);}
.hero .eyebrow{display:inline-flex;align-items:center;gap:8px;color:#a5f3fc;font-size:12px;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:12px;padding:6px 12px;border-radius:999px;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.25);
  animation:rise .5s ease backwards;}
.hero .eyebrow i{width:7px;height:7px;border-radius:50%;background:var(--c1);animation:glow 2s infinite;}
.hero h1{margin:0 0 10px;font-size:2.35rem;line-height:1.12;font-weight:800;color:#fff;}
.hero h1 span{display:inline-block;animation:word .7s cubic-bezier(.2,.8,.2,1) backwards;margin-right:.24em;}
.hero h1 span.g{background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3));-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:#b6bfd3;margin:0;max-width:680px;line-height:1.6;font-size:15.5px;animation:rise .7s ease .35s backwards;}
.orb{position:absolute;border-radius:50%;z-index:-1;opacity:.55;animation:float 7s ease-in-out infinite;}
.spark{position:absolute;bottom:-10px;width:4px;height:4px;border-radius:50%;background:#a5f3fc;z-index:-1;animation:up 6s linear infinite;}

/* ---------- шыны карточкалар ---------- */
.card{position:relative;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:22px 24px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);animation:rise .5s ease backwards;}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 10px;}
.how .card{animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;transition:transform .3s,border-color .3s;}
.how .card:hover{transform:translateY(-4px);border-color:rgba(34,211,238,.45);}
.how .n{font-size:13px;font-weight:800;letter-spacing:.1em;color:var(--c1);}
.how h4{margin:6px 0 4px;font-size:17px;} .how p{margin:0;color:var(--muted);font-size:14px;line-height:1.5;}

/* сұрақ */
.qhead{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:4px 0 10px;}
.qnum{font-size:46px;font-weight:800;line-height:1;letter-spacing:-.04em;
  background:linear-gradient(135deg,#fff,var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent;animation:popIn .6s ease backwards;}
.qnum small{font-size:18px;color:var(--muted);-webkit-text-fill-color:var(--muted);font-weight:600;}
.segs{display:grid;grid-template-columns:repeat(32,1fr);gap:4px;flex:1;max-width:640px;}
.seg{height:8px;border-radius:4px;background:rgba(255,255,255,.08);}
.seg.s-ok{background:linear-gradient(90deg,var(--c1),var(--c2));}
.seg.s-new{animation:wave .5s cubic-bezier(.2,.8,.2,1.4) backwards;}
.seg.s-cur{background:rgba(34,211,238,.35);animation:glow 1.6s ease-in-out infinite;}
.qcard{position:relative;overflow:hidden;border-radius:24px;padding:28px 30px;border:1px solid rgba(255,255,255,.1);
  background:linear-gradient(135deg,rgba(34,211,238,.08),rgba(139,92,246,.08));animation:cardIn .7s cubic-bezier(.2,.8,.2,1) backwards;}
.qcard .wm{position:absolute;right:-6px;bottom:-34px;font-size:150px;font-weight:800;color:rgba(255,255,255,.04);line-height:1;pointer-events:none;}
.chip{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#c4b5fd;
  background:rgba(139,92,246,.14);border:1px solid rgba(139,92,246,.3);padding:5px 11px;border-radius:999px;}
.q-text{font-size:23px;font-weight:700;color:#fff;line-height:1.38;margin-top:14px;position:relative;}

/* жауап нұсқалары — карточкалар */
[class*="st-key-ans_"] [role="radiogroup"],[class*="st-key-r_"] [role="radiogroup"],[class*="st-key-demo_"] [role="radiogroup"]{gap:8px;width:100%;}
[class*="st-key-ans_"] [role="radiogroup"] > *,[class*="st-key-r_"] [role="radiogroup"] > *,[class*="st-key-demo_"] [role="radiogroup"] > *{width:100%;margin:0 !important;animation:slideIn .45s cubic-bezier(.2,.8,.2,1) backwards;}
[class*="st-key-ans_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]),[class*="st-key-r_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]),[class*="st-key-demo_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]){width:100%;box-sizing:border-box;margin:0 !important;padding:14px 18px;border-radius:16px;background:var(--glass);
  border:1px solid var(--line);cursor:pointer;transition:transform .25s ease,border-color .25s,background .25s,box-shadow .25s;}
[class*="st-key-ans_"] [role="radiogroup"] > :nth-child(1){animation-delay:.15s}
[class*="st-key-ans_"] [role="radiogroup"] > :nth-child(2){animation-delay:.23s}
[class*="st-key-ans_"] [role="radiogroup"] > :nth-child(3){animation-delay:.31s}
[class*="st-key-ans_"] [role="radiogroup"] > :nth-child(4){animation-delay:.39s}
[class*="st-key-ans_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]):hover,[class*="st-key-r_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]):hover,[class*="st-key-demo_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]):hover{transform:translateX(6px);border-color:rgba(34,211,238,.5);background:rgba(34,211,238,.06);}
[class*="st-key-ans_"] :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]),[class*="st-key-r_"] :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]),[class*="st-key-demo_"] :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]){border-color:var(--c1);background:linear-gradient(135deg,rgba(34,211,238,.16),rgba(139,92,246,.14));box-shadow:0 8px 30px -10px rgba(34,211,238,.6);}

/* сенімділік — екі түйме */
[class*="st-key-conf_"] [role="radiogroup"]{gap:10px;}
[class*="st-key-conf_"] [role="radiogroup"] > *{margin:0 !important;animation:rise .4s ease .45s backwards;}
[class*="st-key-conf_"] :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]){margin:0 !important;padding:10px 20px;border-radius:999px;border:1px solid var(--line);background:var(--glass);cursor:pointer;transition:all .3s ease;}
[class*="st-key-conf_"] :is(label[data-baseweb="radio"] > div:first-child,[data-testid="stRadioOption"] > div > div:not(:last-child)){display:none !important;}
[class*="st-key-conf_"] [role="radiogroup"] > :nth-child(1):is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]),[class*="st-key-conf_"] [role="radiogroup"] > :nth-child(1) :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]){background:rgba(52,211,153,.16);border-color:#34d399;box-shadow:0 0 22px -6px #34d399;}
[class*="st-key-conf_"] [role="radiogroup"] > :nth-child(2):is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]),[class*="st-key-conf_"] [role="radiogroup"] > :nth-child(2) :is(label[data-baseweb="radio"]:has(input:checked),[data-testid="stRadioOption"]:has(input:checked),[data-testid="stRadioOption"][data-selected="true"]){background:rgba(251,191,36,.16);border-color:#fbbf24;box-shadow:0 0 22px -6px #fbbf24;}

/* статистика және нәтиже сақинасы */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 14px;}
.stat{display:flex;align-items:center;gap:16px;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:18px 20px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;transition:transform .3s;}
.stat:hover{transform:translateY(-4px);}
.stat img{width:58px;height:58px;flex:none;}
.stat .v{font-size:30px;font-weight:800;line-height:1;}
.stat .l{font-size:13px;color:var(--muted);margin-top:4px;}
.result{display:grid;grid-template-columns:auto 1fr;gap:28px;align-items:center;}
.gauge{position:relative;width:170px;height:170px;}
.gauge img{width:100%;height:100%;}
.gauge .pc{position:absolute;inset:0;display:grid;place-items:center;text-align:center;}
.gauge .pc b{font-size:40px;font-weight:800;display:block;line-height:1;animation:popIn .8s ease .5s backwards;}
.gauge .pc span{font-size:12px;color:var(--muted);}

/* ---------- майндмап ---------- */
.map-wrap{overflow-x:auto;background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:22px;padding:6px;animation:rise .5s ease backwards;}
.map{position:relative;height:440px;min-width:800px;}
.map img.lines{position:absolute;inset:0;width:100%;height:100%;}
@keyframes fadeIn{to{opacity:1}}
.node{position:absolute;width:17.5%;opacity:0;animation:pop .6s cubic-bezier(.2,.8,.25,1.2) forwards;}
.node-in{position:relative;padding:12px 13px;border-radius:15px;border:1.5px solid;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  animation:float 6s ease-in-out infinite;transition:box-shadow .25s ease, scale .25s ease;cursor:default;}
.node-in:hover{scale:1.06;box-shadow:0 14px 34px -8px currentColor;}
.node.weak .node-in::after{content:"";position:absolute;inset:-2px;border-radius:16px;border:2px solid #f87171;animation:pulse 1.8s ease-out infinite;}
.node .id{font-size:11px;font-weight:800;letter-spacing:.08em;}
.node .nm{font-size:13px;font-weight:600;line-height:1.28;color:#fff;margin-top:3px;}
.node .bar{height:4px;border-radius:4px;background:rgba(255,255,255,.1);margin-top:9px;overflow:hidden;}
.node .bar i{display:block;height:100%;border-radius:4px;animation:grow 1.1s ease backwards;}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 6px 4px;color:var(--muted);font-size:13px;align-items:center;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;box-shadow:0 0 8px currentColor;}

/* ---------- кесте ---------- */
.tbl{width:100%;border-collapse:separate;border-spacing:0;background:var(--glass);border:1px solid var(--line);border-radius:20px;overflow:hidden;margin-top:6px;}
.tbl th{font-size:12px;font-weight:700;color:var(--muted);text-align:left;padding:13px 16px;background:rgba(255,255,255,.03);border-bottom:1px solid var(--line);letter-spacing:.05em;text-transform:uppercase;}
.tbl td{padding:13px 16px;border-bottom:1px solid rgba(255,255,255,.05);font-size:14px;color:var(--ink);}
.tbl tr{transition:background .2s;} .tbl tr:hover td{background:rgba(255,255,255,.035);}
.tbl tr:last-child td{border-bottom:none;}
.pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;font-size:12.5px;font-weight:700;white-space:nowrap;}
.pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor;}
.mini{height:6px;width:120px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px;}
.mini i{display:block;height:100%;border-radius:6px;animation:grow 1s ease backwards;}

/* ---------- оқу бағыты ---------- */
.route{position:relative;margin:10px 0 0 20px;padding-left:34px;}
.route::before{content:"";position:absolute;left:0;top:6px;bottom:6px;width:2px;border-radius:2px;transform-origin:top;
  background:linear-gradient(var(--c1),var(--c2),var(--c3));animation:growY 1.4s cubic-bezier(.2,.8,.2,1) backwards;}
.step{position:relative;margin-bottom:14px;animation:slideIn .6s cubic-bezier(.2,.8,.2,1) backwards;}
.step .num{position:absolute;left:-54px;top:20px;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#0b1120;font-size:15px;
  box-shadow:0 0 0 5px var(--bg),0 0 22px -2px currentColor;animation:popIn .6s ease backwards;animation-delay:inherit;}
.step .card{transition:transform .3s,border-color .3s;} .step .card:hover{transform:translateX(6px);border-color:rgba(255,255,255,.2);}
.step h4{margin:0 0 4px;font-size:17px;}
.step .why{font-size:13px;font-weight:700;margin-bottom:8px;}
.step p{margin:0 0 10px;color:#b6bfd3;line-height:1.55;font-size:14.5px;}
.step a{color:var(--c1);font-weight:700;text-decoration:none;}
.done{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.4);color:#6ee7b7;border-radius:18px;padding:18px 22px;font-weight:700;animation:enter .6s ease backwards;}

/* ---------- мұғалім ---------- */
.insight{border-left:4px solid #f87171;}
.bars .row{display:grid;grid-template-columns:minmax(160px,300px) 1fr 44px;align-items:center;gap:14px;margin:10px 0;font-size:14px;}
.track{height:12px;background:rgba(255,255,255,.07);border-radius:6px;overflow:hidden;}
.fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#fb7185,#f87171,#fbbf24);box-shadow:0 0 14px rgba(248,113,113,.5);animation:grow 1.2s cubic-bezier(.2,.8,.2,1) backwards;}
.heat{display:grid;grid-template-columns:64px repeat(8,1fr);gap:5px;font-size:11.5px;color:var(--muted);}
.heat .c{height:18px;border-radius:6px;animation:wave .5s cubic-bezier(.2,.8,.2,1.4) backwards;transition:transform .15s;}
.heat .c:hover{transform:scale(1.35);z-index:2;}
.heat .hd{text-align:center;font-weight:700;}

/* ---------- Streamlit виджеттері ---------- */
.stButton>button,.stDownloadButton>button{position:relative;overflow:hidden;border-radius:14px;padding:.68rem 1.5rem;font-weight:700;
  border:1px solid var(--line);background:var(--glass);color:var(--ink);transition:transform .2s, box-shadow .3s, border-color .3s;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);border-color:rgba(34,211,238,.5);box-shadow:0 10px 30px -10px rgba(34,211,238,.6);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--c1),var(--c2));border:none;color:#fff;box-shadow:0 10px 30px -12px rgba(139,92,246,.9);}
.stButton>button[kind="primary"]::after{content:"";position:absolute;top:0;bottom:0;left:0;width:40%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);animation:shine 3.2s ease-in-out infinite;}
.stButton>button:disabled{opacity:.4;box-shadow:none;}
.stButton>button:disabled::after{display:none;}
[data-baseweb="tab-list"]{gap:4px;background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:16px;padding:5px;width:fit-content;max-width:100%;overflow-x:auto;}
button[data-baseweb="tab"]{border-radius:12px !important;padding:8px 18px !important;transition:background .3s;}
button[data-baseweb="tab"][aria-selected="true"]{background:rgba(255,255,255,.08);}
button[data-baseweb="tab"] p{font-weight:700;}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--c1),var(--c2)) !important;height:3px !important;border-radius:3px;}
[data-baseweb="tab-border"]{display:none;}
[data-baseweb="tab-panel"]{animation:enter .55s cubic-bezier(.2,.8,.2,1) backwards;}
[data-testid="stExpander"] details{background:var(--glass);border-radius:16px;border:1px solid var(--line);}
[data-testid="stAlert"]{border-radius:16px;}
[data-testid="stSliderTickBar"]{opacity:1 !important;}
[data-testid="stWidgetLabel"] p,[data-testid="stMain"] [data-testid="stMarkdownContainer"] p{color:var(--ink);}

@media (max-width:760px){
  .stats,.how{grid-template-columns:1fr}.result{grid-template-columns:1fr;justify-items:center;text-align:center}
  .hero{padding:24px 22px}.hero h1{font-size:1.65rem}.q-text{font-size:19px}.qnum{font-size:36px}
  .qhead{flex-direction:column;align-items:stretch}.segs{max-width:none}
  .st-key-nav div[role="radiogroup"]{flex-wrap:wrap;border-radius:20px}.st-key-nav :is(label[data-baseweb="radio"],[data-testid="stRadioOption"]){padding:8px 14px}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;}}
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


def hero(eyebrow, title, text, accent_last=True):
    """Үлкен тақырып: сөздер кезекпен шығады, соңғы сөз — градиентпен."""
    words = title.split()
    spans = "".join(
        f'<span class="{"g" if accent_last and k == len(words) - 1 else ""}" style="animation-delay:{0.08 + k * 0.07:.2f}s">'
        f'{h.escape(w)}</span>' for k, w in enumerate(words))
    sparks = "".join(f'<i class="spark" style="left:{8 + k * 11}%;animation-delay:{-k * 0.8:.1f}s;'
                     f'animation-duration:{5 + k % 3}s"></i>' for k in range(8))
    st.html(f"""
    <div class="hero">
      <div class="orb" style="width:240px;height:240px;right:-60px;top:-90px;background:radial-gradient(circle,rgba(34,211,238,.55),transparent 65%)"></div>
      <div class="orb" style="width:220px;height:220px;right:200px;bottom:-130px;background:radial-gradient(circle,rgba(139,92,246,.6),transparent 65%);animation-delay:-3s"></div>
      {sparks}
      <div class="eyebrow"><i></i>{h.escape(eyebrow)}</div><h1>{spans}</h1><p>{h.escape(text)}</p>
    </div>""")


def svg_img(svg, cls="", style=""):
    """st.html SVG тегтерін өшіреді, сондықтан SVG-ні сурет (data URI) ретінде қосамыз."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="{cls}" style="{style}" alt="" src="data:image/svg+xml;base64,{data}">'


def ring_svg(value, color):
    """value 0..100 — сақина қаншалықты толады (анимациямен)."""
    off = 100 - value
    return svg_img(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
        f'<style>@keyframes r{{from{{stroke-dashoffset:100}}to{{stroke-dashoffset:{off:.1f}}}}}'
        f'.f{{animation:r 1.4s cubic-bezier(.2,.8,.2,1) both}}</style>'
        f'<circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="12"/>'
        f'<circle class="f" cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
        f'pathLength="100" stroke-dasharray="100" stroke-dashoffset="{off:.1f}" transform="rotate(-90 60 60)"/></svg>')


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
        ys = [220] if n == 1 else [90 + i * (260 / (n - 1)) for i in range(n)]
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
            d = f"M{x1:.0f} {y1:.0f} C{xm:.0f} {y1:.0f},{xm:.0f} {y2:.0f},{x2:.0f} {y2:.0f}"
            delay = 0.25 + depth[p] * 0.2
            # 1) сызық сызылады, 2) үстінен «ағып жатқан» нүктелер жүреді
            paths.append(f'<path pathLength="1" vector-effect="non-scaling-stroke" style="animation-delay:{delay:.2f}s" d="{d}"/>'
                         f'<path class="flow" pathLength="1" vector-effect="non-scaling-stroke" '
                         f'style="animation-delay:0s,{delay + 1:.2f}s" d="{d}"/>')
    nodes = []
    for i, t in enumerate(topics["topic_id"]):
        x, y = pos[t]
        lv = levels[t]
        fg, bg = LV[lv]
        score = int(round(feats.loc[t, "score"] * 100))
        nodes.append(
            f'<div class="node{" weak" if lv == 0 else ""}" '
            f'style="left:{x / 10:.1f}%;top:{y / 440 * 100:.1f}%;animation-delay:{depth[t] * 0.2 + i * 0.04:.2f}s">'
            f'<div class="node-in" style="background:linear-gradient({bg},{bg}),#0e1426;border-color:{fg};color:{fg};animation-delay:{-i * 0.7:.1f}s">'
            f'<div class="id">{t} · {LEVEL_NAMES[lv]}</div>'
            f'<div class="nm">{h.escape(topic_name[t])}</div>'
            f'<div class="bar"><i style="width:{score}%;background:{fg};animation-delay:{0.6 + depth[t] * 0.2:.2f}s"></i></div>'
            f'</div></div>')
    legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]};color:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                     for k in (2, 1, 0))
    lines_img = svg_img(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 440" preserveAspectRatio="none"><style>'
        'path{fill:none;stroke:rgba(255,255,255,.18);stroke-width:2;stroke-dasharray:1;stroke-dashoffset:1;animation:d 1s ease forwards}'
        'path.flow{stroke:url(#gr);stroke-width:2.6;stroke-dasharray:.04 .1;stroke-dashoffset:0;opacity:0;animation:f 2.4s linear infinite,o .6s ease forwards}'
        '@keyframes d{to{stroke-dashoffset:0}}@keyframes f{to{stroke-dashoffset:-1}}@keyframes o{to{opacity:1}}</style>'
        '<defs><linearGradient id="gr" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>'
        + "".join(paths) + '</svg>', cls="lines")
    st.html(f"""
    <div class="map-wrap"><div class="map">
      {lines_img}
      {''.join(nodes)}
    </div></div>
    <div class="legend">{legend}<span style="margin-left:auto">Жолақ — дұрыс жауаптар үлесі · сызық базалық тақырыптан күрделіге қарай</span></div>""")


def show_table(levels, feats):
    rows = []
    for i, t in enumerate(feats.index):
        s = int(round(feats.loc[t, "score"] * 100))
        g = int(round(feats.loc[t, "guess_rate"] * 100))
        fg = LV[levels[t]][0]
        rows.append(f'<tr style="animation:slideIn .45s ease backwards;animation-delay:{i * 0.06:.2f}s">'
                    f'<td><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</td>'
                    f'<td><span class="mini"><i style="width:{s}%;background:{fg};animation-delay:{0.2 + i * 0.06:.2f}s"></i></span>{s}%</td>'
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
        <div class="step" style="animation-delay:{0.15 + i * 0.14:.2f}s">
          <div class="num" style="background:{fg};color:{fg}"><span style="color:#0b1120">{i}</span></div>
          <div class="card" style="animation:none">
            <h4>{s['topic_id']} · {h.escape(s['name'])}</h4>
            <div class="why" style="color:{fg}">{h.escape(s['reason'])}</div>
            <p>{h.escape(s['explanation'])}</p>
            <div style="font-size:13.5px;color:#8b93a7">{h.escape(lecture[s['topic_id']])} ·
              <a href="{h.escape(s['material'])}" target="_blank">қосымша оқу →</a></div>
          </div>
        </div>""")
    st.html(f'<p style="color:#b6bfd3;margin:4px 0 6px">Сізге <b style="color:#fff">{len(route)} тақырыпты</b> осы ретпен оқу ұсынылады — '
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
    n = len(levels) or 1
    tiles = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.1:.2f}s">{ring_svg(counts[k] / n * 100, LV[k][0])}'
        f'<div><div class="v" style="color:{LV[k][0]}">{counts[k]}<span style="font-size:15px;color:#8b93a7"> / {n}</span></div>'
        f'<div class="l">{LEVEL_NAMES[k]}</div></div></div>' for i, k in enumerate((2, 1, 0)))
    st.html(f'<div class="stats">{tiles}</div>')


def result_tabs(levels, feats, key_prefix):
    """Карта / Кесте / Оқу бағыты — бір-біріне анимациямен ауысады."""
    t1, t2, t3 = st.tabs(["Білім картасы", "Кесте", "Оқу бағыты"])
    with t1:
        show_map(levels, feats)
    with t2:
        show_table(levels, feats)
    with t3:
        show_route(levels, key_prefix=key_prefix)


# ---------- Жоғарғы бренд пен мәзір ----------
LOGO = svg_img(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 42 42"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>'
    '<rect x="1" y="1" width="40" height="40" rx="12" fill="#0b1120" stroke="url(#g)" stroke-width="2"/>'
    '<g stroke="#cbd5e1" stroke-width="1.6"><line x1="13" y1="14" x2="21" y2="21"/><line x1="29" y1="13" x2="21" y2="21"/><line x1="21" y1="21" x2="21" y2="30"/></g>'
    '<circle cx="13" cy="14" r="3.2" fill="#22d3ee"/><circle cx="29" cy="13" r="3.2" fill="#22d3ee"/>'
    '<circle cx="21" cy="21" r="3.8" fill="#8b5cf6"/><circle cx="21" cy="30" r="3.2" fill="#22d3ee"/></svg>', cls="logo")
st.html(f"""
<div class="topbar">
  {LOGO}
  <div><div class="brand">Білім картасы</div><div class="brand-sub">ICT1103 · топтық жоба</div></div>
</div>""")
page = st.radio("Бөлім", ["Тест", "Менің картам", "Оқу бағытым", "Мұғалім", "Демо"],
                horizontal=True, label_visibility="collapsed", key="nav")

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
        st.html("""<div class="how">
          <div class="card" style="animation-delay:.15s"><div class="n">01</div><h4>Тест</h4><p>32 сұрақ, әр жауапқа сенімділік белгісі</p></div>
          <div class="card" style="animation-delay:.27s"><div class="n">02</div><h4>Білім картасы</h4><p>Модель әр тақырыптың деңгейін анықтайды</p></div>
          <div class="card" style="animation-delay:.39s"><div class="n">03</div><h4>Оқу бағыты</h4><p>Әлсіз тақырыптар дұрыс ретпен, материалмен</p></div>
        </div>""")
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
        n = len(test_q)
        q = test_q.iloc[i]
        segs = "".join(
            f'<div class="seg{" s-ok" if k < i else ""}{" s-new" if k == i - 1 else ""}{" s-cur" if k == i else ""}"></div>'
            for k in range(n))
        st.html(f'<div class="qhead"><div class="qnum">{i + 1:02d}<small> / {n}</small></div>'
                f'<div class="segs" style="grid-template-columns:repeat({n},1fr)">{segs}</div></div>')
        st.html(f'<div class="qcard"><div class="wm">{q["topic_id"]}</div>'
                f'<span class="chip">{q["topic_id"]} · {h.escape(topic_name[q["topic_id"]])}</span>'
                f'<div class="q-text">{h.escape(q["question"])}</div></div>')
        opts = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"]}
        choice = st.radio("Жауапты таңдаңыз", list(opts), format_func=lambda k: f"{k})  {opts[k]}",
                          index=None, key=f"ans_{i}")
        conf = st.radio("Жауабыңызға сенімдісіз бе?", ["Сенімдімін", "Болжадым"], index=None,
                        horizontal=True, key=f"conf_{i}")
        if st.button("Келесі сұрақ  →", type="primary", disabled=(choice is None or conf is None)):
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
        right = int(answers["correct"].sum())
        pct = right / len(answers) * 100
        st.html(f"""<div class="hero"><div class="result">
          <div class="gauge">{ring_svg(pct, "#22d3ee")}
            <div class="pc"><div><b>{pct:.0f}%</b><span>{right} / {len(answers)} дұрыс</span></div></div></div>
          <div><div class="eyebrow"><i></i>Тест аяқталды</div>
            <h1><span>Сіздің</span><span class="g" style="animation-delay:.1s">білім картаңыз</span></h1>
            <p>Модель әр тақырыптың деңгейін анықтады. Төменде — карта, кесте және жеке оқу бағыты.</p></div>
        </div></div>""")
        show_stats(levels, feats)
        result_tabs(levels, feats, key_prefix="r")

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
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        levels, feats = st.session_state.result
        show_stats(levels, feats)
        t1, t2 = st.tabs(["Білім картасы", "Кесте"])
        with t1:
            show_map(levels, feats)
        with t2:
            show_table(levels, feats)


# ---------- 3. Оқу бағытым ----------
elif page == "Оқу бағытым":
    hero("Жеке жоспар", "Менің оқу бағытым",
         "Әлсіз тақырыптар дұрыс ретпен: алдымен басқа тақырыптарға негіз болатындары.")
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
    else:
        show_route(st.session_state.result[0])


# ---------- 4. Мұғалім ----------
elif page == "Мұғалім":
    hero("Мұғалім панелі", "Топтың нәтижесі",
         "Мысал ретінде — тест бөлігіндегі 60 жасанды студент (data/test_students.csv).")

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
    st.html(f'<div class="card insight"><b style="color:#fff">Ең қиын тақырып: {worst} · '
            f'{h.escape(topic_name[worst])}</b><br><span style="color:#8b93a7">{weak.max()} студент ({weak.max() * 100 // total}%) '
            f'бұл тақырыпты білмейді — оны қайта түсіндіру ұсынылады.</span></div>')

    t1, t2 = st.tabs(["Әлсіз тақырыптар", "Жылу картасы"])
    with t1:
        rows = "".join(
            f'<div class="row"><div><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</div>'
            f'<div class="track"><div class="fill" style="width:{weak[t] / total * 100:.0f}%;animation-delay:{0.1 + i * 0.08:.2f}s"></div></div>'
            f'<div style="text-align:right;font-weight:700">{weak[t]}</div></div>'
            for i, t in enumerate(weak.sort_values(ascending=False).index))
        st.html(f'<div class="card"><h3 style="margin:0 0 6px;font-size:18px">'
                f'Қанша студент тақырыпты білмейді</h3><div class="bars">{rows}</div></div>')
    with t2:
        cells = ['<div></div>'] + [f'<div class="hd">{t}</div>' for t in grid.columns]
        for r, (sid, row) in enumerate(grid.head(20).iterrows()):
            cells.append(f'<div>{sid}</div>')
            for c, t in enumerate(grid.columns):
                cells.append(f'<div class="c" title="{sid} · {t}: {LEVEL_NAMES[row[t]]}" '
                             f'style="background:{LV[row[t]][0]}b3;border:1px solid {LV[row[t]][0]};'
                             f'animation-delay:{(r + c) * 0.03:.2f}s"></div>')
        legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]};color:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                         for k in (2, 1, 0))
        st.html(f'<div class="card"><h3 style="margin:0 0 12px;font-size:18px">'
                f'Студент × тақырып <span style="font-weight:400;color:#8b93a7;font-size:14px">(алғашқы 20 студент)</span></h3>'
                f'<div class="heat">{"".join(cells)}</div><div class="legend">{legend}</div></div>')


# ---------- 5. Демо ----------
elif page == "Демо":
    hero("Қорғауға арналған", "Үш студент — үш түрлі жағдай",
         "Тест бөлігіндегі жасанды студенттер: модель оларды оқыту кезінде көрмеген.")

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
    choice = st.radio("Мысал", list(demos), horizontal=True, label_visibility="collapsed", key="demo")
    sid = demos[choice]
    answers = pd.read_csv("data/students.csv")
    levels, feats = levels_from_answers(answers[answers["student_id"] == sid])
    st.session_state.result = (levels, feats)
    st.html(f'<span class="chip" style="animation:rise .4s ease backwards">Студент коды: {sid}</span>')
    show_stats(levels, feats)
    result_tabs(levels, feats, key_prefix=f"demo_{sid}")
