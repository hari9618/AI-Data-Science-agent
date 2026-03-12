"""
AI Autonomous Data Science Agent - Streamlit Frontend
Ultra-premium 3D animated dashboard with Gemini 2.5 Flash
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
import json
import time

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND = "http://127.0.0.1:8000"

# ─── Mega CSS / 3D Animations ─────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background: #020510 !important;
    color: #e0e8ff !important;
}

/* ── Animated Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d1b4b 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, #1a0d4b 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, #0d2b4b 0%, transparent 50%),
                #020510 !important;
    min-height: 100vh;
}

/* ── Animated Stars Background ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(120,180,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 60%, rgba(180,120,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 30%, rgba(120,255,220,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 75%, rgba(255,180,120,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 10%, rgba(120,180,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 50%, rgba(180,120,255,0.6) 0%, transparent 100%),
        radial-gradient(2px 2px at 15% 85%, rgba(120,255,180,0.3) 0%, transparent 100%),
        radial-gradient(2px 2px at 90% 90%, rgba(255,120,180,0.3) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
    animation: twinkle 4s ease-in-out infinite alternate;
}

@keyframes twinkle {
    0%   { opacity: 0.6; }
    100% { opacity: 1; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050d2a 0%, #0a0520 50%, #050d2a 100%) !important;
    border-right: 1px solid rgba(100,150,255,0.2) !important;
    box-shadow: 4px 0 30px rgba(80,130,255,0.15) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 40px,
        rgba(80,130,255,0.03) 40px,
        rgba(80,130,255,0.03) 41px
    );
    pointer-events: none;
}

/* ── HERO HEADER ── */
.hero-container {
    position: relative;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg,
        rgba(10,20,60,0.95) 0%,
        rgba(20,10,60,0.95) 50%,
        rgba(10,30,50,0.95) 100%);
    border: 1px solid rgba(100,150,255,0.3);
    overflow: hidden;
    box-shadow:
        0 0 60px rgba(80,130,255,0.15),
        0 0 120px rgba(120,80,255,0.08),
        inset 0 0 60px rgba(80,130,255,0.05);
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(
        from 0deg at 50% 50%,
        transparent 0deg,
        rgba(80,130,255,0.05) 60deg,
        transparent 120deg,
        rgba(120,80,255,0.05) 180deg,
        transparent 240deg,
        rgba(80,200,255,0.05) 300deg,
        transparent 360deg
    );
    animation: rotateBg 20s linear infinite;
    pointer-events: none;
}

@keyframes rotateBg {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.hero-container::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4080ff, #8040ff, #40c0ff, transparent);
    animation: scanline 3s ease-in-out infinite;
}

@keyframes scanline {
    0%, 100% { opacity: 0.3; transform: scaleX(0.5); }
    50%       { opacity: 1;   transform: scaleX(1); }
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #4080ff 0%, #8040ff 40%, #40c8ff 80%, #40ffb0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    text-shadow: none;
    letter-spacing: 2px;
    animation: titlePulse 3s ease-in-out infinite;
}

@keyframes titlePulse {
    0%, 100% { filter: brightness(1); }
    50%       { filter: brightness(1.2); }
}

.hero-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.95rem;
    color: rgba(160,200,255,0.8);
    margin-top: 0.5rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-badges {
    display: flex;
    gap: 12px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    border: 1px solid;
    animation: badgeFloat 2s ease-in-out infinite alternate;
}

.badge-blue  { background: rgba(64,128,255,0.15); border-color: rgba(64,128,255,0.5); color: #80b0ff; }
.badge-purple{ background: rgba(128,64,255,0.15); border-color: rgba(128,64,255,0.5); color: #b080ff; }
.badge-cyan  { background: rgba(64,200,255,0.15); border-color: rgba(64,200,255,0.5); color: #80d8ff; }
.badge-green { background: rgba(64,255,160,0.15); border-color: rgba(64,255,160,0.5); color: #80ffc0; }

@keyframes badgeFloat {
    from { transform: translateY(0px); }
    to   { transform: translateY(-3px); }
}

/* ── 3D CARDS ── */
.card-3d {
    position: relative;
    padding: 1.8rem;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(15,25,60,0.9), rgba(8,15,40,0.95));
    border: 1px solid rgba(100,150,255,0.2);
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 2px 8px rgba(80,130,255,0.1),
        inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.card-3d::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(100,150,255,0.6), transparent);
}

.card-3d:hover {
    transform: perspective(1000px) rotateX(-2deg) rotateY(2deg) translateY(-4px);
    border-color: rgba(100,150,255,0.4);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.5),
        0 8px 24px rgba(80,130,255,0.2),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

.card-glow-blue::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 60px;
    background: rgba(64,128,255,0.15);
    filter: blur(20px);
    border-radius: 50%;
}

.card-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── METRIC HOLOGRAM ── */
.metric-holo {
    position: relative;
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(15,25,60,0.8), rgba(8,15,40,0.9));
    border: 1px solid rgba(100,150,255,0.2);
    text-align: center;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-holo::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
    animation: shimmer 3s ease-in-out infinite;
}

@keyframes shimmer {
    0%   { left: -100%; }
    100% { left: 200%; }
}

.metric-holo:hover {
    transform: scale(1.03) translateY(-2px);
    border-color: rgba(100,150,255,0.5);
    box-shadow: 0 8px 30px rgba(80,130,255,0.2);
}

.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #40ffb0, #40c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: rgba(160,200,255,0.6);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* ── STEP PIPELINE ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
    border: 1px solid transparent;
}

.step-active {
    background: rgba(64,128,255,0.1);
    border-color: rgba(64,128,255,0.3);
    animation: stepPulse 1.5s ease-in-out infinite;
}

@keyframes stepPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(64,128,255,0.2); }
    50%       { box-shadow: 0 0 20px rgba(64,128,255,0.3); }
}

.step-done {
    background: rgba(64,255,160,0.08);
    border-color: rgba(64,255,160,0.2);
}

.step-pending {
    background: rgba(255,255,255,0.02);
    border-color: rgba(255,255,255,0.05);
    opacity: 0.5;
}

.step-number {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}

.step-number-active  { background: rgba(64,128,255,0.3); color: #80b0ff; border: 2px solid rgba(64,128,255,0.6); }
.step-number-done    { background: rgba(64,255,160,0.3); color: #80ffc0; border: 2px solid rgba(64,255,160,0.6); }
.step-number-pending { background: rgba(100,100,100,0.2); color: #888; border: 2px solid rgba(100,100,100,0.3); }

.step-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(100,150,255,0.2);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(100,150,255,0.3), transparent);
}

/* ── TERMINAL / INSIGHTS BOX ── */
.terminal-box {
    background: rgba(0,10,30,0.9);
    border: 1px solid rgba(64,200,100,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #a0e0c0;
    position: relative;
    overflow: hidden;
}

.terminal-box::before {
    content: '● ● ●';
    display: block;
    color: rgba(255,100,100,0.6);
    font-size: 0.8rem;
    margin-bottom: 1rem;
    letter-spacing: 4px;
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    padding: 3rem;
    border-radius: 20px;
    border: 2px dashed rgba(100,150,255,0.3);
    text-align: center;
    background: rgba(10,20,60,0.4);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: rgba(100,150,255,0.6);
    background: rgba(10,20,60,0.6);
    box-shadow: 0 0 40px rgba(80,130,255,0.1);
}

.upload-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 1rem;
    animation: iconFloat 3s ease-in-out infinite;
}

@keyframes iconFloat {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}

/* ── MODEL BADGE ── */
.model-winner {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 2rem;
    border-radius: 50px;
    background: linear-gradient(135deg, rgba(64,128,255,0.2), rgba(128,64,255,0.2));
    border: 2px solid;
    border-image: linear-gradient(135deg, #4080ff, #8040ff, #40c8ff) 1;
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #80c0ff;
    letter-spacing: 2px;
    box-shadow: 0 0 40px rgba(80,130,255,0.2);
    animation: modelGlow 2s ease-in-out infinite alternate;
}

@keyframes modelGlow {
    from { box-shadow: 0 0 20px rgba(80,130,255,0.2), 0 0 40px rgba(80,130,255,0.1); }
    to   { box-shadow: 0 0 40px rgba(80,130,255,0.4), 0 0 80px rgba(80,130,255,0.2); }
}

/* ── PROGRESS BAR ── */
.progress-bar-container {
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
    margin: 4px 0;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4080ff, #8040ff, #40c8ff);
    position: relative;
    transition: width 1s ease;
}

.progress-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 100%;
    background: rgba(255,255,255,0.4);
    border-radius: 4px;
    filter: blur(4px);
    animation: progressGlow 1.5s ease-in-out infinite;
}

@keyframes progressGlow {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 1; }
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a2a6c 0%, #2a1a6c 50%, #1a3a5c 100%) !important;
    color: #80b8ff !important;
    border: 1px solid rgba(100,150,255,0.4) !important;
    border-radius: 12px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2a3a8c 0%, #3a2a8c 50%, #2a4a7c 100%) !important;
    border-color: rgba(100,150,255,0.8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(80,130,255,0.3), 0 0 50px rgba(80,130,255,0.1) !important;
    color: #c0d8ff !important;
}

/* ── DATA TABLE ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(100,150,255,0.2) !important;
}

/* ── Selectbox / Input ── */
[data-testid="stSelectbox"] > div {
    background: rgba(10,20,60,0.8) !important;
    border: 1px solid rgba(100,150,255,0.3) !important;
    border-radius: 10px !important;
    color: #a0c0ff !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(10,20,60,0.5) !important;
    border: 1px solid rgba(100,150,255,0.15) !important;
    border-radius: 12px !important;
}

/* ── Success / Error / Info ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ── Sidebar content ── */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid rgba(100,150,255,0.15);
    margin-bottom: 1.5rem;
}

.sidebar-logo-icon {
    display: block;
    margin: 0 auto;
    animation: logoFloat 4s ease-in-out infinite;
    filter: drop-shadow(0 0 16px rgba(80,130,255,0.7)) drop-shadow(0 0 32px rgba(120,80,255,0.4));
}

@keyframes logoFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    25%       { transform: translateY(-6px) rotate(3deg); }
    75%       { transform: translateY(-3px) rotate(-3deg); }
}

.sidebar-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #6090ff;
    letter-spacing: 3px;
    margin-top: 0.5rem;
    text-transform: uppercase;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    margin-bottom: 6px;
}

.status-online  { background: rgba(64,255,160,0.1); color: #40ffa0; border: 1px solid rgba(64,255,160,0.3); }
.status-offline { background: rgba(255,80,80,0.1);  color: #ff8080; border: 1px solid rgba(255,80,80,0.3); }

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: dotPulse 1.5s ease-in-out infinite;
}

.dot-green { background: #40ffa0; box-shadow: 0 0 6px #40ffa0; }
.dot-red   { background: #ff8080; box-shadow: 0 0 6px #ff8080; }

@keyframes dotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(100,150,255,0.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(10,20,60,0.4) !important;
    border: 2px dashed rgba(100,150,255,0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(100,150,255,0.6) !important;
    background: rgba(10,20,60,0.6) !important;
}

/* ── Tooltips / captions ── */
.stCaption { color: rgba(160,200,255,0.5) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.72rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4080ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(10,20,60,0.3); }
::-webkit-scrollbar-thumb { background: rgba(80,130,255,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(80,130,255,0.7); }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Plotly Theme ──────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(5,12,35,0.0)",
    plot_bgcolor="rgba(5,12,35,0.0)",
    font=dict(family="Rajdhani", color="#a0c0ff", size=12),
    title_font=dict(family="Orbitron", color="#80b0ff", size=14),
    legend=dict(
        bgcolor="rgba(10,20,60,0.7)",
        bordercolor="rgba(100,150,255,0.3)",
        borderwidth=1,
        font=dict(color="#a0c0ff"),
    ),
    colorway=["#4080ff","#8040ff","#40c8ff","#40ffb0","#ff8040","#ff40b0","#c0ff40"],
    # Passing them here AND in update_layout() causes:
    # TypeError: got multiple values for keyword argument 'yaxis'
    # They are now applied separately via update_xaxes() / update_yaxes()
    margin=dict(l=50, r=30, t=60, b=50),
)
# never inside update_layout(**PLOTLY_LAYOUT) to avoid duplicate kwarg crash.
AXIS_STYLE = dict(
    gridcolor="rgba(100,150,255,0.08)",
    linecolor="rgba(100,150,255,0.2)",
    tickfont=dict(color="#7090c0"),
)


def apply_theme(fig):
    """Apply consistent dark theme to any 2D Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # ── Massive 3D SVG Logo ─────────────────────────────────────────────
        st.markdown("""
        <div class="sidebar-logo">
            <svg class="sidebar-logo-icon" width="90" height="90" viewBox="0 0 90 90" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <radialGradient id="bgGrad" cx="50%" cy="40%" r="55%">
                  <stop offset="0%"  stop-color="#6080ff" stop-opacity="0.25"/>
                  <stop offset="100%" stop-color="#0a0f30" stop-opacity="0"/>
                </radialGradient>
                <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%"  stop-color="#1a2a7c"/>
                  <stop offset="50%" stop-color="#2a1a6c"/>
                  <stop offset="100%" stop-color="#0d1a50"/>
                </linearGradient>
                <linearGradient id="faceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%"  stop-color="#0d1a50"/>
                  <stop offset="100%" stop-color="#1a0d40"/>
                </linearGradient>
                <linearGradient id="eyeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%"  stop-color="#40ffff"/>
                  <stop offset="100%" stop-color="#4080ff"/>
                </linearGradient>
                <linearGradient id="topGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%"  stop-color="#4080ff"/>
                  <stop offset="50%" stop-color="#8040ff"/>
                  <stop offset="100%" stop-color="#40c8ff"/>
                </linearGradient>
                <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%"  stop-color="#40ffb0"/>
                  <stop offset="100%" stop-color="#4080ff"/>
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="1.5" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="strongGlow">
                  <feGaussianBlur stdDeviation="3" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>
              <!-- Outer glow ring -->
              <circle cx="45" cy="45" r="42" fill="url(#bgGrad)" stroke="url(#topGrad)" stroke-width="0.5" opacity="0.6"/>
              <!-- Orbit ring -->
              <ellipse cx="45" cy="45" rx="40" ry="12" fill="none" stroke="url(#topGrad)" stroke-width="0.6" opacity="0.4" transform="rotate(-20 45 45)"/>
              <!-- Orbit dot -->
              <circle cx="12" cy="38" r="2.5" fill="#40ffb0" filter="url(#glow)" opacity="0.9"/>
              <!-- Body / torso -->
              <rect x="22" y="46" width="46" height="32" rx="8" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="0.8"/>
              <!-- Body sheen -->
              <rect x="22" y="46" width="46" height="10" rx="8" fill="rgba(255,255,255,0.04)"/>
              <!-- Head -->
              <rect x="26" y="18" width="38" height="32" rx="7" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="1"/>
              <!-- Head sheen -->
              <rect x="26" y="18" width="38" height="10" rx="7" fill="rgba(255,255,255,0.06)"/>
              <!-- Antenna base -->
              <rect x="43" y="10" width="4" height="10" rx="2" fill="url(#topGrad)"/>
              <!-- Antenna tip glow -->
              <circle cx="45" cy="8" r="4" fill="#40c8ff" filter="url(#strongGlow)" opacity="0.9"/>
              <circle cx="45" cy="8" r="2" fill="white"/>
              <!-- Left eye -->
              <rect x="30" y="26" width="12" height="8" rx="3" fill="url(#eyeGrad)" filter="url(#glow)"/>
              <rect x="32" y="28" width="4" height="4" rx="1" fill="white" opacity="0.8"/>
              <!-- Right eye -->
              <rect x="48" y="26" width="12" height="8" rx="3" fill="url(#eyeGrad)" filter="url(#glow)"/>
              <rect x="50" y="28" width="4" height="4" rx="1" fill="white" opacity="0.8"/>
              <!-- Mouth / visor bar -->
              <rect x="31" y="38" width="28" height="4" rx="2" fill="url(#accentGrad)" opacity="0.7" filter="url(#glow)"/>
              <!-- Ear left -->
              <rect x="18" y="24" width="8" height="14" rx="3" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="0.6"/>
              <rect x="20" y="29" width="4" height="4" rx="1" fill="#4080ff" filter="url(#glow)"/>
              <!-- Ear right -->
              <rect x="64" y="24" width="8" height="14" rx="3" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="0.6"/>
              <rect x="66" y="29" width="4" height="4" rx="1" fill="#8040ff" filter="url(#glow)"/>
              <!-- Neck connector -->
              <rect x="38" y="50" width="14" height="4" rx="2" fill="url(#topGrad)" opacity="0.6"/>
              <!-- Chest panel -->
              <rect x="29" y="54" width="32" height="18" rx="4" fill="url(#faceGrad)" stroke="rgba(100,150,255,0.3)" stroke-width="0.5"/>
              <!-- Chest circuits -->
              <line x1="33" y1="60" x2="57" y2="60" stroke="url(#accentGrad)" stroke-width="0.8" opacity="0.5"/>
              <line x1="33" y1="64" x2="49" y2="64" stroke="#4080ff" stroke-width="0.8" opacity="0.5"/>
              <!-- Chest gem -->
              <circle cx="45" cy="63" r="5" fill="none" stroke="url(#topGrad)" stroke-width="1" opacity="0.7"/>
              <circle cx="45" cy="63" r="3" fill="#40c8ff" filter="url(#strongGlow)" opacity="0.8"/>
              <!-- Arm left -->
              <rect x="13" y="48" width="9" height="22" rx="4" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="0.6"/>
              <circle cx="17" cy="72" r="3" fill="url(#accentGrad)" filter="url(#glow)"/>
              <!-- Arm right -->
              <rect x="68" y="48" width="9" height="22" rx="4" fill="url(#bodyGrad)" stroke="url(#topGrad)" stroke-width="0.6"/>
              <circle cx="73" cy="72" r="3" fill="url(#accentGrad)" filter="url(#glow)"/>
              <!-- Bottom energy bar -->
              <rect x="29" y="76" width="32" height="3" rx="1.5" fill="url(#accentGrad)" opacity="0.6" filter="url(#glow)"/>
              <!-- Corner bolt details -->
              <circle cx="25" cy="49" r="1.5" fill="#4080ff" opacity="0.6"/>
              <circle cx="65" cy="49" r="1.5" fill="#8040ff" opacity="0.6"/>
              <circle cx="25" cy="76" r="1.5" fill="#40c8ff" opacity="0.6"/>
              <circle cx="65" cy="76" r="1.5" fill="#40ffb0" opacity="0.6"/>
            </svg>
            <div class="sidebar-title">AI · DS · AGENT</div>
        </div>
        """, unsafe_allow_html=True)

        # Backend health check
        try:
            r = requests.get(f"{BACKEND}/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                st.markdown("""
                <div class="status-indicator status-online">
                    <div class="status-dot dot-green"></div>Backend Online
                </div>""", unsafe_allow_html=True)
                xgb = "✅" if data.get("xgboost") else "⚠️"
                gem = "✅" if data.get("gemini_configured") else "⚠️"
                st.markdown(f"""
                <div style="padding: 0.5rem; font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: rgba(160,200,255,0.6);">
                {xgb} XGBoost Available<br>
                {gem} Gemini API Configured
                </div>""", unsafe_allow_html=True)
            else:
                raise Exception()
        except Exception:
            st.markdown("""
            <div class="status-indicator status-offline">
                <div class="status-dot dot-red"></div>Backend Offline
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        st.markdown("""
        <div style="font-family: 'Orbitron', monospace; font-size: 0.7rem; color: rgba(100,150,255,0.6);
             letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem;">Navigation</div>
        """, unsafe_allow_html=True)

        pages = {
            "🏠 Dashboard": "dashboard",
            "📤 Upload Data": "upload",
            "🔍 AI Insights": "insights",
            "⚙️ ML Pipeline": "pipeline",
            "📊 Results": "results",
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "dashboard"

        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}"):
                st.session_state.current_page = key
                st.rerun()

        st.markdown("---")

        # Pipeline steps status
        st.markdown("""
        <div style="font-family: 'Orbitron', monospace; font-size: 0.7rem; color: rgba(100,150,255,0.6);
             letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem;">Pipeline Status</div>
        """, unsafe_allow_html=True)

        steps = [
            ("Data Uploaded", "upload_result"),
            ("AI Analysis", "analyze_result"),
            ("ML Pipeline", "pipeline_result"),
        ]

        for step_name, step_key in steps:
            done = step_key in st.session_state
            icon  = "✅" if done else "○"
            color = "#40ffa0" if done else "rgba(160,200,255,0.3)"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:6px 0;
                font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:{color};">
                {icon} {step_name}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 0.8rem 0.5rem;">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
                 color: rgba(100,150,255,0.4); letter-spacing: 1px; margin-bottom: 4px;">
                AI DATA SCIENCE AGENT
            </div>
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
                 color: rgba(80,200,255,0.35); letter-spacing: 1px; margin-bottom: 10px;">
                POWERED BY GEMINI 2.5 FLASH
            </div>
            <div style="padding: 8px 14px; border-radius: 10px;
                 background: linear-gradient(135deg, rgba(64,128,255,0.10), rgba(128,64,255,0.10));
                 border: 1px solid rgba(100,150,255,0.25); margin-bottom: 6px;">
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
                     color: rgba(160,200,255,0.4); letter-spacing: 2px; margin-bottom: 4px;">
                    ✦ AUTHOR
                </div>
                <div style="font-family: 'Orbitron', monospace; font-size: 0.78rem;
                     font-weight: 700; letter-spacing: 2px;
                     background: linear-gradient(135deg, #4080ff, #8040ff, #40c8ff);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                     background-clip: text;">
                    HARI KRISHNA
                </div>
            </div>
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
                 color: rgba(100,150,255,0.25); letter-spacing: 1px;">
                v2.0.0 · 2025
            </div>
        </div>""", unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 28px; flex-wrap: wrap;">
            <div style="flex-shrink: 0;">
              <svg width="120" height="120" viewBox="0 0 90 90" fill="none" xmlns="http://www.w3.org/2000/svg"
                style="filter: drop-shadow(0 0 20px rgba(64,128,255,0.8)) drop-shadow(0 0 40px rgba(120,80,255,0.4)); animation: logoFloat 4s ease-in-out infinite;">
                <defs>
                  <radialGradient id="hbgGrad" cx="50%" cy="40%" r="55%">
                    <stop offset="0%"  stop-color="#6080ff" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="#0a0f30" stop-opacity="0"/>
                  </radialGradient>
                  <linearGradient id="hbodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%"  stop-color="#1a2a8c"/>
                    <stop offset="50%" stop-color="#3a1a7c"/>
                    <stop offset="100%" stop-color="#0d1a60"/>
                  </linearGradient>
                  <linearGradient id="hfaceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%"  stop-color="#0d1a60"/>
                    <stop offset="100%" stop-color="#2a0d50"/>
                  </linearGradient>
                  <linearGradient id="heyeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%"  stop-color="#80ffff"/>
                    <stop offset="100%" stop-color="#4080ff"/>
                  </linearGradient>
                  <linearGradient id="htopGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%"  stop-color="#4080ff"/>
                    <stop offset="50%" stop-color="#8040ff"/>
                    <stop offset="100%" stop-color="#40c8ff"/>
                  </linearGradient>
                  <linearGradient id="haccentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%"  stop-color="#40ffb0"/>
                    <stop offset="100%" stop-color="#4080ff"/>
                  </linearGradient>
                  <filter id="hglow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                  <filter id="hstrongGlow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                </defs>
                <circle cx="45" cy="45" r="42" fill="url(#hbgGrad)" stroke="url(#htopGrad)" stroke-width="0.6" opacity="0.7"/>
                <ellipse cx="45" cy="45" rx="40" ry="13" fill="none" stroke="url(#htopGrad)" stroke-width="0.7" opacity="0.5" transform="rotate(-20 45 45)"/>
                <circle cx="12" cy="38" r="3" fill="#40ffb0" filter="url(#hglow)" opacity="0.9"/>
                <circle cx="78" cy="52" r="2" fill="#8040ff" filter="url(#hglow)" opacity="0.9"/>
                <rect x="22" y="46" width="46" height="32" rx="8" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="0.9"/>
                <rect x="22" y="46" width="46" height="11" rx="8" fill="rgba(255,255,255,0.05)"/>
                <rect x="26" y="18" width="38" height="32" rx="7" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="1.1"/>
                <rect x="26" y="18" width="38" height="11" rx="7" fill="rgba(255,255,255,0.07)"/>
                <rect x="43" y="10" width="4" height="10" rx="2" fill="url(#htopGrad)"/>
                <circle cx="45" cy="8" r="5" fill="#40c8ff" filter="url(#hstrongGlow)" opacity="0.9"/>
                <circle cx="45" cy="8" r="2.5" fill="white"/>
                <rect x="30" y="26" width="12" height="8" rx="3" fill="url(#heyeGrad)" filter="url(#hglow)"/>
                <rect x="32" y="28" width="4" height="4" rx="1" fill="white" opacity="0.9"/>
                <rect x="48" y="26" width="12" height="8" rx="3" fill="url(#heyeGrad)" filter="url(#hglow)"/>
                <rect x="50" y="28" width="4" height="4" rx="1" fill="white" opacity="0.9"/>
                <rect x="31" y="38" width="28" height="4" rx="2" fill="url(#haccentGrad)" opacity="0.8" filter="url(#hglow)"/>
                <rect x="18" y="24" width="8" height="14" rx="3" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="0.7"/>
                <rect x="20" y="29" width="4" height="4" rx="1" fill="#4080ff" filter="url(#hglow)"/>
                <rect x="64" y="24" width="8" height="14" rx="3" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="0.7"/>
                <rect x="66" y="29" width="4" height="4" rx="1" fill="#8040ff" filter="url(#hglow)"/>
                <rect x="38" y="50" width="14" height="4" rx="2" fill="url(#htopGrad)" opacity="0.7"/>
                <rect x="29" y="54" width="32" height="18" rx="4" fill="url(#hfaceGrad)" stroke="rgba(100,150,255,0.3)" stroke-width="0.6"/>
                <line x1="33" y1="60" x2="57" y2="60" stroke="url(#haccentGrad)" stroke-width="0.8" opacity="0.6"/>
                <line x1="33" y1="65" x2="49" y2="65" stroke="#4080ff" stroke-width="0.8" opacity="0.5"/>
                <circle cx="45" cy="63" r="5" fill="none" stroke="url(#htopGrad)" stroke-width="1.1" opacity="0.8"/>
                <circle cx="45" cy="63" r="3" fill="#40c8ff" filter="url(#hstrongGlow)" opacity="0.9"/>
                <rect x="13" y="48" width="9" height="22" rx="4" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="0.7"/>
                <circle cx="17" cy="72" r="3.5" fill="url(#haccentGrad)" filter="url(#hglow)"/>
                <rect x="68" y="48" width="9" height="22" rx="4" fill="url(#hbodyGrad)" stroke="url(#htopGrad)" stroke-width="0.7"/>
                <circle cx="73" cy="72" r="3.5" fill="url(#haccentGrad)" filter="url(#hglow)"/>
                <rect x="29" y="76" width="32" height="3" rx="1.5" fill="url(#haccentGrad)" opacity="0.7" filter="url(#hglow)"/>
                <circle cx="25" cy="49" r="1.8" fill="#4080ff" opacity="0.7"/>
                <circle cx="65" cy="49" r="1.8" fill="#8040ff" opacity="0.7"/>
                <circle cx="25" cy="76" r="1.8" fill="#40c8ff" opacity="0.7"/>
                <circle cx="65" cy="76" r="1.8" fill="#40ffb0" opacity="0.7"/>
              </svg>
            </div>
            <div>
                <h1 class="hero-title">AI AUTONOMOUS<br>DATA SCIENCE AGENT</h1>
                <p class="hero-subtitle">⚡ Powered by Gemini 2.5 Flash · Scikit-Learn · XGBoost</p>
                <div class="hero-badges">
                    <span class="badge badge-blue">🧠 AUTO ML</span>
                    <span class="badge badge-purple">🔮 GEMINI 2.5</span>
                    <span class="badge badge-cyan">📊 SCIKIT-LEARN</span>
                    <span class="badge badge-green">⚡ XGBOOST</span>
                    <span class="badge badge-blue">🎯 ZERO CODE</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Dashboard Page ────────────────────────────────────────────────────────────

def render_dashboard():
    render_hero()

    col1, col2, col3, col4 = st.columns(4)
    dash_metrics = [
        ("🧬", "Auto-Detect", "Target Column", "#40ffb0"),
        ("🤖", "4 Models", "Evaluated", "#40c8ff"),
        ("⚡", "6 Steps", "ML Pipeline", "#8040ff"),
        ("🎯", "100%", "Autonomous", "#ff8040"),
    ]
    for col, (icon, val, label, color) in zip([col1, col2, col3, col4], dash_metrics):
        with col:
            st.markdown(f"""
            <div class="metric-holo">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #ffffff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="card-3d card-glow-blue">
            <div class="card-title" style="color: #80b0ff;">🔬 HOW IT WORKS</div>
            <div style="display: flex; flex-direction: column; gap: 10px;">
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("1", "Upload CSV Dataset", "Any tabular data file", "done"),
            ("2", "Gemini AI Analysis", "Deep insights & explanations", "done"),
            ("3", "Auto Data Cleaning", "Missing values, encoding, duplicates", "done"),
            ("4", "Feature Engineering", "Scaling, selection, transformation", "done"),
            ("5", "Model Selection", "RF, XGBoost, LR, LinearReg", "done"),
            ("6", "Evaluate & Report", "Metrics, charts, insights", "done"),
        ]

        for num, title, desc, status in pipeline_steps:
            st.markdown(f"""
            <div class="pipeline-step step-done">
                <div class="step-number step-number-done">{num}</div>
                <div>
                    <div class="step-text" style="color: #80ffc0;">{title}</div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: rgba(160,200,255,0.5); margin-top: 2px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="card-3d">
            <div class="card-title" style="color: #80c0ff;">🎯 CAPABILITIES</div>
        """, unsafe_allow_html=True)

        caps = [
            ("🔍", "Classification", "Multi-class & binary"),
            ("📈", "Regression", "Continuous predictions"),
            ("🧹", "Auto Cleaning", "Smart imputation"),
            ("🏆", "Best Model", "Auto-selection"),
            ("📊", "3D Charts", "Interactive visuals"),
            ("🤖", "AI Reasoning", "Gemini explanations"),
        ]
        for icon, title, desc in caps:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0;
                border-bottom: 1px solid rgba(100,150,255,0.08);">
                <span style="font-size: 1.4rem;">{icon}</span>
                <div>
                    <div style="font-family: 'Rajdhani'; font-weight: 600; font-size: 0.9rem; color: #a0c0ff;">{title}</div>
                    <div style="font-size: 0.75rem; color: rgba(160,200,255,0.4); font-family: 'Share Tech Mono';">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ─── Upload Page ───────────────────────────────────────────────────────────────

def render_upload():
    st.markdown('<div class="section-header" style="color: #80c8ff;">📤 DATASET UPLOAD</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <span class="upload-icon">📂</span>
        <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #80b0ff; margin-bottom: 0.5rem;">
            DROP YOUR CSV DATASET
        </div>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; color: rgba(160,200,255,0.5);">
            Supports CSV · Auto-detects target · Any domain
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("🔄 Processing dataset..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                r = requests.post(f"{BACKEND}/upload", files=files, timeout=60)
                if r.status_code == 200:
                    result = r.json()
                    st.session_state.upload_result = result
                    st.success("✅ Dataset uploaded successfully!")
                else:
                    st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
                    return
            except Exception as e:
                st.error(f"Connection error: {e}")
                return

    if "upload_result" in st.session_state:
        res = st.session_state.upload_result

        # Summary cards
        col1, col2, col3, col4, col5 = st.columns(5)
        summary = res.get("summary", {})
        shape   = summary.get("shape", [0, 0])

        for col, (icon, val, label) in zip(
            [col1, col2, col3, col4, col5],
            [
                ("📋", f"{shape[0]:,}", "Rows"),
                ("🔢", str(shape[1]), "Columns"),
                ("🔵", str(len(summary.get("numeric_columns", []))), "Numeric"),
                ("🏷️", str(len(summary.get("categorical_columns", []))), "Categorical"),
                ("⚠️", str(sum(summary.get("missing_values", {}).values())), "Missing"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-holo" style="padding: 1rem;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <div class="metric-value" style="font-size: 1.4rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_info, col_target = st.columns([1, 1])
        with col_info:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title" style="color: #80c0ff;">📌 DETECTED CONFIGURATION</div>
                <div style="font-family: 'Share Tech Mono', monospace; line-height: 2;">
                    <span style="color: rgba(160,200,255,0.5);">FILE:</span>
                    <span style="color: #a0e0ff; margin-left: 8px;">{res.get('filename', 'N/A')}</span><br>
                    <span style="color: rgba(160,200,255,0.5);">TARGET:</span>
                    <span style="color: #40ffb0; margin-left: 8px; font-weight: 700;">{res.get('target_column', 'N/A')}</span><br>
                    <span style="color: rgba(160,200,255,0.5);">TASK:</span>
                    <span style="color: #ff8040; margin-left: 8px; font-weight: 700; text-transform: uppercase;">{res.get('problem_type', 'N/A')}</span><br>
                    <span style="color: rgba(160,200,255,0.5);">SESSION:</span>
                    <span style="color: #8080ff; margin-left: 8px; font-size: 0.7rem;">{res.get('session_id', 'N/A')}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_target:
            # Column selection override
            st.markdown('<div class="card-3d"><div class="card-title" style="color: #80c0ff;">⚙️ OVERRIDE TARGET</div>', unsafe_allow_html=True)
            cols = res["summary"]["columns"]
            default_idx = cols.index(res["target_column"]) if res["target_column"] in cols else 0
            chosen = st.selectbox("Target Column", cols, index=default_idx)
            st.session_state.target_column = chosen
            st.markdown("</div>", unsafe_allow_html=True)

        # Preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="color: #80c8ff; font-size: 1rem;">🔎 DATA PREVIEW</div>', unsafe_allow_html=True)
        preview_df = pd.DataFrame(res.get("preview", []))
        st.dataframe(preview_df, use_container_width=True, height=300)

        # Correlation Heatmap
        corr_data = res.get("correlation", {})
        if corr_data and corr_data.get("columns"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="color: #80c8ff; font-size: 1rem;">🔥 CORRELATION HEATMAP</div>', unsafe_allow_html=True)
            # earlier 'cols = res["summary"]["columns"]' variable on line above.
            corr_cols = corr_data["columns"]
            corr_vals = corr_data["values"]

            fig = go.Figure(go.Heatmap(
                z=corr_vals, x=corr_cols, y=corr_cols,
                colorscale=[
                    [0.0,  "#1a0d4b"],
                    [0.25, "#1a2a6c"],
                    [0.5,  "#0d3b6b"],
                    [0.75, "#1a5c8c"],
                    [1.0,  "#40c8ff"],
                ],
                text=[[f"{v:.2f}" for v in row] for row in corr_vals],
                texttemplate="%{text}",
                textfont=dict(size=9, color="white"),
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="#7090c0"),
                    bordercolor="rgba(100,150,255,0.3)",
                    borderwidth=1,
                    bgcolor="rgba(5,12,35,0.8)",
                ),
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=450,
                **PLOTLY_LAYOUT,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)


# ─── Insights Page ─────────────────────────────────────────────────────────────

def render_insights():
    st.markdown('<div class="section-header" style="color: #b080ff;">🔮 GEMINI AI INSIGHTS</div>', unsafe_allow_html=True)

    if "upload_result" not in st.session_state:
        st.markdown("""
        <div class="card-3d" style="text-align: center; padding: 3rem;">
            <span style="font-size: 3rem;">📤</span>
            <div style="font-family: 'Orbitron', monospace; color: #6080a0; margin-top: 1rem; font-size: 0.9rem;">
                Upload a dataset first to enable AI analysis.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    if st.button("🔮 ANALYZE WITH GEMINI 2.5 FLASH"):
        session_id = st.session_state.upload_result["session_id"]
        with st.spinner("🧠 Gemini is thinking deeply..."):
            try:
                r = requests.post(
                    f"{BACKEND}/analyze",
                    json={"session_id": session_id},
                    timeout=90
                )
                if r.status_code == 200:
                    st.session_state.analyze_result = r.json()
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"Error: {r.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "analyze_result" in st.session_state:
        res = st.session_state.analyze_result
        insights_text = res.get("gemini_insights", "No insights available.")

        # Show clear warning if Gemini returned an error
        is_error = insights_text.startswith("⚠️") or "error" in insights_text[:80].lower()
        if is_error:
            st.warning(f"""**Gemini API Note:** {insights_text}

💡 **Fix:** Make sure `GEMINI_API_KEY` is set correctly. The backend auto-tries `gemini-2.5-flash`, `gemini-2.0-flash`, and `gemini-1.5-flash` in order. Verify your key at https://aistudio.google.com/app/apikey""")
            return

        col1, col2 = st.columns([1, 3])
        with col1:
            pt = res.get("problem_type", "").upper()
            color = "#ff8040" if "reg" in pt.lower() else "#40ffb0"
            st.markdown(f"""
            <div class="metric-holo" style="padding: 1.5rem;">
                <span style="font-size: 2.5rem;">🎯</span>
                <div class="metric-value" style="font-size: 1.1rem; background: linear-gradient(135deg, {color}, #ffffff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 0.5rem;">{pt}</div>
                <div class="metric-label">Problem Type</div>
                <div style="margin-top: 1rem; font-family: 'Share Tech Mono'; font-size: 0.7rem; color: rgba(160,200,255,0.5);">
                    TARGET<br><span style="color: #40ffb0; font-size: 0.8rem;">{res.get('target_column', 'N/A')}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title" style="color: #c080ff;">🤖 GEMINI 2.5 FLASH ANALYSIS</div>
                <div class="terminal-box">{insights_text}</div>
            </div>""", unsafe_allow_html=True)


# ─── Pipeline Page ─────────────────────────────────────────────────────────────

def render_pipeline():
    st.markdown('<div class="section-header" style="color: #40ffb0;">⚙️ AUTONOMOUS ML PIPELINE</div>', unsafe_allow_html=True)

    if "upload_result" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
        return

    # Pipeline visualization
    steps_info = [
        ("🔍", "UNDERSTAND", "Detect columns, target, problem type"),
        ("🧹", "CLEAN",      "Handle nulls, duplicates, encode"),
        ("⚙️", "ENGINEER",   "Scale, select top features"),
        ("🤖", "SELECT",     "Evaluate all candidate models"),
        ("🏋️", "TRAIN",     "Fit best model on training data"),
        ("📊", "EVALUATE",   "Compute metrics and insights"),
    ]

    cols = st.columns(6)
    for i, (col, (icon, title, desc)) in enumerate(zip(cols, steps_info)):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0.5rem;
                border-radius: 12px; border: 1px solid rgba(100,150,255,0.15);
                background: rgba(10,20,60,0.4); position: relative;">
                <span style="font-size: 2rem;">{icon}</span>
                <div style="font-family: 'Orbitron', monospace; font-size: 0.6rem; color: #6090ff;
                    margin-top: 0.5rem; letter-spacing: 1px;">{title}</div>
                <div style="font-size: 0.65rem; color: rgba(160,200,255,0.4);
                    font-family: 'Share Tech Mono'; margin-top: 0.3rem; line-height: 1.4;">{desc}</div>
                {f'<div style="position:absolute;top:-1px;right:-1px;width:8px;height:8px;border-radius:50%;background:#4080ff;box-shadow:0 0 8px #4080ff;"></div>' if i < 5 else ''}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card-3d" style="text-align: center; padding: 2.5rem;">
        <span style="font-size: 4rem; display: block; margin-bottom: 1rem;
            animation: iconFloat 3s ease-in-out infinite;">🚀</span>
        <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem; color: #80c0ff;
            margin-bottom: 0.5rem; letter-spacing: 3px;">READY TO LAUNCH</div>
        <div style="font-family: 'Share Tech Mono'; font-size: 0.8rem; color: rgba(160,200,255,0.5);
            margin-bottom: 1.5rem;">
            Auto-select best ML model · Gemini-powered reasoning · Full evaluation report
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 RUN AUTONOMOUS ML AGENT"):
        session_id = st.session_state.upload_result["session_id"]
        target     = st.session_state.get("target_column", st.session_state.upload_result.get("target_column"))

        progress_bar = st.progress(0)
        status_text  = st.empty()

        pipeline_steps = [
            (0.15, "🔍 Understanding dataset structure..."),
            (0.30, "🧹 Cleaning and preprocessing data..."),
            (0.50, "⚙️ Engineering features..."),
            (0.70, "🤖 Training and evaluating models..."),
            (0.90, "📊 Generating insights with Gemini..."),
            (1.00, "✅ Pipeline complete!"),
        ]

        try:
            result_holder = [None]
            error_holder  = [None]

            def call_backend():
                try:
                    r = requests.post(
                        f"{BACKEND}/run-pipeline",
                        json={"session_id": session_id, "target_column": target},
                        timeout=180
                    )
                    result_holder[0] = r.json() if r.status_code == 200 else None
                    if r.status_code != 200:
                        error_holder[0] = r.json().get("detail", "Unknown error")
                except Exception as e:
                    error_holder[0] = str(e)

            thread = threading.Thread(target=call_backend)
            thread.start()

            # Animate progress while waiting
            step_idx = 0
            while thread.is_alive():
                if step_idx < len(pipeline_steps) - 1:
                    prog, msg = pipeline_steps[step_idx]
                    progress_bar.progress(prog)
                    status_text.markdown(f"""
                    <div style="font-family: 'Share Tech Mono'; font-size: 0.85rem; color: #80c0ff; padding: 0.5rem;">
                        {msg}
                    </div>""", unsafe_allow_html=True)
                    step_idx += 1
                time.sleep(1.5)

            thread.join()
            progress_bar.progress(1.0)
            status_text.markdown("""<div style="color: #40ffb0; font-family: 'Share Tech Mono'; font-size: 0.85rem;">✅ Pipeline complete!</div>""", unsafe_allow_html=True)

            if error_holder[0]:
                st.error(f"Pipeline failed: {error_holder[0]}")
            elif result_holder[0]:
                st.session_state.pipeline_result = result_holder[0]
                st.success("🎉 Autonomous ML pipeline completed! Navigate to Results →")
                st.balloons()
            else:
                st.error("No result returned from pipeline.")

        except Exception as e:
            st.error(f"Error: {e}")


# ─── Results Page ──────────────────────────────────────────────────────────────

def render_results():
    st.markdown('<div class="section-header" style="color: #40c8ff;">📊 RESULTS & ANALYSIS</div>', unsafe_allow_html=True)

    if "pipeline_result" not in st.session_state:
        st.markdown("""
        <div class="card-3d" style="text-align: center; padding: 3rem;">
            <span style="font-size: 3rem;">⚙️</span>
            <div style="font-family: 'Orbitron', monospace; color: #6080a0; margin-top: 1rem; font-size: 0.9rem;">
                Run the ML pipeline first to see results.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    res = st.session_state.pipeline_result
    metrics         = res.get("metrics", {})
    feature_imp     = res.get("feature_importance", {})
    model_comparison= res.get("model_comparison", {})
    problem_type    = res.get("problem_type", "classification")
    best_model      = res.get("best_model", "Unknown")
    ds              = res.get("dataset_summary", {})

    # ── Winner Banner ──────────────────────────────────────────────────────────
    model_colors = {
        "RandomForest":    ("🌲", "#40ffa0"),
        "XGBoost":         ("⚡", "#ffb040"),
        "LogisticRegression": ("📉", "#40c8ff"),
        "LinearRegression": ("📈", "#c040ff"),
    }
    icon, mcolor = model_colors.get(best_model, ("🤖", "#80b0ff"))

    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-family: 'Share Tech Mono'; font-size: 0.75rem; color: rgba(160,200,255,0.5);
            letter-spacing: 3px; margin-bottom: 1rem;">🏆 WINNER MODEL</div>
        <div class="model-winner" style="display: inline-flex;">
            {icon} {best_model}
        </div>
        <div style="font-family: 'Share Tech Mono'; font-size: 0.7rem; color: rgba(160,200,255,0.4);
            margin-top: 0.8rem;">TASK: {problem_type.upper()} · TARGET: {res.get('target_column','N/A').upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    metric_items = []

    if problem_type == "classification":
        for key, icon, label, color in [
            ("accuracy",  "🎯", "ACCURACY",  "#40ffb0"),
            ("f1_score",  "⚡", "F1 SCORE",  "#40c8ff"),
            ("precision", "🔬", "PRECISION", "#8040ff"),
            ("recall",    "📡", "RECALL",    "#ff8040"),
        ]:
            val = metrics.get(key, 0)
            metric_items.append((icon, f"{val:.2%}", label, color))
    else:
        r2  = metrics.get("r2_score", 0)
        rmse= metrics.get("rmse", 0)
        metric_items = [
            ("📐", f"{r2:.4f}", "R² SCORE",   "#40ffb0"),
            ("📏", f"{rmse:.4f}", "RMSE",      "#40c8ff"),
            ("📦", f"{ds.get('train_samples',0):,}", "TRAIN SAMPLES", "#8040ff"),
            ("🧪", f"{ds.get('test_samples',0):,}",  "TEST SAMPLES",  "#ff8040"),
        ]

    cols = st.columns(4)
    for col, (icon, val, label, color) in zip(cols, metric_items):
        with col:
            st.markdown(f"""
            <div class="metric-holo">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #ffffff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_fi, col_mc = st.columns([3, 2])

    with col_fi:
        if feature_imp:
            names  = list(feature_imp.keys())[:12]
            values = [feature_imp[n] for n in names]
            max_v  = max(values) if values else 1

            fig = go.Figure()
            for i, (n, v) in enumerate(zip(names, values)):
                alpha = 0.4 + 0.6 * (v / max_v)
                fig.add_trace(go.Bar(
                    x=[v], y=[n],
                    orientation="h",
                    marker=dict(
                        color=f"rgba(64,{128+i*8},{255-i*10},{alpha})",
                        line=dict(color="rgba(100,150,255,0.3)", width=1),
                    ),
                    text=f"{v:.4f}",
                    textposition="outside",
                    textfont=dict(color="#a0c0ff", size=10),
                    showlegend=False,
                    name=n,
                ))
            # PLOTLY_LAYOUT's yaxis key → TypeError duplicate keyword argument).
            # autorange="reversed" now applied via update_yaxes() separately.
            fig.update_layout(
                title="🔬 Feature Importance",
                height=420,
                barmode="overlay",
                **PLOTLY_LAYOUT,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(autorange="reversed", **AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

    with col_mc:
        if model_comparison:
            model_names  = list(model_comparison.keys())
            model_scores = list(model_comparison.values())
            colors_mc    = ["#40ffb0" if n == best_model else "#4080ff" for n in model_names]

            fig2 = go.Figure(go.Bar(
                x=model_scores,
                y=model_names,
                orientation="h",
                marker=dict(
                    color=colors_mc,
                    line=dict(color="rgba(100,150,255,0.3)", width=1),
                ),
                text=[f"{s:.4f}" for s in model_scores],
                textposition="outside",
                textfont=dict(color="#a0c0ff", size=11),
            ))
            fig2.update_layout(
                title="🏆 Model Competition",
                height=420,
                **PLOTLY_LAYOUT,
            )
            fig2.update_xaxes(**AXIS_STYLE)
            fig2.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Radar Chart for Classification ─────────────────────────────────────────
    if problem_type == "classification" and metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        col_r, col_d = st.columns([1, 1])

        with col_r:
            cats  = ["Accuracy", "F1 Score", "Precision", "Recall"]
            vals  = [
                metrics.get("accuracy", 0),
                metrics.get("f1_score", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
            ]
            vals_closed = vals + [vals[0]]
            cats_closed = cats + [cats[0]]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor="rgba(64,128,255,0.15)",
                line=dict(color="#4080ff", width=2),
                marker=dict(color="#40c8ff", size=8),
                name=best_model,
            ))
            fig3.update_layout(
                title="🕸️ Performance Radar",
                polar=dict(
                    bgcolor="rgba(5,12,35,0.5)",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        tickfont=dict(color="#6090c0", size=9),
                        gridcolor="rgba(100,150,255,0.1)",
                        linecolor="rgba(100,150,255,0.2)",
                    ),
                    angularaxis=dict(
                        tickfont=dict(color="#80a0e0", size=11),
                        gridcolor="rgba(100,150,255,0.1)",
                        linecolor="rgba(100,150,255,0.2)",
                    ),
                ),
                height=380,
                # PLOTLY_LAYOUT no longer contains xaxis/yaxis (fixed in Bug #1)
                # so it is safe to unpack directly here.
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            # Dataset summary donut
            n_num = ds.get("numeric_features", 0)
            n_cat = ds.get("categorical_features", 0)
            # if all pie values are 0. Default to 1 so the chart renders.
            safe_num = max(n_num, 0)
            safe_cat = max(n_cat, 0)
            if safe_num == 0 and safe_cat == 0:
                safe_num, safe_cat = 1, 1  # placeholder to avoid crash
            fig4 = go.Figure(go.Pie(
                labels=["Numeric Features", "Categorical Features"],
                values=[safe_num, safe_cat],
                hole=0.6,
                marker=dict(
                    colors=["#4080ff","#8040ff"],
                    line=dict(color="rgba(5,12,35,1)", width=3),
                ),
                textfont=dict(color="white", size=11),
            ))
            fig4.add_annotation(
                text=f"<b>{n_num+n_cat}</b><br>Features",
                x=0.5, y=0.5,
                font=dict(size=18, color="#80c0ff", family="Orbitron"),
                showarrow=False,
            )
            fig4.update_layout(
                title="📦 Feature Composition",
                height=380,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_stats, col_reason = st.columns([1, 1])

    with col_stats:
        st.markdown("""
        <div class="card-3d">
            <div class="card-title" style="color: #80c0ff;">📦 PIPELINE SUMMARY</div>
        """, unsafe_allow_html=True)

        stat_rows = [
            ("Total Rows",         f"{ds.get('rows',0):,}"),
            ("Total Columns",      f"{ds.get('columns',0):,}"),
            ("Features Selected",  f"{ds.get('features_selected',0):,}"),
            ("Train Samples",      f"{ds.get('train_samples',0):,}"),
            ("Test Samples",       f"{ds.get('test_samples',0):,}"),
            ("Missing Values",     f"{ds.get('missing_values',0):,}"),
            ("Duplicates Removed", f"{ds.get('duplicates_removed',0):,}"),
        ]
        for label, val in stat_rows:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                padding: 10px 0; border-bottom: 1px solid rgba(100,150,255,0.07);">
                <span style="font-family:'Share Tech Mono'; font-size:0.78rem; color:rgba(160,200,255,0.5);">{label}</span>
                <span style="font-family:'Orbitron'; font-size:0.85rem; font-weight:700; color:#80c8ff;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_reason:
        reasoning = res.get("gemini_model_reasoning", "")
        if reasoning:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title" style="color: #c080ff;">🤖 GEMINI REASONING</div>
                <div class="terminal-box" style="color: #b0d0ff; font-size: 0.85rem;">
                    {reasoning[:1500]}{'...' if len(reasoning) > 1500 else ''}
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Feature Importance 3D Scatter ─────────────────────────────────────────
    if feature_imp and len(feature_imp) >= 3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="color: #40c8ff; font-size: 1rem;">🌐 3D FEATURE IMPORTANCE</div>', unsafe_allow_html=True)

        fi_items = list(feature_imp.items())[:15]
        fi_names = [x[0] for x in fi_items]
        fi_vals  = [x[1] for x in fi_items]
        n = len(fi_vals)
        angles = [i * 2 * np.pi / n for i in range(n)]

        fig5 = go.Figure(go.Scatter3d(
            x=[np.cos(a) * v * 10 for a, v in zip(angles, fi_vals)],
            y=[np.sin(a) * v * 10 for a, v in zip(angles, fi_vals)],
            z=fi_vals,
            mode="markers+text",
            marker=dict(
                size=[v * 40 + 5 for v in fi_vals],
                color=fi_vals,
                colorscale=[[0,"#1a0d4b"],[0.5,"#4080ff"],[1,"#40ffb0"]],
                opacity=0.85,
                showscale=True,
                colorbar=dict(tickfont=dict(color="#7090c0"), bgcolor="rgba(5,12,35,0.8)"),
            ),
            text=fi_names,
            textfont=dict(color="#a0c0ff", size=10),
            hovertemplate="<b>%{text}</b><br>Importance: %{z:.4f}<extra></extra>",
        ))
        fig5.update_layout(
            title="3D Feature Importance Visualization",
            scene=dict(
                bgcolor="rgba(5,12,35,0.8)",
                xaxis=dict(backgroundcolor="rgba(5,12,35,0)", gridcolor="rgba(100,150,255,0.1)", color="#7090c0"),
                yaxis=dict(backgroundcolor="rgba(5,12,35,0)", gridcolor="rgba(100,150,255,0.1)", color="#7090c0"),
                zaxis=dict(backgroundcolor="rgba(5,12,35,0)", gridcolor="rgba(100,150,255,0.1)", color="#7090c0",
                           title=dict(text="Importance", font=dict(color="#a0c0ff"))),
            ),
            height=500,
            paper_bgcolor="rgba(5,12,35,0)",
            font=dict(color="#a0c0ff", family="Rajdhani"),
            title_font=dict(family="Orbitron", color="#80b0ff", size=14),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    report = {
        "model": best_model,
        "problem_type": problem_type,
        "target_column": res.get("target_column"),
        "metrics": metrics,
        "feature_importance": feature_imp,
        "model_comparison": model_comparison,
        "dataset_summary": ds,
    }
    st.download_button(
        "⬇️ DOWNLOAD RESULTS JSON",
        data=json.dumps(report, indent=2),
        file_name="ml_pipeline_results.json",
        mime="application/json",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    page = st.session_state.get("current_page", "dashboard")

    if page == "dashboard":
        render_dashboard()
    elif page == "upload":
        render_upload()
    elif page == "insights":
        render_insights()
    elif page == "pipeline":
        render_pipeline()
    elif page == "results":
        render_results()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
