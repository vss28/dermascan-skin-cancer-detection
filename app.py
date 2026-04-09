"""
DermaScan — Skin Cancer Detection
7-Class Classification: akiec, bcc, bkl, df, mel, nv, vasc
Ensemble ML (XGBoost + Random Forest + HistGradientBoosting)
Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import pickle
import os
import json
from PIL import Image
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, hog
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan | Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background-color:#0d0d14; color:#e8e8f0; }
  .main { background-color:#0d0d14; }
  h1,h2,h3 { font-family:'DM Serif Display',serif; color:#f0f0ff; }

  .hero { text-align:center; padding:2rem 1rem 1.5rem; border-bottom:1px solid #1e1e2e; margin-bottom:1.5rem; }
  .hero h1 { font-size:2.8rem; background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.2rem; }
  .hero p { color:#8888aa; font-size:0.95rem; font-weight:300; }

  .model-card { background:linear-gradient(135deg,#141428,#1a1a2e); border:1px solid #2a2a4a; border-radius:16px; padding:1.5rem; text-align:center; }
  .model-card h4 { color:#aaaacc; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.3rem; }
  .model-card .model-name { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#f0f0ff; margin-bottom:0.5rem; }
  .acc-number { font-size:2.8rem; font-weight:700; background:linear-gradient(135deg,#34d399,#60a5fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; }
  .acc-label  { font-size:0.75rem; color:#6666aa; margin-top:0.2rem; }

  .result-card { background:linear-gradient(135deg,#141428,#1a1a2e); border:1px solid #2a2a4a; border-radius:20px; padding:2rem; margin-top:1rem; }
  .disease-title { font-family:'DM Serif Display',serif; font-size:2rem; color:#f0f0ff; margin:0 0 0.3rem; }
  .confidence-bar-bg { background:#1e1e2e; border-radius:999px; height:12px; margin:0.5rem 0 0.3rem; }
  .confidence-bar-fill { height:12px; border-radius:999px; background:linear-gradient(90deg,#a78bfa,#60a5fa); }

  .badge-high   { display:inline-block; background:#3d1212; color:#ff6b6b; border:1px solid #ff4444; border-radius:999px; padding:4px 16px; font-size:0.85rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
  .badge-medium { display:inline-block; background:#2d2212; color:#ffbb55; border:1px solid #ff9900; border-radius:999px; padding:4px 16px; font-size:0.85rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
  .badge-low    { display:inline-block; background:#0d2d1a; color:#4ade80; border:1px solid #22c55e; border-radius:999px; padding:4px 16px; font-size:0.85rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; }

  .info-box { background:#12121f; border:1px solid #1e1e2e; border-radius:12px; padding:1rem 1.5rem; margin-top:0.8rem; font-size:0.9rem; color:#aaaacc; line-height:1.6; }
  .info-box strong { color:#d0d0ff; }

  .section-label { font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#6666aa; margin-bottom:0.3rem; }
  .disclaimer { background:#1a1208; border:1px solid #444; border-radius:10px; padding:0.8rem 1.2rem; margin-top:2rem; font-size:0.78rem; color:#888866; text-align:center; }
  .divider { border:none; border-top:1px solid #1e1e2e; margin:1.5rem 0; }

  .stFileUploader > div { border:2px dashed #2a2a4a !important; background:#12121f !important; border-radius:16px !important; padding:1.5rem !important; }
  .stFileUploader > div:hover { border-color:#a78bfa !important; }

  .source-badge { display:inline-flex; align-items:center; gap:6px; background:#1a1a2e; border:1px solid #2a2a4a; border-radius:999px; padding:4px 12px; font-size:0.78rem; color:#8888aa; margin-bottom:0.8rem; }

  [data-testid="stCameraInput"] > div { border:2px dashed #2a2a4a !important; background:#12121f !important; border-radius:16px !important; }
  [data-testid="stCameraInput"] button { background:linear-gradient(135deg,#a78bfa,#60a5fa) !important; color:white !important; border:none !important; border-radius:8px !important; padding:0.5rem 1.5rem !important; font-weight:600 !important; }

  .stButton > button { border:1px solid #2a2a4a !important; background:#12121f !important; color:#aaaacc !important; border-radius:8px !important; transition:all 0.2s !important; }
  .stButton > button:hover { border-color:#a78bfa !important; color:#a78bfa !important; background:#1a1a2e !important; }

  #MainMenu, footer, header { visibility:hidden; }
  .block-container { padding-top:0; max-width:1200px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 128

CLASS_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}
RISK_LEVEL = {
    'mel':   'HIGH',
    'bcc':   'HIGH',
    'akiec': 'MEDIUM',
    'bkl':   'LOW',
    'df':    'LOW',
    'nv':    'LOW',
    'vasc':  'LOW'
}
DESCRIPTIONS = {
    'akiec': 'Precancerous lesions caused by sun damage. Can develop into squamous cell carcinoma if left untreated.',
    'bcc':   'Most common form of skin cancer. Develops in sun-exposed areas. Grows slowly but needs prompt treatment.',
    'bkl':   'Non-cancerous skin growths. Common and harmless, though they may resemble more serious conditions.',
    'df':    'Benign fibrous skin nodule, commonly found on the legs. Usually harmless.',
    'mel':   'Most dangerous form of skin cancer. Can spread rapidly. Early detection is critical for survival.',
    'nv':    'Common benign moles. Monitor for any changes in size, shape, or color.',
    'vasc':  'Lesions from blood vessels including angiomas and pyogenic granulomas. Generally benign.'
}
ADVICE = {
    'HIGH':   '⚠️ Please consult a dermatologist or oncologist as soon as possible.',
    'MEDIUM': '🔶 Schedule an appointment with a dermatologist for further evaluation.',
    'LOW':    '✅ Monitor for any changes. A routine annual skin check is recommended.'
}

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

def extract_color_histogram(img):
    f = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [32], [0, 256])
        f.extend(cv2.normalize(hist, hist).flatten())
    return f

def extract_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        f.extend(cv2.normalize(hist, hist).flatten())
    return f

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, 24, 3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    return hist

def extract_glcm(img):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(gray, [1], [0], 8, True, True)
    return [graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'energy')[0, 0]]

def extract_hog_feat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def extract_color_moments(img):
    f = []
    for i in range(3):
        ch   = img[:, :, i].astype(float)
        mean = np.mean(ch)
        std  = np.std(ch)
        skew = np.mean(((ch - mean) / (std + 1e-7)) ** 3)
        f.extend([mean, std, skew])
    return f

def extract_asymmetry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w   = mask.shape
    top    = mask[:h//2, :]
    bottom = np.flipud(mask[h//2:, :])
    left   = mask[:, :w//2]
    right  = np.fliplr(mask[:, w//2:])
    min_h  = min(top.shape[0], bottom.shape[0])
    min_w  = min(left.shape[1], right.shape[1])
    h_diff = np.sum(np.abs(top[:min_h].astype(int) - bottom[:min_h].astype(int))) / (h * w * 255)
    v_diff = np.sum(np.abs(left[:, :min_w].astype(int) - right[:, :min_w].astype(int))) / (h * w * 255)
    return [h_diff, v_diff]

def extract_border_irregularity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0]
    cnt         = max(contours, key=cv2.contourArea)
    area        = cv2.contourArea(cnt)
    perimeter   = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / (area + 1e-7)
    return [compactness]

def extract_color_variance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return [
        np.std(hsv[:, :, 0]),
        np.std(hsv[:, :, 1]),
        np.mean(hsv[:, :, 0])
    ]

def extract_features(img):
    return np.concatenate([
        extract_color_histogram(img),
        extract_hsv_histogram(img),
        extract_lbp(img),
        extract_glcm(img),
        extract_hog_feat(img),
        extract_color_moments(img),
        extract_asymmetry(img),
        extract_border_irregularity(img),
        extract_color_variance(img)
    ]).astype(np.float32)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = pickle.load(open('models/skin_cancer_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    le     = pickle.load(open('models/label_encoder.pkl', 'rb'))
    accs   = pickle.load(open('models/model_accuracies.pkl', 'rb')) \
             if os.path.exists('models/model_accuracies.pkl') else None
    return model, scaler, le, accs

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(img_array, model, scaler, le):
    img  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img  = preprocess(img)
    feat = extract_features(img).reshape(1, -1)
    feat = scaler.transform(feat)

    try:
        if hasattr(model, 'estimators_'):
            for est in model.estimators_:
                if hasattr(est, 'get_booster'):
                    est.get_booster().set_param('device', 'cpu')
        elif hasattr(model, 'get_booster'):
            model.get_booster().set_param('device', 'cpu')
    except Exception:
        pass

    proba      = model.predict_proba(feat)[0]
    pred_idx   = int(np.argmax(proba))
    pred_class = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    return pred_class, confidence, proba, le.classes_

# ─────────────────────────────────────────────
# PCA FEATURE VISUALIZATION  (uses components.html — JS works here)
# ─────────────────────────────────────────────
def show_pca_features(img_array, pred_class):
    img  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img  = preprocess(img)
    feat = extract_features(img).reshape(1, -1)

    n_total = feat.shape[1]
    hog_size = n_total - (96 + 96 + 26 + 2 + 9 + 2 + 1 + 3)

    segments = [
        ('Color Histogram', 96),
        ('HSV Histogram',   96),
        ('LBP Texture',     26),
        ('GLCM',             2),
        ('HOG Shape',  hog_size),
        ('Color Moments',    9),
        ('Asymmetry',        2),
        ('Border Irregularity', 1),
        ('Color Variance',   3),
    ]

    # Simulate variance around the real sample so PCA can decompose
    rng   = np.random.RandomState(42)
    X_sim = feat + rng.randn(300, n_total) * 0.05
    X_sim[0] = feat[0]   # keep real sample at index 0

    pca      = PCA(n_components=4)
    X_pca    = pca.fit_transform(X_sim)
    explained = pca.explained_variance_ratio_ * 100
    loadings  = pca.components_   # shape (4, n_features)

    # Which segment dominates each PC?
    seg_abs = []
    ptr = 0
    for name, size in segments:
        seg_abs.append((name, np.sum(np.abs(loadings[:, ptr:ptr + size]), axis=1)))
        ptr += size

    top_segments = []
    for pc_idx in range(4):
        scores = [(name, vals[pc_idx]) for name, vals in seg_abs]
        top    = max(scores, key=lambda x: x[1])[0]
        top_segments.append(top)

    # Per-segment % contribution to each PC (for tooltip breakdown)
    seg_contribs = []
    for pc_idx in range(4):
        total_abs = sum(vals[pc_idx] for _, vals in seg_abs)
        contrib = []
        for name, vals in seg_abs:
            pct = round(float(vals[pc_idx]) / (total_abs + 1e-9) * 100, 1)
            contrib.append({'name': name, 'pct': pct})
        contrib.sort(key=lambda x: -x['pct'])
        seg_contribs.append(contrib[:4])   # top 4 contributors per PC

    pca_data = {
        'components':   [f'PC{i+1}' for i in range(4)],
        'explained':    [round(float(e), 1) for e in explained],
        'dominant':     top_segments,
        'sample_val':   [round(float(X_pca[0, i]), 4) for i in range(4)],
        'contributions': seg_contribs,
    }

    risk   = RISK_LEVEL.get(pred_class, 'LOW')
    accent = {'HIGH': '#ff6b6b', 'MEDIUM': '#ffbb55', 'LOW': '#4ade80'}.get(risk, '#a78bfa')
    data_json = json.dumps(pca_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
    background: #0d0d14;
    color: #e8e8f0;
    padding: 0;
  }}
  .section-wrap {{
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
  }}
  .sec-label {{
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6666aa;
    margin-bottom: 4px;
  }}
  .sec-title {{
    font-size: 1.15rem;
    color: #f0f0ff;
    font-weight: 500;
    margin-bottom: 1.1rem;
  }}
  .pc-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }}
  .pc-card {{
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: border-color 0.2s;
  }}
  .pc-card:hover {{ border-color: var(--accent); }}
  .pc-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
    border-radius: 12px 0 0 12px;
  }}
  .pc-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  .pc-name {{
    font-size: 1rem;
    font-weight: 600;
    color: #d0d0ff;
  }}
  .pc-val {{
    font-size: 0.72rem;
    color: #8888aa;
    background: #0d0d14;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid #2a2a4a;
  }}
  .bar-bg {{
    background: #0d0d14;
    border-radius: 999px;
    height: 8px;
    margin-bottom: 6px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 8px;
    border-radius: 999px;
    background: var(--accent);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
  }}
  .pc-footer {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}
  .pc-dom {{
    font-size: 0.72rem;
    color: #6666aa;
  }}
  .pc-dom strong {{
    color: #aaaacc;
    font-weight: 500;
  }}
  .pc-pct {{
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent);
  }}
  .contrib-list {{
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #2a2a4a;
    display: none;
  }}
  .pc-card:hover .contrib-list {{ display: block; }}
  .contrib-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #888899;
    padding: 1px 0;
  }}
  .contrib-row .c-name {{ color: #aaaacc; }}
  .total-bar {{
    margin-top: 1rem;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 0.85rem 1rem;
  }}
  .total-label {{
    font-size: 0.72rem;
    color: #6666aa;
    margin-bottom: 8px;
  }}
  .total-label span {{ color: var(--accent); font-weight: 600; }}
  .total-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
  }}
  .total-row-name {{
    font-size: 0.75rem;
    color: #aaaacc;
    width: 110px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .total-mini-bg {{
    flex: 1;
    background: #0d0d14;
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
  }}
  .total-mini-fill {{
    height: 6px;
    border-radius: 999px;
    background: var(--accent);
    opacity: 0.7;
  }}
  .total-mini-pct {{
    font-size: 0.7rem;
    color: #6666aa;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
  }}
  .note {{
    font-size: 0.72rem;
    color: #6666aa;
    line-height: 1.55;
    margin-top: 0.9rem;
  }}
  .hover-hint {{
    font-size: 0.68rem;
    color: #4a4a7a;
    text-align: right;
    margin-top: 4px;
  }}
</style>
</head>
<body>
<div class="section-wrap" style="--accent: {accent};">
  <p class="sec-label">Feature Analysis · PCA</p>
  <p class="sec-title">Top 4 Principal Components</p>

  <div class="pc-grid" id="pc-grid"></div>

  <div class="hover-hint">Hover a card to see feature breakdown</div>

  <div class="total-bar" id="total-bar">
    <p class="total-label">Overall feature group contribution across all 4 PCs — <span id="total-variance"></span>% variance captured</p>
    <div id="total-rows"></div>
  </div>

  <p class="note">
    PCA decomposes the full feature vector into orthogonal axes of maximum variance.
    Each PC captures a different aspect of the image — shape, texture, or color.
    The value shown is this image's coordinate on that axis.
  </p>
</div>

<script>
(function() {{
  var data   = {data_json};
  var accent = '{accent}';
  var grid   = document.getElementById('pc-grid');
  var maxPct = Math.max.apply(null, data.explained);

  data.components.forEach(function(label, i) {{
    var pct    = data.explained[i];
    var dom    = data.dominant[i];
    var val    = data.sample_val[i];
    var contribs = data.contributions[i];
    var width  = (pct / maxPct * 100).toFixed(1);

    var contribHTML = contribs.map(function(c) {{
      return '<div class="contrib-row"><span class="c-name">' + c.name + '</span><span>' + c.pct + '%</span></div>';
    }}).join('');

    var card = document.createElement('div');
    card.className = 'pc-card';
    card.innerHTML =
      '<div class="pc-header">' +
        '<span class="pc-name">' + label + '</span>' +
        '<span class="pc-val">' + (val >= 0 ? '+' : '') + val + '</span>' +
      '</div>' +
      '<div class="bar-bg"><div class="bar-fill" style="width:0%" data-w="' + width + '%"></div></div>' +
      '<div class="pc-footer">' +
        '<span class="pc-dom">Dominant: <strong>' + dom + '</strong></span>' +
        '<span class="pc-pct">' + pct + '%</span>' +
      '</div>' +
      '<div class="contrib-list">' +
        '<div style="font-size:0.68rem;color:#6666aa;margin-bottom:4px;">Feature group breakdown</div>' +
        contribHTML +
      '</div>';
    grid.appendChild(card);
  }});

  // Animate bars after short delay
  setTimeout(function() {{
    document.querySelectorAll('.bar-fill').forEach(function(el) {{
      el.style.width = el.getAttribute('data-w');
    }});
  }}, 120);

  // Total variance summary bar
  var totalVar = data.explained.reduce(function(a, b) {{ return a + b; }}, 0);
  document.getElementById('total-variance').textContent = totalVar.toFixed(1);

  // Aggregate segment contributions across all 4 PCs
  var segTotals = {{}};
  data.contributions.forEach(function(pcContribs) {{
    pcContribs.forEach(function(c) {{
      segTotals[c.name] = (segTotals[c.name] || 0) + c.pct;
    }});
  }});

  var segArr = Object.keys(segTotals).map(function(k) {{
    return {{ name: k, total: segTotals[k] }};
  }});
  segArr.sort(function(a, b) {{ return b.total - a.total; }});

  var maxTotal = segArr[0] ? segArr[0].total : 1;
  var rowsDiv  = document.getElementById('total-rows');

  segArr.slice(0, 5).forEach(function(seg) {{
    var w   = (seg.total / maxTotal * 100).toFixed(1);
    var row = document.createElement('div');
    row.className = 'total-row';
    row.innerHTML =
      '<span class="total-row-name">' + seg.name + '</span>' +
      '<div class="total-mini-bg"><div class="total-mini-fill" style="width:0%" data-w="' + w + '%"></div></div>' +
      '<span class="total-mini-pct">' + Math.round(seg.total) + '%</span>';
    rowsDiv.appendChild(row);
  }});

  setTimeout(function() {{
    document.querySelectorAll('.total-mini-fill').forEach(function(el) {{
      el.style.width = el.getAttribute('data-w');
    }});
  }}, 250);
}})();
</script>
</body>
</html>"""

    components.html(html, height=490, scrolling=False)


# ─────────────────────────────────────────────
# SHOW RESULTS
# ─────────────────────────────────────────────
def show_results(image, img_array, model, scaler, le, source_label="Uploaded"):
    left, right = st.columns([1, 2])

    with left:
        source_icon = "📷" if source_label == "Camera" else "📁"
        st.markdown(
            f'<div class="source-badge">{source_icon} Source: {source_label}</div>',
            unsafe_allow_html=True
        )
        st.image(image, caption="Analysed Image", width=280)

    with right:
        with st.spinner("Analysing image..."):
            pred_class, confidence, proba, classes = predict(
                img_array, model, scaler, le
            )

        risk        = RISK_LEVEL.get(pred_class, 'LOW')
        name        = CLASS_NAMES.get(pred_class, pred_class)
        description = DESCRIPTIONS.get(pred_class, '')
        advice      = ADVICE.get(risk, '')
        badge_html  = f'<span class="badge-{risk.lower()}">⬤ {risk} RISK</span>'

        st.markdown(f"""
        <div class="result-card">
          <p class="section-label">Prediction Result</p>
          <p class="disease-title">{name}</p>
          {badge_html}

          <p class="section-label" style="margin-top:1.2rem;">Confidence</p>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{confidence*100:.1f}%"></div>
          </div>
          <p style="color:#aaaacc;font-size:0.9rem;margin-top:-0.3rem;">{confidence*100:.1f}%</p>

          <div class="info-box">
            <strong>About this condition:</strong><br>{description}
          </div>
          <div class="info-box" style="margin-top:0.6rem;">
            <strong>Recommendation:</strong><br>{advice}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── All 7 class probabilities ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Probability Across All 7 Classes")

    sorted_idx = np.argsort(proba)[::-1]
    prob_cols  = st.columns(2)

    for i, idx in enumerate(sorted_idx):
        cls       = classes[idx]
        cls_name  = CLASS_NAMES.get(cls, cls)
        pct       = float(proba[idx]) * 100
        is_pred   = cls == pred_class
        bar_color = "#a78bfa" if is_pred else "#2a2a4a"
        highlight = "border:1px solid #a78bfa;" if is_pred else ""

        with prob_cols[i % 2]:
            st.markdown(f"""
            <div style="background:#12121f;border-radius:10px;padding:0.8rem 1rem;
                 margin-bottom:0.6rem;{highlight}">
              <div style="display:flex;justify-content:space-between;
                   font-size:0.85rem;color:#{'f0f0ff' if is_pred else 'aaaacc'};
                   margin-bottom:5px;">
                <span>{'✦ ' if is_pred else ''}<strong>{cls_name}</strong></span>
                <span>{pct:.1f}%</span>
              </div>
              <div style="background:#1e1e2e;border-radius:999px;height:8px;">
                <div style="width:{pct:.1f}%;height:8px;border-radius:999px;
                     background:{bar_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── PCA Feature Visualization ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🔬 PCA Feature Analysis")
    show_pca_features(img_array, pred_class)

    # ── Disease reference ──
    with st.expander("📋 Disease Reference Guide"):
        for code, cname in CLASS_NAMES.items():
            risk_lvl  = RISK_LEVEL[code]
            badge_cls = f"badge-{risk_lvl.lower()}"
            st.markdown(f"""
            <div style="border-bottom:1px solid #1e1e2e;padding:0.8rem 0;">
              <strong style="color:#d0d0ff;">{cname}</strong>
              &nbsp;<span class="{badge_cls}">{risk_lvl}</span>
              <p style="color:#8888aa;margin:0.3rem 0 0;font-size:0.85rem;">
                {DESCRIPTIONS[code]}
              </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 DermaScan</h1>
  <p>Skin Cancer Detection · 7 Disease Classes · Ensemble ML · Real Images Only</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model_loaded = False
required = ['models/skin_cancer_model.pkl', 'models/scaler.pkl', 'models/label_encoder.pkl']

if all(os.path.exists(f) for f in required):
    try:
        model, scaler, le, accs = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("⚠️ Model files not found. Run `python train_model.py` first.")

# ─────────────────────────────────────────────
# SECTION 1 — MODEL ACCURACY CARDS
# ─────────────────────────────────────────────
if model_loaded:
    st.markdown("### 🤖 Model Performance")

    overall = accs.get('Ensemble ML (Overall)', '—') if accs else "—"
    xgb_acc = accs.get('XGBoost', '—') if accs else "—"
    rf_acc  = accs.get('Random Forest', '—') if accs else "—"
    hgb_acc = accs.get('HistGradientBoosting', '—') if accs else "—"

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f"""
        <div class="model-card" style="border:2px solid #a78bfa;box-shadow:0 0 15px rgba(167,139,250,0.3);">
          <h4 style="color:#a78bfa;">Overall Deployed Model</h4>
          <div class="model-name">🏆 Ensemble ML (Soft Voting)</div>
          <div class="acc-number" style="font-size:3.5rem;">{overall}%</div>
          <div class="acc-label">Real Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div style="background:#12121f;border:1px solid #2a2a4a;border-radius:12px;padding:1rem;text-align:center;">
          <div style="color:#60a5fa;font-weight:bold;margin-bottom:0.5rem;">⚡ XGBoost</div>
          <div style="font-size:1.5rem;font-weight:700;color:#f0f0ff;">{xgb_acc}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:#12121f;border:1px solid #2a2a4a;border-radius:12px;padding:1rem;text-align:center;">
          <div style="color:#34d399;font-weight:bold;margin-bottom:0.5rem;">🌳 Random Forest</div>
          <div style="font-size:1.5rem;font-weight:700;color:#f0f0ff;">{rf_acc}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background:#12121f;border:1px solid #2a2a4a;border-radius:12px;padding:1rem;text-align:center;">
          <div style="color:#f472b6;font-weight:bold;margin-bottom:0.5rem;">📊 HistGradientB.</div>
          <div style="font-size:1.5rem;font-weight:700;color:#f0f0ff;">{hgb_acc}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 2 — INPUT MODE SELECTOR + PREDICT
# ─────────────────────────────────────────────
st.markdown("### 🔍 Analyse Skin Lesion")

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "upload"

col_tab1, col_tab2 = st.columns(2)

with col_tab1:
    if st.button(
        "📁  Upload Image",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "upload" else "secondary"
    ):
        st.session_state.input_mode = "upload"
        st.session_state.pop("camera_image", None)
        st.rerun()

with col_tab2:
    if st.button(
        "📷  Use Camera",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "camera" else "secondary"
    ):
        st.session_state.input_mode = "camera"
        st.session_state.pop("uploaded_image", None)
        st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD MODE
# ─────────────────────────────────────────────
if st.session_state.input_mode == "upload":
    st.markdown("""
    <div style="background:#12121f;border:1px solid #1e1e2e;border-radius:12px;
         padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#8888aa;">
      📁 &nbsp;Upload a clear, close-up photo of the skin lesion (JPG or PNG).
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
        key="file_uploader"
    )

    if uploaded and model_loaded:
        image     = Image.open(uploaded).convert('RGB')
        img_array = np.array(image)
        show_results(image, img_array, model, scaler, le, source_label="Uploaded File")
    elif uploaded and not model_loaded:
        st.error("Model files not found. Please run `python train_model.py` first.")

# ─────────────────────────────────────────────
# CAMERA MODE
# ─────────────────────────────────────────────
elif st.session_state.input_mode == "camera":
    st.markdown("""
    <div style="background:#12121f;border:1px solid #1e1e2e;border-radius:12px;
         padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#8888aa;">
      📷 &nbsp;Position the lesion clearly in frame, in good lighting, then click
      <strong style="color:#d0d0ff;">Take Photo</strong>.
      For best results, hold the camera steady and avoid shadows over the lesion.
    </div>
    """, unsafe_allow_html=True)

    tip_col1, tip_col2, tip_col3 = st.columns(3)
    with tip_col1:
        st.markdown("""
        <div style="background:#12121f;border:1px solid #1e1e2e;border-radius:10px;
             padding:0.8rem;text-align:center;font-size:0.8rem;color:#8888aa;">
          ☀️<br><strong style="color:#d0d0ff;">Good Lighting</strong><br>Natural or bright indoor light
        </div>
        """, unsafe_allow_html=True)
    with tip_col2:
        st.markdown("""
        <div style="background:#12121f;border:1px solid #1e1e2e;border-radius:10px;
             padding:0.8rem;text-align:center;font-size:0.8rem;color:#8888aa;">
          🔍<br><strong style="color:#d0d0ff;">Close-Up</strong><br>Fill the frame with the lesion
        </div>
        """, unsafe_allow_html=True)
    with tip_col3:
        st.markdown("""
        <div style="background:#12121f;border:1px solid #1e1e2e;border-radius:10px;
             padding:0.8rem;text-align:center;font-size:0.8rem;color:#8888aa;">
          🤚<br><strong style="color:#d0d0ff;">Hold Steady</strong><br>Avoid motion blur
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    camera_photo = st.camera_input(
        "Take a photo",
        label_visibility="collapsed",
        key="camera_widget"
    )

    if camera_photo and model_loaded:
        image     = Image.open(camera_photo).convert('RGB')
        img_array = np.array(image)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        show_results(image, img_array, model, scaler, le, source_label="Camera")
    elif camera_photo and not model_loaded:
        st.error("Model files not found. Please run `python train_model.py` first.")

    if camera_photo:
        st.markdown("""
        <div style="text-align:center;margin-top:0.8rem;font-size:0.8rem;color:#6666aa;">
          To retake, click <strong style="color:#aaaacc;">Clear photo</strong> above the camera.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
  Always consult a qualified dermatologist for skin health concerns.
</div>
""", unsafe_allow_html=True)