"""
Stellar Object Classifier — Streamlit Web Application
======================================================
Loads the final trained Random Forest model exported from star_classification.ipynb
and provides a user-friendly interface for classifying SDSS observations.

Required files (produced by the notebook):
  - final_rf_model.pkl   : trained RandomForestClassifier
  - label_encoder.pkl    : LabelEncoder (int → class name)
  - model_meta.json      : feature list, class names, performance metrics

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌌 SDSS Stellar Classifier",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Space Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0b0c1a; color: #e0e4f0; }
    [data-testid="stSidebar"] { background: #111228; }

    .app-header {
        background: linear-gradient(135deg, #1a1a3e 0%, #0d1b4b 50%, #0a0e24 100%);
        border: 1px solid #2a3060;
        border-radius: 12px;
        padding: 26px 34px;
        margin-bottom: 24px;
        text-align: center;
    }
    .app-header h1 { color: #c8d8ff; font-size: 2.4rem; margin: 0; letter-spacing: 1px; }
    .app-header p  { color: #8899bb; margin: 6px 0 0 0; font-size: 1.05rem; }

    .metric-card {
        background: #141630;
        border: 1px solid #2d3260;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .label { color: #7a8bbf; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #a8c4ff; font-size: 1.9rem; font-weight: 700; margin-top: 4px; }

    .pred-GALAXY { background:#0d2640; border-left:5px solid #4FC3F7; border-radius:8px; padding:18px 22px; }
    .pred-STAR   { background:#2a2000; border-left:5px solid #FFD54F; border-radius:8px; padding:18px 22px; }
    .pred-QSO    { background:#2d0f18; border-left:5px solid #EF9A9A; border-radius:8px; padding:18px 22px; }
    .pred-title  { font-size:1.6rem; font-weight:700; margin-bottom:6px; }

    .section-title {
        color:#9bb5ff; font-size:1.15rem; font-weight:600;
        border-bottom:1px solid #2d3260; padding-bottom:6px; margin:18px 0 12px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg,#1c3d8c,#2a56c6);
        color:white; border:none; border-radius:8px;
        padding:10px 28px; font-size:1rem; font-weight:600; width:100%;
    }
    .stButton>button:hover { background: linear-gradient(90deg,#2a56c6,#3d70e0); }
    label { color:#aabbdd !important; }
    h2, h3 { color:#c8d8ff; }
</style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')
PALETTE    = {'GALAXY': '#4FC3F7', 'STAR': '#FFD54F', 'QSO': '#EF9A9A'}
ICONS      = {'GALAXY': '🌌', 'STAR': '⭐', 'QSO': '💫'}
DESCS      = {
    'GALAXY': 'A massive gravitationally-bound system of stars, gas, dust, and dark matter.',
    'STAR'  : 'A luminous plasma sphere held together by self-gravity — local to the Milky Way.',
    'QSO'   : 'A Quasi-Stellar Object: an extremely luminous active galactic nucleus at cosmological distance.'
}

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR INDEX HELPER (must match notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────
def add_colour_indices(df):
    """Derive the same colour indices engineered in the notebook."""
    d = df.copy()
    d['u_g'] = d['u'] - d['g']
    d['g_r'] = d['g'] - d['r']
    d['r_i'] = d['r'] - d['i']
    d['i_z'] = d['i'] - d['z']
    d['u_z'] = d['u'] - d['z']
    return d

# ──────────────────────────────────────────────────────────────────────────────
# LOAD EXPORTED MODEL ARTIFACTS
# Only loads — never trains. Raises a clear error if files are missing.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔭 Loading model...")
def load_model():
    """Load the pre-trained model artifacts produced by the notebook."""
    required = ['final_rf_model.pkl', 'label_encoder.pkl', 'model_meta.json']
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(
            f"❌ Missing model artifact(s): {missing}\n\n"
            "Please run **star_classification.ipynb** first to train and export the model."
        )
        st.stop()

    model = joblib.load('final_rf_model.pkl')
    le    = joblib.load('label_encoder.pkl')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, le, meta

model, le, meta = load_model()

FEATURE_COLS = meta['feature_cols']    # ['u','g','r','i','z','u_g','g_r','r_i','i_z','u_z','redshift']
CLASS_NAMES  = meta['class_names']     # ['GALAXY', 'QSO', 'STAR']

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: load dataset for EDA page (read-only, never used for training)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading dataset for EDA...")
def try_load_data():
    candidates = [
        'star_classification.csv',
        '/mnt/user-data/uploads/star_classification.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df_raw = try_load_data()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌌 SDSS Stellar Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔭 Predict Object", "📁 Batch Predict", "📊 EDA & Insights", "🤖 Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Loaded Model**")
    st.markdown(f"- Type: `RandomForest`")
    st.markdown(f"- Features: `{len(FEATURE_COLS)}`")
    st.markdown(f"- Macro F1: `{meta['test_macro_f1']:.4f}`")
    st.markdown(f"- Accuracy: `{meta['test_accuracy']:.4f}`")
    st.markdown("---")
    st.markdown("**Best Hyperparameters**")
    for k, v in meta['best_params'].items():
        st.markdown(f"- `{k}`: `{v}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="app-header">
        <h1>🌟 SDSS Stellar Object Classifier</h1>
        <p>Classifying Stars · Galaxies · Quasars using Machine Learning</p>
        <p style="color:#667aaa; font-size:0.9rem; margin-top:8px;">
            Sloan Digital Sky Survey · 100,000 Training Observations · Random Forest
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Model Accuracy",  f"{meta['test_accuracy']:.2%}"),
        ("Macro F1-Score",  f"{meta['test_macro_f1']:.4f}"),
        ("Feature Count",   f"{len(FEATURE_COLS)}"),
        ("Classes",         "3"),
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Class overview
    col_a, col_b, col_c = st.columns(3)
    for col, cls in zip([col_a, col_b, col_c], ['GALAXY', 'STAR', 'QSO']):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{ICONS[cls]} {cls}</div>
                <p style="color:#8899bb; font-size:0.82rem; margin-top:8px;">{DESCS[cls]}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How This App Works")
    st.markdown("""
The model was trained and exported from **`star_classification.ipynb`**. This app loads
the exported model file (`final_rf_model.pkl`) and uses it to classify new observations —
no retraining happens here.

| Page | What you can do |
|------|----------------|
| 🔭 **Predict Object** | Enter photometric measurements and get an instant classification |
| 📁 **Batch Predict** | Upload a CSV of multiple observations for bulk classification |
| 📊 **EDA & Insights** | Explore training data distributions and feature relationships |
| 🤖 **Model Info** | View performance metrics, feature importances, and model details |
    """)

    st.markdown("---")
    st.markdown("### About the Problem")
    st.markdown("""
The Sloan Digital Sky Survey records hundreds of thousands of astronomical observations
nightly. Correctly identifying each light source as a **star**, **galaxy**, or **quasar**
is foundational to cosmological research:

- **Stars** are local to our galaxy — their classification enables Milky Way structure studies.
- **Galaxies** map the large-scale structure of the universe, tracing dark matter and dark energy.
- **Quasars** are the most energetic objects known — their light probes the intergalactic medium
  across billions of light-years. Missing them means losing irreplaceable cosmological data.

Our classifier achieves **Macro-F1 of {:.4f}** — ensuring no class is systematically missed.
    """.format(meta['test_macro_f1']))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT OBJECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔭 Predict Object":
    st.markdown("## 🔭 Classify a New Observation")
    st.markdown(
        "Enter the photometric measurements for a single SDSS observation. "
        "The model will compute colour indices internally and return a predicted class."
    )

    with st.form("predict_form"):
        st.markdown('<div class="section-title">📡 SDSS Photometric Filter Magnitudes</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        u = c1.number_input("u  (Ultraviolet)", value=20.5, step=0.1, format="%.3f")
        g = c2.number_input("g  (Green)",        value=19.8, step=0.1, format="%.3f")
        r = c3.number_input("r  (Red)",          value=19.2, step=0.1, format="%.3f")
        i = c4.number_input("i  (Near-IR)",      value=18.9, step=0.1, format="%.3f")
        z = c5.number_input("z  (Infrared)",     value=18.6, step=0.1, format="%.3f")

        st.markdown('<div class="section-title">🌊 Spectral Measurement</div>',
                    unsafe_allow_html=True)
        redshift = st.number_input(
            "Redshift  (z)",
            value=0.001, min_value=-0.1, max_value=7.0, step=0.001, format="%.5f",
            help="Near 0 → Star (Milky Way).  0.1–1.5 → Galaxy.  >0.5 → likely QSO."
        )
        submitted = st.form_submit_button("🚀 Classify Object")

    if submitted:
        # Build input DataFrame and apply the same colour indices as the notebook
        raw_input = pd.DataFrame({
            'u': [u], 'g': [g], 'r': [r], 'i': [i], 'z': [z], 'redshift': [redshift]
        })
        obs = add_colour_indices(raw_input)[FEATURE_COLS]

        pred_enc   = model.predict(obs)[0]
        pred_cls   = le.inverse_transform([pred_enc])[0]
        proba      = model.predict_proba(obs)[0]
        confidence = max(proba)

        # Result box
        st.markdown(f"""
        <div class="pred-{pred_cls}">
            <div class="pred-title">{ICONS[pred_cls]}  Predicted: {pred_cls}</div>
            <div style="color:#aabbdd;">{DESCS[pred_cls]}</div>
            <div style="color:#c8d8ff; margin-top:8px;">
                Confidence: <strong>{confidence:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("### Prediction Confidence")
        prob_series = pd.Series(proba, index=le.classes_).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0d0e22')
        ax.set_facecolor('#0d0e22')
        bars = ax.barh(
            prob_series.index, prob_series.values,
            color=[PALETTE.get(c, '#888') for c in prob_series.index],
            edgecolor='#222244', height=0.5
        )
        for bar, val in zip(bars, prob_series.values):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=11, color='white')
        ax.set_xlim(0, 1.18)
        ax.set_xlabel('Probability', color='#aab')
        ax.set_title('Class Probabilities', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Show derived colour indices
        st.markdown("### Derived Colour Indices Used by Model")
        ci_df = pd.DataFrame({
            'Index': ['u−g', 'g−r', 'r−i', 'i−z', 'u−z'],
            'Value': [round(u-g, 4), round(g-r, 4), round(r-i, 4), round(i-z, 4), round(u-z, 4)],
            'Meaning': [
                'UV excess (hot objects / QSO)',
                'Stellar colour (standard)',
                'Red to Near-IR slope',
                'NIR to IR (dust / cool objects)',
                'Broadband spectral slope'
            ]
        })
        st.dataframe(ci_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Predict":
    st.markdown("## 📁 Batch Classification")
    st.markdown(
        "Upload a CSV with columns `u, g, r, i, z, redshift`. "
        "Colour indices are computed automatically. Results can be downloaded."
    )

    sample = pd.DataFrame({
        'u': [20.5, 22.3, 18.2],
        'g': [19.8, 21.0, 17.5],
        'r': [19.2, 20.5, 17.0],
        'i': [18.9, 20.1, 16.8],
        'z': [18.6, 19.8, 16.5],
        'redshift': [0.62, 1.35, 0.0001],
    })
    with st.expander("Example CSV format"):
        st.dataframe(sample, hide_index=True)

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            required = ['u', 'g', 'r', 'i', 'z', 'redshift']
            missing_cols = [c for c in required if c not in user_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                fe_df     = add_colour_indices(user_df)
                X_batch   = fe_df[FEATURE_COLS]
                preds_enc = model.predict(X_batch)
                preds_cls = le.inverse_transform(preds_enc)
                proba_all = model.predict_proba(X_batch)

                user_df['predicted_class'] = preds_cls
                user_df['confidence']      = np.max(proba_all, axis=1).round(4)
                for j, cls in enumerate(le.classes_):
                    user_df[f'prob_{cls}'] = proba_all[:, j].round(4)

                st.success(f"✅ Classified {len(user_df):,} observations")

                # Summary metrics
                summary = user_df['predicted_class'].value_counts()
                cols = st.columns(len(summary))
                for col, (cls, cnt) in zip(cols, summary.items()):
                    with col:
                        st.markdown(f"""<div class="metric-card">
                            <div class="value">{ICONS.get(cls,'')} {cls}</div>
                            <div class="label">{cnt} ({cnt/len(user_df):.1%})</div>
                        </div>""", unsafe_allow_html=True)

                # Pie chart of predictions
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#0d0e22')
                ax.pie(
                    summary.values,
                    labels=summary.index,
                    colors=[PALETTE.get(c, '#888') for c in summary.index],
                    autopct='%1.1f%%', startangle=140,
                    wedgeprops={'edgecolor': '#0d0e22'},
                    textprops={'color': 'white'}
                )
                ax.set_title('Predicted Class Distribution', color='white')
                st.pyplot(fig, use_container_width=False)
                plt.close()

                st.markdown("### Preview (first 50 rows)")
                st.dataframe(user_df.head(50), hide_index=True, use_container_width=True)

                st.download_button(
                    "⬇️ Download Full Results CSV",
                    data=user_df.to_csv(index=False).encode('utf-8'),
                    file_name="sdss_predictions.csv",
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown("## 📊 Exploratory Data Analysis")

    if df_raw is None:
        st.warning(
            "Dataset `star_classification.csv` not found in the app directory. "
            "EDA visualisations are based on the training data — place the CSV alongside "
            "this app file to enable this page."
        )
        st.stop()

    ID_COLS = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
               'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
    df = df_raw.drop(columns=ID_COLS, errors='ignore')

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target Distribution",
        "📡 Filter Distributions",
        "🌊 Redshift Analysis",
        "🔗 Correlations"
    ])

    with tab1:
        counts  = df_raw['class'].value_counts()
        colors  = [PALETTE.get(c, '#888') for c in counts.index]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.patch.set_facecolor('#0d0e22')

        ax1.bar(counts.index, counts.values, color=colors, edgecolor='#222244', width=0.55)
        ax1.set_facecolor('#0d0e22'); ax1.tick_params(colors='white')
        ax1.spines[:].set_color('#333'); ax1.set_ylabel('Count', color='#aab')
        ax1.set_title('Count per Class', color='white')
        for i, (c, p) in enumerate(zip(counts.values, counts.values/counts.sum()*100)):
            ax1.text(i, c + 500, f'{c:,}\n({p:.1f}%)', ha='center', color='white', fontsize=9)

        ax2.pie(counts.values, labels=counts.index, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'white'},
                wedgeprops={'edgecolor': '#0d0e22'})
        ax2.set_title('Proportions', color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.info("⚠️ Moderate class imbalance (~59% GALAXY). Macro-F1 was chosen as the "
                "evaluation metric to ensure equal treatment of all three classes.")

    with tab2:
        sel = st.selectbox("Select filter:", ['u', 'g', 'r', 'i', 'z'])
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0d0e22'); ax.set_facecolor('#0d0e22')
        for cls, color in PALETTE.items():
            sub = df_raw[df_raw['class'] == cls][sel]
            sub = sub[(sub > sub.quantile(0.01)) & (sub < sub.quantile(0.99))]
            ax.hist(sub, bins=60, alpha=0.55, label=cls, color=color, density=True)
        ax.set_title(f'{sel}-band magnitude distribution', color='white', fontsize=13)
        ax.set_xlabel('Magnitude', color='#aab'); ax.set_ylabel('Density', color='#aab')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333'); ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("**Insight:** Raw filter bands show heavy class overlap — absolute brightness "
                    "alone cannot reliably classify objects. Colour indices (filter differences) "
                    "capture spectral shape and provide better separation.")

    with tab3:
        # ── Panel 1: Full range with fixed clip (not per-class quantile)
        # Using per-class quantile clipping previously made STAR invisible:
        # STAR q99 ≈ 0.001, so it occupied a sliver of the x-axis (0–5 range).
        # Fix: use a single fixed upper clip (z < 5.5) for all classes,
        # plus a zoomed panel where STAR is clearly visible.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.patch.set_facecolor('#0d0e22')

        for cls, color in PALETTE.items():
            sub = df_raw[df_raw['class'] == cls]['redshift']
            sub = sub[sub < 5.5]   # Fixed clip — same for all classes
            ax1.hist(sub, bins=100, alpha=0.6, label=cls, color=color, density=True)
        ax1.set_facecolor('#0d0e22'); ax1.tick_params(colors='white'); ax1.spines[:].set_color('#333')
        ax1.set_title('Full Range (z < 5.5)  — STAR is the spike at z≈0', color='white'); ax1.legend()
        ax1.set_xlabel('Redshift', color='#aab'); ax1.set_ylabel('Density', color='#aab')

        # ── Panel 2: Zoomed into z ∈ [-0.01, 0.10] so STAR is visible
        for cls, color in PALETTE.items():
            sub = df_raw[df_raw['class'] == cls]['redshift']
            sub = sub[(sub >= -0.01) & (sub <= 0.10)]
            if len(sub) > 0:
                ax2.hist(sub, bins=60, alpha=0.65, label=f'{cls} (n={len(sub):,})',
                         color=color, density=True)
        ax2.set_facecolor('#0d0e22'); ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
        ax2.set_title('Zoomed: z ∈ [-0.01, 0.10]  (STAR clearly visible)', color='white')
        ax2.legend(); ax2.axvline(0, color='white', linestyle='--', alpha=0.4)
        ax2.set_xlabel('Redshift', color='#aab')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        c1, c2, c3 = st.columns(3)
        for col, cls in zip([c1, c2, c3], ['GALAXY', 'STAR', 'QSO']):
            col.metric(f"{ICONS[cls]} {cls} median z",
                       f"{df_raw[df_raw['class']==cls]['redshift'].median():.5f}")
        st.success("✅ Stars cluster at z≈0 (nearby). Galaxies span 0–2. "
                   "Quasars extend to z>5 (distant universe). "
                   "Redshift is the single most discriminative feature in the dataset.")

    with tab4:
        num_cols = ['u', 'g', 'r', 'i', 'z', 'redshift']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0d0e22'); ax.set_facecolor('#0d0e22')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    linewidths=0.5, vmin=-1, vmax=1, square=True,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 10, 'color': 'white'})
        ax.set_title('Feature Correlation', color='white'); ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.warning("⚠️ g, r, i, z are highly correlated (r > 0.93) — they encode "
                   "redundant information. This motivates colour indices (differences) "
                   "which are less correlated and more physically interpretable.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":
    st.markdown("## 🤖 Model Information & Performance")

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1, c2, c3, c4], [
        ("Test Accuracy",   f"{meta['test_accuracy']:.4f}"),
        ("Test Macro F1",   f"{meta['test_macro_f1']:.4f}"),
        ("Baseline LR F1",  f"{meta['baseline_lr_f1']:.4f}"),
        ("Improvement",     f"+{meta['test_macro_f1'] - meta['baseline_lr_f1']:.4f}"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Feature Importance", "📋 Model Details", "📖 Rationale"])

    with tab1:
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values()

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0d0e22'); ax.set_facecolor('#0d0e22')
        clrs = ['#EF9A9A' if v > 0.2 else '#4FC3F7' if v > 0.05 else '#90A4AE'
                for v in importances.values]
        bars = ax.barh(importances.index, importances.values, color=clrs, edgecolor='#222244')
        for bar, val in zip(bars, importances.values):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='white')
        ax.set_xlabel('Gini Importance', color='#aab')
        ax.set_title('Feature Importances — Tuned Random Forest', color='white', fontsize=13)
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.info(
            "**Redshift** dominates because it directly encodes distance — stars are "
            "near (z≈0), quasars are far (z>1). Colour indices (u_g, g_r, etc.) contribute "
            "meaningfully for separating galaxy/quasar spectral shapes. alpha and delta were "
            "excluded after EDA and baseline feature importance both showed near-zero signal."
        )

    with tab2:
        st.markdown("### Best Hyperparameters (from RandomizedSearchCV)")
        hp_df = pd.DataFrame([
            {'Hyperparameter': k, 'Value': str(v), 'What it controls': {
                'n_estimators': 'Number of trees (more = lower variance, higher compute)',
                'max_depth'   : 'Max tree depth (None = fully grown; controls overfitting)'
            }.get(k, '')}
            for k, v in meta['best_params'].items()
        ])
        st.dataframe(hp_df, hide_index=True, use_container_width=True)

        st.markdown("### Feature Set")
        feat_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Type': ['Raw filter'] * 5 + ['Colour index'] * 5 + ['Spectral'],
            'Description': [
                'Ultraviolet magnitude', 'Green magnitude', 'Red magnitude',
                'Near-Infrared magnitude', 'Infrared magnitude',
                'u − g (UV-Green)', 'g − r (Green-Red)', 'r − i (Red-NIR)',
                'i − z (NIR-IR)', 'u − z (Broadband slope)',
                'Redshift — distance proxy'
            ]
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("""
### Model Selection Rationale

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression (baseline) | ~0.82 | ~0.80 | Linear; under-fits spectral non-linearities |
| Random Forest (baseline)       | ~0.97 | ~0.97 | Non-linear; robust to outlier magnitudes |
| Gradient Boosting (baseline)   | ~0.97 | ~0.97 | Similar performance, slower inference |
| RF + Colour Indices            | ~0.98 | ~0.98 | Reduced multicollinearity; better QSO recall |
| **RF + Colour Indices (tuned)**| **~0.98** | **~0.98** | **Best overall; selected for deployment** |

**Why Random Forest was chosen over Gradient Boosting:**
Random Forest achieves equivalent or higher macro-F1 while offering significantly faster
inference. For a live prediction service where individual observations are classified
on demand, inference latency matters. Random Forest also parallelises natively across
CPU cores, making it more scalable.

**Why feature engineering helped:**
Colour indices are a well-established tool in observational astronomy — they are not
derived from the data alone but from domain knowledge, making them a reliable and
generalisable enhancement. They specifically improve quasar recall by encoding the
UV excess (high u-g) characteristic of quasar spectra.

**Business impact:**
A Macro-F1 of {:.4f} means all three object classes — including the minority quasar
class — are classified reliably. At SDSS scale (~500K new objects per survey season),
this enables a fully automated classification pipeline, eliminating manual bottlenecks
and ensuring every quasar detection reaches the astronomers who need it.
        """.format(meta['test_macro_f1']))

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5580; font-size:0.8rem;'>"
    "SDSS Stellar Classifier · Model trained in star_classification.ipynb · "
    "Random Forest | scikit-learn · Sloan Digital Sky Survey"
    "</p>",
    unsafe_allow_html=True
)