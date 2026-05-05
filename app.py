import re
import html
import unicodedata

import emoji
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unidecode import unidecode

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Spam Judi Online",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Konstanta ────────────────────────────────────────────────────────────────
MODEL_PATH   = "retrained_models/indobert_v2"
MAX_LENGTH   = 128
THRESHOLD    = 0.5
ID2LABEL     = {0: "Normal", 1: "Judol"}
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAMUS_KATA_JUDOL = {
    "jepe": "jackpot", "jp": "jackpot", "jpot": "jackpot",
    "cuaan": "cuan", "wd": "withdraw", "wede": "withdraw",
    "gacir": "gacor", "gakor": "gacor", "g4c0r": "gacor",
    "g4cor": "gacor", "gac0r": "gacor",
    "sl0t": "slot", "s1ot": "slot",
}

# ─── Preprocessing (identik dengan notebook) ──────────────────────────────────
def normalize_unicode(text: str) -> str:
    text = html.unescape(str(text))
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.category(c).startswith("M")
    )
    text = re.sub(r"[\[【〖〔(]([A-Za-z0-9])[\]】〗〕)]", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = unidecode(text)
    return text

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang(text: str, kamus: dict) -> str:
    words = text.split()
    return " ".join(kamus.get(w, w) for w in words)

def normalize_emoji(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text, delimiters=(" :", ": "))
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text: str, kamus: dict) -> str:
    text = normalize_unicode(text)
    text = clean_text(text)
    text = normalize_slang(text, kamus)
    text = normalize_emoji(text)
    return text

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model IndoBERT v2...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner="Memuat kamus normalisasi...")
def load_kamus() -> dict:
    try:
        df_kamus  = pd.read_csv("kamus_normalisasi_kata.csv")
        kamus_baku = dict(zip(df_kamus["slang"], df_kamus["formal"]))
    except FileNotFoundError:
        kamus_baku = {}
    return {**kamus_baku, **KAMUS_KATA_JUDOL}

# ─── Inferensi ────────────────────────────────────────────────────────────────
def predict(texts: list[str], tokenizer, model, kamus: dict) -> list[dict]:
    results = []
    for raw_text in texts:
        processed = preprocess(raw_text, kamus)
        encoding  = tokenizer(
            processed,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        label_id   = int(np.argmax(probs))
        confidence = float(probs[label_id])

        results.append({
            "text":            raw_text,
            "processed_text":  processed,
            "label":           ID2LABEL[label_id],
            "label_id":        label_id,
            "confidence":      confidence,
            "prob_normal":     float(probs[0]),
            "prob_judol":      float(probs[1]),
        })
    return results

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 960px; }

.badge-judol  { display:inline-block; background:#FEE2E2; color:#991B1B;
                font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500;
                padding:2px 10px; border-radius:4px; letter-spacing:.04em; }
.badge-normal { display:inline-block; background:#DCFCE7; color:#166534;
                font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500;
                padding:2px 10px; border-radius:4px; letter-spacing:.04em; }

.result-card { background:#F9FAFB; border:1px solid #E5E7EB;
               border-radius:8px; padding:1rem 1.25rem; margin-bottom:.75rem; }
.result-card.judol  { border-left:4px solid #EF4444; }
.result-card.normal { border-left:4px solid #22C55E; }
.result-text  { font-size:13px; color:#111827; margin-bottom:.5rem; line-height:1.5; }
.result-meta  { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#6B7280; }

.conf-bar-bg  { background:#E5E7EB; border-radius:4px; height:6px; margin:.4rem 0; }
.conf-bar-fill{ height:6px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Deteksi Spam Judi Online")
st.markdown(
    "Klasifikasi komentar media sosial menggunakan **IndoBERT v2** — "
    "model fine-tuned untuk deteksi konten promosi judi online (*judol*) berbahasa Indonesia."
)
st.divider()

# ─── Load resources ───────────────────────────────────────────────────────────
try:
    tokenizer, model = load_model()
    kamus            = load_kamus()
    model_loaded     = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: `{e}`\n\nPastikan folder `{MODEL_PATH}` tersedia.")
    model_loaded = False

# ─── Tab ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Komentar Tunggal", "Batch Komentar"])

# ── Tab 1: Komentar Tunggal ────────────────────────────────────────────────────
with tab_single:
    st.markdown("#### Masukkan Komentar")
    input_text = st.text_area(
        label="Teks komentar",
        placeholder="Contoh: Main di sl0t ini dijamin g4c0r terus, WD tiap hari 100%",
        height=120,
        label_visibility="collapsed",
    )

    col_btn, col_conf = st.columns([2, 3])
    with col_btn:
        run_single = st.button("Klasifikasi", type="primary", width="stretch", key="btn_single")
    with col_conf:
        threshold = st.slider(
            "Confidence threshold", 0.5, 1.0, THRESHOLD, 0.01,
            help="Prediksi di bawah threshold ditampilkan sebagai 'Tidak pasti'"
        )

    if run_single and input_text.strip():
        if not model_loaded:
            st.warning("Model belum tersedia.")
        else:
            with st.spinner("Memproses..."):
                res = predict([input_text.strip()], tokenizer, model, kamus)[0]

            confident = res["confidence"] >= threshold
            label     = res["label"] if confident else "Tidak pasti"

            st.markdown("---")
            col_l, col_r = st.columns([1, 2])

            with col_l:
                st.markdown("**Hasil Prediksi**")
                if not confident:
                    st.warning(f"⚠️ Tidak pasti ({res['confidence']:.1%})")
                elif label == "Judol":
                    st.error(f"🚨 **JUDOL** ({res['confidence']:.1%})")
                else:
                    st.success(f"✅ **NORMAL** ({res['confidence']:.1%})")

            with col_r:
                st.markdown("**Distribusi Probabilitas**")
                st.progress(res["prob_normal"], text=f"Normal  — {res['prob_normal']:.1%}")
                st.progress(res["prob_judol"],  text=f"Judol   — {res['prob_judol']:.1%}")

            with st.expander("Lihat teks setelah preprocessing"):
                st.code(res["processed_text"], language=None)

    elif run_single:
        st.warning("Masukkan teks komentar terlebih dahulu.")

# ── Tab 2: Batch Komentar ──────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Input Batch")
    input_mode = st.radio(
        "Sumber input", ["Teks manual (satu komentar per baris)", "Upload file CSV"],
        horizontal=True, label_visibility="collapsed"
    )

    texts_to_classify = []

    if input_mode == "Teks manual (satu komentar per baris)":
        batch_text = st.text_area(
            "Komentar (satu per baris)",
            placeholder="Video ini keren banget!\nMain di sl0t ini dijamin gacor\nMakasih bang sharing ilmunya",
            height=180,
            label_visibility="collapsed",
        )
        if batch_text.strip():
            texts_to_classify = [t.strip() for t in batch_text.strip().splitlines() if t.strip()]
            st.caption(f"{len(texts_to_classify)} komentar terdeteksi")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            col_opts  = df_upload.columns.tolist()
            col_sel   = st.selectbox("Pilih kolom komentar", col_opts)
            texts_to_classify = df_upload[col_sel].dropna().astype(str).tolist()
            st.caption(f"{len(texts_to_classify)} komentar ditemukan di kolom `{col_sel}`")
            st.dataframe(df_upload[[col_sel]].head(), width="stretch")

    run_batch = st.button(
        f"Klasifikasi {len(texts_to_classify)} Komentar" if texts_to_classify else "Klasifikasi",
        type="primary",
        width="stretch",
        disabled=not (model_loaded and texts_to_classify),
        key="btn_batch",
    )

    if run_batch and texts_to_classify:
        progress_bar = st.progress(0, "Memproses komentar...")
        results = []

        # Proses per-batch kecil agar progress bar responsif
        chunk_size = 16
        for i in range(0, len(texts_to_classify), chunk_size):
            chunk   = texts_to_classify[i : i + chunk_size]
            results.extend(predict(chunk, tokenizer, model, kamus))
            pct = min((i + chunk_size) / len(texts_to_classify), 1.0)
            progress_bar.progress(pct, f"Memproses... {min(i+chunk_size, len(texts_to_classify))}/{len(texts_to_classify)}")

        progress_bar.empty()
        df_result = pd.DataFrame(results)

        # ── Ringkasan ──
        n_judol  = (df_result["label"] == "Judol").sum()
        n_normal = (df_result["label"] == "Normal").sum()
        avg_conf = df_result["confidence"].mean()

        st.markdown("---")
        st.markdown("**Ringkasan Hasil**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",  len(df_result))
        m2.metric("Normal", n_normal)
        m3.metric("Judol",  n_judol)
        m4.metric("Avg. Confidence", f"{avg_conf:.1%}")

        # ── Hasil per komentar ──
        st.markdown("**Detail Hasil**")
        for _, row in df_result.iterrows():
            cls = "judol" if row["label"] == "Judol" else "normal"
            badge = (
                f'<span class="badge-judol">JUDOL</span>'
                if row["label"] == "Judol"
                else f'<span class="badge-normal">NORMAL</span>'
            )
            bar_color = "#EF4444" if row["label"] == "Judol" else "#22C55E"
            conf_pct  = int(row["confidence"] * 100)

            st.markdown(f"""
            <div class="result-card {cls}">
                <div class="result-text">{row['text']}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    {badge}
                    <span class="result-meta">confidence: {row['confidence']:.1%}</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_pct}%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Download ──
        st.markdown("---")
        csv_out = df_result[["text", "label", "confidence", "prob_normal", "prob_judol"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download hasil (.csv)",
            data=csv_out,
            file_name="hasil_klasifikasi.csv",
            mime="text/csv",
            width="stretch",
        )