import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prospek_core import baca_csv, prediksi, evaluasi

st.set_page_config(page_title="Prospeksi Emas NT", page_icon="⛏️", layout="wide")

# ---------- Gaya tampilan (mengikuti dashboard HTML: emas + charcoal) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; }
.hdr { display:flex; align-items:center; gap:14px; background:linear-gradient(135deg,#090D16,#1F2937);
       border-radius:16px; padding:16px 22px; margin-bottom:14px; }
.hdr .au { width:44px; height:44px; border-radius:12px; background:linear-gradient(135deg,#FACC15,#B45309);
           display:flex; align-items:center; justify-content:center; font-weight:800; font-size:19px; color:#090D16; }
.hdr .judul { color:#FFFFFF; font-weight:800; font-size:20px; letter-spacing:-0.3px; }
.hdr .badge { margin-left:10px; font-size:11px; font-weight:700; color:#B45309; background:#FEF9E7;
              border:1px solid #D4AF37; border-radius:999px; padding:2px 10px; vertical-align:middle; }
.hdr .sub { color:#9CA3AF; font-size:12.5px; margin-top:2px; }
.brand { display:flex; align-items:center; gap:10px; background:#111827; border-radius:14px; padding:12px 14px; margin-bottom:6px; }
.brand .au { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#FACC15,#B45309);
             display:flex; align-items:center; justify-content:center; font-weight:800; color:#090D16; }
.brand .t1 { color:#FFFFFF; font-weight:800; font-size:14px; }
.brand .t2 { color:#9CA3AF; font-size:11px; }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:16px; padding:12px 16px;
                           box-shadow:0 1px 2px rgba(0,0,0,0.04); }
[data-testid="stMetric"]:hover { border-color:#111827; }
[data-testid="stMetricLabel"] p { font-size:11px; font-weight:700; letter-spacing:.4px; text-transform:uppercase; color:#6B7280; }
[data-testid="stMetricValue"] { font-family:'JetBrains Mono', monospace; font-weight:700; color:#111827; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ---------- Muat model (sekali saja) ----------
@st.cache_resource
def muat_model():
    with open("model_prospek_emas.pkl", "rb") as f:
        return pickle.load(f)


try:
    paket = muat_model()
except FileNotFoundError:
    st.error("File `model_prospek_emas.pkl` tidak ditemukan. Taruh di folder yang sama dengan app.py.")
    st.stop()

# Peringatan jika versi scikit-learn berbeda dari saat model dibuat
import sklearn
if paket.get("versi_sklearn") and paket["versi_sklearn"] != sklearn.__version__:
    st.warning(f"Model dibuat dengan scikit-learn {paket['versi_sklearn']}, aplikasi memakai {sklearn.__version__}. "
               "Samakan versinya di requirements.txt agar hasil prediksi benar.")

# ---------- State untuk upload & reset ----------
if "kunci_upload" not in st.session_state:
    st.session_state.kunci_upload = 0


def reset_data():
    st.session_state.kunci_upload += 1          # mengosongkan kotak upload
    for k in ("hasil", "nama_file"):
        st.session_state.pop(k, None)


# ---------- Sidebar ----------
st.sidebar.markdown('<div class="brand"><div class="au">Au</div><div><div class="t1">NTGS PROSPEKSI</div><div class="t2">Geoscience Geochemical Suite</div></div></div>', unsafe_allow_html=True)
file = st.sidebar.file_uploader("Unggah CSV geokimia", type=["csv"],
                                key=f"upload_{st.session_state.kunci_upload}",
                                help="Format standar (koma) atau format asli NTGS (titik koma) sama-sama bisa.")
st.sidebar.button("🔄 Reset data", on_click=reset_data, use_container_width=True)
st.sidebar.divider()

ambang_au = st.sidebar.number_input("Ambang Au prospek (ppb)", min_value=1.0, max_value=5000.0,
                                    value=float(paket.get("ambang_au_ppb", 200)), step=10.0,
                                    help="Sampel dengan Au di atas angka ini dianggap prospek (untuk menilai akurasi). "
                                         "Model dilatih dengan ambang bawaan.")
cutoff = st.sidebar.slider("Cut-off probabilitas", 0.05, 0.95, 0.50, 0.01,
                           help="Sampel dengan probabilitas di atas angka ini diprediksi prospek. "
                                "Turunkan untuk menemukan lebih banyak prospek (tapi lebih banyak salah tebak).")
kecualikan = st.sidebar.checkbox("Kecualikan sampel yang dipakai melatih model", value=True,
                                 help="Sebaiknya aktif. Sampel yang pernah dilihat model membuat akurasi tampak terlalu bagus.")

# ---------- Proses file ----------
@st.cache_data(show_spinner="Memproses data...")
def proses(data: bytes, ambang: float):
    df = baca_csv(data)
    return prediksi(df, paket, ambang)


if file is not None:
    try:
        hasil, ditemukan, unsur = proses(file.getvalue(), ambang_au)
        st.session_state["hasil"], st.session_state["nama_file"] = hasil, file.name
    except Exception as e:
        st.session_state.pop("hasil", None)
        st.error(f"Gagal memproses file: {e}")

# ---------- Halaman utama ----------
st.markdown(f"""
<div class="hdr"><div class="au">Au</div><div>
  <div><span class="judul">PROSPEKSI EMAS NORTHERN TERRITORY (NT)</span><span class="badge">NTGS Geochem ML</span></div>
  <div class="sub">Model Random Forest dari pola unsur geokimia pathfinder (prospek: Au &ge; {ambang_au:g} ppb). Alat bantu menyaring area, bukan pembuktian adanya emas.</div>
</div></div>
""", unsafe_allow_html=True)

if "hasil" not in st.session_state:
    st.info("Unggah file CSV di panel kiri untuk mulai. Klik **Reset data** untuk menghapus data yang sedang dipakai.")
    st.markdown("""
**Kolom yang dibutuhkan:** `LONGITUDE`, `LATITUDE`, kolom unsur seperti `Ag_PPM`, `Cu_PPM`, `As_PPB`, dst., dan `SAMPLE_TYPE`.
**Untuk menguji akurasi:** file perlu punya kolom Au (`Au_PPB` atau `Au_PPM`). Tanpa Au, aplikasi tetap memberi prediksi, tapi akurasi tidak bisa dihitung.
""")
    st.stop()

hasil = st.session_state["hasil"].copy()
hasil["KELAS_PREDIKSI"] = (hasil["PROB_PROSPEK"] >= cutoff).map({True: "Prospek", False: "Tidak prospek"})
st.success(f"Data aktif: **{st.session_state['nama_file']}** ({len(hasil):,} sampel)".replace(",", "."))

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Uji akurasi model", "Peta", "Kandidat teratas"])

# ----- Ringkasan -----
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah sampel", f"{len(hasil):,}".replace(",", "."))
    c2.metric("Punya data Au", f"{hasil['AU_PPB'].notna().sum():,}".replace(",", "."))
    c3.metric("Diprediksi prospek", f"{(hasil['KELAS_PREDIKSI']=='Prospek').sum():,}".replace(",", "."))
    c4.metric("Pernah dipakai melatih", f"{hasil['DIPAKAI_LATIH'].sum():,}".replace(",", "."))
    if hasil["DIPAKAI_LATIH"].any():
        st.warning("Sebagian sampel di file ini adalah data yang dipakai melatih model. "
                   "Untuk uji akurasi yang jujur, biarkan opsi 'Kecualikan sampel yang dipakai melatih' aktif.")
    st.subheader("Distribusi probabilitas prospek")
    st.plotly_chart(px.histogram(hasil, x="PROB_PROSPEK", nbins=40, height=320, color_discrete_sequence=["#D4AF37"],
                                 labels={"PROB_PROSPEK": "Probabilitas prospek"}), use_container_width=True)
    st.subheader("Jumlah sampel per tipe")
    st.dataframe(hasil.groupby("SAMPLE_TYPE").agg(jumlah=("SAMPLE_NO", "count"),
                                                  prospek_prediksi=("KELAS_PREDIKSI", lambda s: (s == "Prospek").sum()))
                 .sort_values("jumlah", ascending=False), use_container_width=True)

# ----- Uji akurasi -----
with tab2:
    ev = evaluasi(st.session_state["hasil"], ambang_au, cutoff, kecualikan_latih=kecualikan)
    if ev["n"] == 0:
        st.warning(ev.get("pesan", "Tidak ada data untuk dievaluasi.") +
                   (f" ({ev['n_dikecualikan']} sampel dikecualikan karena dipakai melatih.)" if ev["n_dikecualikan"] else ""))
    else:
        st.write(f"Dievaluasi pada **{ev['n']}** sampel ber-Au ({ev['n_prospek']} prospek, {ev['n_tidak']} tidak prospek)."
                 + (f" {ev['n_dikecualikan']} sampel dikecualikan karena dipakai melatih." if ev["n_dikecualikan"] else ""))
        if ev["n_prospek"] < 20:
            st.warning("Jumlah sampel prospek kurang dari 20, jadi angka di bawah belum stabil.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{ev['roc_auc']:.3f}" if "roc_auc" in ev else "-")
        m2.metric("Recall", f"{ev['recall']:.2f}")
        m3.metric("Precision", f"{ev['precision']:.2f}")
        m4.metric("F1", f"{ev['f1']:.2f}")
        m5.metric("Akurasi", f"{ev['akurasi']:.2f}")
        st.caption("Akurasi saja menyesatkan karena sampel prospek jarang. Perhatikan recall, precision, dan ROC-AUC.")
        if "pesan" in ev:
            st.info(ev["pesan"])
        col_a, col_b = st.columns(2)
        with col_a:
            cm = ev["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=["Pred: Tidak", "Pred: Prospek"], y=["Aktual: Tidak", "Aktual: Prospek"],
                                       text=cm, texttemplate="%{text}", colorscale=[[0, "#F8FAFC"], [1, "#111827"]], showscale=False))
            fig.update_layout(title="Confusion matrix", height=340, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if "roc" in ev:
                fpr, tpr = ev["roc"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#D4AF37", width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak", line=dict(dash="dash", color="#9CA3AF")))
                fig.update_layout(title="Kurva ROC", height=340, margin=dict(t=50, b=10),
                                  xaxis_title="False positive rate", yaxis_title="True positive rate")
                st.plotly_chart(fig, use_container_width=True)

# ----- Peta -----
with tab3:
    mode = st.radio("Warnai berdasarkan", ["Probabilitas prospek", "Kelas prediksi"], horizontal=True)
    data_peta = hasil if len(hasil) <= 20000 else hasil.sample(20000, random_state=1)
    if len(hasil) > 20000:
        st.caption("Menampilkan 20.000 titik acak agar peta tetap ringan.")
    kw = dict(lat="LATITUDE", lon="LONGITUDE", zoom=4, height=620,
              hover_data={"SAMPLE_NO": True, "SAMPLE_TYPE": True, "AU_PPB": True, "PROB_PROSPEK": ":.3f",
                          "LATITUDE": False, "LONGITUDE": False})
    if mode == "Probabilitas prospek":
        kw.update(color="PROB_PROSPEK", color_continuous_scale=[[0, "#E5E7EB"], [0.5, "#FACC15"], [1, "#B91C1C"]], range_color=(0, 1))
    else:
        kw.update(color="KELAS_PREDIKSI", color_discrete_map={"Prospek": "#B45309", "Tidak prospek": "#9CA3AF"})
    try:
        fig = px.scatter_map(data_peta, map_style="carto-positron", **kw)
    except AttributeError:
        fig = px.scatter_mapbox(data_peta, mapbox_style="carto-positron", **kw)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ----- Kandidat -----
with tab4:
    n = st.slider("Jumlah kandidat", 5, 100, 15)
    pilihan = st.checkbox("Hanya sampel yang belum dipakai melatih", value=True)
    d = hasil[~hasil["DIPAKAI_LATIH"]] if pilihan else hasil
    top = d.sort_values("PROB_PROSPEK", ascending=False).head(n)
    st.dataframe(top[["SAMPLE_NO", "SAMPLE_TYPE", "LATITUDE", "LONGITUDE", "AU_PPB", "PROB_PROSPEK"]],
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh semua hasil prediksi (CSV)",
                       hasil.drop(columns=[]).to_csv(index=False).encode("utf-8"),
                       file_name="hasil_prediksi_emas_NT.csv", mime="text/csv")
