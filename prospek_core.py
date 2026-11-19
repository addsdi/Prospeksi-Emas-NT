"""Fungsi inti: membaca CSV, membersihkan data (sama persis dengan notebook), memprediksi, dan menghitung akurasi.
File ini tidak memakai Streamlit, jadi bisa diuji terpisah."""
import io
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

FAKTOR = {"PPM": 1, "PPB": 0.001, "PCT": 10000}


def baca_csv(data: bytes) -> pd.DataFrame:
    """Membaca CSV format standar (koma) maupun format asli NTGS (titik koma, desimal koma, latin-1)."""
    baris_pertama = data[:5000].decode("latin-1").splitlines()[0]
    if baris_pertama.count(";") > baris_pertama.count(","):
        opsi = dict(sep=";", decimal=",")
    else:
        opsi = dict(sep=",", decimal=".")
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False, **opsi)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("File tidak bisa dibaca. Pastikan formatnya CSV.")
    df.columns = df.columns.str.strip()
    return df


def ambil_unsur(df, unsur):
    kolom = [c for c in df.columns if re.fullmatch(unsur + r"1?_(PPM|PPB|PCT)", c)]
    hasil = pd.Series(np.nan, index=df.index)
    for c in kolom:
        nilai = pd.to_numeric(df[c], errors="coerce")
        nilai = nilai.where(nilai >= 0, nilai.abs() / 2)          # negatif = di bawah batas deteksi
        nilai = nilai * FAKTOR[c.split("_")[-1]]
        hasil = hasil.combine_first(nilai)
    return hasil


def ambil_au(df, ambang):
    hasil = pd.Series(np.nan, index=df.index)
    for kolom, ke_ppb in [("Au_PPB", 1), ("Au_PPM", 1000), ("Au1_PPB", 1), ("Au1_PPM", 1000)]:
        if kolom not in df.columns:
            continue
        nilai = pd.to_numeric(df[kolom], errors="coerce") * ke_ppb
        batas = nilai.abs()
        nilai = nilai.where(nilai >= 0, np.where(batas < ambang, batas / 2, np.nan))
        hasil = hasil.combine_first(nilai)
    return hasil


def prediksi(df: pd.DataFrame, paket: dict, ambang_au: float):
    """Mengubah data mentah -> fitur yang sama dengan notebook -> probabilitas prospek dari model."""
    for k in ("LONGITUDE", "LATITUDE"):
        if k not in df.columns:
            raise ValueError(f"Kolom {k} tidak ditemukan di file.")
    df = df.copy()
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df = df.dropna(subset=["LONGITUDE", "LATITUDE"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Tidak ada baris dengan koordinat yang valid.")
    if "SAMPLE_TYPE" not in df.columns:
        df["SAMPLE_TYPE"] = "UNKNOWN"
    if "SAMPLE_NO" not in df.columns:
        df["SAMPLE_NO"] = [f"BARIS-{i+1}" for i in range(len(df))]

    fitur = paket["fitur"]
    unsur = [c[4:] for c in fitur if c.startswith("log_")]
    ppm = pd.DataFrame({u: ambil_unsur(df, u) for u in unsur})
    ditemukan = [u for u in unsur if ppm[u].notna().any()]
    if len(ditemukan) < max(3, len(unsur) // 2):
        raise ValueError(
            f"Hanya {len(ditemukan)} dari {len(unsur)} unsur yang dipakai model ditemukan di file. "
            "Kolom unsur harus bernama seperti Ag_PPM, Cu_PPM, As_PPB, dst.")

    log = np.log10(ppm.clip(lower=0.001))
    log.columns = ["log_" + u for u in unsur]
    tipe = pd.get_dummies(df["SAMPLE_TYPE"].astype(str), prefix="TIPE").astype(int)
    X = pd.concat([log, tipe], axis=1)
    for kol in fitur:                                   # kolom yang tidak ada di data baru
        if kol not in X.columns:
            X[kol] = np.nan if kol.startswith("log_") else 0
    X = X[fitur]

    X_siap = pd.DataFrame(paket["scaler"].transform(paket["imputer"].transform(X)), columns=fitur)
    prob = paket["model"].predict_proba(X_siap)[:, 1]

    hasil = df[["SAMPLE_NO", "SAMPLE_TYPE", "LATITUDE", "LONGITUDE"]].copy()
    hasil["SAMPLE_NO"] = hasil["SAMPLE_NO"].astype(str)
    hasil["AU_PPB"] = ambil_au(df, ambang_au).round(2)
    hasil["PROB_PROSPEK"] = prob.round(4)
    hasil["DIPAKAI_LATIH"] = hasil["SAMPLE_NO"].isin(set(paket.get("sample_no_latih", [])))
    return hasil, ditemukan, unsur


def evaluasi(hasil: pd.DataFrame, ambang_au: float, cutoff: float, kecualikan_latih: bool = True):
    """Menghitung akurasi model pada sampel yang punya data Au."""
    d = hasil[hasil["AU_PPB"].notna()]
    n_latih = int(d["DIPAKAI_LATIH"].sum())
    if kecualikan_latih:
        d = d[~d["DIPAKAI_LATIH"]]
    out = {"n": len(d), "n_dikecualikan": n_latih if kecualikan_latih else 0}
    if len(d) == 0:
        out["pesan"] = "Tidak ada sampel dengan data Au yang bisa dievaluasi."
        return out
    y = (d["AU_PPB"] >= ambang_au).astype(int).values
    pred = (d["PROB_PROSPEK"] >= cutoff).astype(int).values
    out.update(n_prospek=int(y.sum()), n_tidak=int((1 - y).sum()))
    out["akurasi"] = accuracy_score(y, pred)
    out["precision"] = precision_score(y, pred, zero_division=0)
    out["recall"] = recall_score(y, pred, zero_division=0)
    out["f1"] = f1_score(y, pred, zero_division=0)
    out["cm"] = confusion_matrix(y, pred, labels=[0, 1])
    if y.min() != y.max():
        out["roc_auc"] = roc_auc_score(y, d["PROB_PROSPEK"].values)
        fpr, tpr, _ = roc_curve(y, d["PROB_PROSPEK"].values)
        out["roc"] = (fpr, tpr)
    else:
        out["pesan"] = ("Semua sampel berada di satu kelas (semuanya prospek atau semuanya tidak), "
                        "jadi ROC-AUC tidak bisa dihitung.")
    return out
