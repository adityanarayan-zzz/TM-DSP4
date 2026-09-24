
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open("feature_config.json") as f:
        config = json.load(f)
    cluster_profile = pd.read_csv("cluster_profile.csv", index_col="Cluster")
    return model, scaler, config, cluster_profile


model, scaler, config, cluster_profile = load_artifacts()

RAW_FEATURES = config["raw_features"]
LOG_FEATURES = config["log_transform_features"]
MODEL_FEATURE_ORDER = config["model_feature_order"]



def build_cluster_labels(profile: pd.DataFrame) -> dict:
    labels = {}
    cash_rank = profile["CASH_ADVANCE"].rank(ascending=False)
    purchase_rank = profile["PURCHASES"].rank(ascending=False)
    payment_disc_rank = profile["PRC_FULL_PAYMENT"].rank(ascending=False)

    for idx in profile.index:
        if cash_rank[idx] == 1 and purchase_rank[idx] >= len(profile) - 0:
            name = "Pengguna Cash Advance Berisiko"
            desc = ("Saldo & penarikan tunai (cash advance) tinggi, jarang berbelanja, "
                    "dan jarang melunasi tagihan penuh. Berpotensi berisiko kredit.")
        elif purchase_rank[idx] == 1:
            name = "Pembelanja Aktif Bernilai Tinggi"
            desc = ("Frekuensi & nominal pembelian tinggi (tunai maupun cicilan), limit kredit "
                    "dan pembayaran besar. Nasabah bernilai tinggi (high value).")
        elif payment_disc_rank[idx] == 1:
            name = "Nasabah Konservatif & Disiplin"
            desc = ("Saldo & penggunaan cash advance rendah, transaksi moderat, namun memiliki "
                    "rasio pelunasan penuh (full payment) tertinggi — nasabah paling disiplin.")
        else:
            name = f"Segmen {idx}"
            desc = "Karakteristik transaksi campuran."
        labels[idx] = {"name": name, "desc": desc}
    return labels


CLUSTER_LABELS = build_cluster_labels(cluster_profile)


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col in RAW_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib hilang: {col}")

    for col in ["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    df["AVG_PURCHASE_TRX"] = df["PURCHASES"] / df["PURCHASES_TRX"].replace(0, np.nan)
    df["AVG_PURCHASE_TRX"] = df["AVG_PURCHASE_TRX"].fillna(0)

    df["LIMIT_USAGE"] = df["BALANCE"] / df["CREDIT_LIMIT"].replace(0, np.nan)
    df["LIMIT_USAGE"] = df["LIMIT_USAGE"].fillna(0).clip(upper=3)

    for col in LOG_FEATURES:
        df[col] = np.log1p(df[col].clip(lower=0))

    df_model = df[MODEL_FEATURE_ORDER]
    return df_model


def predict_clusters(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_model = preprocess(df_raw)
    X_scaled = scaler.transform(df_model)
    clusters = model.predict(X_scaled)
    result = df_raw.copy()
    result["Cluster"] = clusters
    result["Segmen"] = [CLUSTER_LABELS[c]["name"] for c in clusters]
    return result



st.title("💳 Segmentasi Nasabah Kartu Kredit")

tab1= st.tabs(["🧍 Input Manual (1 Nasabah)", "📁 Upload CSV (Banyak Nasabah)"])


with tab1:
    st.subheader("Masukkan Data Perilaku Kartu Kredit Nasabah")
    col1, col2, col3 = st.columns(3)

    with col1:
        balance = st.number_input("Balance (saldo)", min_value=0.0, value=1500.0)
        balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.9)
        purchases = st.number_input("Purchases (total pembelian)", min_value=0.0, value=800.0)
        oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=400.0)
        installments_purchases = st.number_input("Installments Purchases", min_value=0.0, value=400.0)
        cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0)

    with col2:
        purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
        oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.3)
        purchases_installments_frequency = st.slider("Installments Purchases Frequency", 0.0, 1.0, 0.3)
        cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0)
        cash_advance_trx = st.number_input("Cash Advance Trx (jumlah transaksi)", min_value=0, value=0, step=1)
        purchases_trx = st.number_input("Purchases Trx (jumlah transaksi)", min_value=0, value=10, step=1)

    with col3:
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=4000.0)
        payments = st.number_input("Payments", min_value=0.0, value=1000.0)
        minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=500.0)
        prc_full_payment = st.slider("Percent Full Payment", 0.0, 1.0, 0.15)
        tenure = st.number_input("Tenure (bulan)", min_value=6, max_value=12, value=12, step=1)

    if st.button("🔍 Prediksi Segmen", type="primary"):
        input_data = pd.DataFrame([{
            "BALANCE": balance,
            "BALANCE_FREQUENCY": balance_frequency,
            "PURCHASES": purchases,
            "ONEOFF_PURCHASES": oneoff_purchases,
            "INSTALLMENTS_PURCHASES": installments_purchases,
            "CASH_ADVANCE": cash_advance,
            "PURCHASES_FREQUENCY": purchases_frequency,
            "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
            "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
            "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
            "CASH_ADVANCE_TRX": cash_advance_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": minimum_payments,
            "PRC_FULL_PAYMENT": prc_full_payment,
            "TENURE": tenure,
        }])

        try:
            result = predict_clusters(input_data)
            cluster_id = int(result.loc[0, "Cluster"])
            info = CLUSTER_LABELS[cluster_id]

            st.success(f"Nasabah termasuk **Segmen {cluster_id}: {info['name']}**")
            st.write(info["desc"])

            st.markdown("**Perbandingan dengan rata-rata tiap segmen:**")
            compare_df = cluster_profile.copy()
            compare_df.loc["Input Anda"] = [
                balance, purchases, oneoff_purchases, installments_purchases,
                cash_advance, credit_limit, payments, prc_full_payment, tenure,
                np.nan,
            ]
            st.dataframe(compare_df, use_container_width=True)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.divider()
