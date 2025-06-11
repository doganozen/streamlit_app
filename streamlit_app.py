import streamlit as st
import pandas as pd
import plotly.express as px

# Başlık
st.title("Churn Model Aylık Performans Takibi")

# Veri yükleme
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Tarihi sıralı göstermek için datetime'e çevir
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    # Ay seçimi
    selected_month = st.selectbox("Ay Seçiniz", df["month"].dt.strftime('%Y-%m').tolist())
    selected_row = df[df["month"].dt.strftime('%Y-%m') == selected_month].iloc[0]

    st.subheader(f"📊 {selected_month} Ayı Performans Göstergeleri")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{selected_row['precision']:.2f}")
    col2.metric("Recall", f"{selected_row['recall']:.2f}")
    col3.metric("F1 Score", f"{selected_row['f1']:.2f}")
    col4.metric("ROC AUC", f"{selected_row['roc_auc']:.2f}")

    st.subheader("🎯 Tahmin vs Gerçek Churn")
    churn_fig = px.bar(
        df,
        x=df["month"].dt.strftime('%Y-%m'),
        y=["predicted_churn", "actual_churn"],
        labels={"value": "Kullanıcı Sayısı", "month": "Ay"},
        barmode="group",
        title="Ay Bazlı Tahmin Edilen vs Gerçek Churn"
    )
    st.plotly_chart(churn_fig)

    st.subheader("📈 Zaman İçinde Model Performansı")
    metric_fig = px.line(
        df,
        x=df["month"].dt.strftime('%Y-%m'),
        y=["precision", "recall", "f1", "roc_auc"],
        markers=True,
        title="Model Metrikleri Zaman İçinde"
    )
    st.plotly_chart(metric_fig)

    st.subheader("🧮 Confusion Matrix Değerleri")
    st.write(f"**True Positives (TP)**: {selected_row['tp']}")
    st.write(f"**False Positives (FP)**: {selected_row['fp']}")
    st.write(f"**False Negatives (FN)**: {selected_row['fn']}")
    st.write(f"**True Negatives (TN)**: {selected_row['tn']}")
else:
    st.info("Lütfen bir CSV dosyası yükleyin. Örnek: churn_model_monthly_metrics_example.csv")
