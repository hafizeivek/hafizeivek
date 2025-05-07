import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime

def create_pdf(prediction_label, acc, input_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    textobject = c.beginText(40, 800)
    textobject.setFont("Helvetica", 12)
    textobject.textLine("Yolcu Memnuniyeti Tahmin Raporu")
    textobject.textLine("----------------------------------")
    textobject.textLine(f"Tahmin Edilen Memnuniyet Seviyesi: {prediction_label}")
    textobject.textLine(f"Model Doğruluğu (Accuracy): %{round(acc*100, 2)}")
    textobject.textLine("")
    textobject.textLine("Kullanıcı Girdi Değerleri:")
    for key, value in input_data.items():
        textobject.textLine(f"{key}: {value}")
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Sayfa Yapılandırması
st.set_page_config(page_title="Yolcu Memnuniyeti Analizi", layout="wide")
st.title("✈️ Havayolu Yolcu Memnuniyeti Analizi")
st.markdown(
    "Havayolu firmaları, müşteri memnuniyetini artırmak adına çeşitli analizler yapmaktadır. Özellikle uçuş deneyimlerine ilişkin toplanan verilerin analizi, hizmet kalitesinin ölçülmesi ve iyileştirilmesinde önemli rol oynar. Bu çalışmada, bir yolcunun uçuş deneyimi sonrasında memnun olup olmadığını belirleyen faktörler analiz edilmiştir.")


# Veriyi Yükleme
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/hafizeivek/hafizeivek/main/train.csv"
    df = pd.read_csv(url)

    # df = pd.read_csv("train.csv")  ← ❌ bunu kaldır
    df.columns = df.columns.str.strip()
    df.drop(columns=["Unnamed: 0", "id"], inplace=True)
    return df


try:
    df = load_data()
    st.success("✅ Veri başarıyla yüklendi.")

    with st.expander("📄 Veri Kümesine Genel Bakış"):
        st.dataframe(df.head())

    # --- Sidebar Filtreleme ---
    with st.sidebar:
        st.header("🔍 Filtreleme Seçenekleri")
        gender = st.selectbox("Cinsiyet Seçiniz", df["Gender"].unique())
        travel_type = st.multiselect("Yolculuk Türü", df["Type of Travel"].unique(),
                                     default=list(df["Type of Travel"].unique()))
        class_type = st.multiselect("Sınıf", df["Class"].unique(), default=list(df["Class"].unique()))
        customer_type = st.multiselect("Müşteri Türü", df["Customer Type"].unique(),
                                       default=list(df["Customer Type"].unique()))
        if st.button("🔄 Filtreleri Sıfırla"):
            st.experimental_rerun()

    # --- Filtre Uygulama ---
    filtered_df = df[
        (df["Gender"] == gender) &
        (df["Type of Travel"].isin(travel_type)) &
        (df["Class"].isin(class_type)) &
        (df["Customer Type"].isin(customer_type))
    ]

    # --- Temel Göstergeler ---
    st.subheader("📈 Temel Göstergeler")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Toplam Yolcu (filtreli)", len(filtered_df))
    col2.metric("Toplam Yolcu (genel)", len(df))
    col3.metric("Memnun Yolcu Oranı",
                f"%{round(filtered_df['satisfaction'].value_counts(normalize=True).get('satisfied', 0) * 100, 2)}")
    col4.metric("Ortalama Uçuş Mesafesi", round(filtered_df["Flight Distance"].mean(), 2))
    col5.metric("Ortalama Çevrimiçi Biniş", round(filtered_df["Online boarding"].mean(), 2))

    # --- Grafik: Memnuniyet Dağılımı ---
    st.subheader("🧭 Yolcu Memnuniyeti Dağılımı")
    fig1 = px.histogram(filtered_df, x="satisfaction", color="satisfaction", title="Memnuniyet Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    # --- Grafik 2: Hizmet Kalitesine Göre Memnuniyet Ortalamaları ---
    st.subheader("📊 Hizmet Kalitesine Göre Memnuniyet Ortalamaları")
    service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'On-board service']
    if filtered_df["satisfaction"].nunique() > 1:
        means = filtered_df.groupby("satisfaction")[service_cols].mean().T
        st.dataframe(means.style.highlight_max(axis=1))
    else:
        st.warning("Grafik oluşturmak için yeterli memnuniyet kategorisi yok. Lütfen filtreleri değiştirin.")

        # Cinsiyete Göre Memnuniyet
    st.subheader("👥 Cinsiyete Göre Memnuniyet Oranı")
    gender_satisfaction = filtered_df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')
    fig2 = px.pie(gender_satisfaction[gender_satisfaction['Gender'] == gender],
                  names='satisfaction', values='count',
                  title=f"{gender} Yolcularının Memnuniyet Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Tahmin Girişi ---
    st.subheader("🔮 Yolcu Memnuniyeti Tahmini")
    input_data = {}
    input_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Food and drink',
                      'Seat comfort', 'Inflight entertainment', 'Online boarding', 'On-board service',
                      'Leg room service', 'Baggage handling']

    for feature in input_features:
        input_data[feature] = st.slider(feature, int(df[feature].min()), int(df[feature].max()),
                                        int(df[feature].mean()))

    # --- Model Eğitimi ---
    model_df = df.copy()
    model_df.dropna(inplace=True)
    model_df = pd.get_dummies(model_df, columns=["Gender", "Customer Type", "Type of Travel", "Class"], drop_first=True)
    model_df["satisfaction"] = model_df["satisfaction"].map(
        {"dissatisfied": 0, "neutral or dissatisfied": 1, "satisfied": 2})

    # Ortak Özellikler (input_data ve model aynı olmalı)
    model_features = input_features  # Sadece numerik olanlar
    X = model_df[model_features]
    y = model_df["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    acc = accuracy_score(y_test, model.predict(X_test))

    # --- Tahmin Sonucu ---
    satisfaction_level = {0: "🔴 Düşük Memnuniyet", 1: "🟡 Orta Memnuniyet", 2: "🔵 Yüksek Memnuniyet"}
    prediction_label = satisfaction_level.get(prediction, "Bilinmeyen")
    st.success(f"📌 Tahmin: {prediction_label}")

    # --- PDF İndirme Butonu ---
    pdf_file = create_pdf(prediction_label, acc, input_data)
    st.download_button(
        label="📥 PDF Raporu İndir",
        data=pdf_file,
        file_name="yolcu_memnuniyet_raporu.pdf",
        mime="application/pdf"
    )

    # Dinamik Stratejik Yorumlar
    if prediction == 2:
        st.subheader("🔵 Yüksek Memnuniyet - Stratejik Yorumlar")
        st.markdown("""
               - **Yüksek memnuniyet**, yolcuların uçuş deneyimlerinden oldukça memnun olduklarını gösteriyor. 
               - Bu seviyenin sürdürülmesi için yolculara sağlanan hizmet kalitesinin sürekli yüksek tutulması gerekir.
               - **Yemek ve içecek servisleri**, **koltuk konforu** ve **eğlence seçenekleri** gibi faktörler yüksek memnuniyetin anahtarıdır. 
               - Uçuş deneyimini daha da iyileştirmek için bu alanlarda sürekli yenilikler yapılabilir.
               - Bu yüksek memnuniyet seviyesinin devamı için **sadık müşteri programları** ve **özelleştirilmiş hizmetler** önerilebilir.
               """)
    elif prediction == 1:
        st.subheader("🟡 Orta Memnuniyet - Stratejik Yorumlar")
        st.markdown("""
               - **Orta memnuniyet** seviyesi, yolcuların uçuş deneyimlerinden genel olarak memnun olduklarını, ancak bazı iyileştirmelere açık olduklarını gösteriyor. 
               - Yolcuların memnuniyet seviyelerini artırmak için, özellikle **yemek ve içecek servisleri** ve **eğlence hizmetleri** gibi alanlarda iyileştirmeler yapılabilir.
               - Koltuk rahatlığı ve uçuş öncesi hizmetler de gözden geçirilmeli. 
               - Orta memnuniyet seviyesinde olan yolculara yönelik özel teklifler ve iyileştirilmiş deneyimler sunmak, memnuniyeti artırabilir.
               """)
    elif prediction == 0:
        st.subheader("🔴 Düşük Memnuniyet - Stratejik Yorumlar")
        st.markdown("""
               - **Düşük memnuniyet** seviyesi, yolcuların uçuş deneyimlerinden olumsuz yönde etkilendiklerini gösteriyor. 
               - Bu seviyedeki yolcuların memnuniyetini artırmak için, özellikle **yemek ve içecek servisi**, **koltuk konforu**, ve **internet erişimi** gibi alanlarda ciddi iyileştirmelere gidilmesi gerekebilir.
               - Koltuk konforu ve hizmet kalitesi gibi unsurlar üzerinde yapılacak iyileştirmeler, yolcu memnuniyetini hızlıca artırabilir.
               - Ayrıca, düşük memnuniyet gösteren yolcular için **hizmet geri bildirimleri** toplanarak, onların deneyimlerini iyileştirmek adına somut adımlar atılabilir.
               """)

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Geçmiş tahminleri kaydet
    st.session_state.history.append({
        "prediction": prediction_label,
        "accuracy": round(acc * 100, 2),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Geçmiş tahminleri gösterme
    st.subheader("Geçmiş Tahminler")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
    else:
        st.info("Henüz bir tahmin yapılmadı.")


except Exception as e:
    st.error(f"❌ Hata oluştu: {str(e)}")
