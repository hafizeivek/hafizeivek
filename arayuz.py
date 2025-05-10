import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime
from sklearn.preprocessing import LabelEncoder


def create_pdf(prediction_label, acc, input_data, classification_rep):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    textobject = c.beginText(40, 800)
    textobject.setFont("Helvetica", 12)
    textobject.textLine("Yolcu Memnuniyeti Tahmin Raporu")
    textobject.textLine("----------------------------------")
    textobject.textLine(f"Tahmin Edilen Memnuniyet Seviyesi: {prediction_label}")
    textobject.textLine(f"Model Doğruluğu (Accuracy): %{round(acc * 100, 2)}")
    textobject.textLine("")
    textobject.textLine("Sınıf Bazlı Doğruluk Raporu:")
    textobject.textLine(classification_rep)
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
    # input_data kategorik değişkenlerle birlikte güncellenmeli
    input_data['Gender'] = gender
    input_data['Customer Type'] = customer_type[0] if isinstance(customer_type, list) else customer_type
    input_data['Type of Travel'] = travel_type[0] if isinstance(travel_type, list) else travel_type
    input_data['Class'] = class_type[0] if isinstance(class_type, list) else class_type
    input_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Food and drink',
                      'Seat comfort', 'Inflight entertainment', 'Online boarding', 'On-board service',
                      'Leg room service', 'Baggage handling']

    # Kullanıcının giriş yapacağı özellikler
    input_features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Food and drink',
                      'Seat comfort', 'Inflight entertainment', 'Online boarding', 'On-board service',
                      'Leg room service', 'Baggage handling']
    for feature in input_features:
        # Önce bu sütunun veri tipi gerçekten sayısal mı kontrol et
        if pd.api.types.is_numeric_dtype(df[feature]):
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            mean_val = int(df[feature].mean())
            input_data[feature] = st.slider(feature, min_val, max_val, mean_val)
        else:
            st.warning(f"{feature} sayısal değil, slider eklenmedi.")

    # Kategorik değişkenleri encode etme (label encoding)
    le = LabelEncoder()
    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        df[col] = le.fit_transform(df[col])
    label_encoders = {}
    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        # Encode input_data kategorik değerleri
        for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    # Eğitim ve test verisi hazırlığı
    model_df = df.copy()
    model_df.dropna(inplace=True)

    # Özellik ve hedef değişkenleri belirle
    model_features = input_features + ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    X = model_df[model_features]
    # Y hedef değişkenini sayısal hale getir
    y = model_df["satisfaction"].map({"dissatisfied": 0, "satisfied": 2})
    # Gender sütununu 0 ve 1 olarak kodlayalım
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Eksik değerleri olan satırları temizle
    X = model_df[model_features]
    valid_idx = y.notna()  # Yani y'nin NaN olmayan indekslerini al
    X = X[valid_idx]
    y = y[valid_idx]

    # Eğitim ve test verisi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model oluştur ve eğit
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Encode input_data (kategorikleri sayısala çevirme)
    encoded_input = {}
    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        encoded_input[col] = label_encoders[col].transform([input_data[col]])[0]
    for col in input_features:
        encoded_input[col] = input_data[col]

    # Veriyi hazırlama
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    prediction = model.predict([list(encoded_input.values())])[0]
    acc = accuracy_score(y_test, model.predict(X_test))
    class_rep = classification_report(y_test, model.predict(X_test), output_dict=False)

    # Sonuç gösterme
    prediction_label = "Memnun" if prediction == 1 else "Memnun Değil"
    st.success(f"Tahmin: {prediction_label} (Doğruluk: %{round(acc * 100, 2)})")

    # PDF oluşturma
    if st.button("📄 Tahmin Raporunu PDF Olarak İndir"):
        pdf_buffer = create_pdf(prediction_label, acc, input_data, class_rep)
        st.download_button(label="PDF'yi İndir", data=pdf_buffer, file_name="tahmin_raporu.pdf")

    # Tahmin Sonucu Göster
    satisfaction_level = {0: "🔴 Düşük Memnuniyet", 2: "🔵 Yüksek Memnuniyet"}
    prediction_label = satisfaction_level.get(prediction, "Bilinmeyen")
    st.success(f"📌 Tahmin: {prediction_label}")

    # PDF Rapor Oluşturma
    classification_rep = classification_report(y_test, model.predict(X_test))
    pdf_file = create_pdf(prediction_label, acc, input_data, classification_rep)
    st.download_button(
        label="📥 PDF Raporu İndir",
        data=pdf_file,
        file_name="yolcu_memnuniyet_raporu.pdf",
        mime="application/pdf"
    )

    # Stratejik Yorumlar (Sadece Düşük ve Yüksek)
    if prediction == 2:
        st.subheader("🔵 Yüksek Memnuniyet - Stratejik Yorumlar")
        st.markdown("""
            - **Yüksek memnuniyet**, yolcuların uçuş deneyimlerinden oldukça memnun olduklarını gösteriyor. 
            - Bu seviyenin sürdürülmesi için hizmet kalitesi sürekli yüksek tutulmalı.
            - **Yemek**, **konfor**, ve **eğlence** faktörlerine odaklanarak sadakat artırılabilir.
        """)
    elif prediction == 0:
        st.subheader("🔴 Düşük Memnuniyet - Stratejik Yorumlar")
        st.markdown("""
            - **Düşük memnuniyet**, yolcuların deneyimlerinde önemli eksiklikler olduğunu gösteriyor.
            - **Hizmet kalitesinin** artırılması, yolcuların uçuş deneyimlerini iyileştirebilir.
            - Özellikle **online biniş**, **baggage handling** ve **yemek servisi** iyileştirilebilir.
        """)

except Exception as e:
    st.error(f"🛑 Veri Yüklenemedi veya Bir Hata Oluştu: {e}")
