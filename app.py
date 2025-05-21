import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime



# PDF oluşturma fonksiyonu
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
    for line in classification_rep.splitlines():
        textobject.textLine(line)
    textobject.textLine("")
    textobject.textLine("Kullanıcı Girdi Değerleri:")
    for key, value in input_data.items():
        textobject.textLine(f"{key}: {value}")
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# Sayfa ayarı
st.set_page_config(page_title="Yolcu Memnuniyeti Analizi", layout="wide")
st.title("✈ Havayolu Yolcu Memnuniyeti Analizi")
st.markdown(
    "Havayolu firmaları, müşteri memnuniyetini artırmak adına çeşitli analizler yapmaktadır. Özellikle uçuş deneyimlerine ilişkin toplanan verilerin analizi, hizmet kalitesinin ölçülmesi ve iyileştirilmesinde önemli rol oynar. Bu çalışmada, bir yolcunun uçuş deneyimi sonrasında memnun olup olmadığını belirleyen faktörler analiz edilmiştir.")


# Veri yükleme
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

        # Veri setinin dağılımını daha detaylı göster
        st.subheader("Memnuniyet Dağılımı")
        satisfaction_counts = df["satisfaction"].value_counts()
        satisfied_count = satisfaction_counts.get('satisfied', 0)
        dissatisfied_count = satisfaction_counts.get('neutral or dissatisfied', 0)
        total_count = satisfied_count + dissatisfied_count

        # Sayısal değerleri göster
        st.write(
            f"Memnun Olmayan Yolcu Sayısı: {dissatisfied_count} ({round(dissatisfied_count / total_count * 100, 2)}%)")
        st.write(f"Memnun Yolcu Sayısı: {satisfied_count} ({round(satisfied_count / total_count * 100, 2)}%)")
        st.write(f"Toplam Yolcu Sayısı: {total_count}")

        # Yatay bar chart ekle
        satisfaction_df = pd.DataFrame({
            'Memnuniyet': ['Memnun', 'Memnun Değil'],
            'Sayı': [satisfied_count, dissatisfied_count],
            'Yüzde': [round(satisfied_count / total_count * 100, 2), round(dissatisfied_count / total_count * 100, 2)]
        })

        fig_bar = px.bar(satisfaction_df, x='Sayı', y='Memnuniyet', text='Yüzde',
                         color='Memnuniyet', orientation='h',
                         labels={'Sayı': 'Yolcu Sayısı', 'Memnuniyet': 'Memnuniyet Durumu'},
                         title='Memnuniyet Dağılımı',
                         text_auto='.2f%')
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Varsayılan filtre değerlerini tanımlıyoruz
    default_gender = df["Gender"].unique()[0]  # İlk cinsiyet seçeneği
    default_travel_type = list(df["Type of Travel"].unique())  # Tüm yolculuk türleri varsayılan
    default_class_type = list(df["Class"].unique())  # Tüm sınıflar varsayılan
    default_customer_type = list(df["Customer Type"].unique())  # Tüm müşteri türleri varsayılan

    # --- Sidebar filtreleme ---
    with st.sidebar:
        st.header("🔍 Filtreleme Seçenekleri")

        # Session state başlatma - bu kısım en üstte olmalı
        if "filter_reset_counter" not in st.session_state:
            st.session_state.filter_reset_counter = 0
        if "gender" not in st.session_state:
            st.session_state.gender = default_gender
        if "travel_type_selected" not in st.session_state:
            st.session_state.travel_type_selected = {option: True for option in default_travel_type}
        if "class_type_selected" not in st.session_state:
            st.session_state.class_type_selected = {option: True for option in default_class_type}
        if "customer_type_selected" not in st.session_state:
            st.session_state.customer_type_selected = {option: True for option in default_customer_type}
        if "prediction_made" not in st.session_state:
            st.session_state.prediction_made = False

        # Filtreleri temizle butonu
        if st.button("🧹 Filtreleri Temizle", key=f"clear_filters_{st.session_state.filter_reset_counter}"):
            # Counter'ı artır - bu widget'ların key'lerini değiştirecek
            st.session_state.filter_reset_counter += 1

            # Session state değişkenlerini varsayılan değerlere sıfırla
            st.session_state.gender = default_gender
            st.session_state.travel_type_selected = {option: True for option in default_travel_type}
            st.session_state.class_type_selected = {option: True for option in default_class_type}
            st.session_state.customer_type_selected = {option: True for option in default_customer_type}
            st.session_state.prediction_made = False

            st.rerun()

        # Reset counter'ı kullanarak unique key'ler oluştur
        reset_key = st.session_state.filter_reset_counter

        # Cinsiyet seçimi
        gender = st.selectbox(
            "Cinsiyet Seçiniz",
            df["Gender"].unique(),
            index=df["Gender"].unique().tolist().index(st.session_state.gender),
            key=f"gender_select_{reset_key}"
        )

        # Yolculuk Türü için checkbox'lar
        st.subheader("Yolculuk Türü")
        travel_type = []
        for i, option in enumerate(default_travel_type):
            selected = st.checkbox(
                option,
                value=st.session_state.travel_type_selected.get(option, True),
                key=f"travel_{option}_{reset_key}_{i}"
            )
            if selected:
                travel_type.append(option)
            st.session_state.travel_type_selected[option] = selected

        # Sınıf için checkbox'lar
        st.subheader("Sınıf")
        class_type = []
        for i, option in enumerate(default_class_type):
            selected = st.checkbox(
                option,
                value=st.session_state.class_type_selected.get(option, True),
                key=f"class_{option}_{reset_key}_{i}"
            )
            if selected:
                class_type.append(option)
            st.session_state.class_type_selected[option] = selected

        # Müşteri Türü için checkbox'lar
        st.subheader("Müşteri Türü")
        customer_type = []
        for i, option in enumerate(default_customer_type):
            selected = st.checkbox(
                option,
                value=st.session_state.customer_type_selected.get(option, True),
                key=f"customer_{option}_{reset_key}_{i}"
            )
            if selected:
                customer_type.append(option)
            st.session_state.customer_type_selected[option] = selected

        # Session_state değerlerini güncelle
        st.session_state.gender = gender

        # Boş liste kontrolü ve varsayılan değer atama
        if not travel_type:
            travel_type = default_travel_type
        if not class_type:
            class_type = default_class_type
        if not customer_type:
            customer_type = default_customer_type

        st.session_state.travel_type = travel_type
        st.session_state.class_type = class_type
        st.session_state.customer_type = customer_type

    # Filtreleri uygula
    travel_type_filter = getattr(st.session_state, 'travel_type', default_travel_type)
    class_type_filter = getattr(st.session_state, 'class_type', default_class_type)
    customer_type_filter = getattr(st.session_state, 'customer_type', default_customer_type)

    filtered_df = df[
        (df["Gender"] == st.session_state.gender) &
        (df["Type of Travel"].isin(travel_type_filter)) &
        (df["Class"].isin(class_type_filter)) &
        (df["Customer Type"].isin(customer_type_filter))
        ]


    # --- Göstergeler ---
    st.subheader("📈 Temel Göstergeler")

    # Memnuniyet durumlarını hesapla
    satisfied_ratio = filtered_df['satisfaction'].value_counts(normalize=True).get('satisfied', 0)
    dissatisfied_ratio = filtered_df['satisfaction'].value_counts(normalize=True).get('neutral or dissatisfied', 0)

    # Metrik göstergeleri iyileştir
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Toplam Yolcu (filtreli)", f"{len(filtered_df):,}")
    col2.metric("Toplam Yolcu (genel)", f"{len(df):,}")

    # Her iki memnuniyet oranı da gösteriliyor
    col3.metric("Memnun Yolcu Oranı", f"%{round(satisfied_ratio * 100, 2)}")
    col4.metric("Memnun Olmayan Yolcu Oranı", f"%{round(dissatisfied_ratio * 100, 2)}")

    col5.metric("Ortalama Uçuş Mesafesi", f"{round(filtered_df['Flight Distance'].mean(), 2)} km")

    # --- Memnuniyet Dağılımı Grafiği (Geliştirilmiş) ---
    st.subheader("🧭 Memnuniyet Dağılımı")

    # Count değerlerini hesapla
    filtered_satisfaction_counts = filtered_df["satisfaction"].value_counts().reset_index()
    filtered_satisfaction_counts.columns = ['Memnuniyet Durumu', 'Sayı']

    # Memnuniyet durumlarını Türkçeleştir
    filtered_satisfaction_counts['Memnuniyet'] = filtered_satisfaction_counts['Memnuniyet Durumu'].map({
        'satisfied': 'Memnun',
        'neutral or dissatisfied': 'Memnun Değil'
    })

    # Yüzde değerlerini ekle
    total = filtered_satisfaction_counts['Sayı'].sum()
    filtered_satisfaction_counts['Yüzde'] = (filtered_satisfaction_counts['Sayı'] / total * 100).round(2)

    # Donut chart ile göster
    fig1 = px.pie(filtered_satisfaction_counts,
                  values='Sayı',
                  names='Memnuniyet',
                  title=f"Filtrelenen Verideki Memnuniyet Dağılımı (Toplam: {total} Yolcu)",
                  color='Memnuniyet',
                  color_discrete_map={'Memnun': '#2E86C1', 'Memnun Değil': '#E74C3C'},
                  hole=0.4)

    # Yüzdeleri daha belirgin göster
    fig1.update_traces(textposition='inside', textinfo='percent+label')

    # Açıklamaları ekle
    annotations = []
    for i, row in filtered_satisfaction_counts.iterrows():
        annotations.append(dict(
            text=f"{row['Sayı']:,} kişi<br>({row['Yüzde']}%)",
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        ))

    st.plotly_chart(fig1, use_container_width=True)

    # Sayısal detayları tablo halinde göster
    st.markdown("#### Sayısal Dağılım")
    st.dataframe(filtered_satisfaction_counts[['Memnuniyet', 'Sayı', 'Yüzde']], use_container_width=True)

    # Cinsiyete Göre Memnuniyet
    st.subheader("👥 Cinsiyete Göre Memnuniyet Oranı")
    gender_satisfaction = filtered_df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')

    if not gender_satisfaction.empty and gender in gender_satisfaction['Gender'].values:
        # Türkçe memnuniyet isimlerini ekle
        gender_satisfaction['Memnuniyet'] = gender_satisfaction['satisfaction'].map({
            'satisfied': 'Memnun',
            'neutral or dissatisfied': 'Memnun Değil'
        })

        # Seçilen cinsiyete göre filtrele
        gender_data = gender_satisfaction[gender_satisfaction['Gender'] == gender]

        # Pie chart güncellendi
        fig2 = px.pie(gender_data,
                      names='Memnuniyet',
                      values='count',
                      title=f"{gender} Yolcularının Memnuniyet Dağılımı",
                      color='Memnuniyet',
                      color_discrete_map={'Memnun': '#2E86C1', 'Memnun Değil': '#E74C3C'})

        # Pasta grafiği geliştir
        fig2.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Hizmet Kalitesine Göre Ortalamalar")
    service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'On-board service']

    if filtered_df["satisfaction"].nunique() > 1:
        # Memnuniyet durumuna göre gruplandırma
        means_by_satisfaction = filtered_df.groupby("satisfaction")[service_cols].mean().T

        # Türkçe sütun isimleri ile yeni DataFrame oluştur
        means_df = means_by_satisfaction.copy()
        means_df.columns = ['Memnun Değil',
                            'Memnun'] if 'neutral or dissatisfied' in means_by_satisfaction.columns else means_df.columns

        # Farkı hesapla
        if means_df.shape[1] == 2:
            means_df['Fark'] = means_df['Memnun'] - means_df['Memnun Değil']

        # Tablo formatında göster
        st.dataframe(means_df.style.highlight_max(axis=1).format("{:.2f}"), use_container_width=True)

        # Hizmet puanlarını görselleştir
        st.markdown("#### Memnuniyete Göre Hizmet Puanları Karşılaştırması")
        service_data = means_df.reset_index()
        service_data.columns.name = None
        service_data = service_data.rename(columns={'index': 'Hizmet'})

        # Bar chart için hazırla
        melted_service = pd.melt(service_data, id_vars=['Hizmet'],
                                 value_vars=['Memnun', 'Memnun Değil'],
                                 var_name='Memnuniyet', value_name='Ortalama Puan')

        # Bar chart oluştur
        fig_service = px.bar(melted_service,
                             x='Hizmet',
                             y='Ortalama Puan',
                             color='Memnuniyet',
                             barmode='group',
                             title='Memnuniyet Durumuna Göre Hizmet Puanları',
                             color_discrete_map={'Memnun': '#2E86C1', 'Memnun Değil': '#E74C3C'})

        st.plotly_chart(fig_service, use_container_width=True)
    else:
        st.warning("Grafik için yeterli kategori yok.")

    # --- Tahmin Girişi ---
    st.subheader("🔮 Yolcu Memnuniyeti Tahmini")

    # Kategorik değişkenler
    input_data = {
        'Gender': gender,
        'Customer Type': customer_type[0] if customer_type else default_customer_type[0],
        'Type of Travel': travel_type[0] if travel_type else default_travel_type[0],
        'Class': class_type[0] if class_type else default_class_type[0]
    }

    # Önemli sayısal değişkenler - Sayıyı azalttık
    key_numerical_cols = [
        'Age', 'Flight Distance', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Inflight wifi service',
        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',

    ]

    # Filtrelere göre sayısal değerlerin ortalamalarını hesapla
    filtered_means = {}
    for feature in key_numerical_cols:
        filtered_means[feature] = int(filtered_df[feature].mean())

    # Önemli sayısal verileri al
    col1, col2, col3 = st.columns(3)

    with col1:
        # İlk sütundaki slider'lar
        for i, feature in enumerate(key_numerical_cols[:6]):
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            mean_val = filtered_means.get(feature, int(df[feature].mean()))
            input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

    with col2:
        # İkinci sütundaki slider'lar
        for i, feature in enumerate(key_numerical_cols[6:12]):
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            mean_val = filtered_means.get(feature, int(df[feature].mean()))
            input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

    with col3:
        # İkinci sütundaki slider'lar
        for i, feature in enumerate(key_numerical_cols[12:]):
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            mean_val = filtered_means.get(feature, int(df[feature].mean()))
            input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

    # Eksik olan sayısal değişkenleri tamamla (model için gerekli)
    all_numerical_cols = [
        'Age', 'Flight Distance', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Inflight wifi service',
        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    # Kullanıcı tarafından seçilmeyen değişkenler için ortalama değerleri kullan
    for feature in all_numerical_cols:
        if feature not in input_data:
            input_data[feature] = filtered_df[feature].mean()

    # --- Label Encoding ---
    label_encoders = {}
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Hedef değişkeni encode et
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

    # Model verisi
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ÖNEMLİ: Model parametrelerini iyileştirme
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   min_samples_leaf=5,  # Overfitting'i azaltmak için
                                   max_depth=10,  # Daha dengeli tahminler için derinliği sınırla
                                   class_weight='balanced')  # Sınıf dengesizliğini gidermek için
    model.fit(X_train, y_train)

    # Model doğruluğunu test et ve bilgi ver
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Özellik önemlerini hesapla ve göster
    with st.expander("📊 Özellik Önem Sıralaması"):
        feature_importance = pd.DataFrame({
            'Özellik': X.columns,
            'Önem': model.feature_importances_
        }).sort_values('Önem', ascending=False)

        # Özellik önemleri için bar grafiği ekle
        fig_importance = px.bar(feature_importance.head(10),
                                x='Önem',
                                y='Özellik',
                                orientation='h',
                                title='En Önemli 10 Özellik')
        st.plotly_chart(fig_importance, use_container_width=True)

        # Tam tabloyu göster
        st.dataframe(feature_importance)

        # En önemli özellikleri vurgula
        st.info(f"En önemli 5 özellik: {', '.join(feature_importance['Özellik'].head(5).tolist())}")

    # Tahmin butonu
    if st.button("🧠 Tahmini Hesapla"):
        # Kullanıcı girişini encode et
        encoded_input = {}
        for col in categorical_cols:
            if input_data[col]:  # Boş olmadığından emin ol
                encoded_input[col] = label_encoders[col].transform([input_data[col]])[0]
            else:
                encoded_input[col] = 0  # Varsayılan değer

        for col in all_numerical_cols:
            encoded_input[col] = input_data[col]

        # Özellikleri doğru sırayla listele
        feature_order = X.columns.tolist()  # Eğitimde kullanılan özelliklerin sırası
        input_list = [encoded_input[col] for col in feature_order]

        # DataFrame'e çevir
        input_df = pd.DataFrame([input_list], columns=feature_order)

        # Model tahmini - Olasılıkları da hesapla
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Tahmini doğrudan değil, olasılık eşiğine göre belirle
        # Özellikle düşük değerlerdeki parametreler için tahmini 'memnun değil' olma yönünde güçlendir
        # Eğer memnuniyet olasılığı %60'dan azsa, "memnun değil" olarak tahmin et
        if prediction_proba[1] < 0.6:  # Eşik değeri: olasılığın %60'ından azı "memnun değil" olarak belirle
            prediction = 0  # Memnun değil

        # Tahmin yapıldı olarak işaretle
        st.session_state.prediction_made = True
        st.session_state.prediction = prediction
        st.session_state.input_data = input_data.copy()
        st.session_state.prediction_proba = prediction_proba.tolist()  # Olasılıkları da kaydet

        # Sayfayı yenile
        st.rerun()

    # Tahmin sonucunu göster
    if st.session_state.prediction_made:
        prediction = st.session_state.prediction
        input_data = st.session_state.input_data
        prediction_proba = st.session_state.prediction_proba if 'prediction_proba' in st.session_state else [0.5, 0.5]

        acc = accuracy_score(y_test, model.predict(X_test))
        class_rep = classification_report(y_test, model.predict(X_test))

        prediction_label = "🔵 Memnun" if prediction == 1 else "🔴 Memnun Değil"

        # Tahmin sonucunu daha dikkat çekici hale getir
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {'#D4EFDF' if prediction == 1 else '#FADBD8'};">
            <h3 style="text-align: center; color: {'#145A32' if prediction == 1 else '#7B241C'};">
                🔍 Tahmin Sonucu: {prediction_label}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Olasılıkları görsel olarak göster - Geliştirilmiş
        st.markdown("#### Tahmin Olasılıkları")

        # Olasılık değerlerini DataFrame'e çevir
        probs_df = pd.DataFrame({
            'Memnuniyet': ['Memnun Değil', 'Memnun'],
            'Olasılık': [prediction_proba[0], prediction_proba[1]],
            'Yüzde': [f"%{round(prediction_proba[0] * 100, 2)}", f"%{round(prediction_proba[1] * 100, 2)}"]
        })


     # Dinamik Stratejik Yorumlar
        if prediction == 1:
            st.subheader("Stratejik Yorumlar")
            st.markdown(""" 
            Yapılan veri analizi sonucunda, yolcuların büyük bir çoğunluğunun havayolu firmasıyla olan genel deneyiminden yüksek düzeyde memnuniyet duyduğu tespit edilmiştir. Bu memnuniyet, markaya olan güvenin ve hizmet kalitesinin doğrudan bir yansımasıdır. Ancak havacılık sektöründeki yoğun rekabet ve değişen müşteri beklentileri göz önüne alındığında, yalnızca mevcut başarıyı korumak değil, hizmet kalitesini sürekli geliştirerek beklentilerin ötesine geçmek de stratejik bir gerekliliktir.

            ### 🌟 Kritik Başarı Alanları ve Gelişim Önerileri:

            - **🍽️ Yemek ve İçecek Hizmetleri:**  
              Mevcut menülerin farklı kültürel ve kişisel tercihlere hitap edecek şekilde çeşitlendirilmesi, vegan, vejetaryen ve alerjen duyarlılığına uygun opsiyonların artırılması önerilmektedir.  
              Ayrıca menü sunumunun görsel ve işitsel deneyimle desteklenmesi (örneğin mobil menüler, sesli açıklamalar) yolcu deneyimini zenginleştirebilir.

            - **💺 Koltuk Konforu ve Fiziksel Alan:**  
              Ergonomik tasarıma sahip, kişisel alanı artıran koltuklar ve modüler oturma çözümleri özellikle uzun mesafeli uçuşlarda müşteri sadakatini etkileyen başlıca faktörlerdendir.  
              Ayarlanabilir başlık, ayak dayama desteği, kişisel USB/şarj noktaları gibi özellikler konfor seviyesini artırabilir.

            - **🎬 Uçuş İçi Eğlence Sistemleri:**  
              Kişiselleştirilebilir arayüzler, farklı yaş gruplarına özel içerikler, çoklu dil desteği ve online içerik güncellemeleri sayesinde sistem daha çekici hale getirilebilir.  
              Ayrıca yolcunun tercih geçmişine göre önerilen içerikler, müşteri bağlılığını artıracak yapay zeka destekli çözümlerle desteklenebilir.

            - **🚀 Yenilikçi Hizmet Yaklaşımları:**  
              - **Mobil Uygulama İyileştirmeleri:** Uçuş içi anketler, anlık geri bildirim alma sistemleri, kişisel hizmet tercihleri yönetimi gibi özellikler eklenmelidir.  
              - **Yapay Zeka Entegrasyonu:** Kişisel öneriler, check-in kolaylığı ve dijital asistan destekli rehberlik gibi uygulamalar yolcu deneyimini bireyselleştirecektir.  
              - **Sanal Gerçeklik ve Artırılmış Gerçeklik Uygulamaları:** Özellikle business ve premium segmentte fark yaratan deneyimler sunabilir.

            ### 📈 Stratejik Sonuç ve Yol Haritası:

            Yüksek memnuniyet seviyesi bir avantaj olmakla birlikte sürdürülebilirliği ancak sistematik bir iyileştirme döngüsü ile sağlanabilir.  
            Bu kapsamda:

            - Müşteri geri bildirimlerinin sürekli ve bütüncül olarak analiz edilmesi,  
            - Veriye dayalı karar alma kültürünün yerleştirilmesi,  
            - Hizmet kalitesinin teknolojik gelişmelerle eşgüdümlü olarak güncellenmesi gereklidir.

            **Sonuç olarak**, yüksek müşteri memnuniyeti sadece bir anlık başarı göstergesi değil, stratejik bir sürdürülebilirlik unsuru olarak ele alınmalıdır.
            """)

        else:
            st.subheader("Stratejik Yorumlar")
            st.markdown(""" 
            Yapılan analizler, bazı yolcuların havayolu firmasının sunduğu hizmetlerden yeterince memnun kalmadığını göstermektedir.  
            Bu durum, müşteri deneyiminde iyileştirme gerektiren kritik noktaların varlığını açıkça ortaya koymaktadır.  

            ### 📉 Belirlenen Başlıca Sorun Alanları

            🔁 **Biniş İşlemleri (Check-in ve Boarding):**  
            ⏳ Uzun bekleme süreleri, ❌ yetersiz yönlendirmeler ve 💻 dijital altyapı eksiklikleri yolcu stresini artırmakta ve memnuniyeti azaltmaktadır.  
            ✅ Daha hızlı ve kullanıcı dostu dijital çözümler acil ihtiyaçlar arasındadır.

            👨‍✈️ **Hizmet Kalitesi:**  
            ✈️ Uçuş öncesi, sırası ve sonrasında sunulan hizmetlerin tutarsız olması, müşteri beklentilerinin karşılanmamasına yol açmaktadır.  
            Özellikle kabin ekibinin 🤝 tutumu, yardımseverliği ve profesyonelliği yolcu deneyiminde belirleyici rol oynamaktadır.

            📞 **İletişim ve Geri Bildirim Mekanizmaları:**  
            🙁 Yolcuların yaşadıkları sorunları kolayca iletebileceği ve çözüm alabileceği etkili sistemlerin eksikliği göze çarpmaktadır.

            ---

            ### ✅ Stratejik İyileştirme Önerileri

            🎓 **Personel Eğitimi:**  
            Kabin ve yer hizmetleri personeline yönelik düzenli eğitimler verilmelidir.  
            👂 Empati, 🧘 stres yönetimi ve 📢 etkili iletişim becerileri odağında geliştirici programlar uygulanmalıdır.

            ⚙️ **Operasyonel Geliştirmeler:**  
            📲 Mobil uygulamalar, 🖥️ self-servis kiosklardan biniş, anlık bilgilendirme sistemleri gibi yeniliklerle süreçler dijitalleştirilmelidir.

            📊 **Geri Bildirim Analitiği:**  
            💡 Yolculardan alınan şikayet, öneri ve değerlendirmeler 🤖 yapay zeka destekli analizlerle işlenmeli ve karar süreçlerine entegre edilmelidir.

            ---

            🚨 **Sonuç:**  
            Düşük memnuniyet düzeyleri, hem müşteri sadakati hem de marka algısı üzerinde olumsuz etkiler yaratmaktadır.  
            Bu nedenle sorun alanları hızla tespit edilmeli, 🔧 iyileştirici adımlar atılmalı ve tüm gelişmeler 🎯 düzenli olarak izlenmelidir.
                    """)

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            "Zaman": timestamp,
            "Tahmin": "Memnun" if prediction == 1 else "Memnun Değil",
            "Girdi": input_data
        }

        # Eğer bu tahmin daha önce eklenmemişse ekle
        if not any(entry["Zaman"] == timestamp for entry in st.session_state['history']):
            st.session_state['history'].append(history_entry)

        with st.expander("🕓 Tahmin Geçmişi"):
            if st.session_state['history']:
                history_df = pd.DataFrame(st.session_state['history'])
                st.dataframe(history_df)
            else:
                st.info("Henüz bir tahmin yapılmadı.")

        # PDF çıktısı oluştur
        if st.button("📄 PDF Raporu Oluştur ve İndir"):
            pdf_buffer = create_pdf(prediction_label, acc, input_data, class_rep)
            st.download_button(label="📥 Raporu İndir", data=pdf_buffer,
                               file_name=f"yolcu_memnuniyet_raporu_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf")
    else:
        # Tahmin yapılmadığında göster
        st.info("⏳ Henüz tahmin yapılmadı. Yukarıdaki parametreleri ayarlayıp 'Tahmini Hesapla' butonuna tıklayınız.")

except Exception as e:
    st.error(f"Veri yükleme veya modelleme sırasında bir hata oluştu: {str(e)}")
