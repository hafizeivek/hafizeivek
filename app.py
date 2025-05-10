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
st.title("✈️ Havayolu Yolcu Memnuniyeti Analizi")
st.markdown("Havayolu firmaları, müşteri memnuniyetini artırmak adına çeşitli analizler yapmaktadır. Özellikle uçuş deneyimlerine ilişkin toplanan verilerin analizi, hizmet kalitesinin ölçülmesi ve iyileştirilmesinde önemli rol oynar. Bu çalışmada, bir yolcunun uçuş deneyimi sonrasında memnun olup olmadığını belirleyen faktörler analiz edilmiştir.")

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

    # Varsayılan filtre değerlerini tanımlıyoruz
    default_gender = df["Gender"].unique()[0]  # İlk cinsiyet seçeneği
    default_travel_type = list(df["Type of Travel"].unique())  # Tüm yolculuk türleri varsayılan
    default_class_type = list(df["Class"].unique())  # Tüm sınıflar varsayılan
    default_customer_type = list(df["Customer Type"].unique())  # Tüm müşteri türleri varsayılan

    # --- Sidebar filtreleme ---
    with st.sidebar:
        st.header("🔍 Filtreleme Seçenekleri")

        # Filtreleme seçeneklerini kaydediyoruz, `session_state` üzerinden
        if "gender" not in st.session_state:
            st.session_state.gender = default_gender
        if "travel_type" not in st.session_state:
            st.session_state.travel_type = default_travel_type
        if "class_type" not in st.session_state:
            st.session_state.class_type = default_class_type
        if "customer_type" not in st.session_state:
            st.session_state.customer_type = default_customer_type

        # Filtreleme seçenekleri
        gender = st.selectbox("Cinsiyet Seçiniz", df["Gender"].unique(),
                              index=df["Gender"].unique().tolist().index(st.session_state.gender))
        travel_type = st.multiselect("Yolculuk Türü", df["Type of Travel"].unique(),
                                     default=st.session_state.travel_type)
        class_type = st.multiselect("Sınıf", df["Class"].unique(), default=st.session_state.class_type)
        customer_type = st.multiselect("Müşteri Türü", df["Customer Type"].unique(),
                                       default=st.session_state.customer_type)



   # Filtreleri uygula
    filtered_df = df[
        (df["Gender"] == st.session_state.gender) &
        (df["Type of Travel"].isin(st.session_state.travel_type)) &
        (df["Class"].isin(st.session_state.class_type)) &
        (df["Customer Type"].isin(st.session_state.customer_type))
        ]

    # --- Göstergeler ---
    st.subheader("📈 Temel Göstergeler")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Toplam Yolcu (filtreli)", len(filtered_df))
    col2.metric("Toplam Yolcu (genel)", len(df))
    col3.metric("Memnun Yolcu Oranı",
                f"%{round(filtered_df['satisfaction'].value_counts(normalize=True).get('satisfied', 0) * 100, 2)}")
    col4.metric("Ortalama Uçuş Mesafesi", round(filtered_df["Flight Distance"].mean(), 2))
    col5.metric("Ortalama Çevrimiçi Biniş", round(filtered_df["Online boarding"].mean(), 2))

    # --- Grafikler ---
    st.subheader("🧭 Memnuniyet Dağılımı")
    fig1 = px.histogram(filtered_df, x="satisfaction", color="satisfaction", title="Memnuniyet Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    # Cinsiyete Göre Memnuniyet
    st.subheader("👥 Cinsiyete Göre Memnuniyet Oranı")
    gender_satisfaction = filtered_df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')
    fig2 = px.pie(gender_satisfaction[gender_satisfaction['Gender'] == gender],
                  names='satisfaction', values='count',
                  title=f"{gender} Yolcularının Memnuniyet Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Hizmet Kalitesine Göre Ortalamalar")
    service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'On-board service']
    if filtered_df["satisfaction"].nunique() > 1:
        means = filtered_df.groupby("satisfaction")[service_cols].mean().T
        st.dataframe(means.style.highlight_max(axis=1))
    else:
        st.warning("Grafik için yeterli kategori yok.")

    # --- Tahmin Girişi ---
    st.subheader("🔮 Yolcu Memnuniyeti Tahmini")
    input_data = {
        'Gender': gender,
        'Customer Type': customer_type[0],
        'Type of Travel': travel_type[0],
        'Class': class_type[0]
    }

    numerical_cols = [
        'Age', 'Flight Distance', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Inflight wifi service',
        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    # Sayısal verileri al
    for feature in numerical_cols:
        min_val = int(df[feature].min())
        max_val = int(df[feature].max())
        mean_val = int(df[feature].mean())
        input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Kullanıcı girişini encode et
    encoded_input = {}
    for col in categorical_cols:
        encoded_input[col] = label_encoders[col].transform([input_data[col]])[0]
    for col in numerical_cols:
        encoded_input[col] = input_data[col]

    # Özellikleri doğru sırayla listele
    feature_order = X.columns.tolist()  # Eğitimde kullanılan özelliklerin sırası
    input_list = [encoded_input[col] for col in feature_order]

    # DataFrame'e çevir
    input_df = pd.DataFrame([input_list], columns=feature_order)

    # Model tahmini
    prediction = model.predict(input_df)[0]

    acc = accuracy_score(y_test, model.predict(X_test))
    class_rep = classification_report(y_test, model.predict(X_test))

    prediction_label = "🔵 Memnun" if prediction == 1 else "🔴Memnun Değil"

    st.markdown(f"### 🔍 Tahmin Sonucu: **{prediction_label}**")

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
        st.subheader(" Stratejik Yorumlar")
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
    st.session_state['history'].append({
        "Zaman": timestamp,
        "Tahmin": "Memnun" if prediction == 1 else "Memnun Değil",
        "Girdi": input_data
    })
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

except Exception as e:
        st.error(f"Veri yükleme veya modelleme sırasında bir hata oluştu: {str(e)}")


