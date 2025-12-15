LAPORAN ASSURANCE OF LEARNING
MACHINE LEARNING

SonicPulse AI
Sistem Klasifikasi Genre Musik Berbasis Deep Learning & Audio Intelligence

Oleh:
Kelas: LJ01
Audya Florencia - 2702224591
Calvin Junaidy - 2702225865
Wandy Reynand Lim - 2702243602

Machine Learning
Semester Genap 2024/2025

BAB I
PENDAHULUAN

1.1 Latar Belakang
Industri musik digital telah mengalami evolusi radikal dalam dua dekade terakhir. Transisi dari media fisik ke layanan streaming digital telah menciptakan ledakan volume data audio yang belum pernah terjadi sebelumnya. Menurut laporan dari Spotify (2024), lebih dari 100.000 lagu baru diunggah ke platform mereka setiap harinya, menambah katalog yang sudah berisi lebih dari 100 juta lagu. Skala ini menciptakan tantangan masif dalam hal manajemen metadata, pengorganisasian konten, dan sistem rekomendasi.

Salah satu elemen paling fundamental dalam taksonomi musik adalah Genre. Bagi platform streaming (Spotify, Apple Music) dan label rekaman, klasifikasi genre yang akurat adalah kunci untuk algoritma rekomendasi (Discovery Weekly), penargetan iklan, dan analisis tren pasar. Namun, metode klasifikasi tradisional yang mengandalkan anotasi manual oleh manusia (human curation) tidak lagi relevan. Proses manual bersifat lambat, tidak dapat diskalakan (unscalable), dan rentan terhadap subjektivitas manusia yang menyebabkan inkonsistensi data.

Kebutuhan akan sistem otomatisasi yang cerdas menjadi sangat mendesak. Di sinilah peran Artificial Intelligence (AI) dan Deep Learning. Dengan memanfaatkan kemampuan komputer untuk memproses sinyal audio mentah, kita dapat mengekstrak pola matematis yang merepresentasikan karakteristik genre—seperti tempo, ritme, dan instrumentasi—tanpa intervensi manusia.

Proyek SonicPulse AI dikembangkan sebagai solusi atas permasalahan ini. Kami membangun sistem klasifikasi end-to-end yang memanfaatkan arsitektur Deep Learning modern. Proyek ini juga selaras dengan Sustainable Development Goals (SDG), khususnya SDG 9 (Industry, Innovation, and Infrastructure) dengan memodernisasi infrastruktur teknologi di industri kreatif, serta SDG 8 (Decent Work and Economic Growth) dengan meningkatkan produktivitas dan efisiensi kerja melalui otomatisasi cerdas.

1.2 Tujuan
Tujuan utama dari proyek ini adalah mengembangkan aplikasi web interaktif yang mampu mengklasifikasikan genre musik secara real-time dengan akurasi tinggi. Secara spesifik, sistem ini bertujuan untuk:
1.  **Implementasi Model Mutakhir**: Menggabungkan Transfer Learning dari model YAMNet (dikembangkan oleh Google) dengan arsitektur Bidirectional GRU dan mekanisme Attention untuk menangkap konteks musik yang kompleks.
2.  **Efisiensi Pemrosesan**: Menggantikan metode spectrogram-image-based yang berat dengan ekstraksi fitur embedding audio yang lebih efisien dan cepat.
3.  **Pengalaman Pengguna (UX)**: Menyediakan antarmuka (UI) berbasis Streamlit yang futuristik dan mudah digunakan, memungkinkan pengguna non-teknis untuk menganalisis file audio mereka dengan satu klik.

BAB II
PENDEKATAN & METODOLOGI

2.1 Audio Deep Learning
Berbeda dengan data tabular atau gambar statis, musik adalah data sekuensial (berurutan) yang memiliki dimensi waktu. Pendekatan konvensional sering menggunakan fitur statistik manual (MFCC, Zero Crossing Rate). Namun, dalam SonicPulse AI, kami menerapkan pendekatan Representation Learning di mana model belajar sendiri fitur-fitur terbaik dari data audio.

2.1.1 Transfer Learning: YAMNet
Alih-alih melatih model dari nol (yang membutuhkan jutaan data), kami menggunakan teknik Transfer Learning dengan YAMNet.
-   **Arsitektur**: YAMNet adalah model deep neural network berbasis arsitektur MobileNetV1 yang menggunakan depthwise-separable convolution.
-   **Pre-training**: Model ini telah dilatih sebelumnya (pre-trained) pada dataset Google AudioSet yang berisi lebih dari 2 juta klip audio YouTube dengan 521 kategori suara.
-   **Fungsi**: Dalam proyek ini, YAMNet berfungsi sebagai *Feature Extractor*. Model ini menerima gelombang audio mentah dan mengubahnya menjadi vektor embeddings berdimensi 1024. Vektor ini adalah representasi numerik padat yang mengandung informasi semantik tentang konten audio.

2.1.2 Temporal Modeling: Bi-GRU & Attention
Untuk memproses urutan embedding dari YAMNet, kami merancang arsitektur backend khusus:
1.  **Bidirectional GRU (Gated Recurrent Unit)**: Musik memiliki struktur temporal. Kami menggunakan GRU dua arah (Bidirectional) untuk memproses urutan musik dari awal ke akhir dan sebaliknya. Ini memungkinkan model memahami konteks lagu secara menyeluruh (misalnya, bagaimana intro terhubung ke chorus).
2.  **Custom Attention Layer**: Tidak semua detik dalam lagu memiliki bobot yang sama dalam menentukan genre. Bagian vokal atau drop mungkin lebih penting daripada silence. Mekanisme Attention (Atensi) yang kami bangun memungkinkan model untuk memberikan "bobot" lebih besar pada segmen-segmen krusial, meningkatkan fokus dan akurasi prediksi.

2.2 Dataset
Kami menggunakan **GTZAN Dataset**, standar global untuk riset Music Information Retrieval (MIR).
-   **Volume**: 1.000 trek audio.
-   **Format**: .wav, durasi 30 detik per trek.
-   **Kelas**: 10 Genre seimbang (100 lagu/genre): Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
-   **Kualitas**: Dataset ini dipilih karena variasi instrumen dan gaya yang representatif, meskipun memiliki tantangan berupa beberapa label yang ambigu (noisy labels).

2.3 Preprocessing Pipeline
Sebelum data masuk ke model, data melalui tahapan pemrosesan sinyal digital (DSP) yang ketat:
1.  **Resampling**: Semua input audio dikonversi laju sampelnya (sample rate) menjadi 16,000 Hz (16kHz). Ini adalah syarat mutlak input YAMNet.
2.  **Sliding Window Segmentation**: Lagu berdurasi 30 detik dipecah menjadi segmen-segmen lebih kecil (10 detik) dengan overlap (hop) 5 detik. Teknik ini melipatgandakan jumlah data latih dan memungkinkan model menangkap detail lokal yang lebih halus.
3.  **Embedding Generation**: Setiap segmen diteruskan ke YAMNet untuk menghasilkan matriks fitur. Matriks ini kemudian disimpan (cached) untuk mempercepat proses pelatihan (training).

BAB III
IMPLEMENTASI

3.1 Teknologi yang Digunakan
Sistem SonicPulse AI dibangun di atas ekosistem Python yang robust:
-   **Core AI**: TensorFlow & Keras (untuk arsitektur Bi-GRU + Attention), TensorFlow Hub (untuk memuat YAMNet).
-   **Audio Processing**: Librosa & Resampy (untuk manipulasi gelombang suara dan resampling).
-   **Backend Logic**: NumPy (manipulasi matriks).
-   **Frontend UI**: Streamlit (framework web app).
-   **Visualisasi**: Pandas (untuk grafik batang probabilitas).

3.2 Fitur User Interface (UI)
Aplikasi web dirancang dengan estetika "Sci-Fi/Cyberpunk" menggunakan kustomisasi CSS injeksi pada Streamlit. Desain ini dipilih untuk mencerminkan sifat futuristik dari teknologi AI.
Fitur utama meliputi:
1.  **Smart Audio Uploader**: Menerima format .wav, .mp3, dan .au. Menggunakan Glassmorphism style untuk tampilan modern.
2.  **Integrated Audio Player**: Memungkinkan pengguna memutar kembali file yang diunggah untuk verifikasi manual.
3.  **Analysis Engine**: Tombol "INITIATE ANALYSIS" memicu pipeline inferensi. Sistem melakukan scanning frekuensi, segmentasi, dan prediksi secara real-time.
4.  **Dynamic Result Card**: Menampilkan Genre Utama dengan animasi floating.
5.  **Audio DNA Chart**: Visualisasi grafik batang yang menunjukkan distribusi probabilitas untuk ke-10 genre, memberikan transparansi mengenai tingkat kepercayaan ("Confidence Level") model.

3.3 Cara Kerja Sistem (Inference Logic)
Ketika pengguna mengunggah lagu:
1.  File dibaca dan di-resample ke 16kHz.
2.  Audio dipecah menjadi segmen-segmen menggunakan logika Sliding Window.
3.  Setiap segmen diubah menjadi embedding oleh YAMNet.
4.  Jika lagu panjang, embeddings dikelompokkan (batched) menjadi urutan (sequences) sesuai input model (misal: 5 segmen per batch).
5.  Model Bi-GRU memprediksi genre untuk setiap batch.
6.  Hasil prediksi dari seluruh batch di-rata-rata (averaged) untuk mendapatkan kesimpulan final genre lagu tersebut.

BAB IV
HASIL & EVALUASI

4.1 Hasil Evaluasi Model
Model dilatih selama 50 epoch dengan Early Stopping untuk mencegah overfitting. Pada data uji (Test Set) yang tidak pernah dilihat sebelumnya, model mencapai:
-   **Akurasi Pengujian (Test Accuracy)**: **84.00%**

Angka ini sangat kompetitif dibandingkan model CNN standar yang biasanya berkisar di angka 60-70% pada dataset yang sama tanpa transfer learning.

4.2 Evaluasi Detail
Evaluasi model dilakukan melalui dua perspektif utama: analisis visual terhadap pola kesalahan (*Confusion Matrix*) dan analisis metrik kuantitatif (*Classification Report*).

### 4.2.1 Analisis Confusion Matrix
Visualisasi *Confusion Matrix* memberikan wawasan mendalam tentang bagaimana model "berpikir" saat terjadi kesalahan prediksi. Dari matriks yang dihasilkan, terlihat pola-pola menarik:
-   **Kekuatan Distingtif**: Genre seperti **Classical** dan **Metal** membentuk diagonal yang kuat dan bersih, menandakan model sangat yakin dengan ciri khas genre tersebut (seperti distorsi berat pada Metal atau ketiadaan drum pada Classical).
-   **Ambiguitas Akustik**: Terjadi kebingungan yang signifikan antara **Country**, **Rock**, dan **Blues**. Misalnya, 3 lagu Country salah diprediksi sebagai Rock, dan 2 sebagai Blues. Ini sangat manusiawi, mengingat ketiga genre ini berbagi akar instrumen yang sama (gitar, bass, drum) dan seringkali memiliki struktur lagu yang tumpang tindih.
-   **Dominasi Rock**: Kolom prediksi "Rock" memiliki banyak *false positives* (prediksi salah). Model cenderung "membuang" lagu yang tidak ia yakini (seperti Pop atau Country) ke dalam kategori Rock, menjadikannya semacam kelas default atau "catch-all".

### 4.2.2 Analisis Classification Report
Tabel berikut merangkum performa kuantitatif model untuk setiap genre (Precision, Recall, dan F1-Score) disertai analisis kualitatifnya:

| Genre | Precision | Recall | F1-Score | Analisis Kualitatif (Human-Readable) |
| :--- | :--- | :--- | :--- | :--- |
| **Blues** | 0.78 | 0.93 | 0.85 | Model mengenali genre ini dengan sangat baik dengan tingkat sensitivitas tinggi. Sedikit kebingungan terjadi dengan standar Rock, yang wajar karena akar sejarah musik Rock berasal dari Blues. |
| **Classical** | 0.94 | 1.00 | 0.97 | **Sempurna**. Model tidak mengalami kesulitan sama sekali. Absennya instrumen perkusi modern dan karakteristik harmonik orkestra menjadi pembeda yang sangat kontras bagi AI. |
| **Country** | 0.83 | 0.67 | 0.74 | **Paling Menantang**. Merupakan genre yang paling sulit bagi model. Banyak lagu Country yang salah diprediksi sebagai Rock atau Blues karena kemiripan instrumen (gitar akustik/elektrik) dan vokal. |
| **Disco** | 0.87 | 0.87 | 0.87 | Stabil dan konsisten. Pola ritme bass dan drum yang sangat khas (*four-on-the-floor*) memudahkan model membedakannya dari genre lain, dengan sedikit *overlap* ke Rock. |
| **Hiphop** | 0.76 | 0.87 | 0.81 | Terdeteksi dengan baik lewat pola *beat* dan vokal ritmis (*rapping*). Kesalahan klasifikasi sesekali terjadi pada lagu yang memiliki unsur melodi Pop yang kuat. |
| **Jazz** | 1.00 | 0.87 | 0.93 | **Sangat Akurat**. Presisi sempurna (1.00) menandakan bahwa jika model menebak Jazz, tebakannya hampir pasti benar. Kompleksitas harmoni Jazz menjadi fitur yang sangat unik bagi model. |
| **Metal** | 1.00 | 0.93 | 0.97 | **Performa Luar Biasa**. Intensitas suara, distorsi gitar yang berat, dan tempo cepat membuat Metal menjadi genre yang paling mudah dibedakan oleh model dibandingkan genre populer lainnya. |
| **Pop** | 0.91 | 0.67 | 0.77 | Presisi tinggi namun banyak yang terlewat (*Low Recall*). Pop adalah genre yang paling "cair" dan menyerap elemen genre lain, sehingga model sering bingung dan mengira lagu Pop sebagai Hiphop atau Rock. |
| **Reggae** | 0.86 | 0.80 | 0.83 | Cukup baik. Aksen ritme *off-beat* (*skank*) sangat membantu identifikasi, meskipun bassline yang berat terkadang membuatnya tertukar dengan Hiphop. |
| **Rock** | 0.60 | 0.80 | 0.69 | **Genre "Jebakan"**. Memiliki presisi terendah. Karena definisi Rock sangat luas, model seringkali menebak lagu Country, Blues, atau Pop yang memiliki gitar dominan sebagai Rock. |

**Kesimpulan Evaluasi**:
Secara umum, genre dengan karakteristik akustik dan instrumen yang unik (Classical, Metal, Jazz) memiliki akurasi di atas 90%. Tantangan terbesar terletak pada "segitiga" genre Rock-Pop-Country yang secara sonik sering berbagi instrumentasi dasar (gitar, bass, drum) dan struktur lagu yang serupa, menyebabkan sebagian misklasifikasi di antara ketiganya.

4.3 Keterbatasan
Sistem SonicPulse AI memiliki beberapa keterbatasan yang dapat dikembangkan di masa depan:
1.  **Ambiguitas Genre**: Musik modern seringkali bersifat hybrid (contoh: Pop-Rock atau Jazz-Fusion). Model saat ini dipaksa memilih satu genre dominan (*Single-label classification*), yang mungkin tidak menangkap nuansa penuh lagu.
2.  **Dataset Lama**: GTZAN dirilis tahun 2002. Model mungkin kurang sensitif terhadap sub-genre musik modern (seperti Trap, Dubstep, atau Lo-Fi) yang tidak terwakili dalam data latih.
3.  **Ketergantungan YAMNet**: Kualitas fitur sangat bergantung pada bagaimana YAMNet dilatih oleh Google. Bias dalam YAMNet terhadap kelas suara tertentu (misal: *speech* vs *music*) dapat mempengaruhi ekstraksi fitur.

BAB V
CONTRIBUTION STATEMENT

**Calvin Junaidy (2702225865) - Modeling Architecture & Optimization**
Bertanggung jawab atas inti kecerdasan buatan. Merancang arsitektur model Custom Attention dan Bidirectional GRU. Melakukan eksperimen *hyperparameter tuning* (learning rate, dropout, batch size) untuk mencapai konvergensi optimal. Memastikan integrasi antara Transfer Learning YAMNet dengan layer klasifikasi kustom berjalan mulus dalam TensorFlow.

**Wandy Reynand Lim (2702243602) - Data Engineering & Preprocessing**
Bertanggung jawab atas pipa data (*data pipeline*). Mengembangkan skrip untuk memuat dataset GTZAN, melakukan *resampling* audio menggunakan Librosa, dan memecah lagu menjadi segmen-segmen (*segmentation logic*). Mengelola ekstraksi fitur embedding YAMNet dan menyimpannya dalam format `.npy` untuk efisiensi pelatihan, serta melakukan analisis Confusion Matrix.

**Audya Florencia (2702224591) - Deployment & UI/UX Design**
Bertanggung jawab atas pengembangan aplikasi web (`app.py`). Mengimplementasikan logika inferensi model ke dalam Streamlit. Merancang antarmuka pengguna (*User Interface*) dengan kustomisasi CSS (animasi pulse, glassmorphism cards) untuk memberikan pengalaman pengguna yang menarik. Menyusun narasi laporan dan memastikan keselarasan antara aspek teknis dan presentasi produk.

BAB VI
SUMBER REFERENSI

1.  Gemmeke, J. F., et al. (2017). "Audio Set: An ontology and human-labeled dataset for audio events." *Proc. IEEE ICASSP 2017*.
2.  Tzanetakis, G., & Cook, P. (2002). "Musical genre classification of audio signals." *IEEE Transactions on Speech and Audio Processing*.
3.  Hershey, S., et al. (2017). "CNN Architectures for Large-Scale Audio Classification." *Google Research*.
4.  Librosa Development Team. (2024). *Librosa: Audio and Music Signal Analysis in Python*. [https://librosa.org/](https://librosa.org/)
5.  TensorFlow Hub. (2024). *YAMNet Model Documentation*. [https://tfhub.dev/google/yamnet/1](https://tfhub.dev/google/yamnet/1)
6.  Spotify Engineering. (2024). *Managing the world’s audio data*. [https://engineering.atspotify.com/](https://engineering.atspotify.com/)
