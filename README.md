
# Membangun Model Klasifikasi dan Clustering untuk Menganalisis Kesejahteraan Pekerja Indonesia

> "Data is the new oil. It’s valuable, but if you don’t have the right tools, it’s useless." — Clive Humby

---

**Apakah Anda tahu data apa yang mempengaruhi kesejahteraan pekerja di Indonesia? Temukan rahasia di balik upah, pengeluaran, dan kemiskinan dengan analisis machine learning!**

---

## Apa yang Membuat Proyek Ini Menarik?

Di Indonesia, kesejahteraan pekerja memainkan peran yang sangat penting dalam menentukan kualitas hidup masyarakat. Namun, dengan banyaknya faktor sosial dan ekonomi yang mempengaruhi, data kesejahteraan pekerja sering kali besar, tidak terstruktur, dan tanpa label yang jelas. Data besar seperti ini bisa mengubah segalanya—**jika Anda tahu bagaimana memanfaatkannya.**

Di sinilah **machine learning** hadir sebagai solusi. Dengan teknik **Clustering** dan **Klasifikasi**, kita dapat menggali wawasan berharga tentang kesejahteraan pekerja Indonesia yang sebelumnya tersembunyi dalam data yang besar. 🌍📊

**"Machine learning is a tool, and how you use it can change the way you see the world."**

---

## Pendekatan yang Digunakan:

### 1. **Clustering (Unsupervised Learning)**

Pada tahap pertama, kita menggunakan **Clustering** untuk mengelompokkan pekerja berdasarkan kesamaan karakteristik, seperti pendapatan dan pengeluaran. Dengan menggunakan algoritma **K-Means**, kita membagi data menjadi beberapa cluster untuk mengidentifikasi pola-pola yang mungkin tidak terdeteksi sebelumnya. Hasil dari clustering ini memberikan informasi tentang segmen-segmen pekerja yang dapat dianalisis lebih lanjut dengan model klasifikasi.

```python
from sklearn.cluster import KMeans
import pandas as pd

# Memuat dataset
df = pd.read_csv('Dataset_clustering.csv')

# Menentukan jumlah cluster
kmeans = KMeans(n_clusters=3, random_state=42)

# Melakukan clustering
df['Cluster'] = kmeans.fit_predict(df[['Pendapatan', 'Pengeluaran']])

# Menampilkan hasil clustering
df.head()
```

### 2. **Klasifikasi (Supervised Learning)**

Setelah memperoleh hasil clustering, label dari cluster digunakan sebagai target (kelas) dalam model **Klasifikasi**. Di sini, kita mengembangkan model klasifikasi untuk memprediksi kelas kesejahteraan pekerja berdasarkan fitur-fitur yang tersedia. Algoritma yang digunakan adalah **Random Forest** dan **Logistic Regression**.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Memisahkan fitur dan label
X = df.drop(columns=['Cluster'])
y = df['Cluster']

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Membuat dan melatih model Logistic Regression
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)

# Evaluasi model
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)

# Metrik evaluasi
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print(f'Random Forest - Accuracy: {accuracy_rf}, F1 Score: {f1_rf}')
print(f'Logistic Regression - Accuracy: {accuracy_lr}, F1 Score: {f1_lr}')
```

---

## Evaluasi Model: Apa yang Ditemukan?

Setelah melatih model, kami mengevaluasi performa model dengan metrik **Accuracy** dan **F1-Score**, dua metrik penting untuk mengetahui bagaimana model bekerja pada data uji. F1-Score memberikan keseimbangan antara **Precision** dan **Recall**, yang sangat penting untuk memastikan model bekerja dengan baik pada data yang tidak seimbang.

Hasil yang diperoleh menunjukkan bahwa **Random Forest** memberikan hasil yang sangat baik, dengan **Accuracy** mencapai 1.0 dan **F1-Score** yang juga 1.0, yang menandakan model mampu mengklasifikasikan data dengan sempurna. Sementara **Logistic Regression** memberikan **Accuracy** 0.97 dan **F1-Score** yang sedikit lebih rendah, tetapi masih menunjukkan performa yang sangat baik.

```python
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_lr = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
```

---

### Tuning Model Klasifikasi (Optional)

Pada langkah ini, kami menggunakan **GridSearchCV** untuk menemukan kombinasi hyperparameter terbaik untuk model **Random Forest**. Hasil GridSearch mengungkapkan parameter terbaik untuk model, yang kemudian digunakan untuk evaluasi ulang.

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid untuk Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV untuk menemukan kombinasi parameter terbaik
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Menampilkan hasil kombinasi parameter terbaik
print("Best parameters:", grid_search.best_params_)
```

---

### Analisis Hasil Evaluasi Model Klasifikasi

Secara keseluruhan, model menunjukkan performa yang sangat baik pada kedua algoritma, terutama dengan **Random Forest**, yang mampu mengklasifikasikan data dengan akurasi 100%. Namun, ada beberapa hal yang perlu dicermati:

- **Precision dan Recall**: Tidak dihitung secara eksplisit, tetapi nilai **F1-Score** yang tinggi menunjukkan bahwa model mampu menyeimbangkan **Precision** dan **Recall** dengan baik.
- **Overfitting dan Underfitting**: Meskipun **Accuracy** dan **F1-Score** sangat tinggi, kita harus memperhatikan kemungkinan **overfitting**, terutama pada model **Random Forest**. Untuk mengatasi hal ini, penggunaan teknik **cross-validation** dan **regularization** dapat diterapkan.

Rekomendasi untuk langkah selanjutnya termasuk mengeksplorasi teknik pengurangan overfitting, seperti **cross-validation** dan **regularization**, serta mencoba algoritma lain seperti **XGBoost** atau **KNN** untuk perbandingan.

---

### Kesimpulan

Proyek ini berhasil menunjukkan bagaimana teknik **Clustering** dan **Klasifikasi** dapat digunakan untuk menganalisis kesejahteraan pekerja Indonesia, dengan mengidentifikasi segmen-segmen pekerja berdasarkan pola data seperti upah dan pengeluaran. Model klasifikasi yang dikembangkan mampu memprediksi kesejahteraan pekerja dengan akurasi yang tinggi, memberikan wawasan yang sangat berharga dalam merancang kebijakan sosial dan ekonomi.

Jika Anda tertarik untuk mengeksplorasi lebih lanjut atau memberikan kontribusi pada proyek ini, Anda bisa mengunduh kode dan datasetnya melalui GitHub. Terus eksplorasi dan optimasi algoritma untuk melihat bagaimana data besar bisa menghasilkan wawasan yang dapat mengubah masa depan pekerja Indonesia!

---

### Let’s Dive Into the Code:
Seluruh kode yang digunakan dalam proyek ini dapat ditemukan dalam file [Klasifikasi] Submission Akhir BMLP_Nama.ipynb dan [Clustering] Submission Akhir BMLP_Nama.ipynb yang tersedia di repositori ini.
```
