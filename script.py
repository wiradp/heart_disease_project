# Machine Learning Workflow

### 1. Importing Data to Python
* Drop Duplicates
### 2. Data Preprocessing
* Input-output Split
* Train - Test Split
* Imputation
* Processing Categorical
* Normalization
### 3. Training Machine Learning
* Choose Score to optimize and Hyperparameter Space
## **Heart Disease Analysis**
* Task : Classification
* Objective : Prediksi pasien yang berpotensi kena serangan jantung
### Data Description
**Informasi tambahan:**

Database ini memiliki 76 atribut, namun semua eksperimen yang dipublikasikan hanya menggunakan subset dari 14 atribut tersebut. Secara khusus, database Cleveland adalah satu-satunya yang telah digunakan oleh peneliti machine learning hingga saat ini. Kolom "target" mengacu pada keberadaan penyakit jantung pada pasien, dengan nilai integer dari 0 (tidak ada penyakit) hingga 4 (kehadiran penyakit). Eksperimen yang dilakukan dengan database Cleveland berfokus pada usaha untuk membedakan antara keberadaan penyakit (nilai 1, 2, 3, 4) dengan tidak adanya penyakit (nilai 0).

**Dataset Atribut**

| Atribut    | Type         | Deskripsi                        |
|------------|--------------|----------------------------------|
| age        | Integer      | Usia pasien                      |
| sex        | Categorical  | Jenis kelamin pasien *(0=wanita, 1=pria)*|
| cp         | Categorical  | Tipe nyeri dada *(1=angina tipikal, 2=angina atipikal, 3=nyeri non-angina, 4=tanpa gejala)*             |
| trestbps   | Integer      | Tekanan darah istirahat (mm Hg)  |
| chol       | Integer      | Kolesterol serum (mg/dl)         |
| fbs        | Categorical  | Gula darah puasa >120 mg/dl (1=benar, 0=salah)|
| restecg    | Categorical  | Hasil elektrokardiografi istirahat *(0=normal, 1=memiliki kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV), 2=menunjukkan hipertrofi ventrikel kiri yang mungkin atau pasti berdasarkan kriteria Estes)*|
| thalach    | Integer      | Denyut jantung max yang dicapai|
| exang      | Categorical  | Angina yang disebabkan oleh olahraga *(1=ya, 0=tidak)*|
| oldpeak    | Integer      | Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat|
| slope      | Categorical  | Kemiringan segmen ST puncak latihan (0-2)|
| ca         | Integer      | Jumlah pembuluh besar yang diwarnai oleh fluoroskopi (0-3)|
| thal       | Categorical  | Thalassemia *(1 = normal; 2 = cacat tetap; 3 = cacat reversibel)*|
| target     | Integer      | Diagnosa penyakit jantung *(0=tidak ada, 1-4=ada)*|


## <b><font color='orange'> 1. Importing Data to Python </b>
# Import library pengolahan struktur data
import pandas as pd

# Import library pengolahan angka
import numpy as np

# Import library untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
# Buat fungsi untuk mengimpor dataset
def ImportData(data_file):
    """
    Fungsi untuk import data & hapus duplikat
    :param data_file: <string> nama file input (format .data)
    :return heart_df: <pandas> sample data
    """
    # Definisikan nama kolom sesuai dengan dokumentasi dataset
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    # baca data
    heart_df = pd.read_csv(data_file, names=column_names)

    # cetak bentuk data
    print('Data asli:',heart_df.shape, '-(#Observasi, #kolom)')
    print('Jumlah baris',heart_df.shape[0], 'dan jumlah kolom',heart_df.shape[1])

    # Cek data duplikat
    duplicate_status = heart_df.duplicated(keep='first')

    if duplicate_status.sum() == 0:
        print('Tidak ada data duplikat')
    else:
        heart_df = heart_df.drop_duplicates()
        print('Data setelah di-drop :', heart_df.shape, '-(#observasi, #kolom)')

    return heart_df

# (data_file) adalah argumen
# Argumen adalah sebuah variable
# Jika fungsi tersebut diberi argumen data_file = "processed.cleveland.data",
# maka semua variable 'data_file' didalam fungsi akan berubah menjadi 'processed.cleveland.data'
# Input argumen
data_file = 'processed.cleveland.data'

# Panggil fungsi
heart_df = ImportData(data_file)
# Cek statistical data
heart_df.describe().T
# Cek Jumlah nilai dan nilai unik pada setip kolom
summary_dict = {}
for i in list(heart_df.columns):
    summary_dict[i] = {
        'Jumlah Nilai': heart_df[i].value_counts().shape[0],
        'Nilai Unik': heart_df[i].unique()
    }
summary_df = pd.DataFrame(summary_dict).T

summary_df
>Terdapat nilai **'?'** pada kolom `ca` dan `thal`. Kita perlu merubah nilai tersebut menjadi NA/NaN
print('Jumlah nilai "?" pada kolom ca   :', (heart_df['ca'] == '?').sum())
print('Jumlah nilai "?" pada kolom thal :', (heart_df['thal'] == '?').sum())
# Lihat semua kolom yang mengandung nilai '?'
heart_df_unique = heart_df[heart_df.isin(['?']).any(axis=1)]
heart_df_unique

sns.histplot(heart_df['ca'])
plt.show()
sns.histplot(heart_df['thal'])
plt.show()
# Penganganan missing value
def handle_missing_value(df):
    """
    Fungsi untuk menangani missing value yang ditandai dengan '?'
    param df: <pandas dataframe> data input
    return df: <pandas dataframe> data dengan missing value yang sudah diganti
    """
    # Ganti '?' dengan NaN
    df.replace('?', np.NaN, inplace=True)

    # Tampilkan jumlah missing value per kolom
    print('Jumlah missing value per kolom:\n', df.isnull().sum())

    return df
# Panggil fungsi untuk menangai missing value
heart_df = handle_missing_value(heart_df)
heart_df.head()
sns.histplot(heart_df['ca'])
plt.show()
sns.histplot(heart_df['thal'])
plt.show()
print('Nilai unik kolom ca   :', heart_df['ca'].unique())
print('Nilai unik kolom thal :', heart_df['thal'].unique())
## Data Visualisasi
sns.countplot(data=heart_df, x='sex', hue='sex')
plt.show()
Insight:
- Pria = 1, Wanita = 0
- Jumlah pria lebih banyak daripada wanita
sns.countplot(data=heart_df, x='cp')

plt.show()
## <b><font color='orange'> 2. Data Preprocessing:</font></b>
---
* Input-Output Split, Train-Test Split
* Processing Categorical
* Imputation, Normalization, Drop Duplicates
### **Input-Output Split**

- Fitur `y` adalah output variabel dari target
- yang lainnya menjadi input
Buat output data
# Buat data yang berisikan data target
# Pilih data dengan nama kolom 'target' sebagai output data
output_data = heart_df['target']

output_data.head()
**Buat data input**

- DATA = INPUT + OUTPUT
- DATA - OUTPUT = INPUT
- Jadi kalau dari data, kita drop VARIABLE OUTPUT, maka tersisa hanya variabel INPUT.
def extractInputOutput(data, output_column_name, column_to_drop=None):
    """
    Fungsi untuk memisahkan data input dan output
    :param data: <pandas dataframe> data seluruh sample
    :param output_column_name: <string> nama kolom output
    :param column_to_drop: daftar nama kolom yang ingin dihapus sebelum memisahkan
    :return input_data: <pandas dataframe> data input, <pandas series> data output
    """
    # drop data yang tidak diperlukan jika ada
    if column_to_drop:
        data = data.drop(columns=column_to_drop)

    # pisahkan data output
    output_data = data[output_column_name]

    # drop kolom output dari data untuk mendapatkan input_dataa
    input_data = data.drop(columns=output_column_name, axis=1)

    return input_data, output_data

# (data, output_column_name) adalah argumen
# Argumen adalah sebuah variable
# Jika fungsi tsb diberi argumen data = heart_df
# maka semua variable 'data' didalam fungsi akan berubah menjadi heart_data
# input_data, output_data = extractInputOutput(heart_df, 'target')
x, y = extractInputOutput(heart_df, 'target')
**Selalu sanity check**
x.head()
y.head()
**Check count value data**
# Cek Jumlah nilai dan nilai unik pada input_data (x)
summary_dict = {}
for i in list(x.columns):
    summary_dict[i] = {
        'Jumlah Nilai': x[i].value_counts().shape[0],
        'Nilai Unik': x[i].unique()
    }
summary_df = pd.DataFrame(summary_dict).T

summary_df
### **Train-Test Split**

- **Kenapa?**
  - Karena tidak mau overfit data training
  - Test data akan menjadi future data
  - Kita akan latih model ML di data training, dengan CV (Cross-validation)
  - Selanjutnya melakukan evaluasi di data testing
# Import train-test splitting library dari sklearn
from sklearn.model_selection import train_test_split
**Train Test Split Function**
1. `X` adalah input
2. `y` adalah output (target)
3. `test_size` adalah seberapa besar proporsi data test dari keseluruhan data. Contoh `test_size = 0.2` artinya data test akan berisi 20% data.
4. `random_state` adalah kunci untuk random. Harus di-setting sama. Misal `random_state = 12`.
5. Output:
   - `x_train` = input dari data training
   - `x_test` = input dari data testing
   - `y_train` = output dari data training
   - `y_test` = output dari data testing
6. Urutan outputnya: `x_train, x_test, y_train, y_test`. Tidak boleh terbalik

> Readmore: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=12)
print('Bentuk dari x_train adalah :', x_train.shape)
print('Bentuk dari x_test adalah  :', x_test.shape)
print('Bentuk dari y_train adalah :', y_train.shape)
print('Bentuk dari y_test adalah  :', y_test.shape)
# Ratio
x_test.shape[0] / x.shape[0]

# Hasil 0.20 --> sudah sesuai dengan test_size
**Congrats kita sudah punya data train & test**
### **Data Imputation**

- Proses pengisian data yang kosong (NaN)
- Ada 2 hal yang diperhatikan:
  - Numerical Imputation
  - Categorical Imputation
# Cek data x_train yang kosong
x_train.isnull().sum() / x_train.shape[0]*100
**Bedakan antara data categorical dan numerical**
x_train.head()
**Data Categorical**
- sex
- cp
- fbs
- restecg
- exang

Sisanya adalah numerical
# Lihat kolom pada x_train
x_train.columns
# Buat kolom numerical dan categorical
categorical_column = ['sex', 'cp', 'fbs', 'restecg', 'exang']

numerical_column = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                    'slope', 'ca', 'thal']
# Lihat hasil pengkategorian
print(categorical_column)
print(numerical_column)
# Seleksi dataframe x_train numerikal
x_train_numerical = x_train[numerical_column]
x_train_numerical.head()
**Cek apakah ada data numerical yang kosong**
x_train_numerical.isnull().any()

>Terdapat dua kolom yang mempunyai nilai kosong yaitu ca dan thal
**Gunakan Imputer dari sklearn untuk data Imputation numerik saja**
- fit : imputer agar mengetahui mean atau median dari tiap kolom
- transform : isi data dengan median atau mean
- output dari transform adalah pandas dataframe
- kembalikan dataFrame yang sudah memiliki kolom dan indeks yang sama seperti data asli
# Import library untuk melakukan impute
from sklearn.impute import SimpleImputer
def impute_missing_values(data, strategy="median"):
    """
    Fungsi untuk melakukan imputasi missing value pada data numerik
    param data: <pandas dataframe> Data numerik yang ingin di imputasi
    param strategy: <string> Strategi imputasi, default adalah "median"
                    Pilihan lain: "mean", "most_frequent", "constant"
    return data_imputed: <pandas dataframe> Data numerik dengan missing values yang sudah diimputasi
    """
    # Inisialisasi SimpeImputer dengan strategi yang dipilih
    imputer = SimpleImputer(missing_values=np.NaN, strategy="median")

    # Fit imputer pada data dan trasformasi
    imputed_data = imputer.fit_transform(data)

    # Konversi hasil transformasi kembali ke dataframe
    data_imputed = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)

    return data_imputed
# Jalankan fungsi
x_train_numerical_imputed = impute_missing_values(x_train_numerical)
# Cek missing value setelah di impute
x_train_numerical_imputed.isnull().any()
>Setelah di impute tidak ada lagi data yang kosong
# Seleksi dataframe x_train kategorical
x_train_categorical = x_train[categorical_column]
x_train_categorical.head()
# Cek missing value pada data kategorikal
x_train_categorical.isnull().sum()
>Tidak ada missing value pada data kategorikal
**Preprocessing Categorical Variables**
- Kita tidak bisa memasukkan data categorical jika tidak diubah menjadi numerical
- Solusi --> One Hot Encoding (OHE)
def extractCategorical(data, categorical_column):
    """
    Fungsi untuk ekstrak data kategorikal dengan One Hot Encoding (OHE)
    :param data: <pandas dataframe> data sample
    :param categorical_column: <list> list kolom kategorik
    :param categorical_ohe: <pandas dataframe> data sample dengan OHE
    :return result_data: <pandas dataframe> hasil penggabungan data sample dengan data ketegorik OHE
    """
    # Lakukan One-Hot Encoding pada kolom kategorikal
    # categorical_ohe = pd.get_dummies(data[categorical_column])
    categorical_ohe = pd.get_dummies(x_train_categorical)
    
    # Gabungkan hasil OHE dengan kolom lainnya (jika ada)
    # result_data = pd.concat([data.drop(columns=categorical_column), categorical_ohe], axis=1)
    
    return categorical_ohe
# Panggil fungsi untuk melakukan encoding
x_train_categorical_ohe = extractCategorical(data=x_train, categorical_column=categorical_column)
x_train_categorical_ohe.head()
# Simpan kolom OHE untuk diimplementasikan dalam testing data
# Agar shape nya konsisten
ohe_columns = x_train_categorical_ohe.columns
### Join data Numerical dan Categorical
- Data numerik & kategorik harus digabungkan kembali
- Penggabungan dengan `pd.concat`
# Lakukan penggabungan data numerik dan data kategorik yang sudah di encoded
x_train_concat = pd.concat([x_train_numerical_imputed, x_train_categorical_ohe], axis=1)
# Lihat hasilnya
x_train_concat.head()
x_train_concat.isnull().sum()
>Tidak ada missing value pada penggabungan data numerik dan data kategorik
### Standardizing Variables
- Menyamakan skala dari variable input
- `fit` : imputer agar mengetahui mean dan standar deviasi dari setiap kolom
- `transform` : isi data dngan value yang sudah di normalisasi
- output dari transform berupa pandas dataframe
- normalize dikeluarkan karena akan digunakan pada data test

from sklearn.preprocessing import StandardScaler

def standardizerData(data):
    """
    Fungsi untuk melakukan standarisasi data
    :param data: <pandas dataframe> data sample
    :return standardized_data: <pandas dataframe> data sample standart
    :return standarizer: method untuk standarisasi data
    """
    data_columns = data.columns # agar nama kolom tidak hilang
    data_index = data.index # agar index tidak hilang

    # Buat (fit) Standardizer
    standarizer = StandardScaler()
    standarizer.fit(data)

    # Transform data
    standarized_data_raw = standarizer.transform(data)
    standarized_data = pd.DataFrame(standarized_data_raw)
    standarized_data.columns = data_columns
    standarized_data.index = data_index

    return standarized_data, standarizer
# Jalankan fungsi
x_train_clean, standardizer = standardizerData(data=x_train_concat)
x_train_clean.head()
# Cek missing value
x_train_clean.isnull().sum()
>Tidak ada missing value dari data yang sudah di standarisasi
## <b><font color='orange'> 3. Training Machine Learning </b>
* Choose score to optimize and Hyperparameter Space
* Cross-Validation: Random vs Grid Search CV
* Kita  harus mengalahkan benchmark

### **Benchmark / Baseline**
- Baseline untuk evaluasi nanti
- Karena inii klarifikasi, bisa kita ambil dari proporsi kelas target yang terbesar
- Dengan kata lain, menebak hasil output marketing response dengan nilai "no" semua tanpa modeling
y_train.value_counts(normalize=True)
### 1. Import Model
- Kita akan gunakan 3 model ML untuk klarifikasi:
    - K-nearest neighbor (K-NN)
    - Logistic Regression
    - Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

### 2. Fitting Model
- Cara fitting / training model mengikuti yang dokumentasi model
# Model K-nearst neighbor (KNN)
knn = KNeighborsClassifier()
knn.fit(x_train_clean, y_train)
# Model Logistic Regression
logreg = LogisticRegression(random_state=123)
logreg.fit(x_train_clean, y_train)
# Model Random Forest
random_forest = RandomForestClassifier(random_state=123)
random_forest.fit(x_train_clean, y_train)
# Model Random Forest Classifier 1
# Ubah hyperparameter dari random forest --> n_estimator
# Tambahkan n_estimator = 500
random_forest_1 = RandomForestClassifier(random_state=123, n_estimators=500)
random_forest_1.fit(x_train_clean, y_train)
### 3. Prediction
- Saatnya melakukan prediksi
# Prediksi KNN
predicted_knn = pd.DataFrame(knn.predict(x_train_clean))
predicted_knn.head()
# Prediksi Logistic Regression
predicted_logreg = pd.DataFrame(logreg.predict(x_train_clean))
predicted_logreg.head()
# Prediksi Random Forest
predicted_random_forest = pd.DataFrame(random_forest.predict(x_train_clean))
predicted_random_forest.head()
predicted_random_forest_1 = pd.DataFrame(random_forest_1.predict(x_train_clean))
predicted_random_forest_1.head()
### 4. Cek performa model di data training
benchmark = y_train.value_counts(normalize=True)
benchmark
# Cek Akurasi tiap-tiap model
print('Akurasi KNN                  :',knn.score(x_train_clean, y_train))
print('Akurasi Logistric Regression :',logreg.score(x_train_clean, y_train))
print('Akurasi Random Forest        :',random_forest.score(x_train_clean, y_train))
print('Akurasi Random Forest 1      :',random_forest_1.score(x_train_clean, y_train))
### 6. Test Prediction
1. Siapkan file test dataset
2. Lakukan preprocessing yang sama dengan yang dilakukan di train dataset
3. Gunakan `imputer_numerical` dan `standarizer` yang telah di fit di train dataset
# Cek nilai kosong pada x_test
x_test.isnull().any()
# Cek nilai kosong pada y_test
y_test.isnull().any()
def extractTest(data, numerical_column, categorical_column, ohe_columns,
                impute_missing_values, standardizer):
    """
    Fungsi untuk mengekstrak & membersihkan test data
    :param data: <pandas dataframe sample dataset
    :param numerical_column: <list> kolom numerik
    :param categorical_column: <list> kolom kategorik
    :param ohe_column: <list> kolom one_hot_encoding dari kategorik kolom
    :param impute_missing_values: <sklearn method> imputer data numerik
    :param standardizer: <sklearn method> standarizer data
    :return cleaned_data: <pandas dataframe> data final
    """
    # Filter data
    numerical_data = data[numerical_column]
    categorical_data = data[categorical_column]

    # Proses data numerik
    numerical_data = pd.DataFrame(impute_missing_values.transform(numerical_data))
    numerical_data.columns = numerical_column
    numerical_data.index = data.index
    
    # Proses data ketegorik
    categorical_data.index = data.index
    categorical_data = pd.get_dummies(categorical_data)
    categorical_data.reindex(index=categorical_data.index, 
                             columns=ohe_columns)
    
    # Gabungkan data
    concat_data = pd.concat([numerical_data, categorical_data], axis=1)
    cleaned_data = pd.DataFrame(standardizer.transform(concat_data))
    cleaned_data.columns = concat_data.columns

    return cleaned_data
x_text_clean = extractTest(data=x_test,
                           numerical_column=numerical_column,
                           categorical_column=categorical_column,
                           ohe_columns=ohe_columns,
                           impute_missing_values=impute_missing_values,
                           standardizer=standardizer)
