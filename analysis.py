#
# Ebubekir Tosun
#
# Ödev: Veri Analizi ve Manipülasyonu


#################
# Gereksinimler:
#################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
warnings.simplefilter(action="ignore")


#########################
# KULLANILAN FONKSİYONLAR
#########################

# Genel kontrol fonksiyonu:
def check_df(dataframe, head=5):

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    num_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    print("##################### Quantiles #####################")
    print(dataframe[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Değişken tiplerini belirlemek için:
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# Aykırı değerlerin belirlenmesi:
def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin baskılanması fonksiyonu:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Eksik veri tablosu oluşturmak için:
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

# Trend, Seasonality, Stationarity
def plot_trend_analysis(data, window_size=30):
    """
    Hareketli ortalama (trend) analizi yapar ve grafiği çizer.

    Parameters:
        data (pd.Series): Günlük satış verileri
        window_size (int): Hareketli ortalama penceresi
    """
    rolling_mean = data.rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Günlük Satışlar', alpha=0.5)
    plt.plot(rolling_mean, label=f'{window_size} Günlük Hareketli Ortalama (Trend)', color='red')
    plt.title('Trend Analizi - Günlük Satışlar ve Hareketli Ortalama')
    plt.xlabel('Tarih')
    plt.ylabel('Satış')
    plt.legend()
    plt.grid()
    plt.show()
def plot_seasonality_analysis(data, period=365):
    """
    Mevsimsellik analizi için dekompozisyon yapar ve grafiği çizer.

    Parameters:
        data (pd.Series): Günlük satış verileri
        period (int): Mevsimsel periyot
    """
    decomposition = seasonal_decompose(data, model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 10)
    plt.show()
def perform_stationarity_test(data):
    """
    Durağanlık analizi için ADF testi uygular ve sonuçları yazdırır.

    Parameters:
        data (pd.Series): Günlük satış verileri
    """
    adf_test = adfuller(data.dropna())  # NA değerlerini temizleyerek ADF testi
    print("ADF Test Sonuçları:")
    print(f"Test İstatistiği: {adf_test[0]}")
    print(f"p-Değeri: {adf_test[1]}")
    print(f"Kritik Değerler: {adf_test[4]}")

    if adf_test[1] <= 0.05:
        print("\nSonuç: Zaman serisi durağandır (p <= 0.05).")
    else:
        print("\nSonuç: Zaman serisi durağan değildir (p > 0.05).")

# Haftalık ve Aylık Toplam Satış:
def total_weekly_sales(data):
    # Haftalık toplam satışları hesaplar
    weekly_sales = data.resample('W', on='TARIH')['TOPLAM_SATIS'].sum()

    # Haftalık toplam satışları çizdirir
    plt.figure(figsize=(14, 6))
    plt.plot(weekly_sales.index, weekly_sales.values, label='Haftalık Satışlar', color='blue')
    plt.title('Haftalık Toplam Satış Trendleri')
    plt.xlabel('Tarih')
    plt.ylabel('Satış (TL)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
def total_monthly_sales(data):
    # Aylık toplam satışları hesaplar
    monthly_sales = data.resample('M', on='TARIH')['TOPLAM_SATIS'].sum()

    # Aylık toplam satışları çizdirir
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales.index, monthly_sales.values, label='Aylık Satışlar', color='green')
    plt.title('Aylık Toplam Satış Trendleri')
    plt.xlabel('Tarih')
    plt.ylabel('Satış (TL)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Haftalık ve Aylık Ürün Satışları
def weekly_sales_trends(data):
    weekly_product_sales = data.groupby([pd.Grouper(key='TARIH', freq='W'), 'ÜRÜN_ADI'])['TOPLAM_SATIS'].sum()
    weekly_product_sales = weekly_product_sales.unstack(level='ÜRÜN_ADI').fillna(0)

    for product in weekly_product_sales.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_product_sales.index, weekly_product_sales[product], label=product)

        # Grafiği başlıklandırma ve etiketleme
        plt.title(f'Haftalık Satışlar - {product}')
        plt.xlabel('Tarih')
        plt.ylabel('Toplam Satış')
        plt.legend(title='Ürünler')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
def monthly_sales_trends(data):
    # Aylık satışları hesaplar
    monthly_product_sales = data.groupby([pd.Grouper(key='TARIH', freq='M'), 'ÜRÜN_ADI'])['TOPLAM_SATIS'].sum()
    monthly_product_sales = monthly_product_sales.unstack(level='ÜRÜN_ADI').fillna(0)

    # Her bir ürün için ayrı bir grafik oluşturur
    for product in monthly_product_sales.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_product_sales.index, monthly_product_sales[product], label=product)

        # Grafiği başlıklandırma ve etiketleme
        plt.title(f'Aylık Satışlar - {product}')
        plt.xlabel('Tarih')
        plt.ylabel('Toplam Satış')
        plt.legend(title='Ürünler')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Aylık Satışlar ve Hareketli Ortalama
def monthly_sales_averages(data):
    # Aylık toplam satışları hesaplar
    monthly_sales = data.resample('M', on='TARIH')['TOPLAM_SATIS'].sum()

    # Hareketli ortalama (örneğin 3 aylık) hesaplar
    moving_average = monthly_sales.rolling(window=3).mean()

    # Grafik 1: Aylık Satışlar ve Hareketli Ortalama
    plt.figure(figsize=(14, 6))

    # Aylık satışlar grafiği
    plt.plot(monthly_sales.index, monthly_sales, label='Aylık Satışlar', color='blue')

    # Hareketli ortalama grafiği
    plt.plot(moving_average.index, moving_average, label='3 Aylık Hareketli Ortalama', color='red', linestyle='--')

    # Başlık ve etiketler
    plt.title('Aylık Satış Trendleri ve Hareketli Ortalama')
    plt.xlabel('Tarih')
    plt.ylabel('Satış (TL)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Göster
    plt.show()

# Aylık Satış Değişim Yüzdesi
def monthly_sales_change(data):
    # Aylık toplam satışları hesaplar
    monthly_sales = data.resample('M', on='TARIH')['TOPLAM_SATIS'].sum()

    # Aylık satış değişim yüzdesini hesaplar
    monthly_sales_change = monthly_sales.pct_change() * 100

    # Grafik 2: Aylık Satış Değişim Yüzdesi
    plt.figure(figsize=(14, 6))

    # Aylık satış değişim yüzdesi grafiği (çizgi şeklinde)
    plt.plot(monthly_sales_change.index, monthly_sales_change, label='Aylık Değişim (%)', color='green', linestyle='-', linewidth=2)

    # Başlık ve etiketler
    plt.title('Aylık Satış Değişim Yüzdesi')
    plt.xlabel('Tarih')
    plt.ylabel('Değişim (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Göster
    plt.show()

# Kategorisel ve Sayısal Analiz
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    total_sales = dataframe[target].sum()
    summary = pd.DataFrame({
        "Total_Sales": dataframe.groupby(categorical_col)[target].sum(),
        "Count": dataframe[categorical_col].value_counts(),
        "Sales_Ratio": 100 * dataframe.groupby(categorical_col)[target].sum() / total_sales  # Kategorinin oranı
    })
    print(summary, end="\n\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print('Ortalama:')
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# İleri Düzey Veri Manipülasyonu
def sehir_bazinda_harcama(df):
    # Şehir bazında toplam harcama miktarını hesaplama
    sehir_harcama = df.groupby('SEHIR')['HARCAMA_MIKTARI'].sum()

    # Şehirleri harcama miktarına göre azalan sırayla sıralama
    sehir_harcama_sorted = sehir_harcama.sort_values(ascending=False)

    return sehir_harcama_sorted
def urun_satis_artis_orani(df, pilot=False):
    # 'TARIH' sütununu datetime formatına çevirme
    df['TARIH'] = pd.to_datetime(df['TARIH'])

    # Ürün bazında aylık toplam satışları hesaplama
    df['AY'] = df['TARIH'].dt.to_period('M')  # Aylık periyod oluşturma
    monthly_sales_per_product = df.groupby(['ÜRÜN_ADI', 'AY'])['TOPLAM_SATIS'].sum().unstack(fill_value=0)

    # Satış değişim yüzdesi hesaplama: [(Bu Ay - Geçen Ay) / Geçen Ay] * 100
    monthly_sales_change = monthly_sales_per_product.pct_change(axis='columns') * 100

    # Ortalama satış artış oranını hesaplama
    avg_sales_increase = monthly_sales_change.mean(axis=1)

    # Pilot olarak grafik çizdirme
    if pilot:
        plt.figure(figsize=(10, 6))
        avg_sales_increase.head(20).plot(kind='bar')
        plt.title('Ürünlerin Ortalama Satış Artışı Oranı')
        plt.xlabel('Ürün Kodu')
        plt.ylabel('Ortalama Satış Artışı (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    return avg_sales_increase
def kategori_aylik_satis_ve_degisiklik(df):
    # 'TARIH' sütununu datetime formatına çevirme
    df['TARIH'] = pd.to_datetime(df['TARIH'])

    # Aylık periyot oluşturma
    df['AY'] = df['TARIH'].dt.to_period('M')

    # Her bir kategorinin aylık toplam satışlarını hesaplama
    kategori_aylik_satis = df.groupby(['KATEGORI', 'AY'])['TOPLAM_SATIS'].sum().unstack()

    # Her bir kategori için aylık değişim oranını hesaplama
    kategori_aylik_degisiklik = kategori_aylik_satis.pct_change(axis=1) * 100

    # Grafik 1: Her bir kategorinin aylık toplam satışları
    plt.figure(figsize=(12, 8))
    for kategori in kategori_aylik_satis.index:
        plt.plot(
            kategori_aylik_satis.columns.astype(str),  # Tarih periyotlarını string'e çevir
            kategori_aylik_satis.loc[kategori],
            label=kategori
        )
    plt.title('Her Kategorinin Aylık Toplam Satışları', fontsize=16)
    plt.xlabel('Ay', fontsize=12)
    plt.ylabel('Toplam Satış', fontsize=12)
    plt.legend(title='Kategori', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Grafik 2: Her bir kategorinin aylık satış değişim oranları
    plt.figure(figsize=(12, 8))
    for kategori in kategori_aylik_degisiklik.index:
        plt.plot(
            kategori_aylik_degisiklik.columns.astype(str),  # Tarih periyotlarını string'e çevir
            kategori_aylik_degisiklik.loc[kategori],
            label=kategori
        )
    plt.title('Her Kategorinin Aylık Satış Değişim Oranları', fontsize=16)
    plt.xlabel('Ay', fontsize=12)
    plt.ylabel('Değişim Oranı (%)', fontsize=12)
    plt.legend(title='Kategori', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()







############################################
# GÖREV 1 : VERİ TEMİZLEME VE MANİPÜLASYONU
############################################

# Satış Verisi:
satis_df = pd.read_csv("dataset/satis_verisi_5000.csv")

# Musteri Verisi:
musteri_df = pd.read_csv("dataset/musteri_verisi_5000_utf8.csv")

# Verisetlerini "musteri_id" değişkeni üzerinden birleştiriyorum:
df = pd.merge(musteri_df, satis_df, on='musteri_id', how='inner')
df.info()

# info() ile dtype'ları kontrol ettik. Şimdi yanlış dtype'ları düzeltiyorum:
df["tarih"] = pd.to_datetime(df["tarih"])
df["fiyat"] = pd.to_numeric(df["fiyat"], errors='coerce')
df["toplam_satis"] = pd.to_numeric(df["toplam_satis"], errors='coerce')
df.info()

# Index ifade eden kolonu çıkartıyorum:
df = df.drop(columns=['Unnamed: 0'])

# Tarih değişkenine göre sıralama yapıyorum:
df = df[['tarih'] + [col for col in df.columns if col != 'tarih']]
df = df.sort_values(by='tarih')

# Tüm değişken isimlendirmelerini büyütüyorum:
df.columns = [col.upper() for col in df.columns]


# GENEL RESİM: Verinin özelliklerini kontrol ettiğim fonksiyonu çağırıyorum:
check_df(df)


# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


# AYKIRI DEĞER ANALİZİ

# AŞAMA 1
# Aykırı değer analizi iki aşamada yapılmıştır.
# İlk aşama Fiyat * Adet = Toplam_Satis Hesabını yeniden yaparak bozuk veriler düzenlendi:

df.loc[df['FIYAT'].notnull() & df['ADET'].notnull(), 'TOPLAM_SATIS'] = df['FIYAT'] * df['ADET']


# AŞAMA 2
# İkinci Aşamada gerçek aykırı değer analizi yapıldı:

for col in num_cols:
    if col not in ["TARIH", "MUSTERI_ID"]:
        print(col, check_outlier(df, col))

# Burada TOPLAM_SATIS 'da aykırı değerler gözlemlendi.

###
# NOT : TOPLAM SATIŞ DEĞERLERİNİ FİYAT'LARI HESAPLADIKTAN SONRA BASKILAYACAĞIM!
###


# EKSİK DEĞER ANALİZİ
##################################

# Eksik Gözlem Analizi
df.isnull().sum()

# 1: Veri Setinin Özelliklerini Kullanarak Eksik Veri Doldurulabilir:
# FIYAT ve ADET bilgisini kullanarak doldurma:

# Eksik Toplam Satışları Fiyat * Adet ile doldurma
df.loc[df['TOPLAM_SATIS'].isnull() & df['FIYAT'].notnull(), 'TOPLAM_SATIS'] = (
    df['FIYAT'] * df['ADET']
)

# Eksik Fiyatları Toplam Satış / Adet ile doldurma
df.loc[df['FIYAT'].isnull() & df['TOPLAM_SATIS'].notnull(), 'FIYAT'] = (
    df['TOPLAM_SATIS'] / df['ADET']
)



# AYKIRI DEĞERLERİ BU AŞAMADA BASKILIYORUM. ÇÜNKÜ MATEMATİKSEL OLARAK DOLDURABİLDİĞİM YERLER BİTTİ.
for col in num_cols:
    if col not in ["TARIH", "MUSTERI_ID"]:
        replace_with_thresholds(df,col)


# 2. Matematiksel olarak dolduramayacağım gerçek eksik gözlemleri kontol edelim.

df.isnull().sum()

# Eksik değer tablosu:
na_columns = missing_values_table(df, na_name=True)

# Eksik Değerlerin Doldurulması
for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

# Eksik değerleri median kullanarak doldurduk.
df.isnull().sum()








##################################
# GÖREV 2: ZAMAN SERİSİ ANALİZİ
##################################

# Görevlere geçmeden önce trend, mevsimsellik ve durağanlık analizi yapıldı!

data = df.copy()
data = data.set_index('TARIH', drop=False)

# Günlük Toplam Satışları Hazırlama
daily_sales = data.resample('D')['TOPLAM_SATIS'].sum()

# Zaman Serisi Analiz Fonksiyonlarını Çağıralım:
print("Trend Analizi:")
plot_trend_analysis(daily_sales)

print("\nMevsimsellik Analizi:")
plot_seasonality_analysis(daily_sales)

print("\nDurağanlık Analizi:")
perform_stationarity_test(daily_sales)

# Grafikler elde edildi ve sonuç:
"""
Sonuç: Zaman serisi durağandır (p <= 0.05).
"""

# Görevler:
#############
# 1. Satış verisi üzerinde haftalık ve aylık bazda toplam satış ve ürün satış trendlerini analiz edin.

### Haftalık ve Aylık Toplam Satış Grafikleri:
total_weekly_sales(data)

total_monthly_sales(data)

###  Haftalık ve Aylık Ürün Satışları (Her Bir Ürüne Ayrı Grafikler)
weekly_sales_trends(data)

monthly_sales_trends(data)



##############
# 2: Ayın İlk ve Son Satış Günlerini Bulma

monthly_first_sale = data.groupby(data['TARIH'].dt.to_period('M')).first()
monthly_last_sale = data.groupby(data['TARIH'].dt.to_period('M')).last()
print(monthly_first_sale)
print(monthly_last_sale)

# Haftalık ürün adedi hesapladım (Ürün bazında)
weekly_product_count = data.groupby([pd.Grouper(key='TARIH', freq='W'), 'ÜRÜN_ADI'])['ADET'].sum()
weekly_product_count = weekly_product_count.unstack(level='ÜRÜN_ADI').fillna(0)
print(weekly_product_count)

# Haftalık ürün adedi hesapladım (toplam)
weekly_product_count_total = data.groupby([pd.Grouper(key='TARIH', freq='W')])['ADET'].sum()
print(weekly_product_count_total)


################
# 3: Grafikler çizdirin

# Önceki adımlarda ek olarak grafikler çizdirmiştim.
# Şimdi analizi derinleştirmek için farklı grafikler hazırlıyorum:

# Aylık Satışlar ve Hareketli Ortalama
monthly_sales_averages(data)

# Aylık Satış Değişim Yüzdesi
monthly_sales_change(data)







########################################
# GÖREV 3: KATEGORİSEL VE SAYISAL ANALİZ
########################################

################
# 1: Ürün kategorilerine göre toplam satış miktarını ve her kategorinin tüm satışlar içindeki
# oranını hesaplayın.

kategori_list = ["KATEGORI", "ÜRÜN_ADI"]
for col in kategori_list:
    target_summary_with_cat(df, "TOPLAM_SATIS", col)


################
# 2: Müşterilerin yaş gruplarına göre satış eğilimlerini analiz edin.

# Öncelikle yaş kategori değişkeni oluşturuyorum:
bins = [17, 25, 35, 50, df['YAS'].max()]
labels = ['18-25', '26-35', '36-50', '50+']
df['YAS_GRUBU'] = pd.cut(df['YAS'], bins=bins, labels=labels, right=True)

# Yaş grubuna göre analiz (Ortalama hesabı yapılmıştır.) :
target_summary_with_num(df, "YAS_GRUBU", "TOPLAM_SATIS")



################
# 3: Kadın ve erkek müşterilerin harcama miktarlarını karşılaştırın ve harcama davranışları
# arasındaki farkı tespit edin.

# Burada da kadın ve erkek harcama miktarlarının ortalamaları alınmıştır.
target_summary_with_num(df, "CINSIYET", "HARCAMA_MIKTARI")












#########################################
# GÖREV 4: İLERİ DÜZEY VERİ MANİPÜLASYONU
#########################################


################
# 1. Müşterilerin şehir bazında toplam harcama miktarını bulun ve şehirleri en çok harcama yapan
# müşterilere göre sıralayın.

# Fonksiyonu çağırarak sonucu görüntülüyorum:
print(sehir_bazinda_harcama(df))


################
# 2. Satış verisinde her bir ürün için ortalama satış artışı oranı hesaplayın. Bu oranı
# hesaplamak için her bir üründe önceki aya göre satış değişim yüzdesini kullanın.


# Fonksiyonu çağırarak sonucu görüntülüyorum:
print(urun_satis_artis_orani(df, pilot=True))


################
# 3. Pandas groupby ile her bir kategorinin aylık toplam satışlarını hesaplayın ve değişim
# oranlarını grafikle gösterin.


# Fonksiyonu çağıralım:
kategori_aylik_satis_ve_degisiklik(df)











#########################################
# GÖREV 5: PARETO/ COHORT/ MODELLEME
#########################################


#################
# 1. PARETO
#################
def pareto_analysis(data, product_column, sales_column):
    """
    Pareto analizi: Satışların %80'ini oluşturan ürünleri belirler ve
    farklı ölçekler için ikincil y ekseni ekler.

    Parameters:
        data (DataFrame): Veri seti.
        product_column (str): Ürünleri temsil eden sütun adı.
        sales_column (str): Satış miktarlarını temsil eden sütun adı.

    Returns:
        List: %80'lik dilime giren ürünlerin listesi.
    """
    import matplotlib.pyplot as plt

    # Ürünlere göre toplam satış miktarlarını hesapla ve azalan sırada sırala
    product_sales = data.groupby(product_column)[sales_column].sum().sort_values(ascending=False)

    # Kümülatif toplamını hesapla ve yüzdeye dönüştür
    cumulative_sales = product_sales.cumsum() / product_sales.sum() * 100

    # %80'in altındaki ürünleri seç
    pareto_threshold = cumulative_sales[cumulative_sales <= 80].index

    # Grafik oluşturma
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar grafiği (Ürün satışları için birinci y ekseni)
    ax1.bar(product_sales.index, product_sales, color='lightblue', label='Ürün Satışları')
    ax1.set_ylabel('Satış Miktarı', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Ürün Adı')
    ax1.tick_params(axis='x', rotation=90)

    # İkincil y ekseni (Kümülatif yüzdeler için)
    ax2 = ax1.twinx()
    ax2.plot(cumulative_sales.index, cumulative_sales, color='orange', label='Kümülatif Yüzde', marker='o')
    ax2.axhline(y=80, color='red', linestyle='--', label='%80 Sınırı')
    ax2.set_ylabel('Kümülatif Yüzde (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Başlık ve açıklama
    fig.suptitle('Pareto Analizi: Satışların %80\'ini Oluşturan Ürünler', fontsize=14)
    fig.tight_layout()

    # Efsaneleri birleştir ve göster
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper left')

    plt.show()

    return list(pareto_threshold)

pareto_products = pareto_analysis(data, product_column='ÜRÜN_ADI', sales_column='TOPLAM_SATIS')
print("Satışların %80'ini oluşturan ürünler:", pareto_products)


#################
# 2. COHORT
#################


# Tekrar Alım Oranlarını Hesaplayan Fonksiyon
def calculate_repeat_purchase(data):
    """
    Veri setine ilk satın alma tarihine göre tekrar alım oranlarını ekler.
    - TEKRAR_ALIM: İlk satın alma sonrası yapılan alışverişler.
    """
    # İlk satın alma tarihlerini belirle
    data['ILK_SATINALMA'] = data.groupby('MUSTERI_ID')['TARIH'].transform('min')
    data['COHORT'] = data['ILK_SATINALMA'].dt.to_period('M')
    data['SATINALMA_DONEMI'] = data['TARIH'].dt.to_period('M')

    # Tekrar alım sütunu ekle
    data['TEKRAR_ALIM'] = data['TARIH'] > data['ILK_SATINALMA']
    return data

# Cohort Pivot Tablosu ve Tekrar Alım Analizi
def create_repeat_cohort_analysis(data):
    """
    Cohort bazında tekrar alım oranlarını analiz eder ve bir pivot tablo oluşturur.
    """
    cohort_data = (
        data.groupby(['COHORT', 'TEKRAR_ALIM'])
        .agg(ACTIVE_MUSTERI=('MUSTERI_ID', 'nunique'))
        .reset_index()
    )
    # Pivot tablo oluştur (TEKRAR_ALIM: False ve True sütunları)
    cohort_pivot = cohort_data.pivot(index='COHORT', columns='TEKRAR_ALIM', values='ACTIVE_MUSTERI')

    # Cohort bazında oran hesapla
    cohort_pivot_percentage = cohort_pivot.div(cohort_pivot.sum(axis=1), axis=0)
    return cohort_pivot_percentage

# Cohort Isı Haritası Görselleştirme Fonksiyonu
def plot_repeat_cohort_heatmap(cohort_pivot_percentage):
    """
    Tekrar alım oranlarını ısı haritası olarak görselleştirir.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cohort_pivot_percentage, annot=True, fmt=".0%", cmap="Greens", cbar_kws={'label': 'Tekrar Alım Oranı'}
    )
    plt.title('Cohort Analizi: Tekrar Alım Oranları', fontsize=16)
    plt.xlabel('Tekrar Alım Durumu', fontsize=12)
    plt.ylabel('Cohort (İlk Satın Alma Ayı)', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Hayır', 'Evet'])
    plt.tight_layout()
    plt.show()


# Ana Çalışma
prepared_data = calculate_repeat_purchase(df)

# Tekrar alım oranlarını hesapla
cohort_repeat_percentage = create_repeat_cohort_analysis(prepared_data)

# Analizi görselleştir
plot_repeat_cohort_heatmap(cohort_repeat_percentage)






###############
# 3. Modelleme
###############

### FEATURE ENGINEERING

# Verisetini temizliyorum:

df = df.drop(columns=["ISIM", "ÜRÜN_KODU", "AY", "ILK_SATINALMA", "COHORT", "SATINALMA_DONEMI", "HARCAMA_MIKTARI"])
df.info()

# AY ve HAFTA bilgilerini oluşturuyorum.
df['AY'] = df['TARIH'].dt.to_period('M').astype(str)  # YYYY-MM formatında

# Yeni değişkenler
# 1. Haftanın Günü
df['HAFTANIN_GUNU'] = df['TARIH'].dt.dayofweek  # Pazartesi = 0, Pazar = 6

# 2. Aylık Ortalama Fiyat
df['AYLIK_ORTALAMA_FIYAT'] = df.groupby('AY')['FIYAT'].transform('mean')

# 3. Fiyat / Adet Oranı (Fiyat başına satılan adet)
df['FIYAT_ADET_ORANI'] = df['FIYAT'] / df['ADET']

# 4. Ürün Fiyatı ve Satış Miktarının Çarpımı
df['FIYAT_ADET_CARPI'] = df['FIYAT'] * df['ADET']

# 5. Zamanla Değişen Satışlar (Son 3 Ayın Satış Ortalaması)
df['SON_3_AY_SATIS'] = df.groupby('MUSTERI_ID')['TOPLAM_SATIS'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

# 6. Mevsimsel Satış Durumu (Kış, İlkbahar, Yaz, Sonbahar)
df['MEVSIM'] = df['TARIH'].dt.month % 12 // 3 + 1
df['MEVSIM'] = df['MEVSIM'].map({1: 'Kış', 2: 'İlkbahar', 3: 'Yaz', 4: 'Sonbahar'})

# 7. Yılbaşına Kalan Gün Sayısı
df['YILBASI_KALAN_GUN'] = (pd.to_datetime(df['TARIH'].dt.year.astype(str) + '-12-31') - df['TARIH']).dt.days

# 'repeated_purchase' sütununu 0 ve 1'e dönüştür
df['TEKRAR_ALIM'] = df['TEKRAR_ALIM'].astype(int)


# Tarih bilgisini dönüştür:
# Tarih bilgisini nümerik olarak kullanma
df['AY'] = df['TARIH'].dt.month
df['YIL'] = df['TARIH'].dt.year
df['GUN'] = df['TARIH'].dt.day

cat_cols, num_cols, cat_but_car = grab_col_names(df, 10, 1000)
cat_cols
num_cols
cat_but_car


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, prefix=categorical_cols)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.info()

# Boolean sütunlarını 0 ve 1'e dönüştürme
bool_columns = df.select_dtypes(include='bool').columns

for col in bool_columns:
    df[col] = df[col].astype(int)

# Dönüştürülmüş dataframe'i kontrol et
print(dataframe.head())


####
# BU VERİDE SCALİNG İŞLEMİ DENENDİ!! OVERFİTTİNG'E YOL AÇTIĞI İÇİN KALDIRILDI!!



######## MODEL

X = df.drop('TOPLAM_SATIS','TARIH', 'MUSTERI_ID', axis=1)  # TOPLAM_SATIS hariç tüm sütunlar
y = df['TOPLAM_SATIS']  # Hedef değişken: TOPLAM_SATIS

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


models = [('LR', LinearRegression()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



# Sonuç:  Base
##############################
# RMSE: 760.0062 (LR)
# RMSE: 473.1695 (XGBoost)
##############################


########## Hiperparametre Optimizasyonu


# XGBoost modelini başlatma
xgb_model = XGBRegressor(random_state=46)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametre gridini tanımlama
xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2],         # Öğrenme oranı
    "n_estimators": [100, 500, 1000],          # Ağaç sayısı
    "max_depth": [3, 6, 10],                    # Ağaç derinliği
    "subsample": [0.6, 0.8, 1.0],               # Alt örnekleme oranı
    "colsample_bytree": [0.5, 0.7, 1.0]         # Ağaç başına özellik sayısı
}

# GridSearchCV ile parametre optimizasyonu
xgb_gs_best = GridSearchCV(xgb_model,
                           xgb_params,
                           cv=3,
                           n_jobs=-1,
                           verbose=True).fit(X_train, y_train)

# En iyi parametrelerle modeli eğitme
final_model = xgb_model.set_params(**xgb_gs_best.best_params_).fit(X_train, y_train)

# Modelin performansını test setinde değerlendirme
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(f"RMSE: {rmse}")




# SONUÇ: XGBoost (Hiperparametre Optimizasyonu
#########################
#RMSE: 447.2684377881331
#########################

# HİPERPARAMETRE OPTİMİZASYONU İLE İYİLEŞTİRME YAPILDI!!!

