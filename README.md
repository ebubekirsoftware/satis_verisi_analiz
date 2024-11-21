# Veri Analizi ve Satış Eğilimleri Ödevi

NOT: Tüm görselleştirmeler RAPOR.pdf dosyasında verilmiştir. Bu yüzden ek bir sunum hazırlanmamıştır. Kullanılan gereksinimlerin verssiyon bilgisi requirements.txt'de mevcuttur.


Bu repo, veri analizi ve satış eğilimlerinin incelenmesi amacıyla yapılan bir ödev projesini içermektedir. Proje, çeşitli analiz teknikleri kullanarak satış verilerini incelemekte ve satışların zaman içindeki değişimini, farklı kategoriler arasındaki farkları ve daha fazlasını araştırmaktadır.


## Veri Analizi ve Temizleme

Bu adımda, veri seti üzerinde temel analizler gerçekleştirilmiş ve verinin temizlenmesi sağlanmıştır. Aşağıdaki işlemler yapılmıştır:

- **Veri Seti Özeti**: Gözlem sayısı, değişken türleri ve eksik veri analizi yapılmıştır.
- **Eksik Veri Doldurma**: Eksik değerlerin uygun formüller ve medyan değerlerle doldurulması.
- **Aykırı Değer Tespiti**: Aykırı değerlerin analizi ve eşik değerlerle düzeltilmesi.

## Zaman Serisi Analizi

Zaman serisi verilerinin analizi yapılmıştır. Adımlar şunlardır:

- **Veri Hazırlığı**: Tarihsel verinin düzenlenmesi ve günlük toplam satışların hesaplanması.
- **Trend ve Mevsimsellik Analizi**: Zaman içindeki trend ve mevsimsel örüntülerin belirlenmesi.
- **Durağanlık Testi (ADF Testi)**: Zaman serisinin durağan olup olmadığı test edilmiştir.

## Haftalık ve Aylık Satış Analizi

Bu adımda, satış verileri haftalık ve aylık bazda incelenmiştir:

- **Haftalık Satışlar**: Haftalık toplam satışlar analiz edilmiştir.
- **Aylık Satışlar ve Hareketli Ortalama**: Aylık satışlar ve hareketli ortalamalar hesaplanmıştır.
- **Ürün Bazında Satış Analizi**: Haftalık ve aylık bazda ürün satışları analiz edilmiştir.

## Kategorisel ve Sayısal Analiz

Ürün kategorileri, yaş grupları ve cinsiyet gibi faktörlere göre analizler yapılmıştır:

- **Ürün Kategorilerine Göre Satışlar**: Kategorilere göre toplam satış miktarları ve satış oranları hesaplanmıştır.
- **Yaş Gruplarına Göre Satışlar**: Yaş gruplarına göre ortalama satış analizleri yapılmıştır.
- **Cinsiyet Bazında Harcama**: Kadın ve erkek müşterilerin harcama miktarları karşılaştırılmıştır.

## İleri Düzey Veri Manipülasyonu

Bu adımda daha ileri düzey veri manipülasyonları yapılmıştır:

- **Şehir Bazında Toplam Harcama**: Şehirlere göre toplam harcama miktarları hesaplanmıştır.
- **Ürün Bazında Satış Artışı**: Ürün bazında satış artışı oranları hesaplanmıştır.
- **Kategori Bazında Aylık Satış Değişim Analizi**: Aylık bazda kategorilerin satış değişimleri analiz edilmiştir.
