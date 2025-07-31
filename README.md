# BIST Gelişmiş Filtreli Hisse Tarayıcısı

Bu Python programı, Borsa İstanbul (BIST) hisselerini teknik analiz göstergelerine göre otomatik olarak tarar ve filtreler. Kullanıcı, ister tüm BIST hisselerini isterse seçtiği hisseleri analiz ettirebilir. Sonuçlar, öneriler ve sebepleriyle birlikte ekrana yazdırılır.

## Özellikler

- **RSI, MACD, Parabolik SAR, Hareketli Ortalamalar, Bollinger Bands, Stochastic, Volatilite** gibi teknik göstergelerle gelişmiş filtreleme
- Minimum/maksimum fiyat ve hacim gibi temel filtreler
- Kriterlere uyan hisseler için öneri ve gerekçeleri
- Kriterlere uymayan hisseler için detaylı açıklama
- Kullanıcıdan hisse seçimi veya tüm hisseleri tarama seçeneği

## Kurulum

1. Python 3.8+ yüklü olmalıdır.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install yfinance pandas numpy
   ```

## Kullanım

1. `Hisse Analiz Programı.py` dosyasını çalıştırın:
   ```bash
   python "Hisse Analiz Programı.py"
   ```
2. Program başında mevcut filtre kriterleri ekrana gelir.
3. Tüm hisseleri taramak için `t`, belirli hisseleri taramak için `b` tuşlayın.
4. Belirli hisseleri seçtiyseniz, hisse kodlarını virgülle ayırarak girin (örn: `THYAO,AKBNK`).
5. Sonuçlar ekrana yazdırılır.

## Filtre Kriterleri

Filtreler, kodun başındaki global değişkenlerle kolayca değiştirilebilir. Örneğin:
- RSI aralığı
- MACD pozitif/negatif
- Fiyat, hacim, volatilite aralıkları
- Bollinger Bands pozisyonu
- Stochastic K aralığı
- Fiyatın hareketli ortalamaların üstünde/altında olması

Her bir filtreyi True/False/None veya sayısal aralıklarla özelleştirebilirsiniz.

## Çıktı

- **Kriterlere uyan hisseler**: Öneri (Al/Tut/Sat) ve sebepleriyle birlikte listelenir.
- **Kriterlere uymayan hisseler**: Uymama sebepleriyle birlikte ayrıca gösterilir.

## Notlar

- Program, Yahoo Finance üzerinden veri çeker. Veri eksikliği veya bağlantı sorunlarında bazı hisseler analiz edilemeyebilir.
- Filtreleri değiştirdikçe, analiz sonuçları ve öneriler de değişecektir.

---

Her kod güncellemesinde bu dosya da güncellenecektir.
