# BIST Gelişmiş Filtreli Hisse Tarayıcısı v2.0

Bu Python programı, Borsa İstanbul (BIST) hisselerini teknik analiz göstergelerine göre otomatik olarak tarar ve filtreler. Kullanıcı ister tüm BIST100 hisselerini isterse seçtiği hisseleri analiz ettirebilir. Sonuçlar, tablo halinde özetlenir ve detaylı gerekçelerle birlikte ekrana yazdırılır.

## Özellikler

- **RSI, MACD, EMA, ATR, Destek/Direnç, Hacim** gibi teknik göstergelerle gelişmiş filtreleme
- Fiyat, hacim, volatilite ve teknik seviyelere göre detaylı analiz
- Kriterlere uyan hisseler için tablo ve özet
- Kriterlere uymayan hisseler için detaylı açıklama ve uymama sebepleri
- Kriterlere en yakın hisseler için skor ve özet gösterimi
- Kullanıcıdan hisse seçimi veya tüm BIST100 hisselerini tarama seçeneği

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
3. Tüm BIST100 hisselerini taramak için `t`, belirli hisseleri taramak için `b` tuşlayın.
4. Belirli hisseleri seçtiyseniz, hisse kodlarını virgülle ayırarak girin (örn: `THYAO,AKBNK`).
5. Sonuçlar tablo halinde ve gerekçeleriyle ekrana yazdırılır.

## Filtre Kriterleri

Filtreler kodun başındaki global değişkenlerle kolayca değiştirilebilir:
- RSI aralığı
- MACD crossover ve histogram pozitifliği
- Hacim artışı oranı
- EMA20 ve EMA50 ilişkisi, fiyatın EMA20'ye yakınlığı
- ATR (volatilite) aralığı
- Fiyatın destek/direnç seviyelerine uzaklığı
- Stop-loss ve direnç potansiyeli mesafeleri
- Fiyat aralığı

Her bir filtreyi True/False veya sayısal aralıklarla özelleştirebilirsiniz.

## Çıktı

- **Kriterlere uyan hisseler**: Tablo halinde özetlenir (fiyat, destek/direnç, hacim artışı, ATR vb.)
- **Kriterlere uymayan hisseler**: Uymama sebepleriyle birlikte detaylı olarak gösterilir
- **Kriterlere en yakın hisseler**: Skor ve özet ile birlikte gösterilir (uygun hisse yoksa)

## Notlar

- Program, Yahoo Finance üzerinden veri çeker. Veri eksikliği veya bağlantı sorunlarında bazı hisseler analiz edilemeyebilir.
- Filtreleri değiştirdikçe analiz sonuçları ve öneriler de değişecektir.
- Destek/direnç seviyeleri algoritmik olarak son 20 günlük lokal min/max ile hesaplanır.

---

Her kod güncellemesinde bu dosya da güncellenmelidir.
