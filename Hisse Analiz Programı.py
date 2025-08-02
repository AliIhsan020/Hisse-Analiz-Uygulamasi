import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GÜNCELLENMIŞ FİLTRE KRİTERLERİ
# =============================================================================

# RSI Kriterleri
MIN_RSI = 40          # Minimum RSI değeri
MAX_RSI = 60          # Maksimum RSI değeri (>60 ise alma)

# MACD Kriterleri
MACD_CROSSOVER = True    # MACD çizgisi sinyal çizgisini aşağıdan yukarı kesmiş olmalı
MACD_HISTOGRAM_POSITIVE = True  # MACD histogram pozitif olmalı

# Hacim Kriterleri
MIN_VOLUME = 150000     # Minimum günlük hacim
VOLUME_INCREASE_MIN = 1.2  # Son gün hacim %20 artmış olmalı

# EMA Kriterleri
EMA20_ABOVE_EMA50 = True    # EMA20 > EMA50
PRICE_NEAR_EMA20 = True     # Fiyat EMA20'ye yakın ama çok üstünde olmamalı
MAX_PRICE_EMA20_DISTANCE = 0.05  # Fiyat EMA20'den maksimum %5 uzakta olabilir

# ATR Kriterleri (Günlük hareket)
MIN_ATR_PERCENT = 3.0   # Minimum %3 günlük hareket
MAX_ATR_PERCENT = 6.0   # Maksimum %6 günlük hareket

# Destek/Direnç Kriterleri
NEAR_SUPPORT = True     # Fiyat destek bölgesine yakın olmalı
MAX_SUPPORT_DISTANCE = 0.03  # Destekten maksimum %3 uzakta
MAX_STOP_LOSS_DISTANCE = 0.03  # Stop-loss mesafesi %3'ü geçmemeli
RESISTANCE_POTENTIAL = True   # 2-3 gün içinde direnç seviyesine ulaşma potansiyeli
MAX_RESISTANCE_DISTANCE = 0.08  # Dirençten maksimum %8 uzakta (2-3 günde ulaşılabilir)

# Fiyat Kriterleri
MIN_PRICE = 3.0
MAX_PRICE = 500.0

# BIST 100 hisse kodları (güncellenmiş liste)
BIST100_STOCKS = [
    'THYAO', 'AKBNK', 'ISCTR', 'GARAN', 'VAKBN', 'SASA', 'KCHOL', 'ARCLK', 
    'TUPRS', 'EREGL', 'HALKB', 'TCELL', 'BIMAS', 'SAHOL', 'ASELS', 'KOZAL',
    'PGSUS', 'MGROS', 'SOKM', 'ENKAI', 'OYAKC', 'TOASO', 'PETKM', 'TKFEN',
    'SISE', 'OTKAR', 'AEFES', 'CCOLA', 'ULKER', 'VESTL', 'FROTO', 'GUBRF',
    'ALARK', 'TTKOM', 'KOZAA', 'TAVHL', 'DOHOL', 'ECILC', 'YKBNK', 'ZOREN',
    'LOGO', 'TSKB', 'AKSA', 'AKSEN', 'BRISA', 'CEMTS', 'CIMSA', 'EGEEN',
    'ENJSA', 'FENER', 'GLYHO', 'HEKTS', 'IHLAS', 'IHYAY', 'IZMDC', 'KARSN',
    'KAYSE', 'KLMSN', 'KONYA', 'KRONT', 'MAVI', 'MPARK', 'NETAS', 'PARSN',
    'PINSU', 'PRKME', 'RAYSG', 'RTALB', 'SARKY', 'SELEC', 'SMART', 'SMRTG',
    'TATEN', 'TBORG', 'TEZOL', 'TIRE', 'TRCAS', 'TURSG', 'UFUK', 'ULUSE',
    'UNYEC', 'VAKKO', 'VESBE', 'YAPRK', 'YESIL', 'YGYO', 'ZEDUR', 'ADEL',
    'AGHOL', 'AGROT', 'AHGAZ', 'AKGRT', 'ALCTL', 'ALGYO', 'ALKIM', 'ALTIN',
    'ANSGR', 'ARDYZ', 'ASTOR', 'AVGYO', 'AVHOL', 'AVTUR', 'AYCES', 'AYEN',
    'AYGAZ', 'BAKAB', 'BANVT', 'BASGZ', 'BAYRK', 'BEGYO', 'BERA', 'BEYAZ',
    'BFREN', 'BIGYO', 'BIOEN', 'BJKAS', 'BLCYT', 'BMSCH', 'BMSTL', 'BNTAS',
    'BOBET', 'BOLUC', 'BOSSA', 'BRKO', 'BRKSN', 'BRKVY', 'BSOKE', 'BTCIM'
]

def calculate_rsi(prices, period=14):
    """RSI hesaplama fonksiyonu"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    """EMA hesaplama fonksiyonu"""
    return prices.ewm(span=period).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD hesaplama fonksiyonu"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    """ATR (Average True Range) hesaplama"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def find_support_resistance_levels(prices, window=20):
    """Destek ve direnç seviyelerini bul"""
    # Son window günlük veriler üzerinden yerel min/max bul
    supports = []
    resistances = []
    
    for i in range(window, len(prices) - window):
        # Yerel minimum (destek)
        if prices.iloc[i] == prices.iloc[i-window:i+window].min():
            supports.append(prices.iloc[i])
        
        # Yerel maksimum (direnç)
        if prices.iloc[i] == prices.iloc[i-window:i+window].max():
            resistances.append(prices.iloc[i])
    
    # En son 2 destek ve direnç seviyesini al
    supports = sorted(supports)[-2:] if len(supports) >= 2 else supports
    resistances = sorted(resistances, reverse=True)[:2] if len(resistances) >= 2 else resistances
    
    return supports, resistances

def check_macd_crossover(macd_line, signal_line, lookback=5):
    """MACD çizgisinin sinyal çizgisini aşağıdan yukarı kesip kesmediğini kontrol et"""
    if len(macd_line) < lookback + 1:
        return False
    
    # Son lookback gün içinde crossover var mı?
    for i in range(len(macd_line) - lookback, len(macd_line)):
        if i > 0:
            # Önceki gün MACD < Signal, bugün MACD > Signal
            if (macd_line.iloc[i-1] < signal_line.iloc[i-1] and 
                macd_line.iloc[i] > signal_line.iloc[i]):
                return True
    return False

def analyze_stock_comprehensive(ticker):
    """Kapsamlı hisse analizi"""
    try:
        bist_ticker = ticker.strip().upper()
        if not bist_ticker.endswith('.IS'):
            bist_ticker += '.IS'
        
        stock = yf.Ticker(bist_ticker)
        hist = stock.history(period="6mo")
        
        if len(hist) < 50:
            return None
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        # Temel veriler
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        prev_volume = volume.iloc[-2] if len(volume) > 1 else current_volume
        volume_increase = current_volume / prev_volume if prev_volume > 0 else 1
        
        # EMA hesaplamaları
        ema_20 = calculate_ema(close, 20).iloc[-1]
        ema_50 = calculate_ema(close, 50).iloc[-1] if len(close) >= 50 else None
        
        # RSI
        rsi = calculate_rsi(close).iloc[-1]
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        macd_crossover = check_macd_crossover(macd_line, signal_line)
        
        # ATR
        atr = calculate_atr(high, low, close).iloc[-1]
        atr_percent = (atr / current_price) * 100
        
        # Destek ve direnç seviyeleri
        supports, resistances = find_support_resistance_levels(close)
        
        # Destek ve direnç uzaklıkları
        support_distances = []
        resistance_distances = []
        
        for support in supports:
            distance = ((current_price - support) / support) * 100
            support_distances.append(distance)
        
        for resistance in resistances:
            distance = ((resistance - current_price) / current_price) * 100
            resistance_distances.append(distance)
        
        # En yakın destek ve direnç
        nearest_support = min(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        return {
            'ticker': ticker.upper(),
            'price': current_price,
            'volume_increase': volume_increase,
            'rsi': rsi,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'macd_crossover': macd_crossover,
            'atr_percent': atr_percent,
            'supports': supports,
            'resistances': resistances,
            'support_distances': support_distances,
            'resistance_distances': resistance_distances,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
        
    except Exception as e:
        print(f"Hata {ticker}: {e}")
        return None

def check_new_filters(stock):
    """Yeni filtrelere göre hisse kontrolü"""
    if not stock:
        return False
    
    # Fiyat filtreleri
    if not (MIN_PRICE <= stock['price'] <= MAX_PRICE):
        return False
    
    # RSI filtreleri (40-60 arası, >60 ise alma)
    if not (MIN_RSI <= stock['rsi'] <= MAX_RSI):
        return False
    
    # MACD crossover kontrolü
    if MACD_CROSSOVER and not stock['macd_crossover']:
        return False
    
    # MACD histogram pozitif kontrolü
    if MACD_HISTOGRAM_POSITIVE and stock['histogram'] <= 0:
        return False
    
    # Hacim artış kontrolü
    if stock['volume_increase'] < VOLUME_INCREASE_MIN:
        return False
    
    # EMA20 > EMA50 kontrolü
    if EMA20_ABOVE_EMA50 and stock['ema_50']:
        if stock['ema_20'] <= stock['ema_50']:
            return False
    
    # Fiyat EMA20'ye yakın kontrolü
    if PRICE_NEAR_EMA20:
        price_ema20_distance = abs(stock['price'] - stock['ema_20']) / stock['ema_20']
        if price_ema20_distance > MAX_PRICE_EMA20_DISTANCE:
            return False
    
    # ATR kontrolü
    if not (MIN_ATR_PERCENT <= stock['atr_percent'] <= MAX_ATR_PERCENT):
        return False
    
    # Destek yakınlığı kontrolü
    if NEAR_SUPPORT and stock['nearest_support']:
        support_distance = abs(stock['price'] - stock['nearest_support']) / stock['nearest_support']
        if support_distance > MAX_SUPPORT_DISTANCE:
            return False
    
    # Stop-loss mesafesi kontrolü
    if stock['nearest_support']:
        stop_loss_distance = (stock['price'] - stock['nearest_support']) / stock['price']
        if stop_loss_distance > MAX_STOP_LOSS_DISTANCE:
            return False
    
    # Direnç potansiyeli kontrolü
    if RESISTANCE_POTENTIAL and stock['nearest_resistance']:
        resistance_distance = (stock['nearest_resistance'] - stock['price']) / stock['price']
        if resistance_distance > MAX_RESISTANCE_DISTANCE:
            return False
    
    return True

def calculate_proximity_score(stock):
    """Hissenin kriterlere ne kadar yakın olduğunu hesapla"""
    score = 0
    max_score = 0
    
    # RSI skoru
    max_score += 1
    if MIN_RSI <= stock['rsi'] <= MAX_RSI:
        score += 1
    else:
        if stock['rsi'] < MIN_RSI:
            distance = MIN_RSI - stock['rsi']
            score += max(0, 1 - distance/20)
        else:
            distance = stock['rsi'] - MAX_RSI
            score += max(0, 1 - distance/20)
    
    # MACD crossover skoru
    max_score += 1
    if stock['macd_crossover']:
        score += 1
    else:
        score += 0.3
    
    # MACD histogram skoru
    max_score += 1
    if stock['histogram'] > 0:
        score += 1
    else:
        score += 0.3
    
    # Hacim artış skoru
    max_score += 1
    if stock['volume_increase'] >= VOLUME_INCREASE_MIN:
        score += 1
    else:
        score += max(0, stock['volume_increase'] / VOLUME_INCREASE_MIN)
    
    # EMA skoru
    if stock['ema_50']:
        max_score += 1
        if stock['ema_20'] > stock['ema_50']:
            score += 1
        else:
            score += 0.3
    
    # Fiyat EMA20 yakınlık skoru
    max_score += 1
    price_ema20_distance = abs(stock['price'] - stock['ema_20']) / stock['ema_20']
    if price_ema20_distance <= MAX_PRICE_EMA20_DISTANCE:
        score += 1
    else:
        score += max(0, 1 - price_ema20_distance/0.1)
    
    # ATR skoru
    max_score += 1
    if MIN_ATR_PERCENT <= stock['atr_percent'] <= MAX_ATR_PERCENT:
        score += 1
    else:
        score += 0.3
    
    return score / max_score if max_score > 0 else 0

def format_stock_summary(stock):
    """Hisse özet bilgilerini formatla"""
    # En yakın 2 destek uzaklığı
    support_text = "Yok"
    if len(stock['support_distances']) >= 2:
        support_text = f"{stock['support_distances'][0]:.1f}%, {stock['support_distances'][1]:.1f}%"
    elif len(stock['support_distances']) == 1:
        support_text = f"{stock['support_distances'][0]:.1f}%"
    
    # En yakın 2 direnç uzaklığı
    resistance_text = "Yok"
    if len(stock['resistance_distances']) >= 2:
        resistance_text = f"{stock['resistance_distances'][0]:.1f}%, {stock['resistance_distances'][1]:.1f}%"
    elif len(stock['resistance_distances']) == 1:
        resistance_text = f"{stock['resistance_distances'][0]:.1f}%"
    
    return {
        "Hisse": stock['ticker'],
        "En Yakın 2 Destek Uzaklığı": support_text,
        "En Yakın 2 Direnç Uzaklığı": resistance_text,
        "Hacim Artışı": f"%{(stock['volume_increase']-1)*100:.1f}",
        "ATR": f"%{stock['atr_percent']:.1f}"
    }

def explain_why_not_matching(stock):
    """Hissenin neden filtrelere uymadığını açıklar"""
    reasons = []
    
    if not (MIN_PRICE <= stock['price'] <= MAX_PRICE):
        reasons.append(f"Fiyat {stock['price']:.2f} TL, aralık dışında ({MIN_PRICE}-{MAX_PRICE} TL)")
    
    if not (MIN_RSI <= stock['rsi'] <= MAX_RSI):
        reasons.append(f"RSI {stock['rsi']:.1f}, aralık dışında ({MIN_RSI}-{MAX_RSI})")
    
    if not stock['macd_crossover']:
        reasons.append("MACD çizgisi sinyal çizgisini aşağıdan yukarı kesmemiş")
    
    if stock['histogram'] <= 0:
        reasons.append("MACD histogram pozitif değil")
    
    if stock['volume_increase'] < VOLUME_INCREASE_MIN:
        reasons.append(f"Hacim artışı yetersiz (%{(stock['volume_increase']-1)*100:.1f})")
    
    if stock['ema_50'] and stock['ema_20'] <= stock['ema_50']:
        reasons.append("EMA20 EMA50'nin üstünde değil")
    
    price_ema20_distance = abs(stock['price'] - stock['ema_20']) / stock['ema_20']
    if price_ema20_distance > MAX_PRICE_EMA20_DISTANCE:
        reasons.append(f"Fiyat EMA20'den çok uzak (%{price_ema20_distance*100:.1f})")
    
    if not (MIN_ATR_PERCENT <= stock['atr_percent'] <= MAX_ATR_PERCENT):
        reasons.append(f"ATR aralık dışında (%{stock['atr_percent']:.1f})")
    
    if stock['nearest_support']:
        support_distance = abs(stock['price'] - stock['nearest_support']) / stock['nearest_support']
        if support_distance > MAX_SUPPORT_DISTANCE:
            reasons.append(f"Destekten çok uzak (%{support_distance*100:.1f})")
    
    return reasons

def scan_and_filter_stocks(selected_stocks=None):
    """Hisseleri tara ve filtrele"""
    stocks_to_scan = selected_stocks if selected_stocks else BIST100_STOCKS
    scan_type = "Seçilen" if selected_stocks else "BIST100"
    
    print(f"🔍 {scan_type} hisseler taranıyor...")
    print("Bu işlem birkaç dakika sürebilir...\n")
    
    all_results = []
    filtered_results = []
    processed = 0
    
    for ticker in stocks_to_scan:
        processed += 1
        print(f"İşleniyor: {ticker} ({processed}/{len(stocks_to_scan)})", end='\r')
        
        result = analyze_stock_comprehensive(ticker)
        if result:
            all_results.append(result)
            if check_new_filters(result):
                filtered_results.append(result)
    
    print(f"\n✅ Toplam {len(all_results)} hisse analiz edildi.")
    print(f"🎯 {len(filtered_results)} hisse kriterlere uygun bulundu.\n")
    
    return filtered_results, all_results

def display_results(filtered_results, all_results, is_specific_search=False):
    """Sonuçları göster"""
    
    if is_specific_search:
        # Belirli hisse araması - hem uygun hem uymayanları göster
        if filtered_results:
            print(f"{'='*80}")
            print(f"KRİTERLERE UYGUN HİSSELER ({len(filtered_results)} adet)")
            print(f"{'='*80}")
            
            table_data = []
            for stock in filtered_results:
                table_data.append(format_stock_summary(stock))
            
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        
        # Kriterlere uymayanlar
        non_matching = [s for s in all_results if s not in filtered_results]
        if non_matching:
            print(f"\n{'='*80}")
            print(f"KRİTERLERE UYGUN OLMAYAN HİSSELER ({len(non_matching)} adet)")
            print(f"{'='*80}")
            
            for stock in non_matching:
                print(f"\n📊 {stock['ticker']}:")
                summary = format_stock_summary(stock)
                for key, value in summary.items():
                    if key != "Hisse":
                        print(f"   {key}: {value}")
                
                reasons = explain_why_not_matching(stock)
                print("   Uyumsuzluk Sebepleri:")
                for reason in reasons:
                    print(f"     • {reason}")
    
    else:
        # BIST100 araması - sadece uygun olanları göster
        if filtered_results:
            print(f"{'='*80}")
            print(f"KRİTERLERE UYGUN HİSSELER ({len(filtered_results)} adet)")
            print(f"{'='*80}")
            
            table_data = []
            for stock in filtered_results:
                table_data.append(format_stock_summary(stock))
            
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        
        else:
            # Hiç uygun hisse yok - en yakın 5'i göster
            print("❌ Kriterlere uygun hisse bulunamadı!")
            print("\n🔍 Kriterlere en yakın 5 hisse:\n")
            
            scored_stocks = []
            for stock in all_results:
                score = calculate_proximity_score(stock)
                scored_stocks.append((score, stock))
            
            scored_stocks.sort(reverse=True, key=lambda x: x[0])
            
            for i, (score, stock) in enumerate(scored_stocks[:5], 1):
                print(f"{i}. 📊 {stock['ticker']} (Yakınlık Skoru: {score:.2f})")
                summary = format_stock_summary(stock)
                for key, value in summary.items():
                    if key != "Hisse":
                        print(f"   {key}: {value}")
                
                reasons = explain_why_not_matching(stock)
                print("   Kriterlere Uymama Sebepleri:")
                for reason in reasons:
                    print(f"     • {reason}")
                print()

def show_current_filters():
    """Mevcut filtreleri göster"""
    print(f"\n{'='*60}")
    print(f"GÜNCEL FİLTRE KRİTERLERİ")
    print(f"{'='*60}")
    print(f"RSI: {MIN_RSI}-{MAX_RSI} (>{MAX_RSI} ise alma)")
    print(f"MACD: Çizgi sinyal çizgisini aşağıdan yukarı kesmiş olmalı")
    print(f"MACD Histogram: Pozitif olmalı")
    print(f"Hacim: Son gün %{(VOLUME_INCREASE_MIN-1)*100:.0f} artmış olmalı")
    print(f"EMA: EMA20 > EMA50")
    print(f"Fiyat-EMA20: Maksimum %{MAX_PRICE_EMA20_DISTANCE*100:.0f} uzaklık")
    print(f"ATR: %{MIN_ATR_PERCENT}-{MAX_ATR_PERCENT} günlük hareket")
    print(f"Destek: Maksimum %{MAX_SUPPORT_DISTANCE*100:.0f} uzaklık")
    print(f"Stop-loss: Maksimum %{MAX_STOP_LOSS_DISTANCE*100:.0f}")
    print(f"Direnç Potansiyeli: Maksimum %{MAX_RESISTANCE_DISTANCE*100:.0f} uzaklık")

def main():
    """Ana program"""
    print("🚀 BIST Gelişmiş Filtreli Hisse Tarayıcısı v2.0")
    print("="*60)

    show_current_filters()

    choice = input(f"\nBIST100 taraması için 't', belirli hisseler için 'b' seçin (t/b): ").lower()

    if choice == 'b':
        selected_stocks = input("Hisse kodlarını virgülle ayırarak girin (örn: THYAO,AKBNK): ").split(',')
        selected_stocks = [stock.strip().upper() for stock in selected_stocks]
        filtered_results, all_results = scan_and_filter_stocks(selected_stocks)
        display_results(filtered_results, all_results, is_specific_search=True)
    elif choice == 't':
        filtered_results, all_results = scan_and_filter_stocks()
        display_results(filtered_results, all_results, is_specific_search=False)
    else:
        print("Geçersiz seçim! Program sonlandırılıyor.")
        return

if __name__ == "__main__":
    main()