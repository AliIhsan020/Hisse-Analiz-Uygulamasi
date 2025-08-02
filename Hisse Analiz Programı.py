import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GÜNCELLENMIŞ FİLTRE KRİTERLERİ - İYİLEŞTİRİLMİŞ VERSİYON
# =============================================================================

# 🟢 OLUMLU NOKTALAR (Aynen kalsın)
# RSI Kriterleri - Mükemmel seviye, ne çok dipte ne aşırı alımda
MIN_RSI = 40          # Minimum RSI değeri
MAX_RSI = 60          # Maksimum RSI değeri (>60 ise alma)

# MACD Kriterleri - Momentumun yeni başladığı noktaları yakalar
MACD_CROSSOVER = True    # MACD çizgisi sinyal çizgisini aşağıdan yukarı kesmiş olmalı
MACD_HISTOGRAM_POSITIVE = True  # MACD histogram pozitif olmalı

# 🟡 İYİLEŞTİRİLMİŞ NOKTALAR
# Hacim Kriterleri - Likidite için artırıldı
MIN_VOLUME = 500000     # Minimum günlük hacim (150k'dan 500k'ya çıkarıldı - spread koruması)
VOLUME_INCREASE_MIN = 1.2  # Son gün hacim, son 5 günlük ortalamanın %20 üstü olmalı
VOLUME_LOOKBACK_DAYS = 5   # Hacim karşılaştırması için gün sayısı

# EMA Kriterleri - Kısa vadeli pozitif trend garantisi
EMA20_ABOVE_EMA50 = True    # EMA20 > EMA50
PRICE_NEAR_EMA20 = True     # Fiyat EMA20'ye yakın ama çok üstünde olmamalı
MAX_PRICE_EMA20_DISTANCE = 0.03  # Fiyat EMA20'den maksimum %3 uzakta (5%'den düşürüldü)

# ATR Kriterleri - Ne ölü hisse, ne spekülatif çılgınlık
MIN_ATR_PERCENT = 3.0   # Minimum %3 günlük hareket
MAX_ATR_PERCENT = 6.0   # Maksimum %6 günlük hareket

# 🔴 KRİTİK DENGE NOKTALARI - İyileştirildi
# Destek/Direnç Kriterleri
NEAR_SUPPORT = True     # Fiyat destek bölgesine yakın olmalı
MAX_SUPPORT_DISTANCE = 0.03  # Destekten maksimum %3 uzakta
MAX_STOP_LOSS_DISTANCE = 0.04  # Stop-loss mesafesi %4'ü geçmemeli (gerçekçi seviye)
RESISTANCE_POTENTIAL = True   # 2-3 gün içinde direnç seviyesine ulaşma potansiyeli
MAX_RESISTANCE_DISTANCE = 0.06  # Dirençten maksimum %6 uzakta (8%'den düşürüldü - daha gerçekçi)

# Destek/Direnç analiz parametreleri - 3'e çıkarıldı
SUPPORT_RESISTANCE_COUNT = 3  # Gösterilecek destek/direnç sayısı (2'den 3'e çıkarıldı)

# Fiyat Kriterleri - Uygun, spek hisseleri hariç tutar
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

def calculate_support_strength(prices, support_level, window=20):
    """Destek seviyesinin gücünü hesapla"""
    touches = 0
    bounces = 0
    tolerance = support_level * 0.02  # %2 tolerans
    
    for i in range(len(prices) - window, len(prices)):
        if i > 0:
            # Destek seviyesine yaklaşma
            if abs(prices.iloc[i] - support_level) <= tolerance:
                touches += 1
                # Destek seviyesinden yukarı sıçrama
                if i < len(prices) - 1 and prices.iloc[i+1] > prices.iloc[i]:
                    bounces += 1
    
    if touches == 0:
        return 0
    
    strength = (bounces / touches) * min(touches, 5)  # Max 5 puan
    return min(strength, 5)  # 0-5 arası güç skoru

def find_support_resistance_levels(prices, window=20):
    """Geliştirilmiş destek ve direnç seviyelerini bul - 3 seviye + güç analizi"""
    supports = []
    resistances = []
    
    for i in range(window, len(prices) - window):
        # Yerel minimum (destek)
        if prices.iloc[i] == prices.iloc[i-window:i+window].min():
            strength = calculate_support_strength(prices, prices.iloc[i], window)
            supports.append((prices.iloc[i], strength))
        
        # Yerel maksimum (direnç)
        if prices.iloc[i] == prices.iloc[i-window:i+window].max():
            strength = calculate_support_strength(prices, prices.iloc[i], window)
            resistances.append((prices.iloc[i], strength))
    
    # Güçlü seviyeler öncelikli olmak üzere sırala
    supports = sorted(supports, key=lambda x: (-x[1], -x[0]))  # Güce göre, sonra fiyata göre
    resistances = sorted(resistances, key=lambda x: (-x[1], x[0]))  # Güce göre, sonra fiyata göre
    
    # En güçlü 3 seviyeyi al
    top_supports = supports[:SUPPORT_RESISTANCE_COUNT]
    top_resistances = resistances[:SUPPORT_RESISTANCE_COUNT]
    
    return top_supports, top_resistances

def check_volume_increase(volume, days=VOLUME_LOOKBACK_DAYS):
    """Geliştirilmiş hacim artış kontrolü - son N günlük ortalama ile karşılaştır"""
    if len(volume) < days + 1:
        return 1.0
    
    current_volume = volume.iloc[-1]
    avg_volume = volume.iloc[-(days+1):-1].mean()  # Son N günün ortalaması (bugün hariç)
    
    if avg_volume == 0:
        return 1.0
    
    return current_volume / avg_volume

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
        
        # Geliştirilmiş hacim artış kontrolü
        volume_increase = check_volume_increase(volume)
        
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
        
        # Geliştirilmiş destek ve direnç seviyeleri (güç analizi ile)
        supports_with_strength, resistances_with_strength = find_support_resistance_levels(close)
        
        # Destek ve direnç uzaklıkları
        support_distances = []
        resistance_distances = []
        
        for support_price, strength in supports_with_strength:
            distance = ((current_price - support_price) / support_price) * 100
            support_distances.append(distance)
        
        for resistance_price, strength in resistances_with_strength:
            distance = ((resistance_price - current_price) / current_price) * 100
            resistance_distances.append(distance)
        
        # En yakın destek ve direnç
        nearest_support = supports_with_strength[0][0] if supports_with_strength else None
        nearest_resistance = resistances_with_strength[0][0] if resistances_with_strength else None
        
        return {
            'ticker': ticker.upper(),
            'price': current_price,
            'volume': current_volume,
            'volume_increase': volume_increase,
            'rsi': rsi,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'macd_crossover': macd_crossover,
            'atr_percent': atr_percent,
            'supports_with_strength': supports_with_strength,
            'resistances_with_strength': resistances_with_strength,
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
    
    # Minimum hacim kontrolü
    if stock['volume'] < MIN_VOLUME:
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
    
    # Geliştirilmiş hacim artış kontrolü
    if stock['volume_increase'] < VOLUME_INCREASE_MIN:
        return False
    
    # EMA20 > EMA50 kontrolü
    if EMA20_ABOVE_EMA50 and stock['ema_50']:
        if stock['ema_20'] <= stock['ema_50']:
            return False
    
    # Fiyat EMA20'ye yakın kontrolü (daha sıkı %3)
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
    
    # Stop-loss mesafesi kontrolü (gerçekçi %4)
    if stock['nearest_support']:
        stop_loss_distance = (stock['price'] - stock['nearest_support']) / stock['price']
        if stop_loss_distance > MAX_STOP_LOSS_DISTANCE:
            return False
    
    # Direnç potansiyeli kontrolü (daha gerçekçi %6)
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

def format_support_strength(strength):
    """Destek gücünü formatla"""
    if strength >= 4:
        return "🔴 Çok Güçlü"
    elif strength >= 3:
        return "🟠 Güçlü"
    elif strength >= 2:
        return "🟡 Orta"
    elif strength >= 1:
        return "🟢 Zayıf"
    else:
        return "⚪ Çok Zayıf"

def format_stock_summary(stock):
    """Hisse özet bilgilerini formatla - 3 seviye + güç göstergesi"""
    # En güçlü 3 destek (fiyat, uzaklık ve güç)
    support_text = "Yok"
    if len(stock['supports_with_strength']) >= 3:
        support_parts = []
        for i in range(3):
            price, strength = stock['supports_with_strength'][i]
            distance = stock['support_distances'][i]
            strength_text = format_support_strength(strength)
            support_parts.append(f"{price:.2f}TL (%{distance:.1f}) {strength_text}")
        support_text = " | ".join(support_parts)
    elif len(stock['supports_with_strength']) > 0:
        support_parts = []
        for i, (price, strength) in enumerate(stock['supports_with_strength']):
            distance = stock['support_distances'][i]
            strength_text = format_support_strength(strength)
            support_parts.append(f"{price:.2f}TL (%{distance:.1f}) {strength_text}")
        support_text = " | ".join(support_parts)
    
    # En güçlü 3 direnç (fiyat, uzaklık ve güç)
    resistance_text = "Yok"
    if len(stock['resistances_with_strength']) >= 3:
        resistance_parts = []
        for i in range(3):
            price, strength = stock['resistances_with_strength'][i]
            distance = stock['resistance_distances'][i]
            strength_text = format_support_strength(strength)
            resistance_parts.append(f"{price:.2f}TL (%{distance:.1f}) {strength_text}")
        resistance_text = " | ".join(resistance_parts)
    elif len(stock['resistances_with_strength']) > 0:
        resistance_parts = []
        for i, (price, strength) in enumerate(stock['resistances_with_strength']):
            distance = stock['resistance_distances'][i]
            strength_text = format_support_strength(strength)
            resistance_parts.append(f"{price:.2f}TL (%{distance:.1f}) {strength_text}")
        resistance_text = " | ".join(resistance_parts)
    
    return {
        "Hisse": stock['ticker'],
        "Güncel Fiyat": f"{stock['price']:.2f}TL",
        "En Güçlü 3 Destek": support_text,
        "En Güçlü 3 Direnç": resistance_text,
        "Hacim Artışı": f"%{(stock['volume_increase']-1)*100:.1f}",
        "Günlük Hacim": f"{stock['volume']:,.0f}",
        "ATR": f"%{stock['atr_percent']:.1f}"
    }

def explain_why_not_matching(stock):
    """Hissenin neden filtrelere uymadığını açıklar"""
    reasons = []
    
    if not (MIN_PRICE <= stock['price'] <= MAX_PRICE):
        reasons.append(f"Fiyat {stock['price']:.2f} TL, aralık dışında ({MIN_PRICE}-{MAX_PRICE} TL)")
    
    if stock['volume'] < MIN_VOLUME:
        reasons.append(f"Günlük hacim yetersiz ({stock['volume']:,.0f} < {MIN_VOLUME:,})")
    
    if not (MIN_RSI <= stock['rsi'] <= MAX_RSI):
        reasons.append(f"RSI {stock['rsi']:.1f}, aralık dışında ({MIN_RSI}-{MAX_RSI})")
    
    if not stock['macd_crossover']:
        reasons.append("MACD çizgisi sinyal çizgisini aşağıdan yukarı kesmemiş")
    
    if stock['histogram'] <= 0:
        reasons.append("MACD histogram pozitif değil")
    
    if stock['volume_increase'] < VOLUME_INCREASE_MIN:
        reasons.append(f"Hacim artışı yetersiz (%{(stock['volume_increase']-1)*100:.1f} < %{(VOLUME_INCREASE_MIN-1)*100:.0f})")
    
    if stock['ema_50'] and stock['ema_20'] <= stock['ema_50']:
        reasons.append("EMA20 EMA50'nin üstünde değil")
    
    price_ema20_distance = abs(stock['price'] - stock['ema_20']) / stock['ema_20']
    if price_ema20_distance > MAX_PRICE_EMA20_DISTANCE:
        reasons.append(f"Fiyat EMA20'den çok uzak (%{price_ema20_distance*100:.1f} > %{MAX_PRICE_EMA20_DISTANCE*100:.0f})")
    
    if not (MIN_ATR_PERCENT <= stock['atr_percent'] <= MAX_ATR_PERCENT):
        reasons.append(f"ATR aralık dışında (%{stock['atr_percent']:.1f})")
    
    if stock['nearest_support']:
        support_distance = abs(stock['price'] - stock['nearest_support']) / stock['nearest_support']
        if support_distance > MAX_SUPPORT_DISTANCE:
            reasons.append(f"En yakın destekten çok uzak (%{support_distance*100:.1f} > %{MAX_SUPPORT_DISTANCE*100:.0f})")
    
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
            print(f"{'='*100}")
            print(f"KRİTERLERE UYGUN HİSSELER ({len(filtered_results)} adet)")
            print(f"{'='*100}")
            
            table_data = []
            for stock in filtered_results:
                table_data.append(format_stock_summary(stock))
            
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        
        # Kriterlere uymayanlar
        non_matching = [s for s in all_results if s not in filtered_results]
        if non_matching:
            print(f"\n{'='*100}")
            print(f"KRİTERLERE UYGUN OLMAYAN HİSSELER ({len(non_matching)} adet)")
            print(f"{'='*100}")
            
            for stock in non_matching:
                print(f"\n📊 {stock['ticker']} - Güncel Fiyat: {stock['price']:.2f}TL - Hacim: {stock['volume']:,.0f}")
                
                # Güçlü destek ve direnç bilgileri detaylı göster
                if stock['supports_with_strength']:
                    print("   Güçlü Destekler:")
                    for i, (support, strength) in enumerate(stock['supports_with_strength'][:3]):
                        distance = stock['support_distances'][i]
                        strength_text = format_support_strength(strength)
                        print(f"     {i+1}. {support:.2f}TL (Uzaklık: %{distance:.1f}) - {strength_text}")
                
                if stock['resistances_with_strength']:
                    print("   Güçlü Dirençler:")
                    for i, (resistance, strength) in enumerate(stock['resistances_with_strength'][:3]):
                        distance = stock['resistance_distances'][i]
                        strength_text = format_support_strength(strength)
                        print(f"     {i+1}. {resistance:.2f}TL (Uzaklık: %{distance:.1f}) - {strength_text}")
                
                print(f"   Hacim Artışı: %{(stock['volume_increase']-1)*100:.1f}")
                print(f"   ATR: %{stock['atr_percent']:.1f}")
                
                reasons = explain_why_not_matching(stock)
                print("   Uyumsuzluk Sebepleri:")
                for reason in reasons:
                    print(f"     • {reason}")
    
    else:
        # BIST100 araması - sadece uygun olanları göster
        if filtered_results:
            print(f"{'='*100}")
            print(f"KRİTERLERE UYGUN HİSSELER ({len(filtered_results)} adet)")
            print(f"{'='*100}")
            
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
    print(f"\n{'='*80}")
    print(f"GÜNCEL FİLTRE KRİTERLERİ - İYİLEŞTİRİLMİŞ VERSİYON")
    print(f"{'='*80}")
    print(f"🟢 OLUMLU NOKTALAR (Aynen korundu):")
    print(f"   • RSI: {MIN_RSI}-{MAX_RSI} (>{MAX_RSI} ise alma)")
    print(f"   • MACD: Çizgi sinyal çizgisini aşağıdan yukarı kesmiş olmalı")
    print(f"   • MACD Histogram: Pozitif olmalı")
    print(f"   • EMA: EMA20 > EMA50")
    print(f"   • ATR: %{MIN_ATR_PERCENT}-{MAX_ATR_PERCENT} günlük hareket")
    print(f"   • Fiyat Aralığı: {MIN_PRICE}-{MAX_PRICE} TL")
    
    print(f"\n🟡 İYİLEŞTİRİLEN NOKTALAR:")
    print(f"   • Minimum Hacim: {MIN_VOLUME:,} (150k'dan artırıldı - likidite koruması)")
    print(f"   • Hacim Artışı: Son {VOLUME_LOOKBACK_DAYS} günlük ortalama ile karşılaştırma")
    print(f"   • Fiyat-EMA20: Maksimum %{MAX_PRICE_EMA20_DISTANCE*100:.0f} uzaklık (5%'den düşürüldü)")
    print(f"   • Destek/Direnç: {SUPPORT_RESISTANCE_COUNT} seviye + güç analizi")
    
    print(f"\n🔴 KRİTİK DENGE NOKTALARI:")
    print(f"   • Destek Uzaklığı: Maksimum %{MAX_SUPPORT_DISTANCE*100:.0f}")
    print(f"   • Stop-loss: Maksimum %{MAX_STOP_LOSS_DISTANCE*100:.0f} (gerçekçi seviye)")
    print(f"   • Direnç Potansiyeli: Maksimum %{MAX_RESISTANCE_DISTANCE*100:.0f} (8%'den düşürüldü)")
    
    print(f"\n📊 DESTEK GÜÇ SEVİYELERİ:")
    print(f"   🔴 Çok Güçlü (4-5 puan) | 🟠 Güçlü (3-4 puan)")
    print(f"   🟡 Orta (2-3 puan) | 🟢 Zayıf (1-2 puan) | ⚪ Çok Zayıf (0-1 puan)")

def main():
    """Ana program"""
    print("🚀 BIST Gelişmiş Filtreli Hisse Tarayıcısı v3.0 - İYİLEŞTİRİLMİŞ")
    print("="*80)

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