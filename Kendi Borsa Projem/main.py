import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL FİLTRE KRİTERLERİ - İSTEDİĞİNİZ GİBİ DEĞİŞTİREBİLİRSİNİZ
# =============================================================================

# RSI Kriterleri
MIN_RSI = 30          # Minimum RSI değeri (örn: 30 = aşırı satımdan çıkanlar)
MAX_RSI = 70          # Maksimum RSI değeri (örn: 70 = aşırı alıma girmeyenler)

# MACD Kriterleri
MACD_POSITIVE = True   # True: MACD pozitif olsun, False: negatif olsun, None: fark etmez
MACD_HISTOGRAM_POSITIVE = True  # True: MACD histogram pozitif, False: negatif, None: fark etmez

# Parabolik SAR Kriterleri
PRICE_ABOVE_SAR = True  # True: Fiyat SAR üstünde, False: SAR altında, None: fark etmez
SAR_TREND_UP = True     # True: Yükseliş trendi, False: Düşüş trendi, None: fark etmez

# Hareketli Ortalama Kriterleri
PRICE_ABOVE_MA20 = True   # True: Fiyat MA20 üstünde, False: altında, None: fark etmez
PRICE_ABOVE_MA50 = None   # True: Fiyat MA50 üstünde, False: altında, None: fark etmez
PRICE_ABOVE_MA200 = None  # True: Fiyat MA200 üstünde, False: altında, None: fark etmez

# Hacim Kriterleri
MIN_VOLUME = 100000     # Minimum günlük hacim
MIN_VOLUME_RATIO = 0.5  # Minimum hacim oranı (güncel/20günlük ortalama)
MAX_VOLUME_RATIO = 10.0 # Maksimum hacim oranı (anormal hacim artışını filtrele)

# Fiyat Kriterleri
MIN_PRICE = 1.0       # Minimum hisse fiyatı
MAX_PRICE = 1000.0    # Maksimum hisse fiyatı

# Bollinger Bands Kriterleri
BB_POSITION_MIN = 0.0   # Bollinger Bands içindeki minimum pozisyon (0-100)
BB_POSITION_MAX = 100.0 # Bollinger Bands içindeki maksimum pozisyon (0-100)

# Stochastic Kriterleri
MIN_STOCH_K = 20      # Minimum Stochastic %K değeri
MAX_STOCH_K = 80      # Maksimum Stochastic %K değeri

# Volatilite Kriterleri
MIN_VOLATILITY = 0.0  # Minimum 20 günlük volatilite (%)
MAX_VOLATILITY = 50.0 # Maksimum 20 günlük volatilite (%)

# =============================================================================

# BIST hisse kodları (genişletilmiş liste)
BIST_STOCKS = [
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
    'BOBET', 'BOLUC', 'BOSSA', 'BRKO', 'BRKSN', 'BRKVY', 'BSOKE', 'BTCIM',
    'BUCIM', 'BURCE', 'BURVA', 'CCOLA', 'CEMAS', 'CEMTS', 'CIMSA', 'CLEBI',
    'CMBTN', 'CMENT', 'CONSE', 'COSMO', 'CRDFA', 'CRFSA', 'CUSAN', 'CVKMD',
    'CWENE', 'DAGI', 'DAPGM', 'DARDL', 'DENGE', 'DERHL', 'DERIM', 'DESA'
]

def calculate_rsi(prices, period=14):
    """RSI hesaplama fonksiyonu"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD hesaplama fonksiyonu"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Parabolik SAR hesaplama fonksiyonu"""
    length = len(close)
    sar = np.zeros(length)
    trend = np.zeros(length)
    af = np.zeros(length)
    ep = np.zeros(length)
    
    sar[0] = low.iloc[0]
    trend[0] = 1
    af[0] = af_start
    ep[0] = high.iloc[0]
    
    for i in range(1, length):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if low.iloc[i] <= sar[i]:
                trend[i] = -1
                af[i] = af_start
                sar[i] = ep[i-1]
                ep[i] = low.iloc[i]
            else:
                trend[i] = 1
                if high.iloc[i] > ep[i-1]:
                    af[i] = min(af[i-1] + af_increment, af_max)
                    ep[i] = high.iloc[i]
                else:
                    af[i] = af[i-1]
                    ep[i] = ep[i-1]
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if high.iloc[i] >= sar[i]:
                trend[i] = 1
                af[i] = af_start
                sar[i] = ep[i-1]
                ep[i] = high.iloc[i]
            else:
                trend[i] = -1
                if low.iloc[i] < ep[i-1]:
                    af[i] = min(af[i-1] + af_increment, af_max)
                    ep[i] = low.iloc[i]
                else:
                    af[i] = af[i-1]
                    ep[i] = ep[i-1]
    
    return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index)

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands hesaplama"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator hesaplama"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_drawdown(prices):
    """Geri çekilme oranını hesapla (drawdown %)"""
    max_price = prices.cummax()
    drawdown = ((prices - max_price) / max_price) * 100
    return drawdown.iloc[-1]

def calculate_support_resistance(prices):
    """Destek ve direnç seviyelerini hesapla"""
    support = prices.rolling(window=20).min().iloc[-1]
    resistance = prices.rolling(window=20).max().iloc[-1]
    return support, resistance

def analyze_stock_comprehensive(ticker):
    """Kapsamlı hisse analizi (geliştirilmiş)"""
    try:
        bist_ticker = ticker.strip().upper()
        if not bist_ticker.endswith('.IS'):
            bist_ticker += '.IS'
        
        stock = yf.Ticker(bist_ticker)
        hist = stock.history(period="6mo")  # 6 aylık veri
        
        if len(hist) < 50:
            return None
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        # Temel veriler
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume_20
        
        # Hareketli ortalamalar
        ma_20 = close.rolling(window=20).mean().iloc[-1]
        ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None
        
        # RSI
        rsi = calculate_rsi(close).iloc[-1]
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        current_macd = macd_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Parabolik SAR
        sar_values, sar_trend = calculate_parabolic_sar(high, low, close)
        current_sar = sar_values.iloc[-1]
        current_trend = sar_trend.iloc[-1]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        bb_position = ((current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)) * 100
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        current_stoch_k = stoch_k.iloc[-1]
        
        # Volatilite (20 günlük)
        returns = close.pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100  # Yıllık volatilite %
        
        # Geri Çekilme Oranı ve Destek/Direnç
        drawdown = calculate_drawdown(close)
        support, resistance = calculate_support_resistance(close)
        
        return {
            'ticker': ticker.upper(),
            'price': current_price,
            'volume': current_volume,
            'volume_ratio': volume_ratio,
            'rsi': rsi,
            'macd': current_macd,
            'macd_hist': current_histogram,
            'sar': current_sar,
            'sar_trend': current_trend,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'bb_position': bb_position,
            'stoch_k': current_stoch_k,
            'volatility': volatility,
            'drawdown': drawdown,
            'support': support,
            'resistance': resistance
        }
        
    except Exception as e:
        return None

def check_filters(stock):
    """Global filtrelere göre hisse kontrolü"""
    if not stock:
        return False
    
    # Fiyat filtreleri
    if not (MIN_PRICE <= stock['price'] <= MAX_PRICE):
        return False
    
    # RSI filtreleri
    if not (MIN_RSI <= stock['rsi'] <= MAX_RSI):
        return False
    
    # MACD filtreleri
    if MACD_POSITIVE is not None:
        if MACD_POSITIVE and stock['macd'] <= 0:
            return False
        if not MACD_POSITIVE and stock['macd'] >= 0:
            return False
    
    if MACD_HISTOGRAM_POSITIVE is not None:
        if MACD_HISTOGRAM_POSITIVE and stock['macd_hist'] <= 0:
            return False
        if not MACD_HISTOGRAM_POSITIVE and stock['macd_hist'] >= 0:
            return False
    
    # Parabolik SAR filtreleri
    if PRICE_ABOVE_SAR is not None:
        price_above_sar = stock['price'] > stock['sar']
        if PRICE_ABOVE_SAR != price_above_sar:
            return False
    
    if SAR_TREND_UP is not None:
        trend_up = stock['sar_trend'] == 1
        if SAR_TREND_UP != trend_up:
            return False
    
    # Hareketli ortalama filtreleri
    if PRICE_ABOVE_MA20 is not None:
        price_above_ma20 = stock['price'] > stock['ma_20']
        if PRICE_ABOVE_MA20 != price_above_ma20:
            return False
    
    if PRICE_ABOVE_MA50 is not None and stock['ma_50']:
        price_above_ma50 = stock['price'] > stock['ma_50']
        if PRICE_ABOVE_MA50 != price_above_ma50:
            return False
    
    if PRICE_ABOVE_MA200 is not None and stock['ma_200']:
        price_above_ma200 = stock['price'] > stock['ma_200']
        if PRICE_ABOVE_MA200 != price_above_ma200:
            return False
    
    # Hacim filtreleri
    if stock['volume'] < MIN_VOLUME:
        return False
    
    if not (MIN_VOLUME_RATIO <= stock['volume_ratio'] <= MAX_VOLUME_RATIO):
        return False
    
    # Bollinger Bands filtreleri
    if not (BB_POSITION_MIN <= stock['bb_position'] <= BB_POSITION_MAX):
        return False
    
    # Stochastic filtreleri
    if not (MIN_STOCH_K <= stock['stoch_k'] <= MAX_STOCH_K):
        return False
    
    # Volatilite filtreleri
    if not (MIN_VOLATILITY <= stock['volatility'] <= MAX_VOLATILITY):
        return False
    
    return True

def scan_and_filter_stocks(selected_stocks=None):
    """Seçilen hisseleri veya tüm hisseleri tara ve filtrele"""
    stocks_to_scan = selected_stocks if selected_stocks else BIST_STOCKS
    print(f"🔍 {'Seçilen' if selected_stocks else 'Tüm'} hisseler taranıyor ve filtreleniyor...")
    print("Bu işlem birkaç dakika sürebilir...\n")
    
    all_results = []
    filtered_results = []
    processed = 0
    
    for ticker in stocks_to_scan:  # Corrected indentation
        processed += 1
        print(f"İşleniyor: {ticker} ({processed}/{len(stocks_to_scan)})", end='\r')
        
        result = analyze_stock_comprehensive(ticker)
        if result:
            all_results.append(result)
            if check_filters(result):
                filtered_results.append(result)
    
    print(f"\n✅ Toplam {len(all_results)} hisse analiz edildi.")
    print(f"🎯 {len(filtered_results)} hisse kriterlere uygun bulundu.")
    
    return filtered_results, all_results

def calculate_proximity_score(stock):
    """Hissenin kriterlere ne kadar yakın olduğunu hesapla"""
    score = 0
    max_score = 0
    
    # RSI skorunu hesapla
    max_score += 1
    if MIN_RSI <= stock['rsi'] <= MAX_RSI:
        score += 1
    else:
        # RSI aralığına ne kadar yakın olduğuna göre kısmi puan
        if stock['rsi'] < MIN_RSI:
            distance = MIN_RSI - stock['rsi']
            score += max(0, 1 - distance/20)  # 20 puan fark için lineer azalış
        else:
            distance = stock['rsi'] - MAX_RSI
            score += max(0, 1 - distance/20)
    
    # MACD skorları
    if MACD_POSITIVE is not None:
        max_score += 1
        if (MACD_POSITIVE and stock['macd'] > 0) or (not MACD_POSITIVE and stock['macd'] <= 0):
            score += 1
        else:
            score += 0.5  # Kısmi puan
    
    if MACD_HISTOGRAM_POSITIVE is not None:
        max_score += 1
        if (MACD_HISTOGRAM_POSITIVE and stock['macd_hist'] > 0) or (not MACD_HISTOGRAM_POSITIVE and stock['macd_hist'] <= 0):
            score += 1
        else:
            score += 0.5
    
    # SAR skorları
    if PRICE_ABOVE_SAR is not None:
        max_score += 1
        price_above_sar = stock['price'] > stock['sar']
        if PRICE_ABOVE_SAR == price_above_sar:
            score += 1
        else:
            score += 0.3
    
    if SAR_TREND_UP is not None:
        max_score += 1
        trend_up = stock['sar_trend'] == 1
        if SAR_TREND_UP == trend_up:
            score += 1
        else:
            score += 0.3
    
    # MA skorları
    if PRICE_ABOVE_MA20 is not None:
        max_score += 1
        price_above_ma20 = stock['price'] > stock['ma_20']
        if PRICE_ABOVE_MA20 == price_above_ma20:
            score += 1
        else:
            score += 0.4
    
    if PRICE_ABOVE_MA50 is not None and stock['ma_50']:
        max_score += 1
        price_above_ma50 = stock['price'] > stock['ma_50']
        if PRICE_ABOVE_MA50 == price_above_ma50:
            score += 1
        else:
            score += 0.4
    
    # Fiyat aralığı skoru
    max_score += 1
    if MIN_PRICE <= stock['price'] <= MAX_PRICE:
        score += 1
    else:
        score += 0.2
    
    # Hacim skoru
    max_score += 1
    if stock['volume'] >= MIN_VOLUME and MIN_VOLUME_RATIO <= stock['volume_ratio'] <= MAX_VOLUME_RATIO:
        score += 1
    else:
        # Hacim kriterlerine yakınlık
        volume_score = 0.5 if stock['volume'] >= MIN_VOLUME/2 else 0.2
        ratio_score = 0.5 if MIN_VOLUME_RATIO/2 <= stock['volume_ratio'] <= MAX_VOLUME_RATIO*2 else 0.2
        score += max(volume_score, ratio_score)
    
    # Bollinger Bands skoru
    max_score += 1
    if BB_POSITION_MIN <= stock['bb_position'] <= BB_POSITION_MAX:
        score += 1
    else:
        score += 0.3
    
    # Stochastic skoru
    max_score += 1
    if MIN_STOCH_K <= stock['stoch_k'] <= MAX_STOCH_K:
        score += 1
    else:
        score += 0.3
    
    # Volatilite skoru
    max_score += 1
    if MIN_VOLATILITY <= stock['volatility'] <= MAX_VOLATILITY:
        score += 1
    else:
        score += 0.3
    
    return score / max_score if max_score > 0 else 0

def determine_recommendation(stock):
    """Hisse için öneri belirle (geliştirilmiş)"""
    if stock['rsi'] > 80 or stock['stoch_k'] > 90 or stock['bb_position'] > 90:
        return "Kar Satışlarına Dikkat"
    elif stock['rsi'] < 30 and stock['macd_hist'] > 0 and stock['price'] > stock['ma_20']:
        return "Güçlü Al"
    elif stock['rsi'] > 70 and stock['macd_hist'] < 0 and stock['price'] < stock['ma_20']:
        return "Güçlü Sat"
    elif stock['volatility'] > 70:
        return "Kısa Vadede Riskli"
    else:
        return "Tut"

def determine_investment_horizon(stock):
    """Hisse için net bir vade değerlendirmesi yap"""
    if stock['ma_200'] and stock['price'] > stock['ma_200']:
        if stock['price'] > stock['ma_50']:
            return "Uzun Vade (Güçlü)"
        return "Uzun Vade"
    elif stock['ma_50'] and stock['price'] > stock['ma_50']:
        return "Orta Vade"
    else:
        return "Kısa Vade (Riskli)"

def display_filtered_results(results, all_results):
    """Filtrelenmiş sonuçları ve kriterlere uymayanları göster"""
    if not results:
        print("\n❌ Kriterlere uygun hisse bulunamadı!")
    
    # Kriterlere uygun olanları göster
    if results:
        sorted_results = sorted(results, key=lambda x: x['rsi'], reverse=True)
        print(f"\n{'='*180}")
        print(f"KRİTERLERE UYGUN HİSSELER ({len(results)} adet)")
        print(f"{'='*180}")
        print(f"{'Kod':<6} {'Fiyat':<8} {'RSI':<6} {'MACD':<8} {'SAR':<8} {'Trend':<6} {'MA20':<6} {'BB%':<6} {'Stoch':<6} {'Vol%':<6} {'Hacim':<13} {'Draw%':<8} {'Destek':<8} {'Direnç':<8} {'Öneri':<20} {'Vade':<10}")
        print(f"{'-'*180}")
        
        for stock in sorted_results:
            trend_text = "Yük" if stock['sar_trend'] == 1 else "Düş"
            ma20_status = "Üst" if stock['price'] > stock['ma_20'] else "Alt"
            recommendation = determine_recommendation(stock)
            horizon = determine_investment_horizon(stock)
            
            print(f"{stock['ticker']:<6} {stock['price']:<8.2f} {stock['rsi']:<6.1f} "
                  f"{stock['macd']:<8.4f} {stock['sar']:<8.2f} {trend_text:<6} "
                  f"{ma20_status:<6} {stock['bb_position']:<6.1f} {stock['stoch_k']:<6.1f} "
                  f"{stock['volatility']:<6.1f} {stock['volume']:>12,.0f} {stock['drawdown']:<8.1f} "
                  f"{stock['support']:<8.2f} {stock['resistance']:<8.2f} {recommendation:<20} {horizon:<10}")
    
    # Kriterlere uymayanları göster
    non_matching_results = [stock for stock in all_results if stock not in results]
    if non_matching_results:
        print(f"\n{'='*180}")
        print(f"KRİTERLERE UYMAYAN HİSSELER ({len(non_matching_results)} adet)")
        print(f"{'='*180}")
        print(f"{'Kod':<6} {'Fiyat':<8} {'RSI':<6} {'MACD':<8} {'SAR':<8} {'Trend':<6} {'MA20':<6} {'BB%':<6} {'Stoch':<6} {'Vol%':<6} {'Hacim':<13} {'Draw%':<8} {'Destek':<8} {'Direnç':<8} {'Öneri':<20} {'Vade':<10}")
        print(f"{'-'*180}")
        
        for stock in non_matching_results:
            trend_text = "Yük" if stock['sar_trend'] == 1 else "Düş"
            ma20_status = "Üst" if stock['price'] > stock['ma_20'] else "Alt"
            recommendation = determine_recommendation(stock)
            horizon = determine_investment_horizon(stock)
            
            print(f"{stock['ticker']:<6} {stock['price']:<8.2f} {stock['rsi']:<6.1f} "
                  f"{stock['macd']:<8.4f} {stock['sar']:<8.2f} {trend_text:<6} "
                  f"{ma20_status:<6} {stock['bb_position']:<6.1f} {stock['stoch_k']:<6.1f} "
                  f"{stock['volatility']:<6.1f} {stock['volume']:>12,.0f} {stock['drawdown']:<8.1f} "
                  f"{stock['support']:<8.2f} {stock['resistance']:<8.2f} {recommendation:<20} {horizon:<10}")

def show_current_filters():
    """Mevcut filtreleri göster"""
    print(f"\n{'='*60}")
    print(f"MEVCUT FİLTRE KRİTERLERİ")
    print(f"{'='*60}")
    print(f"RSI: {MIN_RSI} - {MAX_RSI}")
    print(f"MACD Pozitif: {MACD_POSITIVE}")
    print(f"MACD Histogram Pozitif: {MACD_HISTOGRAM_POSITIVE}")
    print(f"Fiyat SAR Üstünde: {PRICE_ABOVE_SAR}")
    print(f"SAR Trend Yukarı: {SAR_TREND_UP}")
    print(f"Fiyat MA20 Üstünde: {PRICE_ABOVE_MA20}")
    print(f"Fiyat MA50 Üstünde: {PRICE_ABOVE_MA50}")
    print(f"Fiyat MA200 Üstünde: {PRICE_ABOVE_MA200}")
    print(f"Fiyat Aralığı: {MIN_PRICE} - {MAX_PRICE} TL")
    print(f"Hacim: Min {MIN_VOLUME:,}")
    print(f"Hacim Oranı: {MIN_VOLUME_RATIO} - {MAX_VOLUME_RATIO}")
    print(f"Bollinger Position: {BB_POSITION_MIN}% - {BB_POSITION_MAX}%")
    print(f"Stochastic K: {MIN_STOCH_K} - {MAX_STOCH_K}")
    print(f"Volatilite: {MIN_VOLATILITY}% - {MAX_VOLATILITY}%")

def main():
    """Ana program"""
    print("🚀 BIST Gelişmiş Filtreli Hisse Tarayıcısı")
    print("="*50)
    
    show_current_filters()
    
    print(f"\n💡 İPUCU: Kodun başındaki global değişkenleri değiştirerek")
    print(f"    filtreleri özelleştirebilirsiniz!")
    
    choice = input(f"\nTüm hisseleri taramak için 't', belirli hisseleri taramak için 'b' seçin (t/b): ").lower()
    
    if choice == 'b':
        selected_stocks = input("Lütfen hisse kodlarını virgülle ayırarak girin (örn: THYAO,AKBNK): ").split(',')
        selected_stocks = [stock.strip().upper() for stock in selected_stocks]
        filtered_results, all_results = scan_and_filter_stocks(selected_stocks)
    elif choice == 't':
        filtered_results, all_results = scan_and_filter_stocks()
    else:
        print("Geçersiz seçim! Program sonlandırılıyor.")
        return
    
    display_filtered_results(filtered_results, all_results)
    
    if filtered_results:
        print(f"\n📊 ÖRNEK ANALİZ:")
        print(f"En yüksek RSI: {max(filtered_results, key=lambda x: x['rsi'])['ticker']} "
              f"(RSI: {max(filtered_results, key=lambda x: x['rsi'])['rsi']:.1f})")
        print(f"En yüksek hacim: {max(filtered_results, key=lambda x: x['volume'])['ticker']} "
              f"(Hacim: {max(filtered_results, key=lambda x: x['volume'])['volume']:,.0f})")
    else:
        print("Kriterlere uygun hisse bulunamadı.")

if __name__ == "__main__":
    main()