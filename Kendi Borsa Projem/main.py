import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# BIST 100 hisse kodları (en likit hisseler)
BIST_STOCKS = [
    'THYAO', 'AKBNK', 'ISCTR', 'GARAN', 'VAKBN', 'SASA', 'KCHOL', 'ARCLK', 
    'TUPRS', 'EREGL', 'HALKB', 'TCELL', 'BIMAS', 'SAHOL', 'ASELS', 'KOZAL',
    'PGSUS', 'MGROS', 'SOKM', 'ENKAI', 'OYAKC', 'TOASO', 'PETKM', 'TKFEN',
    'SISE', 'OTKAR', 'AEFES', 'CCOLA', 'ULKER', 'VESTL', 'FROTO', 'GUBRF',
    'ALARK', 'TTKOM', 'KOZAA', 'TAVHL', 'DOHOL', 'ECILC', 'YKBNK', 'ZOREN',
    'LOGO', 'TSKB', 'AKSA', 'AKSEN', 'BRISA', 'CEMTS', 'CIMSA', 'EGEEN',
    'ENJSA', 'FENER', 'GLYHO', 'HEKTS', 'IHLAS', 'IHYAY', 'IZMDC', 'KARSN',
    'KAYSE', 'KLMSN', 'KONYA', 'KRONT', 'MAVI', 'MPARK', 'NETAS', 'OTKAR',
    'PARSN', 'PINSU', 'PRKME', 'RAYSG', 'RTALB', 'SARKY', 'SELEC', 'SMART',
    'SMRTG', 'TATEN', 'TBORG', 'TEZOL', 'TIRE', 'TOASO', 'TRCAS', 'TURSG',
    'UFUK', 'ULUSE', 'UNYEC', 'VAKKO', 'VESBE', 'YAPRK', 'YESIL', 'YGYO',
    'ZEDUR', 'ADEL', 'AGHOL', 'AGROT', 'AHGAZ', 'AKGRT', 'ALCTL', 'ALGYO',
    'ALKIM', 'ALTIN', 'ANSGR', 'ARDYZ', 'ASTOR', 'AVGYO', 'AVHOL', 'AVTUR',
    'AYCES', 'AYEN', 'AYGAZ', 'BAKAB', 'BANVT', 'BASGZ', 'BAYRK', 'BEGYO',
    'BERA', 'BEYAZ', 'BFREN', 'BIGYO', 'BIOEN', 'BJKAS', 'BLCYT', 'BMSCH',
    'BMSTL', 'BNTAS', 'BOBET', 'BOLUC', 'BOSSA', 'BRKO', 'BRKSN', 'BRKVY'
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
    
    # İlk değerler
    sar[0] = low.iloc[0]
    trend[0] = 1  # 1 = yükseliş, -1 = düşüş
    af[0] = af_start
    ep[0] = high.iloc[0]
    
    for i in range(1, length):
        if trend[i-1] == 1:  # Yükseliş trendi
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
        else:  # Düşüş trendi
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

def analyze_stock_quick(ticker):
    """Hızlı hisse analizi"""
    try:
        # BIST kodları için .IS ekleme
        bist_ticker = ticker.strip().upper()
        if not bist_ticker.endswith('.IS'):
            bist_ticker += '.IS'
        
        # Veri çekme (3 aylık)
        stock = yf.Ticker(bist_ticker)
        hist = stock.history(period="3mo")
        
        if len(hist) < 30:
            return None
        
        # Temel veriler
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        # Son değerler
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        
        # Teknik göstergeler
        rsi = calculate_rsi(close).iloc[-1]
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        current_macd = macd_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Parabolik SAR
        sar_values, sar_trend = calculate_parabolic_sar(high, low, close)
        current_sar = sar_values.iloc[-1]
        current_trend = sar_trend.iloc[-1]
        
        # MA20
        ma_20 = close.rolling(window=20).mean().iloc[-1]
        
        return {
            'ticker': ticker.upper(),
            'price': current_price,
            'volume': current_volume,
            'avg_volume': avg_volume_20,
            'volume_ratio': current_volume / avg_volume_20,
            'rsi': rsi,
            'macd': current_macd,
            'macd_hist': current_histogram,
            'sar': current_sar,
            'trend': 'Yükseliş' if current_trend == 1 else 'Düşüş',
            'ma20': ma_20,
            'price_vs_ma20': 'Üstünde' if current_price > ma_20 else 'Altında',
            'price_vs_sar': 'Üstünde' if current_price > current_sar else 'Altında'
        }
        
    except Exception as e:
        return None

def scan_all_bist_stocks():
    """BIST'teki tüm hisseleri tara"""
    print("🔍 BIST hisseleri taranıyor...")
    print("Bu işlem birkaç dakika sürebilir...\n")
    
    results = []
    processed = 0
    
    for ticker in BIST_STOCKS:
        processed += 1
        print(f"İşleniyor: {ticker} ({processed}/{len(BIST_STOCKS)})", end='\r')
        
        result = analyze_stock_quick(ticker)
        if result:
            results.append(result)
    
    print(f"\n✅ {len(results)} hisse başarıyla analiz edildi.")
    return results

def display_stocks_by_volume(results):
    """Hisseleri hacme göre sıralayıp göster"""
    # Hacme göre sırala (büyükten küçüğe)
    sorted_results = sorted(results, key=lambda x: x['volume'], reverse=True)
    
    print(f"\n{'='*120}")
    print(f"BIST HİSSELERİ - HACİM SIRALAMASINA GÖRE")
    print(f"{'='*120}")
    print(f"{'Sıra':<4} {'Kod':<6} {'Fiyat':<8} {'Hacim':<12} {'H.Oran':<6} {'RSI':<6} {'MACD':<8} {'SAR':<8} {'Trend':<8} {'MA20':<8}")
    print(f"{'-'*120}")
    
    for i, stock in enumerate(sorted_results, 1):
        # RSI durumu
        rsi_status = ""
        if stock['rsi'] > 70:
            rsi_status = "🔴"  # Aşırı alım
        elif stock['rsi'] < 30:
            rsi_status = "🟢"  # Aşırı satım
        else:
            rsi_status = "⚪"  # Nötr
        
        # MACD durumu
        macd_status = "📈" if stock['macd_hist'] > 0 else "📉"
        
        # Trend durumu
        trend_emoji = "📈" if stock['trend'] == 'Yükseliş' else "📉"
        
        print(f"{i:<4} {stock['ticker']:<6} {stock['price']:<8.2f} "
              f"{stock['volume']:>11,.0f} {stock['volume_ratio']:<6.1f} "
              f"{stock['rsi']:<6.1f}{rsi_status} {stock['macd']:<8.4f}{macd_status} "
              f"{stock['sar']:<8.2f} {stock['trend']:<8}{trend_emoji} "
              f"{stock['price_vs_ma20']:<8}")
        
        # Her 10 hissede bir boşluk
        if i % 10 == 0:
            print()

def display_detailed_analysis(results, count=20):
    """En yüksek hacimli hisseler için detaylı analiz"""
    sorted_results = sorted(results, key=lambda x: x['volume'], reverse=True)
    top_stocks = sorted_results[:count]
    
    print(f"\n{'='*80}")
    print(f"EN YÜKSEK HACİMLİ {count} HİSSE - DETAYLI ANALİZ")
    print(f"{'='*80}")
    
    for i, stock in enumerate(top_stocks, 1):
        print(f"\n{i}. {stock['ticker']} - {stock['price']:.2f} TL")
        print(f"   📊 Hacim: {stock['volume']:,.0f} (Ort: {stock['avg_volume']:,.0f}, Oran: {stock['volume_ratio']:.1f}x)")
        
        # RSI analizi
        rsi_desc = ""
        if stock['rsi'] > 70:
            rsi_desc = "(Aşırı Alım ⚠️)"
        elif stock['rsi'] < 30:
            rsi_desc = "(Aşırı Satım 🎯)"
        elif stock['rsi'] > 60:
            rsi_desc = "(Güçlü 💪)"
        elif stock['rsi'] < 40:
            rsi_desc = "(Zayıf 📉)"
        else:
            rsi_desc = "(Nötr ⚪)"
        
        print(f"   🎯 RSI: {stock['rsi']:.1f} {rsi_desc}")
        
        # MACD analizi
        macd_desc = "Pozitif Momentum 📈" if stock['macd_hist'] > 0 else "Negatif Momentum 📉"
        print(f"   📈 MACD: {stock['macd']:.4f} ({macd_desc})")
        
        # SAR analizi
        sar_desc = "Yükseliş Trendi" if stock['trend'] == 'Yükseliş' else "Düşüş Trendi"
        print(f"   🎪 SAR: {stock['sar']:.2f} ({sar_desc}, Fiyat {stock['price_vs_sar']})")
        
        # MA20 analizi
        print(f"   🔄 MA20: Fiyat {stock['price_vs_ma20']}")

def analyze_custom_stocks():
    """Kullanıcının girdiği hisseleri analiz et"""
    ticker_input = input("Analiz etmek istediğiniz hisse kodlarını girin (virgülle ayırın): ")
    ticker_symbols = [t.strip() for t in ticker_input.split(',')]
    
    results = []
    for ticker in ticker_symbols:
        print(f"Analiz ediliyor: {ticker}")
        result = analyze_stock_quick(ticker)
        if result:
            results.append(result)
    
    if results:
        display_detailed_analysis(results, len(results))
    else:
        print("Hiçbir hisse analiz edilemedi.")

def main():
    """Ana program"""
    print("🚀 BIST Kapsamlı Hisse Analiz Programı")
    print("="*50)
    
    choice = input("Seçiminizi yapın:\n1. Tüm BIST hisselerini tara\n2. Belirli hisseleri analiz et\nSeçim (1 veya 2): ")
    
    if choice == "1":
        # Tüm BIST hisselerini tara
        results = scan_all_bist_stocks()
        
        if results:
            # Hacme göre sıralı liste
            display_stocks_by_volume(results)
            
            # En yüksek hacimli 20 hisse için detaylı analiz
            print(f"\n" + "="*80)
            detail_choice = input("En yüksek hacimli hisseler için detaylı analiz görmek ister misiniz? (e/h): ")
            if detail_choice.lower() == 'e':
                count = input("Kaç tane hisse için detaylı analiz? (varsayılan: 20): ")
                try:
                    count = int(count) if count else 20
                except:
                    count = 20
                display_detailed_analysis(results, count)
        else:
            print("Hiçbir hisse analiz edilemedi.")
    
    elif choice == "2":
        # Belirli hisseleri analiz et
        analyze_custom_stocks()
    
    else:
        print("Geçersiz seçim!")

if __name__ == "__main__":
    main()