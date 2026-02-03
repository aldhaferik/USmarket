#!/usr/bin/env python3
"""
Enhanced Technical Analysis Module
- Interactive technical indicators
- Pattern recognition algorithms
- Support/resistance level detection
- Trading signals generation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import talib
from datetime import datetime, timedelta

class EnhancedTechnicalAnalysis:
    def __init__(self):
        self.patterns = {}
        self.signals = {}
        self.support_resistance = {}
        
    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            # Ensure we have OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                st.error("❌ Missing required OHLCV data")
                return None
            
            # Convert to numpy arrays for TA-Lib
            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            volume = df['Volume'].values
            
            indicators = df.copy()
            
            # Trend Indicators
            indicators['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
            indicators['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
            indicators['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
            indicators['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
            indicators['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # MACD
            indicators['MACD'], indicators['MACD_Signal'], indicators['MACD_Hist'] = talib.MACD(close_prices)
            
            # Bollinger Bands
            indicators['BB_Upper'], indicators['BB_Middle'], indicators['BB_Lower'] = talib.BBANDS(close_prices)
            indicators['BB_Width'] = indicators['BB_Upper'] - indicators['BB_Lower']
            indicators['BB_Position'] = (close_prices - indicators['BB_Lower']) / indicators['BB_Width']
            
            # Momentum Indicators
            indicators['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
            indicators['RSI_21'] = talib.RSI(close_prices, timeperiod=21)
            
            # Stochastic
            indicators['Stoch_K'], indicators['Stoch_D'] = talib.STOCH(high_prices, low_prices, close_prices)
            
            # Williams %R
            indicators['Williams_R'] = talib.WILLR(high_prices, low_prices, close_prices)
            
            # CCI (Commodity Channel Index)
            indicators['CCI'] = talib.CCI(high_prices, low_prices, close_prices)
            
            # ADX (Average Directional Index)
            indicators['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
            indicators['DI_Plus'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
            indicators['DI_Minus'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
            
            # Volume Indicators
            indicators['OBV'] = talib.OBV(close_prices, volume)
            indicators['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
            indicators['CHAIKIN'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
            
            # Volatility Indicators
            indicators['ATR'] = talib.ATR(high_prices, low_prices, close_prices)
            indicators['NATR'] = talib.NATR(high_prices, low_prices, close_prices)
            
            # Price Patterns
            indicators['DOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            indicators['HAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            indicators['ENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            indicators['MORNING_STAR'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            indicators['EVENING_STAR'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            # Custom Indicators
            indicators['Price_Change'] = close_prices - np.roll(close_prices, 1)
            indicators['Price_Change_Pct'] = (close_prices / np.roll(close_prices, 1) - 1) * 100
            
            # Trend Strength
            indicators['Trend_Strength'] = self.calculate_trend_strength(close_prices)
            
            # Support/Resistance Levels
            support_levels, resistance_levels = self.find_support_resistance(df)
            indicators['Support_Level'] = self.map_levels_to_dates(df, support_levels)
            indicators['Resistance_Level'] = self.map_levels_to_dates(df, resistance_levels)
            
            return indicators
            
        except Exception as e:
            st.error(f"❌ Error calculating technical indicators: {e}")
            return None
    
    def calculate_trend_strength(self, prices, window=20):
        """Calculate trend strength using linear regression"""
        trend_strength = np.full(len(prices), np.nan)
        
        for i in range(window, len(prices)):
            y = prices[i-window:i]
            x = np.arange(len(y))
            slope, _, r_value, _, _ = linregress(x, y)
            trend_strength[i] = r_value ** 2 * np.sign(slope)
        
        return trend_strength
    
    def find_support_resistance(self, df, window=20, min_touches=2):
        """Find support and resistance levels"""
        try:
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            
            # Find local maxima and minima
            resistance_indices = argrelextrema(high_prices, np.greater, order=window)[0]
            support_indices = argrelextrema(low_prices, np.less, order=window)[0]
            
            # Get resistance levels
            resistance_levels = []
            for idx in resistance_indices:
                level = high_prices[idx]
                # Count how many times price touched this level
                touches = np.sum(np.abs(high_prices - level) < (level * 0.02))  # Within 2%
                if touches >= min_touches:
                    resistance_levels.append({
                        'level': level,
                        'touches': touches,
                        'strength': touches * (1 / np.std(high_prices[max(0, idx-window):idx+window]))
                    })
            
            # Get support levels
            support_levels = []
            for idx in support_indices:
                level = low_prices[idx]
                touches = np.sum(np.abs(low_prices - level) < (level * 0.02))  # Within 2%
                if touches >= min_touches:
                    support_levels.append({
                        'level': level,
                        'touches': touches,
                        'strength': touches * (1 / np.std(low_prices[max(0, idx-window):idx+window]))
                    })
            
            # Sort by strength
            resistance_levels = sorted(resistance_levels, key=lambda x: x['strength'], reverse=True)[:5]
            support_levels = sorted(support_levels, key=lambda x: x['strength'], reverse=True)[:5]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            st.warning(f"⚠️ Error finding support/resistance: {e}")
            return [], []
    
    def map_levels_to_dates(self, df, levels):
        """Map support/resistance levels to all dates"""
        if not levels:
            return np.full(len(df), np.nan)
        
        # Use the strongest level for each date
        strongest_level = levels[0]['level'] if levels else np.nan
        return np.full(len(df), strongest_level)
    
    def detect_chart_patterns(self, df):
        """Detect common chart patterns"""
        patterns = {}
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        try:
            # Head and Shoulders
            patterns['head_shoulders'] = self.detect_head_shoulders(high_prices, close_prices)
            
            # Double Top/Bottom
            patterns['double_top'] = self.detect_double_top(high_prices)
            patterns['double_bottom'] = self.detect_double_bottom(low_prices)
            
            # Triangle Patterns
            patterns['ascending_triangle'] = self.detect_ascending_triangle(high_prices, low_prices)
            patterns['descending_triangle'] = self.detect_descending_triangle(high_prices, low_prices)
            
            # Flag and Pennant
            patterns['bull_flag'] = self.detect_bull_flag(close_prices, high_prices, low_prices)
            patterns['bear_flag'] = self.detect_bear_flag(close_prices, high_prices, low_prices)
            
            return patterns
            
        except Exception as e:
            st.warning(f"⚠️ Error detecting patterns: {e}")
            return {}
    
    def detect_head_shoulders(self, highs, closes, window=10):
        """Detect Head and Shoulders pattern"""
        try:
            peaks = find_peaks(highs, distance=window)[0]
            if len(peaks) < 3:
                return None
            
            # Look for three consecutive peaks where middle is highest
            for i in range(len(peaks) - 2):
                left_shoulder = highs[peaks[i]]
                head = highs[peaks[i + 1]]
                right_shoulder = highs[peaks[i + 2]]
                
                # Head should be higher than both shoulders
                if head > left_shoulder and head > right_shoulder:
                    # Shoulders should be roughly equal (within 5%)
                    if abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05:
                        return {
                            'pattern': 'Head and Shoulders',
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'neckline': min(left_shoulder, right_shoulder),
                            'signal': 'BEARISH'
                        }
            
            return None
            
        except Exception:
            return None
    
    def detect_double_top(self, highs, window=10, tolerance=0.03):
        """Detect Double Top pattern"""
        try:
            peaks = find_peaks(highs, distance=window)[0]
            if len(peaks) < 2:
                return None
            
            for i in range(len(peaks) - 1):
                peak1 = highs[peaks[i]]
                peak2 = highs[peaks[i + 1]]
                
                # Peaks should be roughly equal
                if abs(peak1 - peak2) / max(peak1, peak2) < tolerance:
                    return {
                        'pattern': 'Double Top',
                        'peak1': peak1,
                        'peak2': peak2,
                        'signal': 'BEARISH'
                    }
            
            return None
            
        except Exception:
            return None
    
    def detect_double_bottom(self, lows, window=10, tolerance=0.03):
        """Detect Double Bottom pattern"""
        try:
            troughs = find_peaks(-lows, distance=window)[0]
            if len(troughs) < 2:
                return None
            
            for i in range(len(troughs) - 1):
                trough1 = lows[troughs[i]]
                trough2 = lows[troughs[i + 1]]
                
                # Troughs should be roughly equal
                if abs(trough1 - trough2) / max(trough1, trough2) < tolerance:
                    return {
                        'pattern': 'Double Bottom',
                        'trough1': trough1,
                        'trough2': trough2,
                        'signal': 'BULLISH'
                    }
            
            return None
            
        except Exception:
            return None
    
    def detect_ascending_triangle(self, highs, lows, window=20):
        """Detect Ascending Triangle pattern"""
        try:
            # Resistance should be relatively flat
            recent_highs = highs[-window:]
            resistance_slope = linregress(range(len(recent_highs)), recent_highs)[0]
            
            # Support should be ascending
            recent_lows = lows[-window:]
            support_slope = linregress(range(len(recent_lows)), recent_lows)[0]
            
            if abs(resistance_slope) < 0.1 and support_slope > 0.1:
                return {
                    'pattern': 'Ascending Triangle',
                    'resistance_level': np.mean(recent_highs),
                    'support_slope': support_slope,
                    'signal': 'BULLISH'
                }
            
            return None
            
        except Exception:
            return None
    
    def detect_descending_triangle(self, highs, lows, window=20):
        """Detect Descending Triangle pattern"""
        try:
            # Support should be relatively flat
            recent_lows = lows[-window:]
            support_slope = linregress(range(len(recent_lows)), recent_lows)[0]
            
            # Resistance should be descending
            recent_highs = highs[-window:]
            resistance_slope = linregress(range(len(recent_highs)), recent_highs)[0]
            
            if abs(support_slope) < 0.1 and resistance_slope < -0.1:
                return {
                    'pattern': 'Descending Triangle',
                    'support_level': np.mean(recent_lows),
                    'resistance_slope': resistance_slope,
                    'signal': 'BEARISH'
                }
            
            return None
            
        except Exception:
            return None
    
    def detect_bull_flag(self, closes, highs, lows, window=20):
        """Detect Bull Flag pattern"""
        try:
            if len(closes) < window * 2:
                return None
            
            # Strong upward move (flagpole)
            flagpole_start = len(closes) - window * 2
            flagpole_end = len(closes) - window
            flagpole_return = (closes[flagpole_end] - closes[flagpole_start]) / closes[flagpole_start]
            
            if flagpole_return < 0.1:  # At least 10% move
                return None
            
            # Consolidation (flag)
            flag_highs = highs[-window:]
            flag_lows = lows[-window:]
            
            # Flag should slope slightly downward or sideways
            flag_high_slope = linregress(range(len(flag_highs)), flag_highs)[0]
            flag_low_slope = linregress(range(len(flag_lows)), flag_lows)[0]
            
            if flag_high_slope <= 0 and flag_low_slope <= 0:
                return {
                    'pattern': 'Bull Flag',
                    'flagpole_return': flagpole_return,
                    'flag_slope': (flag_high_slope + flag_low_slope) / 2,
                    'signal': 'BULLISH'
                }
            
            return None
            
        except Exception:
            return None
    
    def detect_bear_flag(self, closes, highs, lows, window=20):
        """Detect Bear Flag pattern"""
        try:
            if len(closes) < window * 2:
                return None
            
            # Strong downward move (flagpole)
            flagpole_start = len(closes) - window * 2
            flagpole_end = len(closes) - window
            flagpole_return = (closes[flagpole_end] - closes[flagpole_start]) / closes[flagpole_start]
            
            if flagpole_return > -0.1:  # At least 10% decline
                return None
            
            # Consolidation (flag)
            flag_highs = highs[-window:]
            flag_lows = lows[-window:]
            
            # Flag should slope slightly upward or sideways
            flag_high_slope = linregress(range(len(flag_highs)), flag_highs)[0]
            flag_low_slope = linregress(range(len(flag_lows)), flag_lows)[0]
            
            if flag_high_slope >= 0 and flag_low_slope >= 0:
                return {
                    'pattern': 'Bear Flag',
                    'flagpole_return': flagpole_return,
                    'flag_slope': (flag_high_slope + flag_low_slope) / 2,
                    'signal': 'BEARISH'
                }
            
            return None
            
        except Exception:
            return None
    
    def generate_trading_signals(self, indicators):
        """Generate comprehensive trading signals"""
        signals = {
            'overall_signal': 'NEUTRAL',
            'signal_strength': 0,
            'individual_signals': {},
            'signal_summary': []
        }
        
        try:
            latest = indicators.iloc[-1]
            prev = indicators.iloc[-2] if len(indicators) > 1 else latest
            
            signal_scores = []
            
            # RSI Signals
            if latest['RSI_14'] < 30:
                signals['individual_signals']['RSI'] = 'OVERSOLD_BUY'
                signal_scores.append(1)
                signals['signal_summary'].append("RSI indicates oversold condition - potential buy signal")
            elif latest['RSI_14'] > 70:
                signals['individual_signals']['RSI'] = 'OVERBOUGHT_SELL'
                signal_scores.append(-1)
                signals['signal_summary'].append("RSI indicates overbought condition - potential sell signal")
            else:
                signals['individual_signals']['RSI'] = 'NEUTRAL'
                signal_scores.append(0)
            
            # MACD Signals
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals['individual_signals']['MACD'] = 'BULLISH_CROSSOVER'
                signal_scores.append(1)
                signals['signal_summary'].append("MACD bullish crossover - buy signal")
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals['individual_signals']['MACD'] = 'BEARISH_CROSSOVER'
                signal_scores.append(-1)
                signals['signal_summary'].append("MACD bearish crossover - sell signal")
            else:
                signals['individual_signals']['MACD'] = 'NEUTRAL'
                signal_scores.append(0)
            
            # Moving Average Signals
            if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
                signals['individual_signals']['MA_Trend'] = 'STRONG_BULLISH'
                signal_scores.append(2)
                signals['signal_summary'].append("Strong bullish trend - all MAs aligned")
            elif latest['Close'] < latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
                signals['individual_signals']['MA_Trend'] = 'STRONG_BEARISH'
                signal_scores.append(-2)
                signals['signal_summary'].append("Strong bearish trend - all MAs aligned")
            elif latest['Close'] > latest['SMA_20']:
                signals['individual_signals']['MA_Trend'] = 'BULLISH'
                signal_scores.append(1)
                signals['signal_summary'].append("Price above short-term MA - bullish")
            elif latest['Close'] < latest['SMA_20']:
                signals['individual_signals']['MA_Trend'] = 'BEARISH'
                signal_scores.append(-1)
                signals['signal_summary'].append("Price below short-term MA - bearish")
            else:
                signals['individual_signals']['MA_Trend'] = 'NEUTRAL'
                signal_scores.append(0)
            
            # Bollinger Bands Signals
            if latest['Close'] < latest['BB_Lower']:
                signals['individual_signals']['BB'] = 'OVERSOLD'
                signal_scores.append(1)
                signals['signal_summary'].append("Price below lower Bollinger Band - oversold")
            elif latest['Close'] > latest['BB_Upper']:
                signals['individual_signals']['BB'] = 'OVERBOUGHT'
                signal_scores.append(-1)
                signals['signal_summary'].append("Price above upper Bollinger Band - overbought")
            else:
                signals['individual_signals']['BB'] = 'NEUTRAL'
                signal_scores.append(0)
            
            # Stochastic Signals
            if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
                signals['individual_signals']['Stochastic'] = 'OVERSOLD'
                signal_scores.append(1)
                signals['signal_summary'].append("Stochastic oversold - potential buy")
            elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
                signals['individual_signals']['Stochastic'] = 'OVERBOUGHT'
                signal_scores.append(-1)
                signals['signal_summary'].append("Stochastic overbought - potential sell")
            else:
                signals['individual_signals']['Stochastic'] = 'NEUTRAL'
                signal_scores.append(0)
            
            # ADX Trend Strength
            if latest['ADX'] > 25:
                if latest['DI_Plus'] > latest['DI_Minus']:
                    signals['individual_signals']['ADX'] = 'STRONG_UPTREND'
                    signal_scores.append(1)
                    signals['signal_summary'].append("Strong uptrend confirmed by ADX")
                else:
                    signals['individual_signals']['ADX'] = 'STRONG_DOWNTREND'
                    signal_scores.append(-1)
                    signals['signal_summary'].append("Strong downtrend confirmed by ADX")
            else:
                signals['individual_signals']['ADX'] = 'WEAK_TREND'
                signal_scores.append(0)
                signals['signal_summary'].append("Weak trend - sideways movement")
            
            # Calculate overall signal
            total_score = sum(signal_scores)
            max_possible_score = len(signal_scores) * 2  # Maximum possible score
            
            signals['signal_strength'] = total_score / max_possible_score if max_possible_score > 0 else 0
            
            if total_score >= 3:
                signals['overall_signal'] = 'STRONG_BUY'
            elif total_score >= 1:
                signals['overall_signal'] = 'BUY'
            elif total_score <= -3:
                signals['overall_signal'] = 'STRONG_SELL'
            elif total_score <= -1:
                signals['overall_signal'] = 'SELL'
            else:
                signals['overall_signal'] = 'NEUTRAL'
            
            return signals
            
        except Exception as e:
            st.error(f"❌ Error generating trading signals: {e}")
            return signals
    
    def create_technical_dashboard(self, df, indicators, patterns, signals):
        """Create comprehensive technical analysis dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'MACD', 'RSI & Stochastic', 'Volume & OBV'),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price chart with moving averages
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Moving averages
            for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
                if ma in indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators.index,
                            y=indicators[ma],
                            mode='lines',
                            name=ma,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # Bollinger Bands
            if all(col in indicators.columns for col in ['BB_Upper', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Support and Resistance levels
            if 'Support_Level' in indicators.columns and not indicators['Support_Level'].isna().all():
                fig.add_hline(
                    y=indicators['Support_Level'].dropna().iloc[-1],
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Support",
                    row=1, col=1
                )
            
            if 'Resistance_Level' in indicators.columns and not indicators['Resistance_Level'].isna().all():
                fig.add_hline(
                    y=indicators['Resistance_Level'].dropna().iloc[-1],
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Resistance",
                    row=1, col=1
                )
            
            # MACD
            if all(col in indicators.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['MACD_Signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=indicators.index,
                        y=indicators['MACD_Hist'],
                        name='MACD Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            
            # RSI and Stochastic
            if 'RSI_14' in indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['RSI_14'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=3, col=1
                )
                
                # RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            if all(col in indicators.columns for col in ['Stoch_K', 'Stoch_D']):
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['Stoch_K'],
                        mode='lines',
                        name='Stoch %K',
                        line=dict(color='orange'),
                        yaxis='y4'
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['Stoch_D'],
                        mode='lines',
                        name='Stoch %D',
                        line=dict(color='brown'),
                        yaxis='y4'
                    ),
                    row=3, col=1
                )
            
            # Volume and OBV
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=4, col=1
            )
            
            if 'OBV' in indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators['OBV'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='darkblue'),
                        yaxis='y5'
                    ),
                    row=4, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="Comprehensive Technical Analysis Dashboard",
                height=1200,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating technical dashboard: {e}")
            return None

# Try to import TA-Lib, provide fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("⚠️ TA-Lib not available. Some advanced indicators may not work. Install with: pip install TA-Lib")
    
    # Create mock talib functions for basic functionality
    class MockTALib:
        @staticmethod
        def SMA(prices, timeperiod):
            return pd.Series(prices).rolling(window=timeperiod).mean().values
        
        @staticmethod
        def EMA(prices, timeperiod):
            return pd.Series(prices).ewm(span=timeperiod).mean().values
        
        @staticmethod
        def RSI(prices, timeperiod=14):
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).values
        
        @staticmethod
        def MACD(prices):
            exp1 = pd.Series(prices).ewm(span=12).mean()
            exp2 = pd.Series(prices).ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return macd.values, signal.values, histogram.values
        
        @staticmethod
        def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2):
            sma = pd.Series(prices).rolling(window=timeperiod).mean()
            std = pd.Series(prices).rolling(window=timeperiod).std()
            upper = sma + (std * nbdevup)
            lower = sma - (std * nbdevdn)
            return upper.values, sma.values, lower.values
        
        @staticmethod
        def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
            lowest_low = pd.Series(low).rolling(window=fastk_period).min()
            highest_high = pd.Series(high).rolling(window=fastk_period).max()
            k_percent = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))
            k_percent = k_percent.rolling(window=slowk_period).mean()
            d_percent = k_percent.rolling(window=slowd_period).mean()
            return k_percent.values, d_percent.values
        
        @staticmethod
        def WILLR(high, low, close, timeperiod=14):
            highest_high = pd.Series(high).rolling(window=timeperiod).max()
            lowest_low = pd.Series(low).rolling(window=timeperiod).min()
            wr = -100 * ((highest_high - pd.Series(close)) / (highest_high - lowest_low))
            return wr.values
        
        @staticmethod
        def CCI(high, low, close, timeperiod=14):
            tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
            sma_tp = tp.rolling(window=timeperiod).mean()
            mad = tp.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci.values
        
        @staticmethod
        def ADX(high, low, close, timeperiod=14):
            # Simplified ADX calculation
            return np.full(len(close), 25.0)  # Default neutral value
        
        @staticmethod
        def PLUS_DI(high, low, close, timeperiod=14):
            return np.full(len(close), 25.0)
        
        @staticmethod
        def MINUS_DI(high, low, close, timeperiod=14):
            return np.full(len(close), 25.0)
        
        @staticmethod
        def OBV(close, volume):
            obv = np.zeros(len(close))
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            return obv
        
        @staticmethod
        def AD(high, low, close, volume):
            clv = ((close - low) - (high - close)) / (high - low)
            ad = np.cumsum(clv * volume)
            return ad
        
        @staticmethod
        def ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10):
            ad = MockTALib.AD(high, low, close, volume)
            fast_ema = pd.Series(ad).ewm(span=fastperiod).mean()
            slow_ema = pd.Series(ad).ewm(span=slowperiod).mean()
            return (fast_ema - slow_ema).values
        
        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            tr1 = pd.Series(high) - pd.Series(low)
            tr2 = abs(pd.Series(high) - pd.Series(close).shift())
            tr3 = abs(pd.Series(low) - pd.Series(close).shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=timeperiod).mean().values
        
        @staticmethod
        def NATR(high, low, close, timeperiod=14):
            atr = MockTALib.ATR(high, low, close, timeperiod)
            return (atr / pd.Series(close)) * 100
        
        # Candlestick patterns (simplified)
        @staticmethod
        def CDLDOJI(open_prices, high, low, close):
            return np.where(abs(close - open_prices) < (high - low) * 0.1, 100, 0)
        
        @staticmethod
        def CDLHAMMER(open_prices, high, low, close):
            body = abs(close - open_prices)
            lower_shadow = np.minimum(open_prices, close) - low
            upper_shadow = high - np.maximum(open_prices, close)
            return np.where((lower_shadow > 2 * body) & (upper_shadow < body), 100, 0)
        
        @staticmethod
        def CDLENGULFING(open_prices, high, low, close):
            return np.zeros(len(close))  # Simplified
        
        @staticmethod
        def CDLMORNINGSTAR(open_prices, high, low, close):
            return np.zeros(len(close))  # Simplified
        
        @staticmethod
        def CDLEVENINGSTAR(open_prices, high, low, close):
            return np.zeros(len(close))  # Simplified
    
    talib = MockTALib()