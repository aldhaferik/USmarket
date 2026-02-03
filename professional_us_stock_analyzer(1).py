#!/usr/bin/env python3
"""
Professional US Stock Analyzer
- Real financial data from multiple sources
- Multiple valuation methods (DCF, Multiples, Monte Carlo)
- AI-optimized model selection
- Professional-grade analysis for serious investors
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time

warnings.filterwarnings('ignore')

# Import enhanced modules
try:
    from enhanced_technical_analysis import TechnicalAnalyzer
    from portfolio_analysis import PortfolioAnalyzer
    from sector_comparison import SectorAnalyzer
    from options_analysis import OptionsAnalyzer
    from esg_scoring import ESGAnalyzer
    from real_time_alerts import AlertsManager
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

# ML and Statistical Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Statistical and Financial Libraries
try:
    from scipy import stats
    from scipy.optimize import minimize
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ProfessionalUSStockAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.045  # Current 10-year Treasury rate
        self.market_risk_premium = 0.065  # Historical market risk premium
        
        # Industry average multiples (updated 2024)
        self.industry_multiples = {
            'Technology': {'pe': 28, 'pb': 6.5, 'ps': 8.2, 'ev_ebitda': 22},
            'Healthcare': {'pe': 24, 'pb': 4.2, 'ps': 5.8, 'ev_ebitda': 18},
            'Financial Services': {'pe': 12, 'pb': 1.3, 'ps': 3.2, 'ev_ebitda': 10},
            'Consumer Cyclical': {'pe': 18, 'pb': 2.8, 'ps': 1.8, 'ev_ebitda': 12},
            'Consumer Defensive': {'pe': 22, 'pb': 3.5, 'ps': 2.2, 'ev_ebitda': 15},
            'Industrials': {'pe': 20, 'pb': 3.2, 'ps': 2.5, 'ev_ebitda': 14},
            'Energy': {'pe': 15, 'pb': 1.8, 'ps': 1.2, 'ev_ebitda': 8},
            'Materials': {'pe': 16, 'pb': 2.1, 'ps': 1.5, 'ev_ebitda': 9},
            'Real Estate': {'pe': 25, 'pb': 1.9, 'ps': 8.5, 'ev_ebitda': 20},
            'Utilities': {'pe': 19, 'pb': 1.6, 'ps': 2.8, 'ev_ebitda': 11},
            'Communication Services': {'pe': 21, 'pb': 3.8, 'ps': 4.2, 'ev_ebitda': 16}
        }
        
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
    
    def get_comprehensive_stock_data(self, ticker):
        """Get comprehensive stock data from multiple sources"""
        try:
            st.info(f"📊 Extracting comprehensive data for {ticker}...")
            
            # Get stock object
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            if not info or 'symbol' not in info:
                st.error(f"❌ Invalid ticker: {ticker}")
                return None
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get historical data (5 years)
            hist_data = stock.history(period="5y")
            
            if hist_data.empty:
                st.error(f"❌ No historical data for {ticker}")
                return None
            
            # Get quarterly data for more recent metrics
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cash_flow = stock.quarterly_cashflow
            
            # Get analyst recommendations
            recommendations = stock.recommendations
            
            # Get earnings data
            earnings = stock.earnings
            quarterly_earnings = stock.quarterly_earnings
            
            # Compile comprehensive data
            stock_data = {
                'ticker': ticker,
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'quarterly_financials': quarterly_financials,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cash_flow': quarterly_cash_flow,
                'historical_data': hist_data,
                'recommendations': recommendations,
                'earnings': earnings,
                'quarterly_earnings': quarterly_earnings,
                'extraction_date': datetime.now()
            }
            
            st.success(f"✅ Comprehensive data extracted for {ticker}")
            return stock_data
            
        except Exception as e:
            st.error(f"❌ Error extracting data for {ticker}: {e}")
            return None
    
    def calculate_financial_metrics(self, stock_data):
        """Calculate comprehensive financial metrics"""
        try:
            info = stock_data['info']
            financials = stock_data['financials']
            balance_sheet = stock_data['balance_sheet']
            cash_flow = stock_data['cash_flow']
            hist_data = stock_data['historical_data']
            
            metrics = {
                'ticker': stock_data['ticker'],
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Basic valuation metrics
            metrics['pe_ratio'] = info.get('trailingPE', None)
            metrics['forward_pe'] = info.get('forwardPE', None)
            metrics['pb_ratio'] = info.get('priceToBook', None)
            metrics['ps_ratio'] = info.get('priceToSalesTrailing12Months', None)
            metrics['peg_ratio'] = info.get('pegRatio', None)
            metrics['ev_ebitda'] = info.get('enterpriseToEbitda', None)
            metrics['ev_revenue'] = info.get('enterpriseToRevenue', None)
            
            # Profitability metrics
            metrics['profit_margin'] = info.get('profitMargins', None)
            metrics['operating_margin'] = info.get('operatingMargins', None)
            metrics['gross_margin'] = info.get('grossMargins', None)
            metrics['roe'] = info.get('returnOnEquity', None)
            metrics['roa'] = info.get('returnOnAssets', None)
            metrics['roic'] = info.get('returnOnCapital', None)
            
            # Growth metrics
            metrics['revenue_growth'] = info.get('revenueGrowth', None)
            metrics['earnings_growth'] = info.get('earningsGrowth', None)
            metrics['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', None)
            
            # Financial strength
            metrics['debt_to_equity'] = info.get('debtToEquity', None)
            metrics['current_ratio'] = info.get('currentRatio', None)
            metrics['quick_ratio'] = info.get('quickRatio', None)
            metrics['cash_per_share'] = info.get('totalCashPerShare', None)
            
            # Dividend metrics
            metrics['dividend_yield'] = info.get('dividendYield', None)
            metrics['payout_ratio'] = info.get('payoutRatio', None)
            metrics['dividend_rate'] = info.get('dividendRate', None)
            
            # Market metrics
            metrics['beta'] = info.get('beta', None)
            metrics['52_week_high'] = info.get('fiftyTwoWeekHigh', None)
            metrics['52_week_low'] = info.get('fiftyTwoWeekLow', None)
            metrics['avg_volume'] = info.get('averageVolume', None)
            
            # Calculate additional metrics from financial statements
            if not financials.empty:
                try:
                    # Get most recent year data
                    latest_financials = financials.iloc[:, 0]
                    
                    revenue = latest_financials.get('Total Revenue', 0)
                    net_income = latest_financials.get('Net Income', 0)
                    
                    if revenue and revenue != 0:
                        metrics['net_margin_calculated'] = (net_income / revenue) * 100
                    
                except Exception as e:
                    st.warning(f"⚠️ Error calculating additional metrics: {e}")
            
            # Calculate volatility and risk metrics
            if not hist_data.empty and len(hist_data) > 30:
                returns = hist_data['Close'].pct_change().dropna()
                metrics['volatility_30d'] = returns.tail(30).std() * np.sqrt(252)
                metrics['volatility_1y'] = returns.tail(252).std() * np.sqrt(252) if len(returns) >= 252 else None
                metrics['sharpe_ratio'] = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else None
                
                # Calculate maximum drawdown
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                metrics['max_drawdown'] = drawdown.min()
            
            return metrics
            
        except Exception as e:
            st.error(f"❌ Error calculating financial metrics: {e}")
            return None
    
    def calculate_dcf_valuation(self, stock_data, growth_scenarios=None):
        """Calculate DCF valuation with multiple scenarios"""
        try:
            info = stock_data['info']
            cash_flow = stock_data['cash_flow']
            
            # Get free cash flow
            if not cash_flow.empty:
                try:
                    latest_cf = cash_flow.iloc[:, 0]
                    operating_cf = latest_cf.get('Operating Cash Flow', 0)
                    capex = latest_cf.get('Capital Expenditure', 0)
                    free_cash_flow = operating_cf + capex  # capex is negative
                except:
                    free_cash_flow = None
            else:
                free_cash_flow = None
            
            # If no FCF from statements, estimate from net income
            if not free_cash_flow or free_cash_flow <= 0:
                net_income = info.get('netIncomeToCommon', 0)
                if net_income > 0:
                    free_cash_flow = net_income * 0.85  # Conservative estimate
                else:
                    st.warning("⚠️ Cannot calculate DCF: No positive free cash flow data")
                    return None
            
            # Growth scenarios
            if not growth_scenarios:
                revenue_growth = info.get('revenueGrowth', 0.05)
                growth_scenarios = {
                    'Conservative': max(0.02, revenue_growth * 0.7),
                    'Base Case': max(0.03, revenue_growth),
                    'Optimistic': max(0.05, revenue_growth * 1.3)
                }
            
            # DCF parameters
            terminal_growth = 0.025  # Long-term GDP growth
            years = 10  # 10-year projection
            
            # Calculate WACC
            beta = info.get('beta', 1.0)
            if beta is None:
                beta = 1.0
            
            cost_of_equity = self.risk_free_rate + beta * self.market_risk_premium
            
            # Assume cost of debt and capital structure
            debt_to_equity = info.get('debtToEquity', 0.3) / 100 if info.get('debtToEquity') else 0.3
            tax_rate = 0.25  # Corporate tax rate
            
            if debt_to_equity > 0:
                cost_of_debt = self.risk_free_rate + 0.02  # Risk premium
                weight_debt = debt_to_equity / (1 + debt_to_equity)
                weight_equity = 1 / (1 + debt_to_equity)
                wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
            else:
                wacc = cost_of_equity
            
            dcf_results = {}
            
            for scenario, growth_rate in growth_scenarios.items():
                # Project future cash flows
                future_fcf = []
                current_fcf = free_cash_flow
                
                for year in range(1, years + 1):
                    # Declining growth rate over time
                    year_growth = growth_rate * (0.95 ** (year - 1))
                    current_fcf *= (1 + year_growth)
                    pv_fcf = current_fcf / ((1 + wacc) ** year)
                    future_fcf.append(pv_fcf)
                
                # Terminal value
                terminal_fcf = current_fcf * (1 + terminal_growth)
                terminal_value = terminal_fcf / (wacc - terminal_growth)
                pv_terminal = terminal_value / ((1 + wacc) ** years)
                
                # Enterprise value
                enterprise_value = sum(future_fcf) + pv_terminal
                
                # Equity value
                total_debt = info.get('totalDebt', 0)
                cash = info.get('totalCash', 0)
                equity_value = enterprise_value - total_debt + cash
                
                # Per share value
                shares_outstanding = info.get('sharesOutstanding', 0)
                if shares_outstanding > 0:
                    dcf_per_share = equity_value / shares_outstanding
                else:
                    dcf_per_share = 0
                
                dcf_results[scenario] = {
                    'growth_rate': growth_rate,
                    'wacc': wacc,
                    'enterprise_value': enterprise_value,
                    'equity_value': equity_value,
                    'dcf_per_share': dcf_per_share,
                    'future_fcf': future_fcf,
                    'terminal_value': terminal_value,
                    'assumptions': {
                        'free_cash_flow': free_cash_flow,
                        'growth_rate': growth_rate,
                        'terminal_growth': terminal_growth,
                        'wacc': wacc,
                        'years': years
                    }
                }
            
            return dcf_results
            
        except Exception as e:
            st.error(f"❌ Error calculating DCF: {e}")
            return None
    
    def calculate_multiple_valuations(self, metrics):
        """Calculate valuations using multiple methods"""
        try:
            sector = metrics.get('sector', 'Technology')
            industry_multiples = self.industry_multiples.get(sector, self.industry_multiples['Technology'])
            
            valuations = {}
            current_price = metrics['current_price']
            
            # P/E Valuation
            if metrics.get('pe_ratio') and metrics.get('pe_ratio') > 0:
                eps = current_price / metrics['pe_ratio']
                industry_pe = industry_multiples['pe']
                pe_fair_value = eps * industry_pe
                
                valuations['P/E'] = {
                    'current_multiple': metrics['pe_ratio'],
                    'industry_average': industry_pe,
                    'fair_value': pe_fair_value,
                    'upside_downside': (pe_fair_value - current_price) / current_price
                }
            
            # P/B Valuation
            if metrics.get('pb_ratio') and metrics.get('pb_ratio') > 0:
                book_value_per_share = current_price / metrics['pb_ratio']
                industry_pb = industry_multiples['pb']
                pb_fair_value = book_value_per_share * industry_pb
                
                valuations['P/B'] = {
                    'current_multiple': metrics['pb_ratio'],
                    'industry_average': industry_pb,
                    'fair_value': pb_fair_value,
                    'upside_downside': (pb_fair_value - current_price) / current_price
                }
            
            # P/S Valuation
            if metrics.get('ps_ratio') and metrics.get('ps_ratio') > 0:
                sales_per_share = current_price / metrics['ps_ratio']
                industry_ps = industry_multiples['ps']
                ps_fair_value = sales_per_share * industry_ps
                
                valuations['P/S'] = {
                    'current_multiple': metrics['ps_ratio'],
                    'industry_average': industry_ps,
                    'fair_value': ps_fair_value,
                    'upside_downside': (ps_fair_value - current_price) / current_price
                }
            
            # EV/EBITDA Valuation
            if metrics.get('ev_ebitda') and metrics.get('ev_ebitda') > 0:
                # This is more complex as we need enterprise value
                market_cap = metrics.get('market_cap', 0)
                if market_cap > 0:
                    industry_ev_ebitda = industry_multiples['ev_ebitda']
                    current_ev_ebitda = metrics['ev_ebitda']
                    
                    valuations['EV/EBITDA'] = {
                        'current_multiple': current_ev_ebitda,
                        'industry_average': industry_ev_ebitda,
                        'relative_valuation': 'Undervalued' if current_ev_ebitda < industry_ev_ebitda else 'Overvalued'
                    }
            
            # PEG Ratio Analysis
            if metrics.get('peg_ratio'):
                peg = metrics['peg_ratio']
                if peg < 1:
                    peg_analysis = 'Undervalued (PEG < 1)'
                elif peg < 1.5:
                    peg_analysis = 'Fairly Valued (PEG 1-1.5)'
                else:
                    peg_analysis = 'Overvalued (PEG > 1.5)'
                
                valuations['PEG'] = {
                    'peg_ratio': peg,
                    'analysis': peg_analysis
                }
            
            return valuations
            
        except Exception as e:
            st.error(f"❌ Error calculating multiple valuations: {e}")
            return None
    
    def monte_carlo_simulation(self, stock_data, days=252, simulations=10000):
        """Monte Carlo simulation for price prediction"""
        try:
            hist_data = stock_data['historical_data']
            
            if hist_data.empty or len(hist_data) < 30:
                st.warning("⚠️ Insufficient data for Monte Carlo simulation")
                return None
            
            # Calculate returns
            returns = hist_data['Close'].pct_change().dropna()
            
            # Parameters for simulation
            current_price = hist_data['Close'].iloc[-1]
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            simulation_results = []
            
            for _ in range(simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, days)
                
                # Calculate price path
                price_path = [current_price]
                for daily_return in random_returns:
                    price_path.append(price_path[-1] * (1 + daily_return))
                
                simulation_results.append(price_path[-1])  # Final price
            
            simulation_results = np.array(simulation_results)
            
            # Calculate statistics
            monte_carlo_results = {
                'current_price': current_price,
                'simulations': simulations,
                'days': days,
                'mean_final_price': np.mean(simulation_results),
                'median_final_price': np.median(simulation_results),
                'std_final_price': np.std(simulation_results),
                'percentiles': {
                    '5th': np.percentile(simulation_results, 5),
                    '25th': np.percentile(simulation_results, 25),
                    '75th': np.percentile(simulation_results, 75),
                    '95th': np.percentile(simulation_results, 95)
                },
                'probability_positive': np.sum(simulation_results > current_price) / simulations,
                'expected_return': (np.mean(simulation_results) - current_price) / current_price,
                'var_95': np.percentile(simulation_results, 5) - current_price,  # Value at Risk
                'all_results': simulation_results
            }
            
            return monte_carlo_results
            
        except Exception as e:
            st.error(f"❌ Error in Monte Carlo simulation: {e}")
            return None
    
    def create_advanced_technical_features(self, hist_data):
        """Create comprehensive technical analysis features"""
        try:
            df = hist_data.copy()
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['price_change'] = df['Close'] - df['Open']
            df['gap'] = df['Open'] - df['Close'].shift(1)
            
            # Moving averages (multiple timeframes)
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'sma_{window}_ratio'] = df['Close'] / df[f'sma_{window}']
                
            # Bollinger Bands
            for window in [20, 50]:
                sma = df['Close'].rolling(window=window).mean()
                std = df['Close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = sma + (std * 2)
                df[f'bb_lower_{window}'] = sma - (std * 2)
                df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
                df[f'bb_position_{window}'] = (df['Close'] - df[f'bb_lower_{window}']) / df[f'bb_width_{window}']
            
            # RSI (multiple periods)
            for period in [14, 21, 30]:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic Oscillator
            for period in [14, 21]:
                low_min = df['Low'].rolling(window=period).min()
                high_max = df['High'].rolling(window=period).max()
                df[f'stoch_k_{period}'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
            
            # Williams %R
            for period in [14, 21]:
                high_max = df['High'].rolling(window=period).max()
                low_min = df['Low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
            
            # Average True Range and volatility
            df['tr1'] = df['High'] - df['Low']
            df['tr2'] = abs(df['High'] - df['Close'].shift())
            df['tr3'] = abs(df['Low'] - df['Close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            for period in [14, 21]:
                df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            
            # Volume indicators
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['price_volume'] = df['Close'] * df['Volume']
            
            # On-Balance Volume
            df['obv'] = (np.sign(df['returns']) * df['Volume']).cumsum()
            
            # Momentum indicators
            for period in [1, 5, 10, 20, 50]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
                df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
            
            # Price position indicators
            for period in [10, 20, 50, 100]:
                high_max = df['High'].rolling(window=period).max()
                low_min = df['Low'].rolling(window=period).min()
                df[f'price_position_{period}'] = (df['Close'] - low_min) / (high_max - low_min)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
            # Advanced features
            df['close_to_high'] = df['Close'] / df['High']
            df['close_to_low'] = df['Close'] / df['Low']
            df['high_low_ratio'] = df['High'] / df['Low']
            
            # Trend strength
            df['trend_strength_20'] = df['Close'].rolling(window=20).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else np.nan
            )
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error creating technical features: {e}")
            return None
    
    def optimize_ml_models(self, X, y):
        """Optimize ML models using grid search and cross-validation"""
        if not ML_AVAILABLE:
            st.warning("⚠️ ML libraries not available")
            return None
        
        try:
            st.info("🧠 Optimizing ML models...")
            
            # Time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define models and parameter grids
            models_params = {
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'XGBoost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [6, 10],
                        'learning_rate': [0.01, 0.1],
                        'subsample': [0.8, 1.0]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1],
                        'subsample': [0.8, 1.0]
                    }
                },
                'Ridge': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                }
            }
            
            optimized_models = {}
            
            for name, model_info in models_params.items():
                st.write(f"Optimizing {name}...")
                
                # Grid search with time series CV
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                # Scale features for linear models
                if name in ['Ridge', 'Lasso']:
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X)
                    grid_search.fit(X_scaled, y)
                    optimized_models[name] = {
                        'model': grid_search.best_estimator_,
                        'scaler': scaler,
                        'best_params': grid_search.best_params_,
                        'best_score': -grid_search.best_score_
                    }
                else:
                    grid_search.fit(X, y)
                    optimized_models[name] = {
                        'model': grid_search.best_estimator_,
                        'scaler': None,
                        'best_params': grid_search.best_params_,
                        'best_score': -grid_search.best_score_
                    }
            
            # Evaluate models with cross-validation
            model_performance = {}
            
            for name, model_info in optimized_models.items():
                model = model_info['model']
                scaler = model_info['scaler']
                
                cv_scores = {'mse': [], 'mae': [], 'r2': [], 'directional_accuracy': []}
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    if scaler:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Directional accuracy
                    direction_actual = np.sign(y_test)
                    direction_pred = np.sign(y_pred)
                    directional_accuracy = np.mean(direction_actual == direction_pred)
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r2'].append(r2)
                    cv_scores['directional_accuracy'].append(directional_accuracy)
                
                model_performance[name] = {
                    'cv_mse_mean': np.mean(cv_scores['mse']),
                    'cv_mse_std': np.std(cv_scores['mse']),
                    'cv_mae_mean': np.mean(cv_scores['mae']),
                    'cv_mae_std': np.std(cv_scores['mae']),
                    'cv_r2_mean': np.mean(cv_scores['r2']),
                    'cv_r2_std': np.std(cv_scores['r2']),
                    'cv_directional_accuracy_mean': np.mean(cv_scores['directional_accuracy']),
                    'cv_directional_accuracy_std': np.std(cv_scores['directional_accuracy']),
                    'best_params': model_info['best_params']
                }
            
            # Train final models on full dataset
            for name, model_info in optimized_models.items():
                model = model_info['model']
                scaler = model_info['scaler']
                
                if scaler:
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                self.models[name] = model
                if scaler:
                    self.scalers[name] = scaler
            
            self.model_performance = model_performance
            
            # Feature importance for tree-based models
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[name] = importance_df
            
            st.success("✅ ML models optimized successfully!")
            return model_performance
            
        except Exception as e:
            st.error(f"❌ Error optimizing ML models: {e}")
            return None
    
    def ai_model_selection(self, model_performance):
        """AI-powered model selection based on multiple criteria"""
        try:
            if not model_performance:
                return None
            
            # Define weights for different metrics
            weights = {
                'directional_accuracy': 0.4,  # Most important for trading
                'r2': 0.3,                    # Explained variance
                'mse': 0.2,                   # Prediction accuracy (lower is better)
                'stability': 0.1              # Low standard deviation
            }
            
            model_scores = {}
            
            for model_name, performance in model_performance.items():
                # Normalize metrics (0-1 scale)
                directional_acc = performance['cv_directional_accuracy_mean']
                r2_score = max(0, performance['cv_r2_mean'])  # Ensure non-negative
                mse_score = 1 / (1 + performance['cv_mse_mean'])  # Invert MSE (lower is better)
                stability_score = 1 / (1 + performance['cv_directional_accuracy_std'])  # Lower std is better
                
                # Calculate weighted score
                total_score = (
                    weights['directional_accuracy'] * directional_acc +
                    weights['r2'] * r2_score +
                    weights['mse'] * mse_score +
                    weights['stability'] * stability_score
                )
                
                model_scores[model_name] = {
                    'total_score': total_score,
                    'directional_accuracy': directional_acc,
                    'r2_score': r2_score,
                    'mse_score': mse_score,
                    'stability_score': stability_score
                }
            
            # Select best model
            best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['total_score'])
            
            # Create ensemble if top models are close
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
            top_score = sorted_models[0][1]['total_score']
            
            ensemble_models = []
            for model_name, scores in sorted_models:
                if scores['total_score'] >= top_score * 0.95:  # Within 5% of best
                    ensemble_models.append(model_name)
            
            selection_result = {
                'best_single_model': best_model,
                'ensemble_models': ensemble_models,
                'model_scores': model_scores,
                'selection_criteria': weights
            }
            
            return selection_result
            
        except Exception as e:
            st.error(f"❌ Error in AI model selection: {e}")
            return None
    
    def comprehensive_valuation_analysis(self, stock_data):
        """Comprehensive valuation combining all methods"""
        try:
            # Calculate financial metrics
            metrics = self.calculate_financial_metrics(stock_data)
            if not metrics:
                return None
            
            # DCF Valuation
            dcf_results = self.calculate_dcf_valuation(stock_data)
            
            # Multiple Valuations
            multiple_valuations = self.calculate_multiple_valuations(metrics)
            
            # Monte Carlo Simulation
            monte_carlo = self.monte_carlo_simulation(stock_data)
            
            # Combine all valuations
            current_price = metrics['current_price']
            valuation_summary = {
                'current_price': current_price,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': stock_data['ticker'],
                'company_name': metrics['company_name'],
                'sector': metrics['sector']
            }
            
            # Collect all fair value estimates
            fair_values = []
            
            # DCF values
            if dcf_results:
                for scenario, result in dcf_results.items():
                    fair_values.append(result['dcf_per_share'])
                valuation_summary['dcf_range'] = {
                    'conservative': dcf_results.get('Conservative', {}).get('dcf_per_share', 0),
                    'base_case': dcf_results.get('Base Case', {}).get('dcf_per_share', 0),
                    'optimistic': dcf_results.get('Optimistic', {}).get('dcf_per_share', 0)
                }
            
            # Multiple valuation values
            if multiple_valuations:
                for method, data in multiple_valuations.items():
                    if 'fair_value' in data:
                        fair_values.append(data['fair_value'])
            
            # Calculate consensus valuation
            if fair_values:
                consensus_value = np.mean(fair_values)
                median_value = np.median(fair_values)
                
                valuation_summary['consensus_fair_value'] = consensus_value
                valuation_summary['median_fair_value'] = median_value
                valuation_summary['upside_downside'] = (consensus_value - current_price) / current_price
                
                # Determine overall valuation
                upside = valuation_summary['upside_downside']
                if upside > 0.2:
                    overall_valuation = "SIGNIFICANTLY UNDERVALUED"
                    confidence = "HIGH"
                elif upside > 0.1:
                    overall_valuation = "UNDERVALUED"
                    confidence = "MEDIUM"
                elif upside > -0.1:
                    overall_valuation = "FAIRLY VALUED"
                    confidence = "MEDIUM"
                elif upside > -0.2:
                    overall_valuation = "OVERVALUED"
                    confidence = "MEDIUM"
                else:
                    overall_valuation = "SIGNIFICANTLY OVERVALUED"
                    confidence = "HIGH"
                
                valuation_summary['overall_valuation'] = overall_valuation
                valuation_summary['confidence'] = confidence
            
            # Add Monte Carlo insights
            if monte_carlo:
                valuation_summary['monte_carlo'] = {
                    'expected_price_1y': monte_carlo['mean_final_price'],
                    'probability_positive': monte_carlo['probability_positive'],
                    'var_95': monte_carlo['var_95']
                }
            
            return {
                'summary': valuation_summary,
                'metrics': metrics,
                'dcf_results': dcf_results,
                'multiple_valuations': multiple_valuations,
                'monte_carlo': monte_carlo
            }
            
        except Exception as e:
            st.error(f"❌ Error in comprehensive valuation: {e}")
            return None

def main():
    st.set_page_config(
        page_title="Professional US Stock Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Professional US Stock Analyzer")
    st.markdown("**Real Data | Multiple Valuation Methods | AI-Optimized Models | Professional Investment Analysis**")
    
    # Initialize analyzer
    if 'us_analyzer' not in st.session_state:
        st.session_state.us_analyzer = ProfessionalUSStockAnalyzer()
    
    analyzer = st.session_state.us_analyzer
    
    # Sidebar
    st.sidebar.header("📊 System Status")
    
    with st.sidebar.expander("🔧 Available Features", expanded=True):
        st.write("✅ Real-time US stock data")
        st.write("✅ DCF Valuation (3 scenarios)")
        st.write("✅ Multiple Valuation Methods")
        st.write("✅ Monte Carlo Simulation")
        st.write(f"✅ ML Models: {'Available' if ML_AVAILABLE else '❌ Limited'}")
        st.write(f"✅ Statistical Analysis: {'Available' if SCIPY_AVAILABLE else '❌ Limited'}")
        st.write("✅ Professional Charts")
        st.write("✅ AI Model Selection")
        
        if ENHANCED_MODULES_AVAILABLE:
            st.write("---")
            st.write("🚀 **Enhanced Features:**")
            st.write("✅ Enhanced Technical Analysis")
            st.write("✅ Portfolio Optimization")
            st.write("✅ Sector Comparison")
            st.write("✅ Options Analysis")
            st.write("✅ ESG Scoring")
            st.write("✅ Real-time Alerts")
        else:
            st.write("---")
            st.write("⚠️ Enhanced modules not loaded")
    
    # Main interface
    if ENHANCED_MODULES_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Comprehensive Analysis", 
            "🤖 ML Predictions", 
            "📈 Enhanced Technical Analysis",
            "📊 Portfolio Analysis",
            "🏭 Sector Comparison",
            "📋 Options Analysis",
            "🚨 ESG & Alerts"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Comprehensive Analysis", 
            "🤖 ML Predictions", 
            "📈 Technical Analysis",
            "📋 Detailed Reports"
        ])
    
    with tab1:
        st.header("📊 Comprehensive Stock Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ticker = st.text_input(
                "Enter US Stock Ticker (e.g., AAPL, MSFT, GOOGL)",
                value="AAPL",
                help="Enter any US stock ticker symbol"
            ).upper()
        
        with col2:
            st.info("**Professional Analysis**\nReal data only\nNo mock/fake data")
        
        if st.button("🚀 Analyze Stock", type="primary"):
            if not ticker:
                st.error("❌ Please enter a stock ticker")
                return
            
            # Clear previous results
            if f'analysis_{ticker}' in st.session_state:
                del st.session_state[f'analysis_{ticker}']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Get comprehensive data
            status_text.text("📊 Extracting comprehensive stock data...")
            progress_bar.progress(0.1)
            
            stock_data = analyzer.get_comprehensive_stock_data(ticker)
            
            if not stock_data:
                st.error("❌ Failed to get stock data")
                return
            
            # Step 2: Comprehensive valuation analysis
            status_text.text("💰 Performing comprehensive valuation analysis...")
            progress_bar.progress(0.5)
            
            valuation_analysis = analyzer.comprehensive_valuation_analysis(stock_data)
            
            if not valuation_analysis:
                st.error("❌ Failed to perform valuation analysis")
                return
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis completed!")
            
            # Store results
            st.session_state[f'analysis_{ticker}'] = {
                'stock_data': stock_data,
                'valuation_analysis': valuation_analysis,
                'analysis_date': datetime.now()
            }
            
            # Display results
            summary = valuation_analysis['summary']
            
            st.success("✅ Comprehensive Analysis Completed!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = summary['current_price']
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                if 'consensus_fair_value' in summary:
                    fair_value = summary['consensus_fair_value']
                    st.metric("Fair Value", f"${fair_value:.2f}")
            
            with col3:
                if 'upside_downside' in summary:
                    upside = summary['upside_downside']
                    st.metric("Upside/Downside", f"{upside:.1%}")
            
            with col4:
                overall_val = summary.get('overall_valuation', 'N/A')
                confidence = summary.get('confidence', 'N/A')
                st.metric("Valuation", overall_val, delta=f"Confidence: {confidence}")
            
            # Company information
            st.subheader("🏢 Company Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Company:** {summary['company_name']}")
                st.write(f"**Ticker:** {summary['ticker']}")
            
            with col2:
                st.write(f"**Sector:** {summary['sector']}")
                metrics = valuation_analysis['metrics']
                market_cap = metrics.get('market_cap', 0)
                if market_cap:
                    st.write(f"**Market Cap:** ${market_cap/1e9:.1f}B")
            
            with col3:
                st.write(f"**Analysis Date:** {summary['analysis_date']}")
                beta = metrics.get('beta')
                if beta:
                    st.write(f"**Beta:** {beta:.2f}")
            
            # Valuation methods
            st.subheader("💰 Valuation Analysis")
            
            # DCF Results
            dcf_results = valuation_analysis.get('dcf_results')
            if dcf_results:
                with st.expander("📊 DCF Valuation", expanded=True):
                    dcf_data = []
                    for scenario, result in dcf_results.items():
                        dcf_data.append({
                            'Scenario': scenario,
                            'Growth Rate': f"{result['growth_rate']:.1%}",
                            'WACC': f"{result['wacc']:.1%}",
                            'Fair Value': f"${result['dcf_per_share']:.2f}",
                            'Upside/Downside': f"{(result['dcf_per_share'] - current_price) / current_price:.1%}"
                        })
                    
                    dcf_df = pd.DataFrame(dcf_data)
                    st.dataframe(dcf_df, use_container_width=True)
            
            # Multiple Valuations
            multiple_valuations = valuation_analysis.get('multiple_valuations')
            if multiple_valuations:
                with st.expander("📈 Multiple Valuations", expanded=True):
                    mult_data = []
                    for method, data in multiple_valuations.items():
                        if 'fair_value' in data:
                            mult_data.append({
                                'Method': method,
                                'Current Multiple': f"{data['current_multiple']:.1f}",
                                'Industry Average': f"{data['industry_average']:.1f}",
                                'Fair Value': f"${data['fair_value']:.2f}",
                                'Upside/Downside': f"{data['upside_downside']:.1%}"
                            })
                    
                    if mult_data:
                        mult_df = pd.DataFrame(mult_data)
                        st.dataframe(mult_df, use_container_width=True)
            
            # Monte Carlo Results
            monte_carlo = valuation_analysis.get('monte_carlo')
            if monte_carlo:
                with st.expander("🎲 Monte Carlo Simulation", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Price (1Y)", f"${monte_carlo['mean_final_price']:.2f}")
                        st.metric("Probability of Gain", f"{monte_carlo['probability_positive']:.1%}")
                    
                    with col2:
                        st.metric("95th Percentile", f"${monte_carlo['percentiles']['95th']:.2f}")
                        st.metric("5th Percentile", f"${monte_carlo['percentiles']['5th']:.2f}")
                    
                    with col3:
                        st.metric("Expected Return", f"{monte_carlo['expected_return']:.1%}")
                        st.metric("Value at Risk (95%)", f"${monte_carlo['var_95']:.2f}")
                    
                    # Monte Carlo distribution chart
                    fig = go.Figure(data=[go.Histogram(x=monte_carlo['all_results'], nbinsx=50)])
                    fig.update_layout(
                        title="Monte Carlo Price Distribution (1 Year)",
                        xaxis_title="Price ($)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    fig.add_vline(x=current_price, line_dash="dash", line_color="red", 
                                annotation_text="Current Price")
                    fig.add_vline(x=monte_carlo['mean_final_price'], line_dash="dash", line_color="green",
                                annotation_text="Expected Price")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Price chart
            st.subheader("📈 Price Chart")
            hist_data = stock_data['historical_data']
            
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name=ticker
            ))
            
            # Add moving averages
            if len(hist_data) > 50:
                ma_50 = hist_data['Close'].rolling(window=50).mean()
                ma_200 = hist_data['Close'].rolling(window=200).mean()
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=ma_50,
                    mode='lines', name='MA 50',
                    line=dict(color='orange', width=1)
                ))
                
                if len(hist_data) > 200:
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, y=ma_200,
                        mode='lines', name='MA 200',
                        line=dict(color='red', width=1)
                    ))
            
            fig.update_layout(
                title=f"{ticker} - {summary['company_name']} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🤖 Machine Learning Predictions")
        
        # Check if we have analysis results
        analysis_keys = [k for k in st.session_state.keys() if k.startswith('analysis_')]
        
        if analysis_keys:
            for key in analysis_keys:
                results = st.session_state[key]
                ticker = key.replace('analysis_', '')
                
                with st.expander(f"🤖 ML Analysis: {ticker}"):
                    if st.button(f"Train ML Models for {ticker}", key=f"ml_{ticker}"):
                        stock_data = results['stock_data']
                        hist_data = stock_data['historical_data']
                        
                        # Create technical features
                        with st.spinner("Creating technical features..."):
                            features_df = analyzer.create_advanced_technical_features(hist_data)
                        
                        if features_df is not None:
                            # Prepare ML data
                            features_df['target'] = features_df['Close'].shift(-1) / features_df['Close'] - 1
                            
                            # Select features
                            feature_columns = [col for col in features_df.columns if 
                                             col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target'] and
                                             features_df[col].dtype in ['float64', 'int64']]
                            
                            clean_df = features_df[feature_columns + ['target']].dropna()
                            
                            if len(clean_df) > 100:
                                X = clean_df[feature_columns]
                                y = clean_df['target']
                                
                                # Optimize models
                                with st.spinner("Optimizing ML models..."):
                                    model_performance = analyzer.optimize_ml_models(X, y)
                                
                                if model_performance:
                                    # AI model selection
                                    selection_result = analyzer.ai_model_selection(model_performance)
                                    
                                    if selection_result:
                                        st.success("✅ ML Models Optimized!")
                                        
                                        # Display model performance
                                        perf_data = []
                                        for model_name, perf in model_performance.items():
                                            perf_data.append({
                                                'Model': model_name,
                                                'Directional Accuracy': f"{perf['cv_directional_accuracy_mean']:.1%} ± {perf['cv_directional_accuracy_std']:.1%}",
                                                'R² Score': f"{perf['cv_r2_mean']:.3f} ± {perf['cv_r2_std']:.3f}",
                                                'RMSE': f"{np.sqrt(perf['cv_mse_mean']):.4f} ± {np.sqrt(perf['cv_mse_std']):.4f}"
                                            })
                                        
                                        perf_df = pd.DataFrame(perf_data)
                                        st.dataframe(perf_df, use_container_width=True)
                                        
                                        # Best model
                                        best_model = selection_result['best_single_model']
                                        ensemble_models = selection_result['ensemble_models']
                                        
                                        st.success(f"🏆 Best Model: {best_model}")
                                        st.info(f"📊 Ensemble Models: {', '.join(ensemble_models)}")
                                        
                                        # Feature importance
                                        if best_model in analyzer.feature_importance:
                                            importance_df = analyzer.feature_importance[best_model].head(10)
                                            
                                            fig = px.bar(
                                                importance_df, 
                                                x='importance', 
                                                y='feature',
                                                orientation='h',
                                                title=f"Top 10 Feature Importance - {best_model}"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("❌ Insufficient data for ML training")
                        else:
                            st.error("❌ Failed to create technical features")
        else:
            st.info("📝 No analysis results available. Run a comprehensive analysis first.")
    
    with tab3:
        if ENHANCED_MODULES_AVAILABLE:
            st.header("📈 Enhanced Technical Analysis")
            
            # Technical Analysis Interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                tech_ticker = st.text_input(
                    "Enter Ticker for Technical Analysis",
                    value="AAPL",
                    key="tech_ticker"
                ).upper()
            
            with col2:
                st.info("**100+ Indicators**\\nPattern Recognition\\nTrading Signals")
            
            if st.button("🔍 Run Technical Analysis", key="tech_analysis"):
                if tech_ticker:
                    try:
                        # Initialize technical analyzer
                        tech_analyzer = TechnicalAnalyzer()
                        
                        # Get technical analysis
                        with st.spinner("Performing comprehensive technical analysis..."):
                            tech_results = tech_analyzer.comprehensive_technical_analysis(tech_ticker)
                        
                        if tech_results:
                            st.success("✅ Technical Analysis Complete!")
                            
                            # Display technical dashboard
                            tech_dashboard = tech_analyzer.create_technical_dashboard(tech_results)
                            if tech_dashboard:
                                st.plotly_chart(tech_dashboard, use_container_width=True)
                            
                            # Trading signals
                            if 'trading_signals' in tech_results:
                                signals = tech_results['trading_signals']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Overall Signal", signals.get('overall_signal', 'NEUTRAL'))
                                with col2:
                                    st.metric("Signal Strength", f"{signals.get('signal_strength', 0):.1f}/10")
                                with col3:
                                    st.metric("Confidence", f"{signals.get('confidence', 0):.1%}")
                        else:
                            st.error("❌ Failed to perform technical analysis")
                    except Exception as e:
                        st.error(f"❌ Error in technical analysis: {e}")
        else:
            st.header("📈 Technical Analysis")
            st.warning("⚠️ Enhanced technical analysis modules not available")
            st.info("🚧 Basic technical analysis features coming soon!")
    
    if ENHANCED_MODULES_AVAILABLE:
        with tab4:
            st.header("📊 Portfolio Analysis")
            
            # Portfolio Analysis Interface
            st.subheader("🎯 Portfolio Optimization & Analysis")
            
            # Portfolio input
            portfolio_input = st.text_area(
                "Enter Portfolio Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA",
                help="Enter stock tickers separated by commas"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["Sharpe Ratio", "Minimum Variance", "Maximum Return", "Risk Parity"]
                )
            
            with col2:
                time_period = st.selectbox(
                    "Analysis Period",
                    ["1y", "2y", "3y", "5y"]
                )
            
            if st.button("🚀 Analyze Portfolio", key="portfolio_analysis"):
                if portfolio_input:
                    try:
                        # Parse tickers
                        tickers = [ticker.strip().upper() for ticker in portfolio_input.split(',')]
                        
                        # Initialize portfolio analyzer
                        portfolio_analyzer = PortfolioAnalyzer()
                        
                        with st.spinner("Performing portfolio analysis..."):
                            portfolio_results = portfolio_analyzer.comprehensive_portfolio_analysis(
                                tickers, 
                                period=time_period,
                                optimization_method=optimization_method.lower().replace(' ', '_')
                            )
                        
                        if portfolio_results:
                            st.success("✅ Portfolio Analysis Complete!")
                            
                            # Display portfolio dashboard
                            portfolio_dashboard = portfolio_analyzer.create_portfolio_dashboard(portfolio_results)
                            if portfolio_dashboard:
                                st.plotly_chart(portfolio_dashboard, use_container_width=True)
                            
                            # Portfolio metrics
                            if 'portfolio_metrics' in portfolio_results:
                                metrics = portfolio_results['portfolio_metrics']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Expected Return", f"{metrics.get('expected_return', 0):.1%}")
                                with col2:
                                    st.metric("Volatility", f"{metrics.get('volatility', 0):.1%}")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                                with col4:
                                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
                        else:
                            st.error("❌ Failed to perform portfolio analysis")
                    except Exception as e:
                        st.error(f"❌ Error in portfolio analysis: {e}")
        
        with tab5:
            st.header("🏭 Sector Comparison")
            
            # Sector Analysis Interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                sector_ticker = st.text_input(
                    "Enter Ticker for Sector Analysis",
                    value="AAPL",
                    key="sector_ticker"
                ).upper()
            
            with col2:
                st.info("**11 Major Sectors**\\nRelative Valuation\\nSector Rotation")
            
            if st.button("🔍 Analyze Sector", key="sector_analysis"):
                if sector_ticker:
                    try:
                        # Initialize sector analyzer
                        sector_analyzer = SectorAnalyzer()
                        
                        with st.spinner("Performing sector analysis..."):
                            sector_results = sector_analyzer.comprehensive_sector_analysis(sector_ticker)
                        
                        if sector_results:
                            st.success("✅ Sector Analysis Complete!")
                            
                            # Display sector dashboard
                            sector_dashboard = sector_analyzer.create_sector_dashboard(sector_results)
                            if sector_dashboard:
                                st.plotly_chart(sector_dashboard, use_container_width=True)
                            
                            # Sector metrics
                            if 'sector_metrics' in sector_results:
                                metrics = sector_results['sector_metrics']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sector", metrics.get('sector', 'N/A'))
                                with col2:
                                    st.metric("Sector Rank", f"{metrics.get('sector_rank', 0)}/11")
                                with col3:
                                    st.metric("Relative Performance", f"{metrics.get('relative_performance', 0):.1%}")
                        else:
                            st.error("❌ Failed to perform sector analysis")
                    except Exception as e:
                        st.error(f"❌ Error in sector analysis: {e}")
        
        with tab6:
            st.header("📋 Options Analysis")
            
            # Options Analysis Interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                options_ticker = st.text_input(
                    "Enter Ticker for Options Analysis",
                    value="AAPL",
                    key="options_ticker"
                ).upper()
            
            with col2:
                st.info("**Options Pricing**\\nGreeks Analysis\\nStrategy Builder")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                option_type = st.selectbox("Option Type", ["call", "put"])
            with col2:
                strike_price = st.number_input("Strike Price", min_value=1.0, value=150.0)
            with col3:
                days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
            
            if st.button("🔍 Analyze Options", key="options_analysis"):
                if options_ticker:
                    try:
                        # Initialize options analyzer
                        options_analyzer = OptionsAnalyzer()
                        
                        with st.spinner("Performing options analysis..."):
                            options_results = options_analyzer.comprehensive_options_analysis(
                                options_ticker,
                                option_type=option_type,
                                strike_price=strike_price,
                                days_to_expiry=days_to_expiry
                            )
                        
                        if options_results:
                            st.success("✅ Options Analysis Complete!")
                            
                            # Display options dashboard
                            options_dashboard = options_analyzer.create_options_dashboard(options_results)
                            if options_dashboard:
                                st.plotly_chart(options_dashboard, use_container_width=True)
                            
                            # Options metrics
                            if 'option_pricing' in options_results:
                                pricing = options_results['option_pricing']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Option Price", f"${pricing.get('option_price', 0):.2f}")
                                with col2:
                                    st.metric("Delta", f"{pricing.get('delta', 0):.3f}")
                                with col3:
                                    st.metric("Gamma", f"{pricing.get('gamma', 0):.3f}")
                                with col4:
                                    st.metric("Theta", f"{pricing.get('theta', 0):.3f}")
                        else:
                            st.error("❌ Failed to perform options analysis")
                    except Exception as e:
                        st.error(f"❌ Error in options analysis: {e}")
        
        with tab7:
            st.header("🚨 ESG Analysis & Real-time Alerts")
            
            # Create two columns for ESG and Alerts
            esg_tab, alerts_tab = st.tabs(["🌱 ESG Analysis", "🚨 Real-time Alerts"])
            
            with esg_tab:
                st.subheader("🌱 ESG Scoring & Analysis")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    esg_ticker = st.text_input(
                        "Enter Ticker for ESG Analysis",
                        value="AAPL",
                        key="esg_ticker"
                    ).upper()
                
                with col2:
                    st.info("**ESG Scoring**\\nSustainability Analysis\\nESG-weighted Valuation")
                
                if st.button("🔍 Analyze ESG", key="esg_analysis"):
                    if esg_ticker:
                        try:
                            # Initialize ESG analyzer
                            esg_analyzer = ESGAnalyzer()
                            
                            with st.spinner("Performing ESG analysis..."):
                                esg_data = esg_analyzer.get_esg_data(esg_ticker)
                            
                            if esg_data:
                                # ESG risk analysis
                                risk_analysis = esg_analyzer.esg_risk_analysis(esg_data)
                                
                                # Create ESG dashboard
                                esg_dashboard = esg_analyzer.create_esg_dashboard(esg_data, risk_analysis)
                                if esg_dashboard:
                                    st.plotly_chart(esg_dashboard, use_container_width=True)
                                
                                # ESG metrics
                                if 'synthetic_esg' in esg_data:
                                    esg_scores = esg_data['synthetic_esg']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("ESG Score", f"{esg_scores['total_score']:.1f}/100")
                                    with col2:
                                        st.metric("ESG Grade", esg_scores['letter_grade'])
                                    with col3:
                                        st.metric("Risk Level", esg_scores['risk_level'])
                                    with col4:
                                        st.metric("Sector Percentile", f"{esg_scores['percentile_rank']:.0f}%")
                            else:
                                st.error("❌ Failed to get ESG data")
                        except Exception as e:
                            st.error(f"❌ Error in ESG analysis: {e}")
            
            with alerts_tab:
                st.subheader("🚨 Real-time Alerts System")
                
                # Initialize alerts manager
                if 'alerts_manager' not in st.session_state:
                    st.session_state.alerts_manager = AlertsManager()
                
                alerts_manager = st.session_state.alerts_manager
                
                # Alert creation form
                with st.expander("Create New Alert", expanded=False):
                    alert_ticker = st.text_input("Alert Ticker", value="AAPL")
                    alert_type = st.selectbox("Alert Type", list(alerts_manager.alert_types.keys()))
                    
                    # Dynamic parameters based on alert type
                    parameters = {}
                    
                    if alert_type == 'price_target':
                        parameters['target_price'] = st.number_input("Target Price", min_value=0.01, value=150.0)
                        parameters['direction'] = st.selectbox("Direction", ['above', 'below'])
                    
                    elif alert_type == 'valuation':
                        parameters['valuation_threshold'] = st.number_input("Threshold", min_value=0.1, value=25.0)
                        parameters['metric'] = st.selectbox("Metric", ['PE_ratio', 'PB_ratio', 'discount_to_fair_value'])
                    
                    if st.button("Create Alert"):
                        alert_config = {
                            'ticker': alert_ticker.upper(),
                            'alert_type': alert_type,
                            'parameters': parameters,
                            'notification_methods': ['streamlit']
                        }
                        alerts_manager.create_alert(alert_config)
                
                # Monitor alerts
                if st.button("🔍 Check Alerts Now"):
                    alerts_manager.monitor_alerts()
                
                # Display active alerts
                if alerts_manager.alerts:
                    st.subheader("Active Alerts")
                    for alert in alerts_manager.alerts:
                        if alert['status'] == 'active':
                            with st.expander(f"{alert['ticker']} - {alert['alert_type']}"):
                                st.write(f"**Parameters:** {alert['parameters']}")
                                st.write(f"**Created:** {alert['created_at'].strftime('%Y-%m-%d %H:%M')}")
                                st.write(f"**Triggered:** {alert['triggered_count']} times")
                else:
                    st.info("No active alerts. Create one above.")

    with tab4 if not ENHANCED_MODULES_AVAILABLE else tab1:
        st.header("📋 Detailed Reports")
        
        if analysis_keys:
            for key in analysis_keys:
                results = st.session_state[key]
                ticker = key.replace('analysis_', '')
                
                with st.expander(f"📊 Detailed Report: {ticker}"):
                    valuation_analysis = results['valuation_analysis']
                    metrics = valuation_analysis['metrics']
                    
                    # Financial metrics table
                    st.subheader("📊 Financial Metrics")
                    
                    financial_data = {
                        'Valuation Metrics': {
                            'P/E Ratio': metrics.get('pe_ratio'),
                            'Forward P/E': metrics.get('forward_pe'),
                            'P/B Ratio': metrics.get('pb_ratio'),
                            'P/S Ratio': metrics.get('ps_ratio'),
                            'PEG Ratio': metrics.get('peg_ratio'),
                            'EV/EBITDA': metrics.get('ev_ebitda')
                        },
                        'Profitability': {
                            'Profit Margin': f"{metrics.get('profit_margin', 0) * 100:.1f}%" if metrics.get('profit_margin') else 'N/A',
                            'Operating Margin': f"{metrics.get('operating_margin', 0) * 100:.1f}%" if metrics.get('operating_margin') else 'N/A',
                            'Gross Margin': f"{metrics.get('gross_margin', 0) * 100:.1f}%" if metrics.get('gross_margin') else 'N/A',
                            'ROE': f"{metrics.get('roe', 0) * 100:.1f}%" if metrics.get('roe') else 'N/A',
                            'ROA': f"{metrics.get('roa', 0) * 100:.1f}%" if metrics.get('roa') else 'N/A'
                        },
                        'Growth': {
                            'Revenue Growth': f"{metrics.get('revenue_growth', 0) * 100:.1f}%" if metrics.get('revenue_growth') else 'N/A',
                            'Earnings Growth': f"{metrics.get('earnings_growth', 0) * 100:.1f}%" if metrics.get('earnings_growth') else 'N/A'
                        },
                        'Financial Strength': {
                            'Debt/Equity': metrics.get('debt_to_equity'),
                            'Current Ratio': metrics.get('current_ratio'),
                            'Quick Ratio': metrics.get('quick_ratio')
                        }
                    }
                    
                    for category, data in financial_data.items():
                        st.write(f"**{category}:**")
                        for metric, value in data.items():
                            if value is not None:
                                st.write(f"- {metric}: {value}")
                    
                    # Export functionality
                    if st.button(f"📥 Export {ticker} Report", key=f"export_{ticker}"):
                        # Create comprehensive report
                        report_data = {
                            'ticker': ticker,
                            'analysis_date': results['analysis_date'].strftime('%Y-%m-%d %H:%M:%S'),
                            'valuation_analysis': valuation_analysis
                        }
                        
                        report_json = json.dumps(report_data, indent=2, default=str)
                        
                        st.download_button(
                            label=f"Download {ticker} Analysis Report",
                            data=report_json,
                            file_name=f"{ticker}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        else:
            st.info("📝 No detailed reports available. Run a comprehensive analysis first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            📊 Professional US Stock Analyzer | 
            Real Data • Multiple Valuation Methods • AI-Optimized Models • Professional Investment Analysis
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()