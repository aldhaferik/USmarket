#!/usr/bin/env python3
"""
Sector Comparison Tools
- Compare stocks within same sector
- Sector performance rankings
- Relative valuation within sectors
- Sector rotation analysis
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
from scipy.stats import percentileofscore

class SectorComparison:
    def __init__(self):
        # Major sector ETFs for benchmarking
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication Services': 'XLC'
        }
        
        # Sector-specific valuation multiples (updated 2024)
        self.sector_multiples = {
            'Technology': {
                'pe_median': 28, 'pe_75th': 45, 'pe_25th': 18,
                'pb_median': 6.5, 'pb_75th': 12, 'pb_25th': 3.2,
                'ps_median': 8.2, 'ps_75th': 15, 'ps_25th': 4.1,
                'ev_ebitda_median': 22, 'ev_ebitda_75th': 35, 'ev_ebitda_25th': 15
            },
            'Healthcare': {
                'pe_median': 24, 'pe_75th': 35, 'pe_25th': 16,
                'pb_median': 4.2, 'pb_75th': 7.5, 'pb_25th': 2.8,
                'ps_median': 5.8, 'ps_75th': 9.2, 'ps_25th': 3.1,
                'ev_ebitda_median': 18, 'ev_ebitda_75th': 28, 'ev_ebitda_25th': 12
            },
            'Financial Services': {
                'pe_median': 12, 'pe_75th': 16, 'pe_25th': 9,
                'pb_median': 1.3, 'pb_75th': 1.8, 'pb_25th': 0.9,
                'ps_median': 3.2, 'ps_75th': 4.8, 'ps_25th': 2.1,
                'ev_ebitda_median': 10, 'ev_ebitda_75th': 14, 'ev_ebitda_25th': 7
            },
            'Consumer Cyclical': {
                'pe_median': 18, 'pe_75th': 28, 'pe_25th': 12,
                'pb_median': 2.8, 'pb_75th': 4.5, 'pb_25th': 1.8,
                'ps_median': 1.8, 'ps_75th': 2.8, 'ps_25th': 1.1,
                'ev_ebitda_median': 12, 'ev_ebitda_75th': 18, 'ev_ebitda_25th': 8
            },
            'Consumer Defensive': {
                'pe_median': 22, 'pe_75th': 30, 'pe_25th': 16,
                'pb_median': 3.5, 'pb_75th': 5.2, 'pb_25th': 2.4,
                'ps_median': 2.2, 'ps_75th': 3.1, 'ps_25th': 1.5,
                'ev_ebitda_median': 15, 'ev_ebitda_75th': 20, 'ev_ebitda_25th': 11
            },
            'Industrials': {
                'pe_median': 20, 'pe_75th': 28, 'pe_25th': 14,
                'pb_median': 3.2, 'pb_75th': 4.8, 'pb_25th': 2.1,
                'ps_median': 2.5, 'ps_75th': 3.8, 'ps_25th': 1.6,
                'ev_ebitda_median': 14, 'ev_ebitda_75th': 20, 'ev_ebitda_25th': 10
            },
            'Energy': {
                'pe_median': 15, 'pe_75th': 22, 'pe_25th': 10,
                'pb_median': 1.8, 'pb_75th': 2.5, 'pb_25th': 1.2,
                'ps_median': 1.2, 'ps_75th': 1.8, 'ps_25th': 0.8,
                'ev_ebitda_median': 8, 'ev_ebitda_75th': 12, 'ev_ebitda_25th': 5
            },
            'Materials': {
                'pe_median': 16, 'pe_75th': 24, 'pe_25th': 11,
                'pb_median': 2.1, 'pb_75th': 3.2, 'pb_25th': 1.4,
                'ps_median': 1.5, 'ps_75th': 2.3, 'ps_25th': 1.0,
                'ev_ebitda_median': 9, 'ev_ebitda_75th': 14, 'ev_ebitda_25th': 6
            },
            'Real Estate': {
                'pe_median': 25, 'pe_75th': 35, 'pe_25th': 18,
                'pb_median': 1.9, 'pb_75th': 2.8, 'pb_25th': 1.3,
                'ps_median': 8.5, 'ps_75th': 12, 'ps_25th': 5.8,
                'ev_ebitda_median': 20, 'ev_ebitda_75th': 28, 'ev_ebitda_25th': 14
            },
            'Utilities': {
                'pe_median': 19, 'pe_75th': 25, 'pe_25th': 15,
                'pb_median': 1.6, 'pb_75th': 2.2, 'pb_25th': 1.2,
                'ps_median': 2.8, 'ps_75th': 3.8, 'ps_25th': 2.1,
                'ev_ebitda_median': 11, 'ev_ebitda_75th': 15, 'ev_ebitda_25th': 8
            },
            'Communication Services': {
                'pe_median': 21, 'pe_75th': 32, 'pe_25th': 14,
                'pb_median': 3.8, 'pb_75th': 6.2, 'pb_25th': 2.1,
                'ps_median': 4.2, 'ps_75th': 7.1, 'ps_25th': 2.5,
                'ev_ebitda_median': 16, 'ev_ebitda_75th': 24, 'ev_ebitda_25th': 11
            }
        }
        
        # Popular stocks by sector for comparison
        self.sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY', 'AMGN'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'F', 'GM'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA', 'EXR'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI']
        }
    
    def get_sector_data(self, sector, custom_tickers=None, period="1y"):
        """Get comprehensive data for a sector"""
        try:
            # Use custom tickers if provided, otherwise use default sector stocks
            tickers = custom_tickers if custom_tickers else self.sector_stocks.get(sector, [])
            
            if not tickers:
                st.error(f"❌ No tickers available for sector: {sector}")
                return None
            
            st.info(f"📊 Fetching data for {len(tickers)} {sector} stocks...")
            
            # Get historical price data
            price_data = yf.download(tickers, period=period, progress=False)
            
            if price_data.empty:
                st.error("❌ No price data retrieved")
                return None
            
            # Handle single ticker case
            if len(tickers) == 1:
                price_data = pd.DataFrame(price_data)
                price_data.columns = pd.MultiIndex.from_product([price_data.columns, tickers])
            
            # Get adjusted close prices
            if 'Adj Close' in price_data.columns.levels[0]:
                prices = price_data['Adj Close']
            else:
                prices = price_data['Close']
            
            # Get sector ETF data for benchmarking
            sector_etf = self.sector_etfs.get(sector)
            etf_data = None
            if sector_etf:
                try:
                    etf_data = yf.download(sector_etf, period=period, progress=False)['Adj Close']
                except:
                    st.warning(f"⚠️ Could not fetch ETF data for {sector_etf}")
            
            # Get fundamental data for each stock
            stock_fundamentals = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    stock_fundamentals[ticker] = {
                        'name': info.get('longName', ticker),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', None),
                        'forward_pe': info.get('forwardPE', None),
                        'pb_ratio': info.get('priceToBook', None),
                        'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                        'peg_ratio': info.get('pegRatio', None),
                        'ev_ebitda': info.get('enterpriseToEbitda', None),
                        'profit_margin': info.get('profitMargins', None),
                        'roe': info.get('returnOnEquity', None),
                        'debt_to_equity': info.get('debtToEquity', None),
                        'revenue_growth': info.get('revenueGrowth', None),
                        'earnings_growth': info.get('earningsGrowth', None),
                        'dividend_yield': info.get('dividendYield', None),
                        'beta': info.get('beta', None),
                        'current_price': prices[ticker].iloc[-1] if ticker in prices.columns else None,
                        '52_week_high': info.get('fiftyTwoWeekHigh', None),
                        '52_week_low': info.get('fiftyTwoWeekLow', None)
                    }
                except Exception as e:
                    st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
                    stock_fundamentals[ticker] = {'name': ticker, 'market_cap': 0}
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            sector_data = {
                'sector': sector,
                'tickers': tickers,
                'prices': prices,
                'returns': returns,
                'fundamentals': stock_fundamentals,
                'etf_data': etf_data,
                'etf_ticker': sector_etf
            }
            
            st.success(f"✅ Data retrieved for {sector} sector")
            return sector_data
            
        except Exception as e:
            st.error(f"❌ Error fetching sector data: {e}")
            return None
    
    def calculate_sector_rankings(self, sector_data):
        """Calculate comprehensive sector rankings"""
        try:
            fundamentals = sector_data['fundamentals']
            prices = sector_data['prices']
            returns = sector_data['returns']
            
            rankings = {}
            
            # Performance rankings
            if not returns.empty:
                # Calculate various return periods
                periods = {
                    '1M': 21,
                    '3M': 63,
                    '6M': 126,
                    '1Y': 252
                }
                
                performance_rankings = {}
                for period_name, days in periods.items():
                    if len(returns) >= days:
                        period_returns = (1 + returns.tail(days)).prod() - 1
                        performance_rankings[period_name] = period_returns.sort_values(ascending=False)
                
                rankings['performance'] = performance_rankings
            
            # Valuation rankings
            valuation_metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'peg_ratio']
            valuation_rankings = {}
            
            for metric in valuation_metrics:
                metric_values = {}
                for ticker, data in fundamentals.items():
                    if data.get(metric) and data[metric] > 0:
                        metric_values[ticker] = data[metric]
                
                if metric_values:
                    # Lower is better for valuation metrics
                    valuation_rankings[metric] = dict(sorted(metric_values.items(), key=lambda x: x[1]))
            
            rankings['valuation'] = valuation_rankings
            
            # Quality rankings (higher is better)
            quality_metrics = ['roe', 'profit_margin']
            quality_rankings = {}
            
            for metric in quality_metrics:
                metric_values = {}
                for ticker, data in fundamentals.items():
                    if data.get(metric) and data[metric] is not None:
                        metric_values[ticker] = data[metric]
                
                if metric_values:
                    # Higher is better for quality metrics
                    quality_rankings[metric] = dict(sorted(metric_values.items(), key=lambda x: x[1], reverse=True))
            
            rankings['quality'] = quality_rankings
            
            # Growth rankings (higher is better)
            growth_metrics = ['revenue_growth', 'earnings_growth']
            growth_rankings = {}
            
            for metric in growth_metrics:
                metric_values = {}
                for ticker, data in fundamentals.items():
                    if data.get(metric) and data[metric] is not None:
                        metric_values[ticker] = data[metric]
                
                if metric_values:
                    growth_rankings[metric] = dict(sorted(metric_values.items(), key=lambda x: x[1], reverse=True))
            
            rankings['growth'] = growth_rankings
            
            # Risk rankings
            if not returns.empty:
                volatilities = returns.std() * np.sqrt(252)  # Annualized volatility
                risk_rankings = volatilities.sort_values()  # Lower volatility is better
                rankings['risk'] = {'volatility': risk_rankings}
            
            return rankings
            
        except Exception as e:
            st.error(f"❌ Error calculating sector rankings: {e}")
            return None
    
    def relative_valuation_analysis(self, sector_data):
        """Perform relative valuation analysis within sector"""
        try:
            sector = sector_data['sector']
            fundamentals = sector_data['fundamentals']
            sector_multiples = self.sector_multiples.get(sector, {})
            
            relative_analysis = {}
            
            valuation_metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda']
            
            for metric in valuation_metrics:
                metric_analysis = {}
                metric_values = []
                
                # Collect all valid values for this metric
                for ticker, data in fundamentals.items():
                    value = data.get(metric)
                    if value and value > 0:
                        metric_values.append(value)
                        
                        # Compare to sector benchmarks
                        sector_median = sector_multiples.get(f'{metric.replace("_ratio", "").replace("_", "_")}_median')
                        sector_25th = sector_multiples.get(f'{metric.replace("_ratio", "").replace("_", "_")}_25th')
                        sector_75th = sector_multiples.get(f'{metric.replace("_ratio", "").replace("_", "_")}_75th')
                        
                        if sector_median:
                            percentile = percentileofscore(metric_values, value) if len(metric_values) > 1 else 50
                            
                            if value < sector_25th:
                                valuation_category = "Undervalued"
                            elif value > sector_75th:
                                valuation_category = "Overvalued"
                            else:
                                valuation_category = "Fairly Valued"
                            
                            metric_analysis[ticker] = {
                                'value': value,
                                'sector_median': sector_median,
                                'percentile': percentile,
                                'category': valuation_category,
                                'discount_premium': (value - sector_median) / sector_median
                            }
                
                if metric_analysis:
                    relative_analysis[metric] = metric_analysis
            
            # Overall valuation score
            overall_scores = {}
            for ticker in fundamentals.keys():
                scores = []
                for metric, analysis in relative_analysis.items():
                    if ticker in analysis:
                        # Convert percentile to score (lower percentile = better for valuation)
                        score = 100 - analysis[ticker]['percentile']
                        scores.append(score)
                
                if scores:
                    overall_scores[ticker] = np.mean(scores)
            
            # Rank by overall valuation attractiveness
            if overall_scores:
                relative_analysis['overall_ranking'] = dict(sorted(overall_scores.items(), key=lambda x: x[1], reverse=True))
            
            return relative_analysis
            
        except Exception as e:
            st.error(f"❌ Error in relative valuation analysis: {e}")
            return None
    
    def sector_rotation_analysis(self, sectors_list, period="1y"):
        """Analyze sector rotation patterns"""
        try:
            st.info("📊 Analyzing sector rotation patterns...")
            
            sector_performance = {}
            
            # Get ETF data for each sector
            for sector in sectors_list:
                etf_ticker = self.sector_etfs.get(sector)
                if etf_ticker:
                    try:
                        etf_data = yf.download(etf_ticker, period=period, progress=False)
                        if not etf_data.empty:
                            prices = etf_data['Adj Close']
                            returns = prices.pct_change().dropna()
                            
                            # Calculate performance metrics
                            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                            volatility = returns.std() * np.sqrt(252)
                            sharpe_ratio = (returns.mean() * 252 - 0.045) / volatility  # Assuming 4.5% risk-free rate
                            max_drawdown = self.calculate_max_drawdown(prices)
                            
                            sector_performance[sector] = {
                                'etf_ticker': etf_ticker,
                                'total_return': total_return,
                                'volatility': volatility,
                                'sharpe_ratio': sharpe_ratio,
                                'max_drawdown': max_drawdown,
                                'prices': prices,
                                'returns': returns
                            }
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch data for {sector}: {e}")
            
            # Analyze rotation patterns
            rotation_analysis = {}
            
            if len(sector_performance) >= 2:
                # Calculate correlations between sectors
                sector_returns = pd.DataFrame()
                for sector, data in sector_performance.items():
                    sector_returns[sector] = data['returns']
                
                correlation_matrix = sector_returns.corr()
                
                # Find sector pairs with low correlation (good for rotation)
                low_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) < 0.3:  # Low correlation threshold
                            low_corr_pairs.append({
                                'sector1': correlation_matrix.columns[i],
                                'sector2': correlation_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                # Rank sectors by recent performance
                recent_performance = {}
                for sector, data in sector_performance.items():
                    # Last 3 months performance
                    if len(data['returns']) >= 63:
                        recent_return = (1 + data['returns'].tail(63)).prod() - 1
                        recent_performance[sector] = recent_return
                
                rotation_analysis = {
                    'sector_performance': sector_performance,
                    'correlation_matrix': correlation_matrix,
                    'low_correlation_pairs': low_corr_pairs,
                    'recent_performance_ranking': dict(sorted(recent_performance.items(), key=lambda x: x[1], reverse=True))
                }
            
            return rotation_analysis
            
        except Exception as e:
            st.error(f"❌ Error in sector rotation analysis: {e}")
            return None
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            cumulative = prices / prices.iloc[0]
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
        except:
            return 0
    
    def create_sector_comparison_dashboard(self, sector_data, rankings, relative_analysis):
        """Create comprehensive sector comparison dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Price Performance Comparison', 'Valuation Metrics Comparison',
                    'Performance Rankings', 'Risk-Return Analysis',
                    'Market Cap Distribution', 'Sector vs ETF Performance'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "scatter"}]
                ]
            )
            
            # Price performance comparison
            prices = sector_data['prices']
            normalized_prices = prices / prices.iloc[0]  # Normalize to starting value
            
            for ticker in normalized_prices.columns:
                fig.add_trace(
                    go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices[ticker],
                        mode='lines',
                        name=ticker,
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add sector ETF if available
            if sector_data['etf_data'] is not None:
                etf_normalized = sector_data['etf_data'] / sector_data['etf_data'].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=etf_normalized.index,
                        y=etf_normalized,
                        mode='lines',
                        name=f"{sector_data['etf_ticker']} (Sector ETF)",
                        line=dict(width=3, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Valuation metrics comparison
            if 'valuation' in rankings and 'pe_ratio' in rankings['valuation']:
                pe_data = rankings['valuation']['pe_ratio']
                tickers = list(pe_data.keys())
                pe_values = list(pe_data.values())
                
                fig.add_trace(
                    go.Bar(
                        x=tickers,
                        y=pe_values,
                        name='P/E Ratio',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Performance rankings (1Y returns)
            if 'performance' in rankings and '1Y' in rankings['performance']:
                perf_data = rankings['performance']['1Y']
                tickers = list(perf_data.keys())
                returns = [r * 100 for r in perf_data.values()]  # Convert to percentage
                
                fig.add_trace(
                    go.Bar(
                        x=tickers,
                        y=returns,
                        name='1Y Returns (%)',
                        marker_color=['green' if r > 0 else 'red' for r in returns],
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Risk-return scatter
            fundamentals = sector_data['fundamentals']
            returns_data = sector_data['returns']
            
            if not returns_data.empty:
                annual_returns = returns_data.mean() * 252
                annual_volatilities = returns_data.std() * np.sqrt(252)
                
                fig.add_trace(
                    go.Scatter(
                        x=annual_volatilities,
                        y=annual_returns,
                        mode='markers+text',
                        text=list(annual_returns.index),
                        textposition='top center',
                        marker=dict(
                            size=[fundamentals[ticker]['market_cap']/1e9 if fundamentals[ticker]['market_cap'] else 1 
                                  for ticker in annual_returns.index],
                            sizemode='diameter',
                            sizeref=2.*max([fundamentals[ticker]['market_cap']/1e9 if fundamentals[ticker]['market_cap'] else 1 
                                           for ticker in annual_returns.index])/(40.**2),
                            sizemin=4
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Market cap distribution
            market_caps = {}
            for ticker, data in fundamentals.items():
                if data['market_cap'] and data['market_cap'] > 0:
                    market_caps[ticker] = data['market_cap'] / 1e9  # Convert to billions
            
            if market_caps:
                fig.add_trace(
                    go.Pie(
                        labels=list(market_caps.keys()),
                        values=list(market_caps.values()),
                        name="Market Cap Distribution",
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"{sector_data['sector']} Sector Analysis Dashboard",
                height=1200,
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
            fig.update_yaxes(title_text="P/E Ratio", row=1, col=2)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
            fig.update_xaxes(title_text="Volatility", row=2, col=2)
            fig.update_yaxes(title_text="Expected Return", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating sector comparison dashboard: {e}")
            return None
    
    def create_sector_rotation_dashboard(self, rotation_analysis):
        """Create sector rotation analysis dashboard"""
        try:
            if not rotation_analysis or 'sector_performance' not in rotation_analysis:
                st.warning("⚠️ No sector rotation data available")
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Sector Performance Comparison', 'Risk-Return by Sector',
                    'Sector Correlation Heatmap', 'Recent Performance Ranking'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "bar"}]
                ]
            )
            
            sector_performance = rotation_analysis['sector_performance']
            
            # Sector performance comparison
            for sector, data in sector_performance.items():
                prices = data['prices']
                normalized_prices = prices / prices.iloc[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices,
                        mode='lines',
                        name=sector,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Risk-return scatter
            sectors = list(sector_performance.keys())
            returns = [data['total_return'] for data in sector_performance.values()]
            volatilities = [data['volatility'] for data in sector_performance.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=volatilities,
                    y=returns,
                    mode='markers+text',
                    text=sectors,
                    textposition='top center',
                    marker=dict(size=12),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Correlation heatmap
            if 'correlation_matrix' in rotation_analysis:
                corr_matrix = rotation_analysis['correlation_matrix']
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        showscale=False
                    ),
                    row=2, col=1
                )
            
            # Recent performance ranking
            if 'recent_performance_ranking' in rotation_analysis:
                recent_perf = rotation_analysis['recent_performance_ranking']
                sectors = list(recent_perf.keys())
                performance = [p * 100 for p in recent_perf.values()]  # Convert to percentage
                
                fig.add_trace(
                    go.Bar(
                        x=sectors,
                        y=performance,
                        marker_color=['green' if p > 0 else 'red' for p in performance],
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Sector Rotation Analysis Dashboard",
                height=800,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
            fig.update_xaxes(title_text="Volatility", row=1, col=2)
            fig.update_yaxes(title_text="Total Return", row=1, col=2)
            fig.update_yaxes(title_text="3M Performance (%)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating sector rotation dashboard: {e}")
            return None