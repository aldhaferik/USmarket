#!/usr/bin/env python3
"""
Options Analysis Module
- Options pricing using Black-Scholes
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Options strategies analysis
- Volatility analysis and implied volatility
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

class OptionsAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.045  # 10-year Treasury rate
        
        # Common options strategies
        self.strategies = {
            'Long Call': {'type': 'bullish', 'risk': 'limited', 'reward': 'unlimited'},
            'Long Put': {'type': 'bearish', 'risk': 'limited', 'reward': 'high'},
            'Covered Call': {'type': 'neutral_bullish', 'risk': 'high', 'reward': 'limited'},
            'Cash-Secured Put': {'type': 'neutral_bullish', 'risk': 'high', 'reward': 'limited'},
            'Bull Call Spread': {'type': 'bullish', 'risk': 'limited', 'reward': 'limited'},
            'Bear Put Spread': {'type': 'bearish', 'risk': 'limited', 'reward': 'limited'},
            'Iron Condor': {'type': 'neutral', 'risk': 'limited', 'reward': 'limited'},
            'Straddle': {'type': 'volatile', 'risk': 'limited', 'reward': 'unlimited'},
            'Strangle': {'type': 'volatile', 'risk': 'limited', 'reward': 'unlimited'},
            'Butterfly': {'type': 'neutral', 'risk': 'limited', 'reward': 'limited'}
        }
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes call option price"""
        try:
            if T <= 0:
                return max(S - K, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return max(call_price, 0)
            
        except Exception as e:
            st.error(f"❌ Error calculating call price: {e}")
            return 0
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Calculate Black-Scholes put option price"""
        try:
            if T <= 0:
                return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return max(put_price, 0)
            
        except Exception as e:
            st.error(f"❌ Error calculating put price: {e}")
            return 0
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {
                    'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:  # put
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:  # put
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def calculate_implied_volatility(self, market_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            if T <= 0:
                return 0
            
            def objective(sigma):
                if option_type == 'call':
                    theoretical_price = self.black_scholes_call(S, K, T, r, sigma)
                else:
                    theoretical_price = self.black_scholes_put(S, K, T, r, sigma)
                return abs(theoretical_price - market_price)
            
            # Use scipy's minimize_scalar for robust optimization
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            
            if result.success:
                return result.x
            else:
                return 0.2  # Default volatility if optimization fails
                
        except Exception as e:
            st.warning(f"⚠️ Error calculating implied volatility: {e}")
            return 0.2
    
    def get_options_data(self, ticker):
        """Get real options data from Yahoo Finance"""
        try:
            st.info(f"📊 Fetching options data for {ticker}...")
            
            stock = yf.Ticker(ticker)
            
            # Get current stock price
            hist = stock.history(period="1d")
            if hist.empty:
                st.error(f"❌ No stock data for {ticker}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Get options expiration dates
            expiration_dates = stock.options
            if not expiration_dates:
                st.error(f"❌ No options data available for {ticker}")
                return None
            
            # Get options chain for the first few expiration dates
            options_data = {}
            
            for i, exp_date in enumerate(expiration_dates[:5]):  # Limit to first 5 expiration dates
                try:
                    options_chain = stock.option_chain(exp_date)
                    
                    calls = options_chain.calls
                    puts = options_chain.puts
                    
                    # Calculate time to expiration
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    time_to_exp = (exp_datetime - datetime.now()).days / 365.0
                    
                    if time_to_exp > 0:
                        options_data[exp_date] = {
                            'calls': calls,
                            'puts': puts,
                            'time_to_expiration': time_to_exp,
                            'expiration_date': exp_datetime
                        }
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not fetch options for {exp_date}: {e}")
                    continue
            
            if not options_data:
                st.error("❌ No valid options data retrieved")
                return None
            
            # Calculate historical volatility
            hist_data = stock.history(period="1y")
            if not hist_data.empty:
                returns = hist_data['Close'].pct_change().dropna()
                historical_volatility = returns.std() * np.sqrt(252)
            else:
                historical_volatility = 0.2
            
            result = {
                'ticker': ticker,
                'current_price': current_price,
                'options_data': options_data,
                'historical_volatility': historical_volatility,
                'risk_free_rate': self.risk_free_rate
            }
            
            st.success(f"✅ Options data retrieved for {ticker}")
            return result
            
        except Exception as e:
            st.error(f"❌ Error fetching options data: {e}")
            return None
    
    def analyze_options_chain(self, options_data):
        """Analyze options chain and calculate theoretical prices"""
        try:
            ticker = options_data['ticker']
            current_price = options_data['current_price']
            historical_vol = options_data['historical_volatility']
            r = options_data['risk_free_rate']
            
            analysis_results = {}
            
            for exp_date, data in options_data['options_data'].items():
                T = data['time_to_expiration']
                calls = data['calls'].copy()
                puts = data['puts'].copy()
                
                # Analyze calls
                if not calls.empty:
                    calls['theoretical_price'] = calls['strike'].apply(
                        lambda K: self.black_scholes_call(current_price, K, T, r, historical_vol)
                    )
                    calls['price_difference'] = calls['lastPrice'] - calls['theoretical_price']
                    calls['implied_volatility'] = calls.apply(
                        lambda row: self.calculate_implied_volatility(
                            row['lastPrice'], current_price, row['strike'], T, r, 'call'
                        ), axis=1
                    )
                    
                    # Calculate Greeks for calls
                    greeks_calls = calls.apply(
                        lambda row: pd.Series(self.calculate_greeks(
                            current_price, row['strike'], T, r, row['implied_volatility'], 'call'
                        )), axis=1
                    )
                    calls = pd.concat([calls, greeks_calls], axis=1)
                
                # Analyze puts
                if not puts.empty:
                    puts['theoretical_price'] = puts['strike'].apply(
                        lambda K: self.black_scholes_put(current_price, K, T, r, historical_vol)
                    )
                    puts['price_difference'] = puts['lastPrice'] - puts['theoretical_price']
                    puts['implied_volatility'] = puts.apply(
                        lambda row: self.calculate_implied_volatility(
                            row['lastPrice'], current_price, row['strike'], T, r, 'put'
                        ), axis=1
                    )
                    
                    # Calculate Greeks for puts
                    greeks_puts = puts.apply(
                        lambda row: pd.Series(self.calculate_greeks(
                            current_price, row['strike'], T, r, row['implied_volatility'], 'put'
                        )), axis=1
                    )
                    puts = pd.concat([puts, greeks_puts], axis=1)
                
                analysis_results[exp_date] = {
                    'calls': calls,
                    'puts': puts,
                    'time_to_expiration': T
                }
            
            return analysis_results
            
        except Exception as e:
            st.error(f"❌ Error analyzing options chain: {e}")
            return None
    
    def calculate_strategy_payoff(self, strategy_name, legs, spot_range):
        """Calculate payoff for options strategies"""
        try:
            payoffs = np.zeros(len(spot_range))
            total_premium = 0
            
            for leg in legs:
                option_type = leg['type']  # 'call' or 'put'
                position = leg['position']  # 'long' or 'short'
                strike = leg['strike']
                premium = leg['premium']
                quantity = leg.get('quantity', 1)
                
                # Calculate premium cost/credit
                if position == 'long':
                    total_premium -= premium * quantity
                else:  # short
                    total_premium += premium * quantity
                
                # Calculate payoff at expiration
                for i, S in enumerate(spot_range):
                    if option_type == 'call':
                        intrinsic_value = max(S - strike, 0)
                    else:  # put
                        intrinsic_value = max(strike - S, 0)
                    
                    if position == 'long':
                        payoffs[i] += intrinsic_value * quantity
                    else:  # short
                        payoffs[i] -= intrinsic_value * quantity
            
            # Add premium to payoff
            net_payoffs = payoffs + total_premium
            
            # Calculate key metrics
            max_profit = np.max(net_payoffs)
            max_loss = np.min(net_payoffs)
            breakeven_points = []
            
            # Find breakeven points (where payoff crosses zero)
            for i in range(len(net_payoffs) - 1):
                if (net_payoffs[i] <= 0 and net_payoffs[i + 1] > 0) or \
                   (net_payoffs[i] >= 0 and net_payoffs[i + 1] < 0):
                    # Linear interpolation to find exact breakeven
                    breakeven = spot_range[i] + (spot_range[i + 1] - spot_range[i]) * \
                               (-net_payoffs[i] / (net_payoffs[i + 1] - net_payoffs[i]))
                    breakeven_points.append(breakeven)
            
            return {
                'spot_range': spot_range,
                'payoffs': net_payoffs,
                'max_profit': max_profit if max_profit != np.inf else 'Unlimited',
                'max_loss': max_loss if max_loss != -np.inf else 'Unlimited',
                'breakeven_points': breakeven_points,
                'net_premium': total_premium
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating strategy payoff: {e}")
            return None
    
    def volatility_analysis(self, options_analysis):
        """Analyze implied volatility patterns"""
        try:
            vol_analysis = {}
            
            for exp_date, data in options_analysis.items():
                calls = data['calls']
                puts = data['puts']
                
                # Volatility smile/skew analysis
                if not calls.empty and 'implied_volatility' in calls.columns:
                    call_vol_data = calls[['strike', 'implied_volatility']].dropna()
                    call_vol_data = call_vol_data[call_vol_data['implied_volatility'] > 0]
                    
                if not puts.empty and 'implied_volatility' in puts.columns:
                    put_vol_data = puts[['strike', 'implied_volatility']].dropna()
                    put_vol_data = put_vol_data[put_vol_data['implied_volatility'] > 0]
                
                # Combine call and put volatility data
                vol_data = pd.concat([
                    call_vol_data.assign(type='call'),
                    put_vol_data.assign(type='put')
                ], ignore_index=True)
                
                if not vol_data.empty:
                    # Calculate volatility statistics
                    avg_iv = vol_data['implied_volatility'].mean()
                    iv_std = vol_data['implied_volatility'].std()
                    
                    # Identify volatility skew
                    if len(vol_data) > 2:
                        # Fit linear regression to identify skew
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(
                            vol_data['strike'], vol_data['implied_volatility']
                        )
                        
                        skew_direction = 'Negative' if slope < -0.001 else 'Positive' if slope > 0.001 else 'Flat'
                    else:
                        skew_direction = 'Insufficient Data'
                        slope = 0
                    
                    vol_analysis[exp_date] = {
                        'volatility_data': vol_data,
                        'average_iv': avg_iv,
                        'iv_std': iv_std,
                        'skew_direction': skew_direction,
                        'skew_slope': slope,
                        'time_to_expiration': data['time_to_expiration']
                    }
            
            return vol_analysis
            
        except Exception as e:
            st.error(f"❌ Error in volatility analysis: {e}")
            return None
    
    def create_options_dashboard(self, options_data, options_analysis, vol_analysis):
        """Create comprehensive options analysis dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Options Chain - Calls vs Puts', 'Implied Volatility Smile',
                    'Greeks Analysis', 'Put-Call Parity',
                    'Volume vs Open Interest', 'Time Decay Analysis'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            current_price = options_data['current_price']
            
            # Get data for the nearest expiration
            nearest_exp = min(options_analysis.keys())
            nearest_data = options_analysis[nearest_exp]
            
            calls = nearest_data['calls']
            puts = nearest_data['puts']
            
            # Options chain visualization
            if not calls.empty:
                fig.add_trace(
                    go.Scatter(
                        x=calls['strike'],
                        y=calls['lastPrice'],
                        mode='markers+lines',
                        name='Call Prices',
                        marker=dict(color='green')
                    ),
                    row=1, col=1
                )
            
            if not puts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=puts['strike'],
                        y=puts['lastPrice'],
                        mode='markers+lines',
                        name='Put Prices',
                        marker=dict(color='red')
                    ),
                    row=1, col=1
                )
            
            # Add current stock price line
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Current Price: ${current_price:.2f}",
                row=1, col=1
            )
            
            # Implied volatility smile
            if vol_analysis and nearest_exp in vol_analysis:
                vol_data = vol_analysis[nearest_exp]['volatility_data']
                
                call_vol = vol_data[vol_data['type'] == 'call']
                put_vol = vol_data[vol_data['type'] == 'put']
                
                if not call_vol.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=call_vol['strike'],
                            y=call_vol['implied_volatility'],
                            mode='markers+lines',
                            name='Call IV',
                            marker=dict(color='green')
                        ),
                        row=1, col=2
                    )
                
                if not put_vol.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=put_vol['strike'],
                            y=put_vol['implied_volatility'],
                            mode='markers+lines',
                            name='Put IV',
                            marker=dict(color='red')
                        ),
                        row=1, col=2
                    )
            
            # Greeks analysis (Delta)
            if not calls.empty and 'delta' in calls.columns:
                fig.add_trace(
                    go.Bar(
                        x=calls['strike'],
                        y=calls['delta'],
                        name='Call Delta',
                        marker_color='green',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            if not puts.empty and 'delta' in puts.columns:
                fig.add_trace(
                    go.Bar(
                        x=puts['strike'],
                        y=puts['delta'],
                        name='Put Delta',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Volume vs Open Interest
            if not calls.empty and 'volume' in calls.columns and 'openInterest' in calls.columns:
                fig.add_trace(
                    go.Scatter(
                        x=calls['volume'],
                        y=calls['openInterest'],
                        mode='markers',
                        name='Calls Vol vs OI',
                        marker=dict(color='green', size=8),
                        text=calls['strike'],
                        textposition='top center'
                    ),
                    row=3, col=1
                )
            
            if not puts.empty and 'volume' in puts.columns and 'openInterest' in puts.columns:
                fig.add_trace(
                    go.Scatter(
                        x=puts['volume'],
                        y=puts['openInterest'],
                        mode='markers',
                        name='Puts Vol vs OI',
                        marker=dict(color='red', size=8),
                        text=puts['strike'],
                        textposition='top center'
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"Options Analysis Dashboard - {options_data['ticker']}",
                height=1200,
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Strike Price", row=1, col=1)
            fig.update_yaxes(title_text="Option Price", row=1, col=1)
            fig.update_xaxes(title_text="Strike Price", row=1, col=2)
            fig.update_yaxes(title_text="Implied Volatility", row=1, col=2)
            fig.update_xaxes(title_text="Strike Price", row=2, col=1)
            fig.update_yaxes(title_text="Delta", row=2, col=1)
            fig.update_xaxes(title_text="Volume", row=3, col=1)
            fig.update_yaxes(title_text="Open Interest", row=3, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating options dashboard: {e}")
            return None
    
    def create_strategy_payoff_diagram(self, strategy_result):
        """Create payoff diagram for options strategy"""
        try:
            if not strategy_result:
                return None
            
            fig = go.Figure()
            
            # Payoff line
            fig.add_trace(
                go.Scatter(
                    x=strategy_result['spot_range'],
                    y=strategy_result['payoffs'],
                    mode='lines',
                    name='Strategy Payoff',
                    line=dict(width=3, color='blue')
                )
            )
            
            # Zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Breakeven"
            )
            
            # Breakeven points
            for i, breakeven in enumerate(strategy_result['breakeven_points']):
                fig.add_vline(
                    x=breakeven,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"BE: ${breakeven:.2f}"
                )
            
            # Profit/Loss areas
            x_vals = strategy_result['spot_range']
            y_vals = strategy_result['payoffs']
            
            # Profit area (green)
            profit_mask = y_vals > 0
            if np.any(profit_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_vals[profit_mask],
                        y=y_vals[profit_mask],
                        fill='tozeroy',
                        fillcolor='rgba(0,255,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Profit Zone',
                        showlegend=False
                    )
                )
            
            # Loss area (red)
            loss_mask = y_vals < 0
            if np.any(loss_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_vals[loss_mask],
                        y=y_vals[loss_mask],
                        fill='tozeroy',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Loss Zone',
                        showlegend=False
                    )
                )
            
            fig.update_layout(
                title="Options Strategy Payoff Diagram",
                xaxis_title="Stock Price at Expiration ($)",
                yaxis_title="Profit/Loss ($)",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating payoff diagram: {e}")
            return None
    
    def options_screener(self, options_analysis, criteria):
        """Screen options based on specified criteria"""
        try:
            screened_options = []
            
            for exp_date, data in options_analysis.items():
                calls = data['calls']
                puts = data['puts']
                
                # Screen calls
                if not calls.empty:
                    filtered_calls = calls.copy()
                    
                    # Apply filters
                    if 'min_volume' in criteria:
                        filtered_calls = filtered_calls[filtered_calls['volume'] >= criteria['min_volume']]
                    
                    if 'max_bid_ask_spread' in criteria:
                        filtered_calls['bid_ask_spread'] = filtered_calls['ask'] - filtered_calls['bid']
                        filtered_calls = filtered_calls[filtered_calls['bid_ask_spread'] <= criteria['max_bid_ask_spread']]
                    
                    if 'min_open_interest' in criteria:
                        filtered_calls = filtered_calls[filtered_calls['openInterest'] >= criteria['min_open_interest']]
                    
                    if 'delta_range' in criteria:
                        min_delta, max_delta = criteria['delta_range']
                        filtered_calls = filtered_calls[
                            (filtered_calls['delta'] >= min_delta) & 
                            (filtered_calls['delta'] <= max_delta)
                        ]
                    
                    # Add to results
                    for _, row in filtered_calls.iterrows():
                        screened_options.append({
                            'expiration': exp_date,
                            'type': 'call',
                            'strike': row['strike'],
                            'last_price': row['lastPrice'],
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest'],
                            'implied_volatility': row.get('implied_volatility', 0),
                            'delta': row.get('delta', 0),
                            'gamma': row.get('gamma', 0),
                            'theta': row.get('theta', 0),
                            'vega': row.get('vega', 0)
                        })
                
                # Screen puts (similar logic)
                if not puts.empty:
                    filtered_puts = puts.copy()
                    
                    # Apply same filters as calls
                    if 'min_volume' in criteria:
                        filtered_puts = filtered_puts[filtered_puts['volume'] >= criteria['min_volume']]
                    
                    if 'max_bid_ask_spread' in criteria:
                        filtered_puts['bid_ask_spread'] = filtered_puts['ask'] - filtered_puts['bid']
                        filtered_puts = filtered_puts[filtered_puts['bid_ask_spread'] <= criteria['max_bid_ask_spread']]
                    
                    if 'min_open_interest' in criteria:
                        filtered_puts = filtered_puts[filtered_puts['openInterest'] >= criteria['min_open_interest']]
                    
                    if 'delta_range' in criteria:
                        min_delta, max_delta = criteria['delta_range']
                        filtered_puts = filtered_puts[
                            (filtered_puts['delta'] >= min_delta) & 
                            (filtered_puts['delta'] <= max_delta)
                        ]
                    
                    # Add to results
                    for _, row in filtered_puts.iterrows():
                        screened_options.append({
                            'expiration': exp_date,
                            'type': 'put',
                            'strike': row['strike'],
                            'last_price': row['lastPrice'],
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest'],
                            'implied_volatility': row.get('implied_volatility', 0),
                            'delta': row.get('delta', 0),
                            'gamma': row.get('gamma', 0),
                            'theta': row.get('theta', 0),
                            'vega': row.get('vega', 0)
                        })
            
            return pd.DataFrame(screened_options)
            
        except Exception as e:
            st.error(f"❌ Error screening options: {e}")
            return pd.DataFrame()