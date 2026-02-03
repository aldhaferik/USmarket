#!/usr/bin/env python3
"""
Portfolio Analysis System
- Multi-stock portfolio optimization
- Risk-return analysis across portfolios
- Correlation analysis between stocks
- Portfolio rebalancing recommendations
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.045  # 10-year Treasury rate
        self.trading_days = 252
        
    def get_portfolio_data(self, tickers, period="2y"):
        """Get historical data for multiple stocks"""
        try:
            st.info(f"📊 Fetching data for {len(tickers)} stocks...")
            
            # Download data for all tickers
            data = yf.download(tickers, period=period, progress=False)
            
            if data.empty:
                st.error("❌ No data retrieved for the given tickers")
                return None
            
            # Handle single ticker case
            if len(tickers) == 1:
                data = pd.DataFrame(data)
                data.columns = pd.MultiIndex.from_product([data.columns, tickers])
            
            # Get adjusted close prices
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            
            # Remove any tickers with insufficient data
            prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)  # At least 80% data
            
            if prices.empty:
                st.error("❌ Insufficient data for portfolio analysis")
                return None
            
            # Get additional data for each stock
            portfolio_data = {
                'prices': prices,
                'returns': prices.pct_change().dropna(),
                'stock_info': {}
            }
            
            # Get basic info for each stock
            for ticker in prices.columns:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    portfolio_data['stock_info'][ticker] = {
                        'name': info.get('longName', ticker),
                        'sector': info.get('sector', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'beta': info.get('beta', 1.0),
                        'pe_ratio': info.get('trailingPE', None),
                        'dividend_yield': info.get('dividendYield', 0)
                    }
                except:
                    portfolio_data['stock_info'][ticker] = {
                        'name': ticker,
                        'sector': 'Unknown',
                        'market_cap': 0,
                        'beta': 1.0,
                        'pe_ratio': None,
                        'dividend_yield': 0
                    }
            
            st.success(f"✅ Data retrieved for {len(prices.columns)} stocks")
            return portfolio_data
            
        except Exception as e:
            st.error(f"❌ Error fetching portfolio data: {e}")
            return None
    
    def calculate_portfolio_metrics(self, returns, weights):
        """Calculate portfolio risk and return metrics"""
        try:
            # Ensure weights sum to 1
            weights = np.array(weights) / np.sum(weights)
            
            # Portfolio return
            portfolio_return = np.sum(returns.mean() * weights) * self.trading_days
            
            # Portfolio volatility
            cov_matrix = returns.cov() * self.trading_days
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Maximum drawdown
            portfolio_returns = (returns * weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'expected_shortfall': es_95,
                'weights': weights
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating portfolio metrics: {e}")
            return None
    
    def optimize_portfolio(self, returns, optimization_method='sharpe'):
        """Optimize portfolio using different methods"""
        try:
            n_assets = len(returns.columns)
            
            # Constraints: weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds: each weight between 0 and 1 (long-only portfolio)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            initial_guess = np.array([1/n_assets] * n_assets)
            
            if optimization_method == 'sharpe':
                # Maximize Sharpe ratio
                def negative_sharpe(weights):
                    metrics = self.calculate_portfolio_metrics(returns, weights)
                    return -metrics['sharpe_ratio'] if metrics else 1000
                
                result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
            elif optimization_method == 'min_variance':
                # Minimize portfolio variance
                def portfolio_variance(weights):
                    cov_matrix = returns.cov() * self.trading_days
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
            elif optimization_method == 'max_return':
                # Maximize expected return
                def negative_return(weights):
                    return -np.sum(returns.mean() * weights) * self.trading_days
                
                result = minimize(negative_return, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
            
            elif optimization_method == 'risk_parity':
                # Risk parity: equal risk contribution
                def risk_parity_objective(weights):
                    cov_matrix = returns.cov() * self.trading_days
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                    contrib = weights * marginal_contrib
                    target_contrib = portfolio_vol / n_assets
                    return np.sum((contrib - target_contrib) ** 2)
                
                result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
            
            else:
                st.error(f"❌ Unknown optimization method: {optimization_method}")
                return None
            
            if result.success:
                optimal_weights = result.x
                optimal_metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'metrics': optimal_metrics,
                    'method': optimization_method,
                    'success': True
                }
            else:
                st.warning(f"⚠️ Optimization failed: {result.message}")
                return None
                
        except Exception as e:
            st.error(f"❌ Error optimizing portfolio: {e}")
            return None
    
    def generate_efficient_frontier(self, returns, num_portfolios=100):
        """Generate efficient frontier"""
        try:
            n_assets = len(returns.columns)
            results = np.zeros((3, num_portfolios))
            weights_array = np.zeros((num_portfolios, n_assets))
            
            # Calculate expected returns and covariance matrix
            mean_returns = returns.mean() * self.trading_days
            cov_matrix = returns.cov() * self.trading_days
            
            # Generate target returns
            min_ret = mean_returns.min()
            max_ret = mean_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            # Constraints and bounds
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            for i, target_return in enumerate(target_returns):
                # Add return constraint
                return_constraint = {'type': 'eq', 'fun': lambda x, target=target_return: np.sum(x * mean_returns) - target}
                current_constraints = constraints + [return_constraint]
                
                # Minimize portfolio variance
                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                # Initial guess
                initial_guess = np.array([1/n_assets] * n_assets)
                
                try:
                    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=current_constraints)
                    
                    if result.success:
                        weights_array[i] = result.x
                        portfolio_return = np.sum(result.x * mean_returns)
                        portfolio_vol = np.sqrt(portfolio_variance(result.x))
                        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                        
                        results[0, i] = portfolio_return
                        results[1, i] = portfolio_vol
                        results[2, i] = sharpe
                    else:
                        # If optimization fails, use NaN
                        results[:, i] = np.nan
                        weights_array[i] = np.nan
                        
                except:
                    results[:, i] = np.nan
                    weights_array[i] = np.nan
            
            # Remove failed optimizations
            valid_indices = ~np.isnan(results[0])
            
            return {
                'returns': results[0][valid_indices],
                'volatilities': results[1][valid_indices],
                'sharpe_ratios': results[2][valid_indices],
                'weights': weights_array[valid_indices]
            }
            
        except Exception as e:
            st.error(f"❌ Error generating efficient frontier: {e}")
            return None
    
    def calculate_correlation_matrix(self, returns):
        """Calculate and analyze correlation matrix"""
        try:
            correlation_matrix = returns.corr()
            
            # Find highly correlated pairs (>0.7)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_corr_pairs.append({
                            'stock1': correlation_matrix.columns[i],
                            'stock2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Calculate average correlation
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            return {
                'matrix': correlation_matrix,
                'high_corr_pairs': high_corr_pairs,
                'avg_correlation': avg_correlation
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating correlation matrix: {e}")
            return None
    
    def monte_carlo_portfolio_simulation(self, returns, weights, num_simulations=10000, time_horizon=252):
        """Monte Carlo simulation for portfolio"""
        try:
            # Portfolio parameters
            portfolio_return = np.sum(returns.mean() * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
            
            # Generate random returns
            random_returns = np.random.normal(portfolio_return, portfolio_vol, (num_simulations, time_horizon))
            
            # Calculate cumulative returns for each simulation
            cumulative_returns = np.cumprod(1 + random_returns, axis=1)
            final_values = cumulative_returns[:, -1]
            
            # Calculate statistics
            simulation_results = {
                'final_values': final_values,
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'percentiles': {
                    '5th': np.percentile(final_values, 5),
                    '25th': np.percentile(final_values, 25),
                    '75th': np.percentile(final_values, 75),
                    '95th': np.percentile(final_values, 95)
                },
                'probability_positive': np.sum(final_values > 1) / num_simulations,
                'probability_loss_10': np.sum(final_values < 0.9) / num_simulations,
                'probability_loss_20': np.sum(final_values < 0.8) / num_simulations,
                'var_95': np.percentile(final_values, 5) - 1,
                'expected_shortfall': np.mean(final_values[final_values <= np.percentile(final_values, 5)]) - 1
            }
            
            return simulation_results
            
        except Exception as e:
            st.error(f"❌ Error in Monte Carlo simulation: {e}")
            return None
    
    def sector_analysis(self, portfolio_data, weights):
        """Analyze portfolio by sectors"""
        try:
            sector_allocation = {}
            sector_weights = {}
            
            for i, ticker in enumerate(portfolio_data['prices'].columns):
                sector = portfolio_data['stock_info'][ticker]['sector']
                weight = weights[i]
                
                if sector in sector_allocation:
                    sector_allocation[sector] += weight
                    sector_weights[sector].append((ticker, weight))
                else:
                    sector_allocation[sector] = weight
                    sector_weights[sector] = [(ticker, weight)]
            
            # Calculate sector diversification score
            sector_values = list(sector_allocation.values())
            herfindahl_index = sum(w**2 for w in sector_values)
            diversification_score = (1 - herfindahl_index) / (1 - 1/len(sector_values)) if len(sector_values) > 1 else 0
            
            return {
                'sector_allocation': sector_allocation,
                'sector_weights': sector_weights,
                'diversification_score': diversification_score,
                'num_sectors': len(sector_allocation)
            }
            
        except Exception as e:
            st.error(f"❌ Error in sector analysis: {e}")
            return None
    
    def rebalancing_analysis(self, current_weights, target_weights, current_prices, portfolio_value):
        """Analyze portfolio rebalancing requirements"""
        try:
            current_weights = np.array(current_weights)
            target_weights = np.array(target_weights)
            current_prices = np.array(current_prices)
            
            # Calculate current and target values
            current_values = current_weights * portfolio_value
            target_values = target_weights * portfolio_value
            
            # Calculate required trades
            value_differences = target_values - current_values
            share_differences = value_differences / current_prices
            
            # Calculate trading costs (assume 0.1% per trade)
            trading_cost_rate = 0.001
            total_trades_value = np.sum(np.abs(value_differences))
            estimated_trading_costs = total_trades_value * trading_cost_rate
            
            # Calculate rebalancing benefit
            current_metrics = self.calculate_portfolio_metrics(
                pd.DataFrame(), current_weights  # Simplified for this calculation
            )
            target_metrics = self.calculate_portfolio_metrics(
                pd.DataFrame(), target_weights  # Simplified for this calculation
            )
            
            rebalancing_recommendations = []
            for i, (ticker, value_diff, share_diff) in enumerate(zip(
                current_prices.index if hasattr(current_prices, 'index') else range(len(current_prices)),
                value_differences, share_differences
            )):
                if abs(value_diff) > portfolio_value * 0.01:  # Only recommend if >1% of portfolio
                    action = "BUY" if value_diff > 0 else "SELL"
                    rebalancing_recommendations.append({
                        'ticker': ticker,
                        'action': action,
                        'value_change': abs(value_diff),
                        'shares': abs(share_diff),
                        'percentage_of_portfolio': abs(value_diff) / portfolio_value * 100
                    })
            
            return {
                'recommendations': rebalancing_recommendations,
                'total_trading_value': total_trades_value,
                'estimated_costs': estimated_trading_costs,
                'cost_percentage': estimated_trading_costs / portfolio_value * 100,
                'rebalancing_benefit': 'Improved diversification' if len(rebalancing_recommendations) > 0 else 'Portfolio already balanced'
            }
            
        except Exception as e:
            st.error(f"❌ Error in rebalancing analysis: {e}")
            return None
    
    def create_portfolio_dashboard(self, portfolio_data, optimization_results, correlation_analysis):
        """Create comprehensive portfolio dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Portfolio Composition', 'Cumulative Returns',
                    'Correlation Heatmap', 'Risk-Return Scatter',
                    'Sector Allocation', 'Performance Metrics'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "table"}]
                ]
            )
            
            # Portfolio composition pie chart
            if optimization_results:
                weights = optimization_results['weights']
                tickers = portfolio_data['prices'].columns
                
                fig.add_trace(
                    go.Pie(
                        labels=tickers,
                        values=weights,
                        name="Portfolio Weights"
                    ),
                    row=1, col=1
                )
            
            # Cumulative returns
            returns = portfolio_data['returns']
            cumulative_returns = (1 + returns).cumprod()
            
            for ticker in cumulative_returns.columns:
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[ticker],
                        mode='lines',
                        name=ticker,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Correlation heatmap
            if correlation_analysis:
                corr_matrix = correlation_analysis['matrix']
                
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
            
            # Risk-return scatter
            if optimization_results:
                annual_returns = returns.mean() * 252
                annual_volatilities = returns.std() * np.sqrt(252)
                
                fig.add_trace(
                    go.Scatter(
                        x=annual_volatilities,
                        y=annual_returns,
                        mode='markers+text',
                        text=tickers,
                        textposition='top center',
                        marker=dict(size=10),
                        name='Individual Stocks',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Add portfolio point
                portfolio_metrics = optimization_results['metrics']
                fig.add_trace(
                    go.Scatter(
                        x=[portfolio_metrics['volatility']],
                        y=[portfolio_metrics['return']],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name='Portfolio',
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Portfolio Analysis Dashboard",
                height=1000,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating portfolio dashboard: {e}")
            return None
    
    def create_efficient_frontier_plot(self, efficient_frontier, optimization_results=None):
        """Create efficient frontier visualization"""
        try:
            fig = go.Figure()
            
            # Plot efficient frontier
            fig.add_trace(
                go.Scatter(
                    x=efficient_frontier['volatilities'],
                    y=efficient_frontier['returns'],
                    mode='lines+markers',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                )
            )
            
            # Add optimal portfolio points
            if optimization_results:
                for method, result in optimization_results.items():
                    if result and result['success']:
                        metrics = result['metrics']
                        fig.add_trace(
                            go.Scatter(
                                x=[metrics['volatility']],
                                y=[metrics['return']],
                                mode='markers',
                                name=f'{method.replace("_", " ").title()}',
                                marker=dict(size=12, symbol='star')
                            )
                        )
            
            # Add risk-free rate line
            max_vol = max(efficient_frontier['volatilities']) if efficient_frontier['volatilities'].size > 0 else 0.3
            fig.add_trace(
                go.Scatter(
                    x=[0, max_vol],
                    y=[self.risk_free_rate, self.risk_free_rate],
                    mode='lines',
                    name='Risk-Free Rate',
                    line=dict(color='green', dash='dash')
                )
            )
            
            fig.update_layout(
                title='Efficient Frontier Analysis',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating efficient frontier plot: {e}")
            return None