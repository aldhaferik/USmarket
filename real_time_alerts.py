#!/usr/bin/env python3
"""
Real-time Alerts System
- Price target alerts
- Valuation alerts
- Technical indicator alerts
- News sentiment alerts
- Portfolio alerts
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
from typing import Dict, List, Optional, Tuple
import threading
import queue

warnings.filterwarnings('ignore')

class AlertsManager:
    def __init__(self):
        self.alerts = []
        self.alert_history = []
        self.monitoring_active = False
        self.alert_queue = queue.Queue()
        
        # Alert types
        self.alert_types = {
            'price_target': {
                'name': 'Price Target Alert',
                'description': 'Alerts when stock reaches target price',
                'parameters': ['target_price', 'direction']
            },
            'valuation': {
                'name': 'Valuation Alert',
                'description': 'Alerts when stock becomes over/undervalued',
                'parameters': ['valuation_threshold', 'metric']
            },
            'technical': {
                'name': 'Technical Indicator Alert',
                'description': 'Alerts based on technical indicators',
                'parameters': ['indicator', 'threshold', 'condition']
            },
            'volume': {
                'name': 'Volume Alert',
                'description': 'Alerts on unusual volume activity',
                'parameters': ['volume_threshold', 'period']
            },
            'news_sentiment': {
                'name': 'News Sentiment Alert',
                'description': 'Alerts on significant news sentiment changes',
                'parameters': ['sentiment_threshold', 'source']
            },
            'portfolio': {
                'name': 'Portfolio Alert',
                'description': 'Portfolio-level alerts',
                'parameters': ['metric', 'threshold', 'portfolio_name']
            }
        }
        
        # Notification methods
        self.notification_methods = {
            'email': {'enabled': False, 'config': {}},
            'webhook': {'enabled': False, 'config': {}},
            'streamlit': {'enabled': True, 'config': {}}
        }
        
        # Technical indicators for alerts
        self.technical_indicators = {
            'RSI': {'threshold_range': [0, 100], 'overbought': 70, 'oversold': 30},
            'MACD': {'threshold_range': [-10, 10], 'signal': 'crossover'},
            'Bollinger_Bands': {'threshold_range': [0, 1], 'upper': 0.95, 'lower': 0.05},
            'Moving_Average': {'threshold_range': [0.8, 1.2], 'crossover': True},
            'Volume': {'threshold_range': [0.5, 5.0], 'unusual': 2.0}
        }
    
    def create_alert(self, alert_config):
        """Create a new alert"""
        try:
            alert = {
                'id': len(self.alerts) + 1,
                'ticker': alert_config['ticker'],
                'alert_type': alert_config['alert_type'],
                'parameters': alert_config['parameters'],
                'created_at': datetime.now(),
                'status': 'active',
                'triggered_count': 0,
                'last_triggered': None,
                'notification_methods': alert_config.get('notification_methods', ['streamlit'])
            }
            
            # Validate alert configuration
            if self.validate_alert(alert):
                self.alerts.append(alert)
                st.success(f"✅ Alert created successfully for {alert['ticker']}")
                return alert
            else:
                st.error("❌ Invalid alert configuration")
                return None
                
        except Exception as e:
            st.error(f"❌ Error creating alert: {e}")
            return None
    
    def validate_alert(self, alert):
        """Validate alert configuration"""
        try:
            # Check if ticker is valid
            ticker = alert['ticker']
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'regularMarketPrice' not in info:
                st.error(f"❌ Invalid ticker: {ticker}")
                return False
            
            # Check alert type
            if alert['alert_type'] not in self.alert_types:
                st.error(f"❌ Invalid alert type: {alert['alert_type']}")
                return False
            
            # Validate parameters based on alert type
            alert_type = alert['alert_type']
            required_params = self.alert_types[alert_type]['parameters']
            
            for param in required_params:
                if param not in alert['parameters']:
                    st.error(f"❌ Missing parameter: {param}")
                    return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Alert validation error: {e}")
            return False
    
    def check_price_target_alert(self, alert, current_data):
        """Check price target alerts"""
        try:
            current_price = current_data['current_price']
            target_price = alert['parameters']['target_price']
            direction = alert['parameters']['direction']
            
            triggered = False
            message = ""
            
            if direction == 'above' and current_price >= target_price:
                triggered = True
                message = f"🎯 Price Target Hit: {alert['ticker']} reached ${current_price:.2f} (target: ${target_price:.2f})"
            elif direction == 'below' and current_price <= target_price:
                triggered = True
                message = f"🎯 Price Target Hit: {alert['ticker']} dropped to ${current_price:.2f} (target: ${target_price:.2f})"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking price target: {e}"
    
    def check_valuation_alert(self, alert, current_data):
        """Check valuation alerts"""
        try:
            ticker = alert['ticker']
            threshold = alert['parameters']['valuation_threshold']
            metric = alert['parameters']['metric']
            
            # Get valuation metrics
            valuation_data = self.calculate_valuation_metrics(ticker, current_data)
            
            if metric not in valuation_data:
                return False, f"Valuation metric {metric} not available"
            
            current_value = valuation_data[metric]
            triggered = False
            message = ""
            
            if metric == 'PE_ratio':
                if current_value > threshold:
                    triggered = True
                    message = f"📊 Valuation Alert: {ticker} P/E ratio {current_value:.2f} exceeds threshold {threshold}"
            elif metric == 'PB_ratio':
                if current_value > threshold:
                    triggered = True
                    message = f"📊 Valuation Alert: {ticker} P/B ratio {current_value:.2f} exceeds threshold {threshold}"
            elif metric == 'discount_to_fair_value':
                if abs(current_value) > threshold:
                    triggered = True
                    direction = "undervalued" if current_value > 0 else "overvalued"
                    message = f"📊 Valuation Alert: {ticker} is {direction} by {abs(current_value):.1f}%"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking valuation: {e}"
    
    def check_technical_alert(self, alert, current_data):
        """Check technical indicator alerts"""
        try:
            ticker = alert['ticker']
            indicator = alert['parameters']['indicator']
            threshold = alert['parameters']['threshold']
            condition = alert['parameters']['condition']
            
            # Get technical indicators
            technical_data = self.calculate_technical_indicators(ticker)
            
            if indicator not in technical_data:
                return False, f"Technical indicator {indicator} not available"
            
            current_value = technical_data[indicator]
            triggered = False
            message = ""
            
            if condition == 'above' and current_value > threshold:
                triggered = True
                message = f"📈 Technical Alert: {ticker} {indicator} ({current_value:.2f}) above threshold ({threshold})"
            elif condition == 'below' and current_value < threshold:
                triggered = True
                message = f"📉 Technical Alert: {ticker} {indicator} ({current_value:.2f}) below threshold ({threshold})"
            elif condition == 'crossover':
                # Check for crossover conditions (simplified)
                if indicator == 'MACD' and abs(current_value) < 0.1:  # Near zero crossover
                    triggered = True
                    message = f"🔄 Technical Alert: {ticker} MACD crossover detected"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking technical indicators: {e}"
    
    def check_volume_alert(self, alert, current_data):
        """Check volume alerts"""
        try:
            ticker = alert['ticker']
            threshold = alert['parameters']['volume_threshold']
            period = alert['parameters'].get('period', 20)
            
            # Get volume data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{period}d")
            
            if hist.empty:
                return False, "No volume data available"
            
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].iloc[:-1].mean()
            volume_ratio = current_volume / avg_volume
            
            triggered = False
            message = ""
            
            if volume_ratio > threshold:
                triggered = True
                message = f"📊 Volume Alert: {ticker} volume {volume_ratio:.1f}x above average ({current_volume:,.0f} vs {avg_volume:,.0f})"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking volume: {e}"
    
    def check_news_sentiment_alert(self, alert, current_data):
        """Check news sentiment alerts"""
        try:
            ticker = alert['ticker']
            threshold = alert['parameters']['sentiment_threshold']
            
            # Get news sentiment (simplified implementation)
            # In a real implementation, you would integrate with news APIs
            sentiment_score = self.get_news_sentiment(ticker)
            
            triggered = False
            message = ""
            
            if sentiment_score is not None:
                if sentiment_score < -threshold:
                    triggered = True
                    message = f"📰 News Alert: {ticker} negative sentiment detected (score: {sentiment_score:.2f})"
                elif sentiment_score > threshold:
                    triggered = True
                    message = f"📰 News Alert: {ticker} positive sentiment detected (score: {sentiment_score:.2f})"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking news sentiment: {e}"
    
    def check_portfolio_alert(self, alert, portfolio_data):
        """Check portfolio-level alerts"""
        try:
            metric = alert['parameters']['metric']
            threshold = alert['parameters']['threshold']
            portfolio_name = alert['parameters'].get('portfolio_name', 'default')
            
            triggered = False
            message = ""
            
            if metric == 'total_return':
                if portfolio_data['total_return'] < threshold:
                    triggered = True
                    message = f"📊 Portfolio Alert: {portfolio_name} return ({portfolio_data['total_return']:.2f}%) below threshold ({threshold}%)"
            elif metric == 'volatility':
                if portfolio_data['volatility'] > threshold:
                    triggered = True
                    message = f"📊 Portfolio Alert: {portfolio_name} volatility ({portfolio_data['volatility']:.2f}%) above threshold ({threshold}%)"
            elif metric == 'max_drawdown':
                if portfolio_data['max_drawdown'] > threshold:
                    triggered = True
                    message = f"📊 Portfolio Alert: {portfolio_name} drawdown ({portfolio_data['max_drawdown']:.2f}%) exceeds threshold ({threshold}%)"
            
            return triggered, message
            
        except Exception as e:
            return False, f"Error checking portfolio metrics: {e}"
    
    def calculate_valuation_metrics(self, ticker, current_data):
        """Calculate valuation metrics for alerts"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {}
            
            # P/E ratio
            if 'trailingPE' in info and info['trailingPE']:
                metrics['PE_ratio'] = info['trailingPE']
            
            # P/B ratio
            if 'priceToBook' in info and info['priceToBook']:
                metrics['PB_ratio'] = info['priceToBook']
            
            # Simple DCF-based fair value estimate
            if 'freeCashflow' in info and 'sharesOutstanding' in info:
                fcf = info['freeCashflow']
                shares = info['sharesOutstanding']
                growth_rate = 0.05  # Assume 5% growth
                discount_rate = 0.10  # Assume 10% discount rate
                
                # Simple DCF calculation
                fair_value = (fcf / shares) * (1 + growth_rate) / (discount_rate - growth_rate)
                current_price = current_data['current_price']
                
                discount_premium = (fair_value - current_price) / current_price * 100
                metrics['discount_to_fair_value'] = discount_premium
                metrics['fair_value'] = fair_value
            
            return metrics
            
        except Exception as e:
            st.warning(f"⚠️ Error calculating valuation metrics: {e}")
            return {}
    
    def calculate_technical_indicators(self, ticker):
        """Calculate technical indicators for alerts"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                return {}
            
            indicators = {}
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            indicators['MACD'] = (macd - signal).iloc[-1]
            
            # Bollinger Bands position
            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (hist['Close'] - lower_band) / (upper_band - lower_band)
            indicators['Bollinger_Bands'] = bb_position.iloc[-1]
            
            # Moving Average ratio
            ma_50 = hist['Close'].rolling(window=50).mean()
            ma_ratio = hist['Close'] / ma_50
            indicators['Moving_Average'] = ma_ratio.iloc[-1]
            
            return indicators
            
        except Exception as e:
            st.warning(f"⚠️ Error calculating technical indicators: {e}")
            return {}
    
    def get_news_sentiment(self, ticker):
        """Get news sentiment score (simplified implementation)"""
        try:
            # This is a simplified implementation
            # In a real system, you would integrate with news APIs like Alpha Vantage, NewsAPI, etc.
            
            # For demonstration, return a random sentiment score
            import random
            random.seed(hash(ticker + str(datetime.now().date())))
            return random.uniform(-1, 1)
            
        except Exception as e:
            return None
    
    def monitor_alerts(self):
        """Monitor all active alerts"""
        try:
            if not self.alerts:
                return
            
            st.info("🔍 Monitoring alerts...")
            
            for alert in self.alerts:
                if alert['status'] != 'active':
                    continue
                
                try:
                    ticker = alert['ticker']
                    
                    # Get current market data
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    current_data = {
                        'current_price': info.get('regularMarketPrice', 0),
                        'volume': info.get('regularMarketVolume', 0),
                        'timestamp': datetime.now()
                    }
                    
                    # Check alert based on type
                    triggered = False
                    message = ""
                    
                    if alert['alert_type'] == 'price_target':
                        triggered, message = self.check_price_target_alert(alert, current_data)
                    elif alert['alert_type'] == 'valuation':
                        triggered, message = self.check_valuation_alert(alert, current_data)
                    elif alert['alert_type'] == 'technical':
                        triggered, message = self.check_technical_alert(alert, current_data)
                    elif alert['alert_type'] == 'volume':
                        triggered, message = self.check_volume_alert(alert, current_data)
                    elif alert['alert_type'] == 'news_sentiment':
                        triggered, message = self.check_news_sentiment_alert(alert, current_data)
                    
                    # If alert triggered, send notification
                    if triggered:
                        self.trigger_alert(alert, message, current_data)
                        
                except Exception as e:
                    st.warning(f"⚠️ Error monitoring alert {alert['id']}: {e}")
                    continue
            
        except Exception as e:
            st.error(f"❌ Error monitoring alerts: {e}")
    
    def trigger_alert(self, alert, message, current_data):
        """Trigger an alert and send notifications"""
        try:
            # Update alert status
            alert['triggered_count'] += 1
            alert['last_triggered'] = datetime.now()
            
            # Create alert record
            alert_record = {
                'alert_id': alert['id'],
                'ticker': alert['ticker'],
                'message': message,
                'triggered_at': datetime.now(),
                'current_price': current_data['current_price'],
                'alert_type': alert['alert_type']
            }
            
            # Add to alert history
            self.alert_history.append(alert_record)
            
            # Send notifications
            for method in alert['notification_methods']:
                if method in self.notification_methods and self.notification_methods[method]['enabled']:
                    self.send_notification(method, alert_record)
            
            # Add to alert queue for display
            self.alert_queue.put(alert_record)
            
            st.success(f"🚨 Alert Triggered: {message}")
            
        except Exception as e:
            st.error(f"❌ Error triggering alert: {e}")
    
    def send_notification(self, method, alert_record):
        """Send notification via specified method"""
        try:
            if method == 'email':
                self.send_email_notification(alert_record)
            elif method == 'webhook':
                self.send_webhook_notification(alert_record)
            elif method == 'streamlit':
                self.send_streamlit_notification(alert_record)
                
        except Exception as e:
            st.warning(f"⚠️ Failed to send {method} notification: {e}")
    
    def send_email_notification(self, alert_record):
        """Send email notification"""
        try:
            config = self.notification_methods['email']['config']
            
            if not all(key in config for key in ['smtp_server', 'smtp_port', 'username', 'password', 'to_email']):
                return
            
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = config['to_email']
            msg['Subject'] = f"Stock Alert: {alert_record['ticker']}"
            
            body = f"""
            Stock Alert Triggered
            
            Ticker: {alert_record['ticker']}
            Alert Type: {alert_record['alert_type']}
            Message: {alert_record['message']}
            Current Price: ${alert_record['current_price']:.2f}
            Time: {alert_record['triggered_at']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            text = msg.as_string()
            server.sendmail(config['username'], config['to_email'], text)
            server.quit()
            
        except Exception as e:
            st.warning(f"⚠️ Email notification failed: {e}")
    
    def send_webhook_notification(self, alert_record):
        """Send webhook notification"""
        try:
            config = self.notification_methods['webhook']['config']
            
            if 'url' not in config:
                return
            
            payload = {
                'ticker': alert_record['ticker'],
                'alert_type': alert_record['alert_type'],
                'message': alert_record['message'],
                'current_price': alert_record['current_price'],
                'triggered_at': alert_record['triggered_at'].isoformat()
            }
            
            response = requests.post(config['url'], json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            st.warning(f"⚠️ Webhook notification failed: {e}")
    
    def send_streamlit_notification(self, alert_record):
        """Send Streamlit notification"""
        try:
            # This will be displayed in the Streamlit interface
            st.sidebar.success(f"🚨 {alert_record['message']}")
            
        except Exception as e:
            st.warning(f"⚠️ Streamlit notification failed: {e}")
    
    def get_alert_dashboard_data(self):
        """Get data for alerts dashboard"""
        try:
            dashboard_data = {
                'total_alerts': len(self.alerts),
                'active_alerts': len([a for a in self.alerts if a['status'] == 'active']),
                'triggered_today': len([h for h in self.alert_history if h['triggered_at'].date() == datetime.now().date()]),
                'alert_types_distribution': {},
                'recent_alerts': self.alert_history[-10:] if self.alert_history else [],
                'alert_performance': {}
            }
            
            # Alert types distribution
            for alert in self.alerts:
                alert_type = alert['alert_type']
                dashboard_data['alert_types_distribution'][alert_type] = dashboard_data['alert_types_distribution'].get(alert_type, 0) + 1
            
            # Alert performance (accuracy, frequency, etc.)
            for alert in self.alerts:
                ticker = alert['ticker']
                if ticker not in dashboard_data['alert_performance']:
                    dashboard_data['alert_performance'][ticker] = {
                        'total_alerts': 0,
                        'triggered_count': 0,
                        'accuracy': 0
                    }
                
                dashboard_data['alert_performance'][ticker]['total_alerts'] += 1
                dashboard_data['alert_performance'][ticker]['triggered_count'] += alert['triggered_count']
            
            return dashboard_data
            
        except Exception as e:
            st.error(f"❌ Error getting dashboard data: {e}")
            return {}
    
    def create_alerts_dashboard(self):
        """Create comprehensive alerts dashboard"""
        try:
            dashboard_data = self.get_alert_dashboard_data()
            
            if not dashboard_data:
                st.warning("⚠️ No dashboard data available")
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Alert Types Distribution', 'Alert Activity Timeline',
                    'Alert Performance by Ticker', 'Recent Alert Triggers'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "table"}]
                ]
            )
            
            # Alert types distribution
            if dashboard_data['alert_types_distribution']:
                types = list(dashboard_data['alert_types_distribution'].keys())
                counts = list(dashboard_data['alert_types_distribution'].values())
                
                fig.add_trace(
                    go.Pie(
                        labels=types,
                        values=counts,
                        name="Alert Types"
                    ),
                    row=1, col=1
                )
            
            # Alert activity timeline
            if self.alert_history:
                dates = [h['triggered_at'].date() for h in self.alert_history]
                date_counts = pd.Series(dates).value_counts().sort_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=date_counts.index,
                        y=date_counts.values,
                        mode='lines+markers',
                        name='Daily Alerts'
                    ),
                    row=1, col=2
                )
            
            # Alert performance by ticker
            if dashboard_data['alert_performance']:
                tickers = list(dashboard_data['alert_performance'].keys())
                triggered_counts = [dashboard_data['alert_performance'][t]['triggered_count'] for t in tickers]
                
                fig.add_trace(
                    go.Bar(
                        x=tickers,
                        y=triggered_counts,
                        name='Triggered Count'
                    ),
                    row=2, col=1
                )
            
            # Recent alerts table
            if dashboard_data['recent_alerts']:
                recent_alerts = dashboard_data['recent_alerts']
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Ticker', 'Type', 'Message', 'Time']),
                        cells=dict(values=[
                            [a['ticker'] for a in recent_alerts],
                            [a['alert_type'] for a in recent_alerts],
                            [a['message'][:50] + '...' if len(a['message']) > 50 else a['message'] for a in recent_alerts],
                            [a['triggered_at'].strftime('%H:%M:%S') for a in recent_alerts]
                        ])
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Real-time Alerts Dashboard",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating alerts dashboard: {e}")
            return None
    
    def export_alerts_config(self):
        """Export alerts configuration"""
        try:
            config = {
                'alerts': self.alerts,
                'notification_methods': self.notification_methods,
                'export_date': datetime.now().isoformat()
            }
            
            return json.dumps(config, indent=2, default=str)
            
        except Exception as e:
            st.error(f"❌ Error exporting alerts config: {e}")
            return None
    
    def import_alerts_config(self, config_json):
        """Import alerts configuration"""
        try:
            config = json.loads(config_json)
            
            if 'alerts' in config:
                self.alerts = config['alerts']
                # Convert string dates back to datetime objects
                for alert in self.alerts:
                    alert['created_at'] = datetime.fromisoformat(alert['created_at'])
                    if alert['last_triggered']:
                        alert['last_triggered'] = datetime.fromisoformat(alert['last_triggered'])
            
            if 'notification_methods' in config:
                self.notification_methods.update(config['notification_methods'])
            
            st.success("✅ Alerts configuration imported successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error importing alerts config: {e}")
            return False

def create_alerts_interface():
    """Create Streamlit interface for alerts management"""
    st.title("🚨 Real-time Alerts System")
    
    # Initialize alerts manager
    if 'alerts_manager' not in st.session_state:
        st.session_state.alerts_manager = AlertsManager()
    
    alerts_manager = st.session_state.alerts_manager
    
    # Sidebar for alert management
    st.sidebar.header("Alert Management")
    
    # Alert creation form
    with st.sidebar.expander("Create New Alert", expanded=False):
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        alert_type = st.selectbox("Alert Type", list(alerts_manager.alert_types.keys()))
        
        # Dynamic parameters based on alert type
        parameters = {}
        
        if alert_type == 'price_target':
            parameters['target_price'] = st.number_input("Target Price", min_value=0.01, value=150.0)
            parameters['direction'] = st.selectbox("Direction", ['above', 'below'])
        
        elif alert_type == 'valuation':
            parameters['valuation_threshold'] = st.number_input("Threshold", min_value=0.1, value=25.0)
            parameters['metric'] = st.selectbox("Metric", ['PE_ratio', 'PB_ratio', 'discount_to_fair_value'])
        
        elif alert_type == 'technical':
            parameters['indicator'] = st.selectbox("Indicator", ['RSI', 'MACD', 'Bollinger_Bands', 'Moving_Average'])
            parameters['threshold'] = st.number_input("Threshold", value=70.0)
            parameters['condition'] = st.selectbox("Condition", ['above', 'below', 'crossover'])
        
        elif alert_type == 'volume':
            parameters['volume_threshold'] = st.number_input("Volume Multiplier", min_value=1.0, value=2.0)
            parameters['period'] = st.number_input("Period (days)", min_value=5, max_value=100, value=20)
        
        elif alert_type == 'news_sentiment':
            parameters['sentiment_threshold'] = st.number_input("Sentiment Threshold", min_value=0.1, max_value=1.0, value=0.7)
        
        notification_methods = st.multiselect("Notification Methods", ['streamlit', 'email', 'webhook'], default=['streamlit'])
        
        if st.button("Create Alert"):
            alert_config = {
                'ticker': ticker.upper(),
                'alert_type': alert_type,
                'parameters': parameters,
                'notification_methods': notification_methods
            }
            alerts_manager.create_alert(alert_config)
    
    # Monitor alerts button
    if st.sidebar.button("🔍 Check Alerts Now"):
        alerts_manager.monitor_alerts()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Active Alerts", "Alert History", "Dashboard", "Settings"])
    
    with tab1:
        st.header("Active Alerts")
        
        if alerts_manager.alerts:
            for i, alert in enumerate(alerts_manager.alerts):
                if alert['status'] == 'active':
                    with st.expander(f"{alert['ticker']} - {alert['alert_type']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Type:** {alert['alert_type']}")
                            st.write(f"**Created:** {alert['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            st.write(f"**Parameters:** {alert['parameters']}")
                            st.write(f"**Triggered:** {alert['triggered_count']} times")
                        
                        with col3:
                            if st.button(f"Delete", key=f"delete_{i}"):
                                alerts_manager.alerts.remove(alert)
                                st.rerun()
                            
                            if alert['last_triggered']:
                                st.write(f"**Last Triggered:** {alert['last_triggered'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No active alerts. Create one using the sidebar.")
    
    with tab2:
        st.header("Alert History")
        
        if alerts_manager.alert_history:
            history_df = pd.DataFrame(alerts_manager.alert_history)
            history_df['triggered_at'] = pd.to_datetime(history_df['triggered_at'])
            history_df = history_df.sort_values('triggered_at', ascending=False)
            
            st.dataframe(
                history_df[['ticker', 'alert_type', 'message', 'current_price', 'triggered_at']],
                use_container_width=True
            )
        else:
            st.info("No alert history available.")
    
    with tab3:
        st.header("Alerts Dashboard")
        
        dashboard_fig = alerts_manager.create_alerts_dashboard()
        if dashboard_fig:
            st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Summary metrics
        dashboard_data = alerts_manager.get_alert_dashboard_data()
        if dashboard_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", dashboard_data['total_alerts'])
            
            with col2:
                st.metric("Active Alerts", dashboard_data['active_alerts'])
            
            with col3:
                st.metric("Triggered Today", dashboard_data['triggered_today'])
            
            with col4:
                accuracy = len(alerts_manager.alert_history) / max(len(alerts_manager.alerts), 1) * 100
                st.metric("Alert Activity", f"{accuracy:.1f}%")
    
    with tab4:
        st.header("Alert Settings")
        
        # Notification settings
        st.subheader("Notification Methods")
        
        # Email settings
        with st.expander("Email Configuration"):
            email_enabled = st.checkbox("Enable Email Notifications")
            if email_enabled:
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587)
                username = st.text_input("Email Username")
                password = st.text_input("Email Password", type="password")
                to_email = st.text_input("Recipient Email")
                
                alerts_manager.notification_methods['email'] = {
                    'enabled': email_enabled,
                    'config': {
                        'smtp_server': smtp_server,
                        'smtp_port': smtp_port,
                        'username': username,
                        'password': password,
                        'to_email': to_email
                    }
                }
        
        # Webhook settings
        with st.expander("Webhook Configuration"):
            webhook_enabled = st.checkbox("Enable Webhook Notifications")
            if webhook_enabled:
                webhook_url = st.text_input("Webhook URL")
                
                alerts_manager.notification_methods['webhook'] = {
                    'enabled': webhook_enabled,
                    'config': {'url': webhook_url}
                }
        
        # Export/Import configuration
        st.subheader("Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Configuration"):
                config_json = alerts_manager.export_alerts_config()
                if config_json:
                    st.download_button(
                        label="Download Configuration",
                        data=config_json,
                        file_name=f"alerts_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            uploaded_file = st.file_uploader("Import Configuration", type=['json'])
            if uploaded_file is not None:
                config_json = uploaded_file.read().decode('utf-8')
                if st.button("Import Configuration"):
                    alerts_manager.import_alerts_config(config_json)

if __name__ == "__main__":
    create_alerts_interface()