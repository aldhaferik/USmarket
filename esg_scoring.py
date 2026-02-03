#!/usr/bin/env python3
"""
ESG Scoring Integration
- Environmental, Social, Governance metrics
- ESG data integration from external sources
- ESG-weighted valuations
- Sustainability analysis
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
import warnings

warnings.filterwarnings('ignore')

class ESGAnalyzer:
    def __init__(self):
        # ESG scoring weights
        self.esg_weights = {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        }
        
        # Industry ESG benchmarks (0-100 scale)
        self.industry_esg_benchmarks = {
            'Technology': {'E': 75, 'S': 80, 'G': 85, 'total': 80},
            'Healthcare': {'E': 70, 'S': 85, 'G': 80, 'total': 78},
            'Financial Services': {'E': 65, 'S': 75, 'G': 90, 'total': 77},
            'Consumer Cyclical': {'E': 60, 'S': 70, 'G': 75, 'total': 68},
            'Consumer Defensive': {'E': 65, 'S': 75, 'G': 80, 'total': 73},
            'Industrials': {'E': 55, 'S': 70, 'G': 75, 'total': 67},
            'Energy': {'E': 45, 'S': 65, 'G': 70, 'total': 60},
            'Materials': {'E': 50, 'S': 65, 'G': 70, 'total': 62},
            'Real Estate': {'E': 70, 'S': 75, 'G': 80, 'total': 75},
            'Utilities': {'E': 60, 'S': 80, 'G': 85, 'total': 75},
            'Communication Services': {'E': 70, 'S': 75, 'G': 80, 'total': 75}
        }
        
        # ESG risk factors by category
        self.esg_factors = {
            'environmental': {
                'carbon_emissions': {'weight': 0.3, 'description': 'Carbon footprint and emissions'},
                'energy_efficiency': {'weight': 0.25, 'description': 'Energy usage and efficiency'},
                'waste_management': {'weight': 0.2, 'description': 'Waste reduction and recycling'},
                'water_usage': {'weight': 0.15, 'description': 'Water conservation and management'},
                'biodiversity': {'weight': 0.1, 'description': 'Impact on biodiversity and ecosystems'}
            },
            'social': {
                'employee_relations': {'weight': 0.3, 'description': 'Employee satisfaction and rights'},
                'diversity_inclusion': {'weight': 0.25, 'description': 'Workplace diversity and inclusion'},
                'community_impact': {'weight': 0.2, 'description': 'Community engagement and impact'},
                'product_safety': {'weight': 0.15, 'description': 'Product safety and quality'},
                'human_rights': {'weight': 0.1, 'description': 'Human rights compliance'}
            },
            'governance': {
                'board_structure': {'weight': 0.3, 'description': 'Board independence and structure'},
                'executive_compensation': {'weight': 0.25, 'description': 'Executive pay alignment'},
                'transparency': {'weight': 0.2, 'description': 'Financial and operational transparency'},
                'ethics_compliance': {'weight': 0.15, 'description': 'Ethics and compliance programs'},
                'shareholder_rights': {'weight': 0.1, 'description': 'Shareholder rights protection'}
            }
        }
        
        # ESG rating agencies mapping
        self.rating_agencies = {
            'MSCI': {'scale': 'AAA-CCC', 'description': 'MSCI ESG Rating'},
            'Sustainalytics': {'scale': '0-100', 'description': 'ESG Risk Score (lower is better)'},
            'S&P Global': {'scale': '0-100', 'description': 'S&P Global ESG Score'},
            'Refinitiv': {'scale': '0-100', 'description': 'Refinitiv ESG Score'},
            'CDP': {'scale': 'A-D', 'description': 'Carbon Disclosure Project Score'}
        }
    
    def get_esg_data(self, ticker):
        """Get ESG data from Yahoo Finance and other sources"""
        try:
            st.info(f"📊 Fetching ESG data for {ticker}...")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get basic company information
            company_data = {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0)
            }
            
            # Try to get ESG scores from Yahoo Finance
            esg_scores = {}
            try:
                # Yahoo Finance sometimes has ESG data in the sustainability module
                sustainability = stock.sustainability
                if sustainability is not None and not sustainability.empty:
                    esg_scores['yahoo_esg'] = {
                        'total_esg': sustainability.get('totalEsg', {}).get('raw', None),
                        'environment_score': sustainability.get('environmentScore', {}).get('raw', None),
                        'social_score': sustainability.get('socialScore', {}).get('raw', None),
                        'governance_score': sustainability.get('governanceScore', {}).get('raw', None),
                        'controversy_level': sustainability.get('highestControlversy', {}).get('raw', None)
                    }
            except:
                st.warning("⚠️ ESG data not available from Yahoo Finance")
            
            # Generate synthetic ESG scores based on industry benchmarks
            # (In a real implementation, you would integrate with ESG data providers)
            synthetic_esg = self.generate_synthetic_esg_scores(company_data)
            
            esg_data = {
                'company_data': company_data,
                'esg_scores': esg_scores,
                'synthetic_esg': synthetic_esg,
                'extraction_date': datetime.now()
            }
            
            st.success(f"✅ ESG data retrieved for {ticker}")
            return esg_data
            
        except Exception as e:
            st.error(f"❌ Error fetching ESG data: {e}")
            return None
    
    def generate_synthetic_esg_scores(self, company_data):
        """Generate synthetic ESG scores based on industry benchmarks"""
        try:
            sector = company_data['sector']
            industry_benchmark = self.industry_esg_benchmarks.get(sector, self.industry_esg_benchmarks['Technology'])
            
            # Add some randomness to make scores more realistic
            np.random.seed(hash(company_data['ticker']) % 2**32)  # Consistent randomness per ticker
            
            # Generate scores with some variation around industry benchmarks
            environmental_score = max(0, min(100, industry_benchmark['E'] + np.random.normal(0, 10)))
            social_score = max(0, min(100, industry_benchmark['S'] + np.random.normal(0, 10)))
            governance_score = max(0, min(100, industry_benchmark['G'] + np.random.normal(0, 10)))
            
            # Calculate weighted total score
            total_score = (
                environmental_score * self.esg_weights['environmental'] +
                social_score * self.esg_weights['social'] +
                governance_score * self.esg_weights['governance']
            )
            
            # Generate detailed factor scores
            detailed_scores = {}
            for category, factors in self.esg_factors.items():
                category_scores = {}
                base_score = {
                    'environmental': environmental_score,
                    'social': social_score,
                    'governance': governance_score
                }[category]
                
                for factor, details in factors.items():
                    # Generate factor score around category average
                    factor_score = max(0, min(100, base_score + np.random.normal(0, 8)))
                    category_scores[factor] = {
                        'score': factor_score,
                        'weight': details['weight'],
                        'description': details['description']
                    }
                
                detailed_scores[category] = category_scores
            
            # Generate risk level
            if total_score >= 80:
                risk_level = 'Low'
            elif total_score >= 60:
                risk_level = 'Medium'
            elif total_score >= 40:
                risk_level = 'High'
            else:
                risk_level = 'Severe'
            
            # Generate letter grade
            if total_score >= 90:
                letter_grade = 'A+'
            elif total_score >= 85:
                letter_grade = 'A'
            elif total_score >= 80:
                letter_grade = 'A-'
            elif total_score >= 75:
                letter_grade = 'B+'
            elif total_score >= 70:
                letter_grade = 'B'
            elif total_score >= 65:
                letter_grade = 'B-'
            elif total_score >= 60:
                letter_grade = 'C+'
            elif total_score >= 55:
                letter_grade = 'C'
            elif total_score >= 50:
                letter_grade = 'C-'
            else:
                letter_grade = 'D'
            
            return {
                'environmental_score': environmental_score,
                'social_score': social_score,
                'governance_score': governance_score,
                'total_score': total_score,
                'detailed_scores': detailed_scores,
                'risk_level': risk_level,
                'letter_grade': letter_grade,
                'industry_benchmark': industry_benchmark,
                'percentile_rank': self.calculate_percentile_rank(total_score, sector)
            }
            
        except Exception as e:
            st.error(f"❌ Error generating synthetic ESG scores: {e}")
            return None
    
    def calculate_percentile_rank(self, score, sector):
        """Calculate percentile rank within sector"""
        try:
            # Simulate distribution of scores within sector
            industry_avg = self.industry_esg_benchmarks.get(sector, {'total': 70})['total']
            
            # Assume normal distribution with industry average and standard deviation of 15
            from scipy.stats import norm
            percentile = norm.cdf(score, loc=industry_avg, scale=15) * 100
            
            return min(99, max(1, percentile))
            
        except:
            return 50  # Default to median
    
    def esg_weighted_valuation(self, base_valuation, esg_data):
        """Apply ESG weighting to base valuation"""
        try:
            if not esg_data or 'synthetic_esg' not in esg_data:
                return base_valuation
            
            esg_scores = esg_data['synthetic_esg']
            total_esg_score = esg_scores['total_score']
            
            # ESG adjustment factor (0.8 to 1.2 range)
            # Higher ESG scores get premium, lower scores get discount
            if total_esg_score >= 80:
                esg_multiplier = 1.1 + (total_esg_score - 80) * 0.005  # Up to 1.2x for perfect score
            elif total_esg_score >= 60:
                esg_multiplier = 1.0 + (total_esg_score - 60) * 0.005  # 1.0x to 1.1x
            elif total_esg_score >= 40:
                esg_multiplier = 0.9 + (total_esg_score - 40) * 0.005  # 0.9x to 1.0x
            else:
                esg_multiplier = 0.8 + total_esg_score * 0.0025  # 0.8x to 0.9x
            
            # Apply ESG adjustment to different valuation methods
            esg_adjusted_valuation = {}
            
            for method, value in base_valuation.items():
                if isinstance(value, (int, float)) and value > 0:
                    esg_adjusted_valuation[method] = {
                        'base_value': value,
                        'esg_multiplier': esg_multiplier,
                        'esg_adjusted_value': value * esg_multiplier,
                        'adjustment_amount': value * (esg_multiplier - 1)
                    }
                else:
                    esg_adjusted_valuation[method] = {
                        'base_value': value,
                        'esg_multiplier': 1.0,
                        'esg_adjusted_value': value,
                        'adjustment_amount': 0
                    }
            
            return {
                'esg_adjusted_valuations': esg_adjusted_valuation,
                'overall_esg_multiplier': esg_multiplier,
                'esg_score_used': total_esg_score,
                'adjustment_rationale': self.get_esg_adjustment_rationale(total_esg_score)
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating ESG-weighted valuation: {e}")
            return base_valuation
    
    def get_esg_adjustment_rationale(self, esg_score):
        """Get rationale for ESG adjustment"""
        if esg_score >= 80:
            return "Premium valuation due to strong ESG performance, lower regulatory risk, and sustainable business practices"
        elif esg_score >= 60:
            return "Neutral to slight premium due to adequate ESG performance"
        elif esg_score >= 40:
            return "Slight discount due to moderate ESG risks"
        else:
            return "Significant discount due to high ESG risks and potential regulatory/reputational issues"
    
    def esg_risk_analysis(self, esg_data):
        """Analyze ESG risks and opportunities"""
        try:
            if not esg_data or 'synthetic_esg' not in esg_data:
                return None
            
            esg_scores = esg_data['synthetic_esg']
            detailed_scores = esg_scores['detailed_scores']
            
            risk_analysis = {
                'overall_risk_level': esg_scores['risk_level'],
                'category_risks': {},
                'top_risks': [],
                'top_opportunities': [],
                'improvement_recommendations': []
            }
            
            # Analyze each category
            for category, factors in detailed_scores.items():
                category_scores = [factor['score'] for factor in factors.values()]
                category_avg = np.mean(category_scores)
                
                # Identify risk level for category
                if category_avg >= 75:
                    risk_level = 'Low'
                elif category_avg >= 60:
                    risk_level = 'Medium'
                elif category_avg >= 45:
                    risk_level = 'High'
                else:
                    risk_level = 'Severe'
                
                risk_analysis['category_risks'][category] = {
                    'average_score': category_avg,
                    'risk_level': risk_level,
                    'factors': factors
                }
                
                # Identify top risks (lowest scores)
                for factor_name, factor_data in factors.items():
                    if factor_data['score'] < 50:
                        risk_analysis['top_risks'].append({
                            'category': category,
                            'factor': factor_name,
                            'score': factor_data['score'],
                            'description': factor_data['description'],
                            'severity': 'High' if factor_data['score'] < 30 else 'Medium'
                        })
                
                # Identify opportunities (high scores that can be leveraged)
                for factor_name, factor_data in factors.items():
                    if factor_data['score'] > 80:
                        risk_analysis['top_opportunities'].append({
                            'category': category,
                            'factor': factor_name,
                            'score': factor_data['score'],
                            'description': factor_data['description']
                        })
            
            # Generate improvement recommendations
            risk_analysis['improvement_recommendations'] = self.generate_esg_recommendations(risk_analysis)
            
            # Sort risks and opportunities by severity/potential
            risk_analysis['top_risks'] = sorted(risk_analysis['top_risks'], key=lambda x: x['score'])[:5]
            risk_analysis['top_opportunities'] = sorted(risk_analysis['top_opportunities'], key=lambda x: x['score'], reverse=True)[:5]
            
            return risk_analysis
            
        except Exception as e:
            st.error(f"❌ Error in ESG risk analysis: {e}")
            return None
    
    def generate_esg_recommendations(self, risk_analysis):
        """Generate ESG improvement recommendations"""
        recommendations = []
        
        try:
            # Recommendations based on category performance
            for category, data in risk_analysis['category_risks'].items():
                if data['risk_level'] in ['High', 'Severe']:
                    if category == 'environmental':
                        recommendations.extend([
                            "Implement carbon reduction initiatives and set science-based targets",
                            "Invest in renewable energy and energy efficiency programs",
                            "Develop comprehensive waste reduction and recycling programs",
                            "Improve water conservation and management practices"
                        ])
                    elif category == 'social':
                        recommendations.extend([
                            "Enhance employee engagement and satisfaction programs",
                            "Strengthen diversity, equity, and inclusion initiatives",
                            "Expand community investment and social impact programs",
                            "Improve product safety and quality assurance processes"
                        ])
                    elif category == 'governance':
                        recommendations.extend([
                            "Increase board independence and diversity",
                            "Align executive compensation with long-term performance",
                            "Enhance transparency in financial and ESG reporting",
                            "Strengthen ethics and compliance programs"
                        ])
            
            # Specific recommendations for top risks
            for risk in risk_analysis['top_risks']:
                factor = risk['factor']
                if factor == 'carbon_emissions':
                    recommendations.append("Set ambitious carbon neutrality targets and develop decarbonization roadmap")
                elif factor == 'diversity_inclusion':
                    recommendations.append("Implement comprehensive D&I strategy with measurable targets")
                elif factor == 'board_structure':
                    recommendations.append("Recruit independent directors with relevant expertise")
                elif factor == 'employee_relations':
                    recommendations.append("Conduct employee satisfaction surveys and address key concerns")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            st.warning(f"⚠️ Error generating recommendations: {e}")
            return ["Conduct comprehensive ESG assessment to identify improvement areas"]
    
    def esg_peer_comparison(self, esg_data, peer_tickers):
        """Compare ESG performance with industry peers"""
        try:
            if not esg_data or not peer_tickers:
                return None
            
            st.info(f"📊 Comparing ESG performance with {len(peer_tickers)} peers...")
            
            # Get ESG data for peers
            peer_esg_data = {}
            for peer_ticker in peer_tickers:
                try:
                    peer_data = self.get_esg_data(peer_ticker)
                    if peer_data and 'synthetic_esg' in peer_data:
                        peer_esg_data[peer_ticker] = peer_data['synthetic_esg']
                except:
                    continue
            
            if not peer_esg_data:
                st.warning("⚠️ No peer ESG data available for comparison")
                return None
            
            # Create comparison dataframe
            comparison_data = []
            
            # Add target company
            target_esg = esg_data['synthetic_esg']
            comparison_data.append({
                'ticker': esg_data['company_data']['ticker'],
                'company_name': esg_data['company_data']['name'],
                'environmental': target_esg['environmental_score'],
                'social': target_esg['social_score'],
                'governance': target_esg['governance_score'],
                'total_esg': target_esg['total_score'],
                'letter_grade': target_esg['letter_grade'],
                'is_target': True
            })
            
            # Add peers
            for peer_ticker, peer_esg in peer_esg_data.items():
                comparison_data.append({
                    'ticker': peer_ticker,
                    'company_name': peer_ticker,  # Would get actual name in real implementation
                    'environmental': peer_esg['environmental_score'],
                    'social': peer_esg['social_score'],
                    'governance': peer_esg['governance_score'],
                    'total_esg': peer_esg['total_score'],
                    'letter_grade': peer_esg['letter_grade'],
                    'is_target': False
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Calculate rankings
            comparison_df['esg_rank'] = comparison_df['total_esg'].rank(ascending=False)
            comparison_df['env_rank'] = comparison_df['environmental'].rank(ascending=False)
            comparison_df['social_rank'] = comparison_df['social'].rank(ascending=False)
            comparison_df['gov_rank'] = comparison_df['governance'].rank(ascending=False)
            
            # Calculate percentiles
            total_companies = len(comparison_df)
            target_row = comparison_df[comparison_df['is_target']].iloc[0]
            
            peer_analysis = {
                'comparison_df': comparison_df,
                'target_company': esg_data['company_data']['ticker'],
                'total_peers': len(peer_esg_data),
                'rankings': {
                    'total_esg': int(target_row['esg_rank']),
                    'environmental': int(target_row['env_rank']),
                    'social': int(target_row['social_rank']),
                    'governance': int(target_row['gov_rank'])
                },
                'percentiles': {
                    'total_esg': (total_companies - target_row['esg_rank'] + 1) / total_companies * 100,
                    'environmental': (total_companies - target_row['env_rank'] + 1) / total_companies * 100,
                    'social': (total_companies - target_row['social_rank'] + 1) / total_companies * 100,
                    'governance': (total_companies - target_row['gov_rank'] + 1) / total_companies * 100
                },
                'peer_averages': {
                    'environmental': comparison_df[~comparison_df['is_target']]['environmental'].mean(),
                    'social': comparison_df[~comparison_df['is_target']]['social'].mean(),
                    'governance': comparison_df[~comparison_df['is_target']]['governance'].mean(),
                    'total_esg': comparison_df[~comparison_df['is_target']]['total_esg'].mean()
                }
            }
            
            return peer_analysis
            
        except Exception as e:
            st.error(f"❌ Error in ESG peer comparison: {e}")
            return None
    
    def create_esg_dashboard(self, esg_data, risk_analysis, peer_comparison=None):
        """Create comprehensive ESG analysis dashboard"""
        try:
            if not esg_data or 'synthetic_esg' not in esg_data:
                st.error("❌ No ESG data available for dashboard")
                return None
            
            esg_scores = esg_data['synthetic_esg']
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'ESG Score Breakdown', 'ESG vs Industry Benchmark',
                    'Risk Factor Analysis', 'ESG Score Distribution',
                    'Category Performance', 'ESG Trends (Simulated)'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "heatmap"}, {"type": "histogram"}],
                    [{"type": "radar"}, {"type": "scatter"}]
                ]
            )
            
            # ESG Score Breakdown
            categories = ['Environmental', 'Social', 'Governance']
            scores = [
                esg_scores['environmental_score'],
                esg_scores['social_score'],
                esg_scores['governance_score']
            ]
            colors = ['green', 'blue', 'orange']
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=colors,
                    name='ESG Scores',
                    text=[f'{score:.1f}' for score in scores],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # ESG vs Industry Benchmark
            benchmark = esg_scores['industry_benchmark']
            benchmark_scores = [benchmark['E'], benchmark['S'], benchmark['G']]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=benchmark_scores,
                    marker_color='lightgray',
                    name='Industry Benchmark',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=colors,
                    name='Company Scores',
                    opacity=0.8
                ),
                row=1, col=2
            )
            
            # Risk Factor Heatmap
            if risk_analysis and 'category_risks' in risk_analysis:
                risk_data = []
                risk_labels = []
                
                for category, data in risk_analysis['category_risks'].items():
                    for factor, factor_data in data['factors'].items():
                        risk_data.append([factor_data['score']])
                        risk_labels.append(f"{category.title()}: {factor.replace('_', ' ').title()}")
                
                if risk_data:
                    fig.add_trace(
                        go.Heatmap(
                            z=risk_data,
                            y=risk_labels,
                            x=['Score'],
                            colorscale='RdYlGn',
                            zmin=0,
                            zmax=100,
                            showscale=False
                        ),
                        row=2, col=1
                    )
            
            # ESG Score Distribution (if peer comparison available)
            if peer_comparison:
                peer_scores = peer_comparison['comparison_df']['total_esg'].tolist()
                
                fig.add_trace(
                    go.Histogram(
                        x=peer_scores,
                        nbinsx=10,
                        name='Peer Distribution',
                        opacity=0.7
                    ),
                    row=2, col=2
                )
                
                # Add target company line
                target_score = esg_scores['total_score']
                fig.add_vline(
                    x=target_score,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Company: {target_score:.1f}",
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"ESG Analysis Dashboard - {esg_data['company_data']['name']}",
                height=1200,
                showlegend=True
            )
            
            # Update axes
            fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=1)
            fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=2)
            fig.update_yaxes(title_text="ESG Factors", row=2, col=1)
            fig.update_xaxes(title_text="ESG Score", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating ESG dashboard: {e}")
            return None
    
    def esg_investment_screening(self, tickers, screening_criteria):
        """Screen investments based on ESG criteria"""
        try:
            st.info(f"📊 Screening {len(tickers)} investments based on ESG criteria...")
            
            screening_results = []
            
            for ticker in tickers:
                try:
                    esg_data = self.get_esg_data(ticker)
                    if not esg_data or 'synthetic_esg' not in esg_data:
                        continue
                    
                    esg_scores = esg_data['synthetic_esg']
                    company_data = esg_data['company_data']
                    
                    # Apply screening criteria
                    passes_screen = True
                    screening_details = {}
                    
                    if 'min_total_esg' in screening_criteria:
                        passes_screen &= esg_scores['total_score'] >= screening_criteria['min_total_esg']
                        screening_details['total_esg_pass'] = esg_scores['total_score'] >= screening_criteria['min_total_esg']
                    
                    if 'min_environmental' in screening_criteria:
                        passes_screen &= esg_scores['environmental_score'] >= screening_criteria['min_environmental']
                        screening_details['environmental_pass'] = esg_scores['environmental_score'] >= screening_criteria['min_environmental']
                    
                    if 'min_social' in screening_criteria:
                        passes_screen &= esg_scores['social_score'] >= screening_criteria['min_social']
                        screening_details['social_pass'] = esg_scores['social_score'] >= screening_criteria['min_social']
                    
                    if 'min_governance' in screening_criteria:
                        passes_screen &= esg_scores['governance_score'] >= screening_criteria['min_governance']
                        screening_details['governance_pass'] = esg_scores['governance_score'] >= screening_criteria['min_governance']
                    
                    if 'excluded_sectors' in screening_criteria:
                        passes_screen &= company_data['sector'] not in screening_criteria['excluded_sectors']
                        screening_details['sector_pass'] = company_data['sector'] not in screening_criteria['excluded_sectors']
                    
                    screening_results.append({
                        'ticker': ticker,
                        'company_name': company_data['name'],
                        'sector': company_data['sector'],
                        'total_esg_score': esg_scores['total_score'],
                        'environmental_score': esg_scores['environmental_score'],
                        'social_score': esg_scores['social_score'],
                        'governance_score': esg_scores['governance_score'],
                        'letter_grade': esg_scores['letter_grade'],
                        'risk_level': esg_scores['risk_level'],
                        'passes_screen': passes_screen,
                        'screening_details': screening_details
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not screen {ticker}: {e}")
                    continue
            
            # Convert to DataFrame and sort by ESG score
            screening_df = pd.DataFrame(screening_results)
            if not screening_df.empty:
                screening_df = screening_df.sort_values('total_esg_score', ascending=False)
            
            # Calculate screening statistics
            total_screened = len(screening_df)
            passed_screen = len(screening_df[screening_df['passes_screen']]) if not screening_df.empty else 0
            pass_rate = (passed_screen / total_screened * 100) if total_screened > 0 else 0
            
            screening_summary = {
                'total_screened': total_screened,
                'passed_screen': passed_screen,
                'pass_rate': pass_rate,
                'screening_criteria': screening_criteria,
                'results_df': screening_df
            }
            
            st.success(f"✅ ESG screening completed: {passed_screen}/{total_screened} investments passed ({pass_rate:.1f}%)")
            return screening_summary
            
        except Exception as e:
            st.error(f"❌ Error in ESG investment screening: {e}")
            return None