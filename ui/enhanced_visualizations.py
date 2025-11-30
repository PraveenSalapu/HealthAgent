"""
Enhanced visualization functions for health metrics with actionable insights.

This module provides improved visualizations:
- Feature importance chart (model explainability)
- Top risk factors comparison (simplified)
- Risk trajectory simulator (what-if analysis)
- Enhanced insights with action items
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple
from config.settings import (
    FEATURE_CONFIGS,
    FEATURE_NAMES,
    DEFAULT_DIABETIC_AVERAGES
)


def get_feature_contributions(user_data: Dict[str, float],
                              avg_data: Dict[str, float],
                              feature_importance: Dict[str, float]) -> List[Tuple[str, float, float, float]]:
    """
    Calculate feature contributions combining user deviation and model importance.

    Returns:
        List of (feature, deviation_score, importance, combined_score)
    """
    contributions = []

    for feature, config in FEATURE_CONFIGS.items():
        user_value = float(user_data.get(feature, avg_data.get(feature, 0)))
        avg_value = float(avg_data.get(feature, DEFAULT_DIABETIC_AVERAGES.get(feature, user_value)))
        importance = feature_importance.get(feature, 0) * 100  # Convert to percentage

        # Calculate normalized deviation
        if config["type"] == "number":
            span = float(config["max"] - config["min"])
            if span == 0:
                deviation = 0
            else:
                deviation = abs(user_value - avg_value) / span * 100
        else:
            deviation = abs(user_value - avg_value) * 100

        # Combined score: importance weighted by deviation
        combined_score = importance * (1 + deviation / 100)

        contributions.append((feature, deviation, importance, combined_score))

    # Sort by combined score
    contributions.sort(key=lambda x: x[3], reverse=True)
    return contributions


def create_feature_importance_chart(feature_importance: Dict[str, float],
                                    user_data: Dict[str, float],
                                    avg_data: Dict[str, float]) -> go.Figure:
    """
    Create feature importance chart showing what the model considers most important.
    Enhanced with user's actual values.
    """
    # Get contributions
    contributions = get_feature_contributions(user_data, avg_data, feature_importance)

    # Take top 8 features
    top_features = contributions[:8]

    features = [FEATURE_NAMES[f] for f, _, _, _ in top_features]
    importances = [imp for _, _, imp, _ in top_features]

    # Determine if user's value is risk-increasing or risk-decreasing
    colors = []
    for feature_key, dev, _, _ in top_features:
        user_val = user_data.get(feature_key, 0)
        avg_val = avg_data.get(feature_key, 0)

        # Risk-increasing features (higher is worse)
        risk_increasing = ['GenHlth', 'HighBP', 'HighChol', 'BMI', 'PhysHlth',
                          'HeartDiseaseorAttack', 'DiffWalk', 'Age']

        if feature_key in risk_increasing:
            colors.append('#f87171' if user_val > avg_val else '#10b981')
        else:  # Risk-decreasing (e.g., PhysActivity - higher is better)
            colors.append('#10b981' if user_val >= avg_val else '#f87171')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=importances,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        text=[f'{v:.1f}%' for v in importances],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=12, family='"Inter","Segoe UI",sans-serif', weight=600),
        hovertemplate='<b>%{y}</b><br>Model Importance: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="🎯 What Drives Your Risk Score? (Model's Top Factors)",
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        xaxis=dict(
            title=dict(
                text="Model Importance (%)",
                font=dict(color='#94a3b8', size=13)
            ),
            gridcolor="rgba(100, 116, 255, 0.15)",
            tickfont=dict(color='#cbd5e1', size=11),
            range=[0, max(importances) * 1.2]
        ),
        yaxis=dict(
            tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif'),
            autorange="reversed"
        ),
        height=450,
        margin=dict(l=200, r=80, t=80, b=60),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        plot_bgcolor="rgba(21, 29, 53, 0.5)",
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        showlegend=False,
        annotations=[
            dict(
                text="🔴 Red = Higher risk than average | 🟢 Green = Lower risk than average",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=11, color='#94a3b8'),
                xanchor='center'
            )
        ]
    )

    return fig


def create_top_factors_comparison(user_data: Dict[str, float],
                                  avg_data: Dict[str, float],
                                  feature_importance: Dict[str, float]) -> go.Figure:
    """
    Create simplified comparison chart showing only top 6 most impactful factors.
    """
    # Get contributions
    contributions = get_feature_contributions(user_data, avg_data, feature_importance)

    # Take top 6 features
    top_features = [f for f, _, _, _ in contributions[:6]]

    features = [FEATURE_NAMES[f] for f in top_features]
    user_values = [user_data[f] for f in top_features]
    avg_values = [avg_data.get(f, 0) for f in top_features]

    fig = go.Figure()

    # User values
    fig.add_trace(go.Bar(
        name='Your Values',
        x=features,
        y=user_values,
        marker=dict(
            color='#00d9ff',
            line=dict(color='#00b8d4', width=1.5)
        ),
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=13, family='"Inter","Segoe UI",sans-serif', weight=600)
    ))

    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Diabetic Avg',
        x=features,
        y=avg_values,
        marker=dict(
            color='#7c3aed',
            line=dict(color='#6d28d9', width=1.5)
        ),
        text=[f'{v:.1f}' for v in avg_values],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=13, family='"Inter","Segoe UI",sans-serif', weight=500)
    ))

    fig.update_layout(
        title=dict(
            text='Your Top Risk Factors vs. Average',
            font=dict(size=16, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(
            text='Key Health Factors',
            font=dict(size=12, color='#94a3b8')
        ),
        yaxis_title=dict(
            text='Value',
            font=dict(size=12, color='#94a3b8')
        ),
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=11)
        ),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=50, r=30, t=60, b=100),
        bargap=0.2,
        bargroupgap=0.1
    )

    fig.update_xaxes(
        tickangle=-20,
        showgrid=False,
        linecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=11, family='"Inter","Segoe UI",sans-serif')
    )
    fig.update_yaxes(
        gridcolor='rgba(100, 116, 255, 0.15)',
        zerolinecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=11)
    )

    return fig


def create_risk_simulator_data(user_data: Dict[str, float],
                               predictor,
                               feature_importance: Dict[str, float]) -> Dict:
    """
    Calculate what-if scenarios for top modifiable factors.

    Returns dictionary with simulation results.
    """
    # Get top 5 modifiable features (exclude Age, Education, Income)
    non_modifiable = ['Age', 'Education', 'Income', 'HeartDiseaseorAttack']
    modifiable_features = {k: v for k, v in feature_importance.items()
                          if k not in non_modifiable}

    # Sort by importance
    top_modifiable = sorted(modifiable_features.items(), key=lambda x: x[1], reverse=True)[:5]

    scenarios = []
    baseline_prob = predictor.predict(user_data)[0]

    for feature, importance in top_modifiable:
        # Create improved scenario
        improved_data = user_data.copy()
        config = FEATURE_CONFIGS[feature]

        if config['type'] == 'number':
            # Improve by 10% towards ideal
            current_val = user_data[feature]
            if feature == 'BMI':
                ideal_val = 22.0
                improved_val = current_val - (current_val - ideal_val) * 0.1
            elif feature in ['GenHlth', 'PhysHlth']:
                # Lower is better
                improved_val = max(config['min'], current_val - (current_val - config['min']) * 0.2)
            else:
                improved_val = current_val * 0.9
            improved_data[feature] = improved_val
        else:
            # Binary - flip to good value
            if feature == 'PhysActivity':
                improved_data[feature] = 1  # Active
            else:
                improved_data[feature] = 0  # No condition

        # Predict with improvement
        improved_prob = predictor.predict(improved_data)[0]
        impact = baseline_prob - improved_prob

        scenarios.append({
            'feature': feature,
            'feature_name': FEATURE_NAMES[feature],
            'current_value': user_data[feature],
            'improved_value': improved_data[feature],
            'baseline_risk': baseline_prob,
            'improved_risk': improved_prob,
            'impact': impact,
            'importance': importance * 100
        })

    return {
        'baseline_risk': baseline_prob,
        'scenarios': sorted(scenarios, key=lambda x: x['impact'], reverse=True)
    }


def create_risk_simulator_chart(simulator_data: Dict) -> go.Figure:
    """
    Create interactive risk simulator showing potential improvements.
    """
    scenarios = simulator_data['scenarios'][:5]  # Top 5
    baseline = simulator_data['baseline_risk']

    features = [s['feature_name'] for s in scenarios]
    impacts = [s['impact'] for s in scenarios]
    improved_risks = [s['improved_risk'] for s in scenarios]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Impact bars
    fig.add_trace(go.Bar(
        name='Potential Risk Reduction',
        x=features,
        y=impacts,
        marker=dict(
            color=['#10b981' if i > 0 else '#f87171' for i in impacts],
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        text=[f'-{i:.1f}%' if i > 0 else f'+{abs(i):.1f}%' for i in impacts],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=12, family='"Inter","Segoe UI",sans-serif', weight=600),
        yaxis='y',
        hovertemplate='<b>%{x}</b><br>Risk Reduction: %{y:.1f}%<extra></extra>'
    ))

    # Baseline line
    fig.add_trace(go.Scatter(
        name=f'Current Risk ({baseline:.1f}%)',
        x=features,
        y=[baseline] * len(features),
        mode='lines',
        line=dict(color='#00d9ff', width=2, dash='dash'),
        yaxis='y2',
        hovertemplate='<b>Current Risk</b>: %{y:.1f}%<extra></extra>'
    ))

    # Improved risk line
    fig.add_trace(go.Scatter(
        name='Potential New Risk',
        x=features,
        y=improved_risks,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10, symbol='diamond'),
        yaxis='y2',
        hovertemplate='<b>Improved Risk</b>: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="What-If Simulator",
            font=dict(size=16, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700),
            x=0.5,
            xanchor='center'
        ),
        yaxis=dict(
            title=dict(
                text="Risk Reduction (%)",
                font=dict(color='#10b981', size=11)
            ),
            tickfont=dict(color='#cbd5e1', size=10),
            gridcolor="rgba(100, 116, 255, 0.15)",
            side='left'
        ),
        yaxis2=dict(
            title=dict(
                text="Risk Probability (%)",
                font=dict(color='#00d9ff', size=11)
            ),
            tickfont=dict(color='#cbd5e1', size=10),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            tickangle=-20,
            tickfont=dict(color='#cbd5e1', size=10, family='"Inter","Segoe UI",sans-serif'),
            showgrid=False
        ),
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=10)
        ),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=55, r=55, t=60, b=100)
    )

    return fig


def generate_actionable_insights(user_data: Dict[str, float],
                                 avg_data: Dict[str, float],
                                 feature_importance: Dict[str, float],
                                 risk_probability: float) -> List[Dict[str, str]]:
    """
    Generate enhanced insights with actionable recommendations.

    Returns:
        List of insight dictionaries with finding, impact, and action.
    """
    insights = []

    # Get contributions
    contributions = get_feature_contributions(user_data, avg_data, feature_importance)

    # Action item templates
    action_templates = {
        'BMI': {
            'high': {
                'finding': "Your BMI ({val:.1f}) is in the {category} range",
                'impact': "Higher BMI significantly increases diabetes risk",
                'action': "🎯 Target: Lose 5-10% body weight over 6 months through diet and exercise"
            },
            'normal': {
                'finding': "Your BMI ({val:.1f}) is in the healthy range",
                'impact': "Maintaining healthy weight protects against diabetes",
                'action': "✅ Keep up your current healthy habits"
            }
        },
        'PhysActivity': {
            'inactive': {
                'finding': "You reported no physical activity in the past 30 days",
                'impact': "Inactivity increases diabetes risk by ~25%",
                'action': "🚶 Start with 150 minutes/week of moderate activity (brisk walking counts!)"
            },
            'active': {
                'finding': "You're physically active",
                'impact': "Regular activity reduces diabetes risk significantly",
                'action': "✅ Maintain your activity level and consider resistance training"
            }
        },
        'HighBP': {
            'yes': {
                'finding': "You have high blood pressure",
                'impact': "Hypertension doubles diabetes risk",
                'action': "💊 Follow up with doctor for BP management (target: <130/80)"
            },
            'no': {
                'finding': "Your blood pressure is normal",
                'impact': "Good BP control reduces diabetes risk",
                'action': "✅ Monitor BP regularly and limit sodium intake"
            }
        },
        'HighChol': {
            'yes': {
                'finding': "You have high cholesterol",
                'impact': "High cholesterol increases cardiovascular and diabetes risk",
                'action': "🥗 Focus on heart-healthy fats (olive oil, fish, nuts) and consider statins"
            },
            'no': {
                'finding': "Your cholesterol is normal",
                'impact': "Healthy lipid levels protect heart and metabolic health",
                'action': "✅ Continue eating whole grains, fruits, and vegetables"
            }
        },
        'GenHlth': {
            'poor': {
                'finding': "You rated your general health as {category}",
                'impact': "Self-rated health strongly predicts diabetes risk",
                'action': "🏥 Schedule comprehensive checkup with A1C, lipids, and BP tests"
            }
        },
        'PhysHlth': {
            'high': {
                'finding': "You reported {val:.0f} days of poor physical health",
                'impact': "Chronic health issues increase diabetes susceptibility",
                'action': "🩺 Discuss symptom management and diabetes screening with your doctor"
            }
        },
        'HeartDiseaseorAttack': {
            'yes': {
                'finding': "You have history of heart disease/attack",
                'impact': "Cardiovascular disease and diabetes often coexist",
                'action': "❤️ Critical: Work with cardiologist AND endocrinologist for dual management"
            }
        }
    }

    # Generate insights for top 5 factors
    for feature, deviation, importance, combined in contributions[:6]:
        user_val = user_data[feature]
        avg_val = avg_data.get(feature, 0)

        insight = None

        if feature == 'BMI':
            category = 'obese' if user_val >= 30 else 'overweight' if user_val >= 25 else 'healthy'
            if user_val >= 25:
                template = action_templates['BMI']['high']
                insight = {
                    'icon': '⚖️',
                    'finding': template['finding'].format(val=user_val, category=category),
                    'impact': template['impact'],
                    'action': template['action']
                }

        elif feature == 'PhysActivity':
            template = action_templates['PhysActivity']['inactive' if user_val == 0 else 'active']
            insight = {
                'icon': '🏃' if user_val == 1 else '🛋️',
                'finding': template['finding'],
                'impact': template['impact'],
                'action': template['action']
            }

        elif feature == 'HighBP' and importance > 5:
            template = action_templates['HighBP']['yes' if user_val == 1 else 'no']
            insight = {
                'icon': '🩸',
                'finding': template['finding'],
                'impact': template['impact'],
                'action': template['action']
            }

        elif feature == 'HighChol' and importance > 5:
            template = action_templates['HighChol']['yes' if user_val == 1 else 'no']
            insight = {
                'icon': '🧬',
                'finding': template['finding'],
                'impact': template['impact'],
                'action': template['action']
            }

        elif feature == 'GenHlth' and user_val >= 4:
            category = 'fair' if user_val == 4 else 'poor'
            template = action_templates['GenHlth']['poor']
            insight = {
                'icon': '🏥',
                'finding': template['finding'].format(category=category),
                'impact': template['impact'],
                'action': template['action']
            }

        elif feature == 'PhysHlth' and user_val >= 10:
            template = action_templates['PhysHlth']['high']
            insight = {
                'icon': '🤒',
                'finding': template['finding'].format(val=user_val),
                'impact': template['impact'],
                'action': template['action']
            }

        elif feature == 'HeartDiseaseorAttack' and user_val == 1:
            template = action_templates['HeartDiseaseorAttack']['yes']
            insight = {
                'icon': '❤️',
                'finding': template['finding'],
                'impact': template['impact'],
                'action': template['action']
            }

        if insight:
            insights.append(insight)

    # Add overall risk context
    if risk_probability >= 60:
        insights.insert(0, {
            'icon': '⚠️',
            'finding': f"Your risk score ({risk_probability:.0f}%) is in the elevated range",
            'impact': "This suggests you may benefit from diabetes screening",
            'action': "📋 Schedule A1C test or oral glucose tolerance test within 1-2 months"
        })

    return insights[:6]  # Top 6 insights


__all__ = [
    'create_feature_importance_chart',
    'create_top_factors_comparison',
    'create_risk_simulator_chart',
    'create_risk_simulator_data',
    'generate_actionable_insights',
    'get_feature_contributions'
]
