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
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700),
            x=0.5,
            xanchor='center',
            pad=dict(t=15, b=20)
        ),
        xaxis=dict(
            title=dict(
                text="Model Importance (%)",
                font=dict(color='#94a3b8', size=13),
                standoff=20
            ),
            gridcolor="rgba(100, 116, 255, 0.15)",
            tickfont=dict(color='#cbd5e1', size=11),
            range=[0, max(importances) * 1.3]
        ),
        yaxis=dict(
            tickfont=dict(color='#cbd5e1', size=13, family='"Inter","Segoe UI",sans-serif'),
            autorange="reversed",
            automargin=True
        ),
        height=520,
        margin=dict(l=250, r=120, t=100, b=90),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        plot_bgcolor="rgba(21, 29, 53, 0.5)",
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        showlegend=False,
        annotations=[
            dict(
                text="🔴 Red = Higher risk than average | 🟢 Green = Lower risk than average",
                xref="paper", yref="paper",
                x=0.5, y=-0.14,
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
    Enhanced with contextual labels showing actual values with units.
    """
    # Get contributions
    contributions = get_feature_contributions(user_data, avg_data, feature_importance)

    # Take top 6 features
    top_features = [f for f, _, _, _ in contributions[:6]]

    features = [FEATURE_NAMES[f] for f in top_features]
    user_values = [user_data[f] for f in top_features]
    avg_values = [avg_data.get(f, 0) for f in top_features]

    # Create contextual labels based on feature type
    def format_value_label(feature_key, value):
        """Format value with appropriate context and units."""
        if feature_key == 'BMI':
            return f'{value:.1f}'
        elif feature_key in ['Age', 'Education', 'Income']:
            return f'{int(value)}'
        elif feature_key in ['PhysHlth', 'MentHlth']:
            return f'{int(value)} days'
        elif feature_key in ['HighBP', 'HighChol', 'PhysActivity', 'DiffWalk', 'HeartDiseaseorAttack']:
            return 'Yes' if value == 1 else 'No'
        elif feature_key == 'GenHlth':
            health_map = {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
            return health_map.get(int(value), f'{int(value)}')
        else:
            return f'{value:.1f}'

    user_labels = [format_value_label(f, v) for f, v in zip(top_features, user_values)]
    avg_labels = [format_value_label(f, v) for f, v in zip(top_features, avg_values)]

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
        text=user_labels,
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=13, family='"Inter","Segoe UI",sans-serif', weight=600),
        customdata=[[f, uv, ul] for f, uv, ul in zip(top_features, user_values, user_labels)],
        hovertemplate='<b>%{x}</b><br>Your Value: %{customdata[2]}<extra></extra>'
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
        text=avg_labels,
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=13, family='"Inter","Segoe UI",sans-serif', weight=500),
        customdata=[[f, av, al] for f, av, al in zip(top_features, avg_values, avg_labels)],
        hovertemplate='<b>%{x}</b><br>Diabetic Avg: %{customdata[2]}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='📊 Your Top Risk Factors vs. Diabetic Average',
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700),
            x=0.5,
            xanchor='center',
            pad=dict(t=15, b=20)
        ),
        xaxis_title=dict(
            text='Key Health Factors',
            font=dict(size=13, color='#94a3b8'),
            standoff=20
        ),
        yaxis_title=dict(
            text='Value',
            font=dict(size=13, color='#94a3b8'),
            standoff=15
        ),
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=13)
        ),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=100, r=100, t=100, b=120),
        bargap=0.3,
        bargroupgap=0.15
    )

    fig.update_xaxes(
        tickangle=-30,
        showgrid=False,
        linecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif'),
        automargin=True
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
    Create horizontal bullet chart showing potential risk improvements.
    More intuitive than dual-axis - shows current vs potential risk as horizontal bars.
    """
    scenarios = simulator_data['scenarios'][:5]  # Top 5
    baseline = simulator_data['baseline_risk']

    # Reverse order so most impactful is on top
    scenarios = list(reversed(scenarios))

    features = [s['feature_name'] for s in scenarios]
    current_risks = [baseline] * len(scenarios)
    improved_risks = [s['improved_risk'] for s in scenarios]
    impacts = [s['impact'] for s in scenarios]

    fig = go.Figure()

    # Background bar (100% scale)
    fig.add_trace(go.Bar(
        name='Risk Scale',
        y=features,
        x=[100] * len(features),
        orientation='h',
        marker=dict(
            color='rgba(30, 41, 59, 0.3)',
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Current risk bars (semi-transparent)
    fig.add_trace(go.Bar(
        name=f'Current Risk',
        y=features,
        x=current_risks,
        orientation='h',
        marker=dict(
            color='rgba(239, 68, 68, 0.6)',
            line=dict(color='#ef4444', width=2)
        ),
        text=[f'{baseline:.1f}%'] * len(features),
        textposition='inside',
        textfont=dict(color='#ffffff', size=11, weight=700),
        hovertemplate='<b>%{y}</b><br>Current Risk: %{x:.1f}%<extra></extra>'
    ))

    # Potential improved risk bars
    fig.add_trace(go.Bar(
        name='Potential Risk (If Improved)',
        y=features,
        x=improved_risks,
        orientation='h',
        marker=dict(
            color='#10b981',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1.5),
            pattern=dict(shape="/", solidity=0.3)
        ),
        text=[f'{r:.1f}% (↓{i:.1f})' for r, i in zip(improved_risks, impacts)],
        textposition='inside',
        textfont=dict(color='#ffffff', size=11, weight=700),
        hovertemplate='<b>%{y}</b><br>Potential Risk: %{x:.1f}%<br>Reduction: %{customdata:.1f}%<extra></extra>',
        customdata=impacts
    ))

    fig.update_layout(
        title=dict(
            text="💡 What-If Simulator: Your Risk Reduction Potential",
            font=dict(size=17, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700),
            x=0.5,
            xanchor='center',
            pad=dict(t=15, b=20)
        ),
        xaxis=dict(
            title=dict(
                text="Risk Probability (%)",
                font=dict(color='#94a3b8', size=12),
                standoff=20
            ),
            range=[0, 100],
            gridcolor="rgba(100, 116, 255, 0.15)",
            tickfont=dict(color='#cbd5e1', size=11),
            showgrid=True
        ),
        yaxis=dict(
            tickfont=dict(color='#cbd5e1', size=13, family='"Inter","Segoe UI",sans-serif'),
            automargin=True
        ),
        height=580,
        hovermode='y unified',
        showlegend=True,
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=12)
        ),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=200, r=50, t=90, b=140)
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
