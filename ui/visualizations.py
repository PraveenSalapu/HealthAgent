"""
Visualization functions for health metrics.

This module provides all the visualization functions used in the health assessment:
- Risk gauge chart
- Comparison chart
- Contribution waterfall
- Wellness radar
- Insights generation
"""

import plotly.graph_objects as go
from typing import Dict, List, Tuple
from config.settings import (
    FEATURE_CONFIGS,
    FEATURE_NAMES,
    RADAR_FEATURES,
    IDEAL_PROFILE,
    DEFAULT_DIABETIC_AVERAGES
)


def create_risk_gauge(probability: float) -> go.Figure:
    """Create risk level gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Diabetes Risk Probability",
            'font': {'size': 22, 'color': '#f1f5f9', 'family': '"Inter","Segoe UI",sans-serif', 'weight': 700}
        },
        number={
            'suffix': "%",
            'font': {'size': 52, 'color': '#00d9ff', 'family': '"Inter","Segoe UI",sans-serif', 'weight': 800}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1.5,
                'tickcolor': "rgba(100, 116, 255, 0.3)",
                'tickfont': {'color': '#cbd5e1', 'family': '"Inter","Segoe UI",sans-serif', 'size': 11}
            },
            'bar': {
                'color': "#00d9ff",
                'thickness': 0.8
            },
            'bgcolor': "rgba(21, 29, 53, 0.5)",
            'borderwidth': 2,
            'bordercolor': "rgba(100, 116, 255, 0.3)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(248, 113, 113, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#7c3aed", 'width': 5},
                'thickness': 0.8,
                'value': probability
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='rgba(10, 14, 26, 0)',
        font=dict(family='"Inter","Segoe UI",sans-serif')
    )
    return fig


def create_comparison_chart(user_data: Dict[str, float],
                           avg_data: Dict[str, float]) -> Tuple[go.Figure, List[float]]:
    """Create interactive comparison chart."""
    features = list(user_data.keys())
    user_values = [user_data[f] for f in features]
    avg_values = [avg_data.get(f, 0) for f in features]

    # Calculate differences
    differences = [user_values[i] - avg_values[i] for i in range(len(features))]

    fig = go.Figure()

    # User values
    fig.add_trace(go.Bar(
        name='Your Values',
        x=features,
        y=user_values,
        marker=dict(
            color='#00d9ff',
            line=dict(color='#00b8d4', width=1.5),
            pattern=dict(shape="")
        ),
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=13, family='"Inter","Segoe UI",sans-serif', weight=600)
    ))

    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Avg. Diabetic Population',
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
            text='Your Health Metrics vs. Average Diabetic Population',
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        xaxis_title='Health Factors',
        yaxis_title='Value',
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=12)
        ),
        uniformtext=dict(mode="show", minsize=12),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(
        tickangle=-35,
        showgrid=False,
        linecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif')
    )
    fig.update_yaxes(
        gridcolor='rgba(100, 116, 255, 0.15)',
        zerolinecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif')
    )

    return fig, differences


def prepare_feature_contributions(user_data: Dict[str, float],
                                  avg_data: Dict[str, float]) -> List[Tuple[str, float]]:
    """Compute normalized deltas for each feature to feed the waterfall chart."""
    contributions: List[Tuple[str, float]] = []

    for feature, config in FEATURE_CONFIGS.items():
        user_value = float(user_data.get(feature, avg_data.get(feature, 0)))
        avg_value = float(avg_data.get(feature, DEFAULT_DIABETIC_AVERAGES.get(feature, user_value)))

        if config["type"] == "number":
            span = float(config["max"] - config["min"])
            if span == 0:
                continue
            delta = (user_value - avg_value) / span * 100
        else:
            delta = (user_value - avg_value) * 100

        contributions.append((feature, delta))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions


def create_contribution_waterfall(user_data: Dict[str, float],
                                  avg_data: Dict[str, float]) -> go.Figure:
    """Build a waterfall chart showing how each factor shifts risk relative to diabetic average."""
    contributions = prepare_feature_contributions(user_data, avg_data)[:8]
    labels = [FEATURE_NAMES[f] for f, _ in contributions]
    deltas = [round(delta, 2) for _, delta in contributions]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(deltas),
            y=labels,
            x=deltas,
            connector={"mode": "spanning", "line": {"color": "rgba(100, 116, 255, 0.3)", "width": 2}},
            decreasing={"marker": {"color": "#10b981", "line": {"color": "#059669", "width": 1}}},
            increasing={"marker": {"color": "#f87171", "line": {"color": "#ef4444", "width": 1}}},
            textposition="outside",
            textfont=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif', size=11)
        )
    )

    fig.update_layout(
        title=dict(
            text="Feature shifts versus diabetic average (normalized)",
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        showlegend=False,
        height=420,
        margin=dict(l=120, r=30, t=60, b=40),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        plot_bgcolor="rgba(21, 29, 53, 0.5)",
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        xaxis=dict(
            title="Relative shift (percentage points)",
            gridcolor="rgba(100, 116, 255, 0.15)",
            zerolinecolor="rgba(100, 116, 255, 0.3)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            tickfont=dict(color="#cbd5e1"),
            gridcolor="rgba(100, 116, 255, 0.1)"
        ),
    )

    return fig


def score_feature_for_radar(value: float, feature: str) -> float:
    """Translate raw feature values into a 0-1 wellness score (1 = favorable)."""
    config = FEATURE_CONFIGS[feature]

    if feature == "GenHlth":
        min_val, max_val = 1, 5
        return max(0.0, min(1.0, 1 - (value - min_val) / (max_val - min_val)))

    if feature == "BMI":
        # Ideal BMI approximated at 22 within allowable range.
        ideal = 22.0
        spread = max(config["max"] - config["min"], 1)
        return max(0.0, min(1.0, 1 - abs(value - ideal) / (spread / 2)))

    if feature == "PhysHlth":
        return max(0.0, min(1.0, 1 - value / max(config["max"], 1)))

    if feature == "PhysActivity":
        return 1.0 if value >= 1 else 0.0

    if feature in {"HighBP", "HighChol"}:
        return 1.0 - min(1.0, max(0.0, value))

    return 0.5


def create_wellness_radar(user_data: Dict[str, float],
                          avg_data: Dict[str, float]) -> go.Figure:
    """Create a radar chart comparing the user to archetypes."""
    categories = [FEATURE_NAMES[f] for f in RADAR_FEATURES]

    user_scores = [
        score_feature_for_radar(float(user_data.get(f, avg_data.get(f, IDEAL_PROFILE.get(f, 0)))), f)
        for f in RADAR_FEATURES
    ]

    ideal_scores = [
        score_feature_for_radar(float(IDEAL_PROFILE.get(f, avg_data.get(f, 0))), f)
        for f in RADAR_FEATURES
    ]

    diabetic_scores = [
        score_feature_for_radar(float(avg_data.get(f, IDEAL_PROFILE.get(f, 0))), f)
        for f in RADAR_FEATURES
    ]

    # Close the loop for polar plot
    categories_closed = categories + [categories[0]]
    traces = {
        "Your profile": user_scores + [user_scores[0]],
        "Ideal baseline": ideal_scores + [ideal_scores[0]],
        "Diabetic average": diabetic_scores + [diabetic_scores[0]],
    }

    fig = go.Figure()
    palette = {
        "Your profile": "#00d9ff",
        "Ideal baseline": "#10b981",
        "Diabetic average": "#f97316",
    }

    for label, values in traces.items():
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill="toself",
                name=label,
                line=dict(color=palette[label], width=2),
                opacity=0.5 if label != "Your profile" else 0.7,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                showticklabels=True,
                ticks="",
                tickfont=dict(size=10, color="#cbd5e1"),
                gridcolor="rgba(100, 116, 255, 0.2)",
                linecolor="rgba(100, 116, 255, 0.3)",
            ),
            bgcolor="rgba(21, 29, 53, 0.5)",
            angularaxis=dict(
                gridcolor="rgba(100, 116, 255, 0.2)",
                linecolor="rgba(100, 116, 255, 0.3)",
                tickfont=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif', size=11)
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
            bgcolor="rgba(26, 35, 66, 0.7)",
            bordercolor="rgba(100, 116, 255, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        title=dict(
            text="Wellness balance across key factors",
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        height=450,
    )

    return fig


def generate_insights(user_data: Dict[str, float],
                      avg_data: Dict[str, float],
                      differences: List[float]) -> List[str]:
    """Generate textual insights from comparison."""
    insights = []
    features = list(user_data.keys())

    for i, feature in enumerate(features):
        diff = differences[i]
        user_val = user_data[feature]
        avg_val = avg_data.get(feature, 0)

        if feature in ["Age", "BMI", "GenHlth"]:
            if abs(diff) > 0.1 * avg_val:  # 10% threshold
                direction = "higher" if diff > 0 else "lower"
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: Your value ({user_val:.1f}) is "
                    f"{direction} than the average diabetic population ({avg_val:.1f})"
                )
        else:  # Binary features
            if user_val == 1 and avg_val > 0.5:
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: You share this risk factor with "
                    f"{avg_val*100:.0f}% of the diabetic population"
                )
            elif user_val == 0 and avg_val > 0.5:
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: You don't have this risk factor, "
                    f"which is positive (present in {avg_val*100:.0f}% of diabetic population)"
                )

    return insights[:5]  # Top 5 insights


__all__ = [
    'create_risk_gauge',
    'create_comparison_chart',
    'create_contribution_waterfall',
    'create_wellness_radar',
    'generate_insights',
]
