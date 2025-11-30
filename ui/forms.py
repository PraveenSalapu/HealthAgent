"""
Form components for user input.

Note: Form rendering is temporarily handled in main app.
TODO: Extract form components here for better modularity.
"""

import streamlit as st
from typing import Dict
from config.settings import FEATURE_CONFIGS, FEATURE_NAMES, FEATURE_INFO, FORM_SECTIONS


def render_prediction_form() -> Dict[str, float]:
    """
    Render the complete prediction form.

    Returns:
        dict: User input data if form submitted, None otherwise
    """
    # Hide text cursor, blue focus box, and info icons
    st.markdown("""
        <style>
        /* Hide cursor in selectbox */
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
            user-select: none;
        }
        /* Remove blue focus outline/box */
        div[data-baseweb="select"] input:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        /* Remove focus border on select container */
        div[data-baseweb="select"]:focus-within {
            border-color: rgb(250, 250, 250, 0.2) !important;
        }
        /* Hide info icons (ⓘ symbols) */
        .stTooltipIcon {
            display: none !important;
        }
        button[data-testid="stFormSubmitButton"] svg {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    user_data = {}

    with st.form("prediction_form"):
        for section in FORM_SECTIONS:
            st.markdown(f"### {section['title']}")

            # Enhanced info card with helpful context
            section_tips = {
                "Wellness Snapshot": "📊 **What to know:** These metrics reflect your current health status. Be honest about how you feel - this helps us provide accurate risk assessment.",
                "Cardiometabolic History": "🩺 **What to know:** Pre-existing conditions significantly impact diabetes risk. Select 'Yes' if you've been diagnosed by a healthcare provider.",
                "Lifestyle & Support": "🏠 **What to know:** These factors help us understand your health context. All information is confidential and used only for risk calculation."
            }

            tip = section_tips.get(section['title'], section['description'])
            st.info(f"ℹ️ {tip}")

            cols = st.columns(2)
            for idx, feature in enumerate(section['features']):
                col = cols[idx % 2]
                config = FEATURE_CONFIGS[feature]
                display_name = FEATURE_NAMES[feature]
                help_text = config.get('help', FEATURE_INFO.get(feature, ''))
                
                with col:
                    if config['type'] == 'select':
                        # Create formatted options
                        options = config['options']
                        labels = config['labels']
                        formatted_options = [f"{labels[i]}" for i in range(len(options))]

                        selected_label = st.selectbox(
                            display_name,
                            options=formatted_options,
                            key=feature
                        )
                        # Map back to numeric value
                        selected_idx = formatted_options.index(selected_label)
                        user_data[feature] = options[selected_idx]
                    else:
                        user_data[feature] = st.number_input(
                            display_name,
                            min_value=config['min'],
                            max_value=config['max'],
                            value=config['default'],
                            step=config['step'],
                            key=feature
                        )

                    # Add helpful hints for specific fields
                    field_hints = {
                        "BMI": "💡 *Healthy range: 18.5-24.9*",
                        "PhysHlth": "💡 *Count days with pain, illness, or injury*",
                        "GenHlth": "💡 *Rate your overall health honestly*",
                        "PhysActivity": "💡 *Any exercise in the past 30 days?*"
                    }

                    if feature in field_hints:
                        st.caption(field_hints[feature])
            
            if section != FORM_SECTIONS[-1]:
                st.markdown("---")
        
        submitted = st.form_submit_button("🔍 Assess Risk", type="primary")
        
        if submitted:
            return user_data
    
    return None


__all__ = ['render_prediction_form']
