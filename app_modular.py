"""
Health AI Chatbot - Modular Multi-Agent Application

A diabetes risk prediction app with multi-agent chatbot support:
- Gemini Agent: Generic health insights
- Lightweight RAG Agent: Clinical insights from medical literature

Run with: streamlit run app_modular.py
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Modular imports
import pandas as pd
from config.settings import (
    MODEL_JSON_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    AVERAGES_PATH,
    CHAT_MODEL_GEMINI,
    FEATURE_NAMES,
    CHAT_MODEL_INFO,
    AGE_LABELS,
)
from models import load_model_components, load_diabetic_averages, DiabetesPredictor
from agents import AgentManager
from ui.forms import render_prediction_form
from ui.visualizations import create_risk_gauge
from ui.enhanced_visualizations import (
    create_feature_importance_chart,
    create_top_factors_comparison,
    create_risk_simulator_chart,
    create_risk_simulator_data,
    generate_actionable_insights,
)
from ui.chat_interface import render_model_selector, render_agent_status
from ui.styles import get_custom_css, get_progress_indicator_html, get_theme_toggle_html
from utils.helpers import classify_risk, build_profile_summary


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'prediction_prob' not in st.session_state:
        st.session_state.prediction_prob = 0.0
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = None
    if 'diabetic_averages' not in st.session_state:
        st.session_state.diabetic_averages = {}
    if 'active_chat_model' not in st.session_state:
        st.session_state.active_chat_model = CHAT_MODEL_GEMINI
    if 'chatbot_initialized' not in st.session_state:
        # Track initialization PER AGENT, not globally
        st.session_state.chatbot_initialized = {}
    if 'agent_welcome_cache' not in st.session_state:
        # Cache welcome messages per agent to avoid re-fetching
        st.session_state.agent_welcome_cache = {}
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"


def reset_app():
    """Reset application state for new assessment."""
    st.session_state.prediction_made = False
    st.session_state.user_data = {}
    st.session_state.prediction_prob = 0.0
    st.session_state.chatbot_initialized = {}  # Reset all agent initializations

    # Clear agent conversation histories and cache
    if st.session_state.agent_manager:
        st.session_state.agent_manager.clear_conversation_history("all")
    st.session_state.agent_welcome_cache = {}


def load_models():
    """Load ML models and initialize agents."""
    if st.session_state.predictor is None:
        with st.spinner("Loading models..."):
            # Load ML model components
            model, preprocessor, threshold = load_model_components(
                MODEL_JSON_PATH,
                PREPROCESSOR_PATH,
                THRESHOLD_PATH
            )

            if model and preprocessor:
                st.session_state.predictor = DiabetesPredictor(model, preprocessor, threshold)

            # Load diabetic averages
            st.session_state.diabetic_averages = load_diabetic_averages(AVERAGES_PATH)

    # Initialize agent manager
    if st.session_state.agent_manager is None:
        with st.spinner("Initializing AI agents..."):
            st.session_state.agent_manager = AgentManager()


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Health AI Chatbot - Multi-Agent System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Load models
    load_models()

    # Sidebar
    st.sidebar.title("🏥 Health AI Chatbot")
    st.sidebar.markdown("### Multi-Agent System")

    # Theme toggle in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Theme")
    theme_col1, theme_col2 = st.sidebar.columns(2)
    with theme_col1:
        if st.button("☀️ Light", key="light_theme", use_container_width=True, type="primary" if st.session_state.theme == "light" else "secondary"):
            if st.session_state.theme != "light":
                st.session_state.theme = "light"
                st.rerun()
    with theme_col2:
        if st.button("🌙 Dark", key="dark_theme", use_container_width=True, type="primary" if st.session_state.theme == "dark" else "secondary"):
            if st.session_state.theme != "dark":
                st.session_state.theme = "dark"
                st.rerun()

    st.sidebar.markdown("---")

    # Show agent status in sidebar
    if st.session_state.agent_manager:
        render_agent_status(st.session_state.agent_manager)

        # COMMENTED OUT: Model selector radio buttons - using agent cards instead
        # The main UI has agent card buttons which are more intuitive
        # selected_model = render_model_selector(st.session_state.active_chat_model)
        # if selected_model != st.session_state.active_chat_model:
        #     st.session_state.agent_manager.switch_agent(selected_model)
        #     st.session_state.active_chat_model = selected_model

        # # COMMENTED OUT: Knowledge Base Management (Only for RAG Agent - not in use)
        # if st.session_state.active_chat_model == "rag":
        #     st.sidebar.markdown("---")
        #     st.sidebar.markdown("### 📚 Knowledge Base")
        #
        #     if st.sidebar.button("🔄 Sync/Re-index Documents", help="Upload new files from data/clinical_docs to Qdrant"):
        #         with st.spinner("Indexing documents..."):
        #             agent = st.session_state.agent_manager.get_active_agent()
        #             if agent and hasattr(agent, "index_documents"):
        #                 status = agent.index_documents()
        #                 if "Error" in status:
        #                     st.sidebar.error(status)
        #                 else:
        #                     st.sidebar.success(status)
        #             else:
        #                 st.sidebar.error("Agent does not support indexing.")

    # Enhanced CSS with accessibility, responsive design, and theming
    st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

    # Check if models loaded
    if st.session_state.predictor is None:
        st.error("⚠️ Models failed to load. Please check model files.")
        return

    # Hero Header
    prediction_made = st.session_state.prediction_made

    if prediction_made:
        probability = st.session_state.prediction_prob
        risk_level, risk_badge_class, risk_message = classify_risk(probability)
        hero_metric = f"{probability:.0f}%"
        hero_badge_label = risk_level
    else:
        risk_level, risk_badge_class, risk_message = (
            "Pending",
            "risk-badge-neutral",
            "Complete the assessment to unlock your personalized guidance.",
        )
        hero_metric = "--"
        hero_badge_label = "Awaiting input"

    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-left">
                <span class="hero-pill">AI-Powered Multi-Agent System</span>
                <h1>Diabetes Risk Navigator</h1>
                <p class="hero-subtitle">
                    Understand your health indicators, benchmark against diabetic population data,
                    and receive personalized guidance from specialized AI healthcare agents.
                </p>
                <div class="hero-steps">
                    <div class="hero-step">1. Complete Assessment</div>
                    <div class="hero-step">2. Review Analysis</div>
                    <div class="hero-step">3. Chat with AI Agents</div>
                </div>
            </div>
            <div class="hero-highlight">
                <div class="hero-metric-label">Current Risk Estimate</div>
                <div class="hero-metric-value">{}</div>
                <div class="hero-metric-badge {}">{}</div>
                <p class="hero-highlight-note">{}</p>
            </div>
        </div>
        """.format(hero_metric, risk_badge_class, hero_badge_label, risk_message),
        unsafe_allow_html=True,
    )

    # Add reset button if prediction was made
    if st.session_state.prediction_made:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col3:
            if st.button("🔄 Start New Assessment", type="secondary", use_container_width=True):
                reset_app()
                st.rerun()

    st.markdown("")

    # ============ FLOW-BASED UI: Form → Analysis & Visualizations → AI Assistant ============

    if not st.session_state.prediction_made:
        # Progress Indicator - Step 1: Assessment
        st.markdown(get_progress_indicator_html(1), unsafe_allow_html=True)

        # ============ TABLEAU DASHBOARD: PRE-ANALYSIS INSIGHTS ============
        st.markdown("## Diabetes Patterns & Insights")
        st.markdown("""
        <div class="info-box">
            <p>
                📈 <strong>Explore population-level diabetes patterns</strong> before taking your personal assessment.
                This interactive dashboard shows key trends and risk factors from our training data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tableau Dashboard Embed
        tableau_html = """
        <div class='tableauPlaceholder' id='viz1738170000000' style='position: relative; margin: 20px 0;'>
            <noscript>
                <a href='#'>
                    <img alt='Dashboard' src='https://public.tableau.com/static/images/Di/DiabetesPatterns/Dashboard/1_rss.png' style='border: none' />
                </a>
            </noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='DiabetesPatterns/Dashboard' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/Di/DiabetesPatterns/Dashboard/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1738170000000');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
                vizElement.style.width='100%';
                vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            } else if (divElement.offsetWidth > 500) {
                vizElement.style.width='100%';
                vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            } else {
                vizElement.style.width='100%';
                vizElement.style.height='977px';
            }
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """

        # Use an expander to make the dashboard collapsible
        with st.expander("🔍 View Interactive Diabetes Patterns Dashboard", expanded=True):
            st.components.v1.html(tableau_html, height=900, scrolling=False)

        st.markdown("---")

        # ============ STEP 1: ASSESSMENT FORM ============
        st.markdown("## Health Risk Assessment")
        st.markdown("Complete the form below to assess your diabetes risk.")

        # Render prediction form
        user_data = render_prediction_form()

        if user_data:
            # Make prediction
            with st.spinner("Analyzing your health data..."):
                probability, risk_level, badge_class, guidance = st.session_state.predictor.predict(user_data)

                # Store results
                st.session_state.prediction_made = True
                st.session_state.user_data = user_data
                st.session_state.prediction_prob = probability

                # Set prediction context for agents
                if st.session_state.agent_manager:
                    st.session_state.agent_manager.set_prediction_context(probability, user_data)

                # Reset chatbot initialization to trigger auto-welcome for all agents
                st.session_state.chatbot_initialized = {}

                st.success("✅ Assessment complete! Scroll down to see your results and chat with AI agents.")
                st.rerun()
    else:
        # Progress Indicator - Step 3: Complete (Analysis & AI Guidance)
        st.markdown(get_progress_indicator_html(3), unsafe_allow_html=True)

        # ============ STEP 2 & 3: ANALYSIS + AI ASSISTANT (Side-by-side) ============
        risk_level, badge_class, guidance = classify_risk(st.session_state.prediction_prob)

        st.success("✅ Assessment complete! Your personalized analysis and AI assistant are ready.")

        with st.expander("Review your submitted profile", expanded=False):
            summary_df = pd.DataFrame([st.session_state.user_data])
            summary_df.rename(columns=FEATURE_NAMES, inplace=True)
            st.dataframe(summary_df)

        # Create two-column layout: Analysis (left) + Chatbot (right)
        results_col, assistant_col = st.columns([7, 5], gap="large")

        with results_col:
            # ============ ANALYSIS & VISUALIZATIONS ============
            st.markdown("## Your Risk Assessment Results")

            # Key Health Metrics Cards (3 columns)
            st.markdown("### Key Health Indicators")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                bmi_value = st.session_state.user_data.get('BMI', 0)
                bmi_delta = bmi_value - st.session_state.diabetic_averages.get('BMI', 28.0)
                st.metric(
                    label="Body Mass Index (BMI)",
                    value=f"{bmi_value:.1f}",
                    delta=f"{bmi_delta:+.1f} vs avg",
                    delta_color="inverse"
                )

            with metric_col2:
                age_value = st.session_state.user_data.get('Age', 0)
                st.metric(
                    label="Age Range",
                    value=AGE_LABELS.get(int(age_value), "N/A"),
                    delta=None
                )

            with metric_col3:
                phys_health = st.session_state.user_data.get('PhysHlth', 0)
                st.metric(
                    label="Physical Health (poor days/month)",
                    value=f"{int(phys_health)} days",
                    delta=None
                )

            with metric_col4:
                phys_activity = st.session_state.user_data.get('PhysActivity', 0)
                st.metric(
                    label="Physical Activity",
                    value="Active" if phys_activity == 1 else "Inactive",
                    delta="Good" if phys_activity == 1 else "Needs Improvement",
                    delta_color="normal" if phys_activity == 1 else "inverse"
                )

            # Risk Factors Summary
            st.markdown("### ⚠️ Risk Factors Present")
            risk_factors = []
            if st.session_state.user_data.get('HighBP', 0) == 1:
                risk_factors.append("❌ High Blood Pressure")
            if st.session_state.user_data.get('HighChol', 0) == 1:
                risk_factors.append("❌ High Cholesterol")
            if st.session_state.user_data.get('HeartDiseaseorAttack', 0) == 1:
                risk_factors.append("❌ Heart Disease/Attack History")
            if st.session_state.user_data.get('DiffWalk', 0) == 1:
                risk_factors.append("⚠️ Difficulty Walking")
            if st.session_state.user_data.get('PhysActivity', 0) == 0:
                risk_factors.append("⚠️ No Physical Activity")
            if st.session_state.user_data.get('BMI', 0) >= 30:
                risk_factors.append("⚠️ BMI ≥ 30 (Obese)")
            elif st.session_state.user_data.get('BMI', 0) >= 25:
                risk_factors.append("⚠️ BMI ≥ 25 (Overweight)")

            if risk_factors:
                risk_factor_cols = st.columns(min(len(risk_factors), 3))
                for idx, factor in enumerate(risk_factors[:6]):  # Limit to 6
                    with risk_factor_cols[idx % 3]:
                        st.info(factor)
            else:
                st.success("✅ No major risk factors detected!")

            st.markdown("")

            # Display risk gauge
            col1, col2 = st.columns([1, 1])
            with col1:
                fig_gauge = create_risk_gauge(st.session_state.prediction_prob)
                st.plotly_chart(fig_gauge, use_container_width=True, key="risk_gauge")

            with col2:
                st.markdown(f"### Risk Level: {risk_level}")
                st.markdown(guidance)

            # Enhanced Visualizations
            st.markdown("---")

            # Get feature importance from model
            feature_importance = st.session_state.predictor.get_feature_importance()

            # Feature Importance Chart (full width)
            st.markdown("### Understanding Your Risk Score")
            fig_importance = create_feature_importance_chart(
                feature_importance,
                st.session_state.user_data,
                st.session_state.diabetic_averages
            )
            st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False}, key="importance_chart")

            # Two charts side by side
            viz_col1, viz_col2 = st.columns(2, gap="large")

            with viz_col1:
                # Top Factors Comparison
                fig_comparison = create_top_factors_comparison(
                    st.session_state.user_data,
                    st.session_state.diabetic_averages,
                    feature_importance
                )
                st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False}, key="comparison_chart")

            with viz_col2:
                # Risk Simulator
                simulator_data = create_risk_simulator_data(
                    st.session_state.user_data,
                    st.session_state.predictor,
                    feature_importance
                )
                fig_simulator = create_risk_simulator_chart(simulator_data)
                st.plotly_chart(fig_simulator, use_container_width=True, config={'displayModeBar': False}, key="simulator_chart")

            # Enhanced Actionable Insights
            st.markdown("### 💡 Personalized Action Plan")
            st.markdown("""
            <div class="info-box">
                <p>
                    Based on your assessment, here are your most impactful next steps:
                </p>
            </div>
            """, unsafe_allow_html=True)

            insights = generate_actionable_insights(
                st.session_state.user_data,
                st.session_state.diabetic_averages,
                feature_importance,
                st.session_state.prediction_prob
            )

            for insight in insights:
                st.markdown(f"""
                <div class="action-card">
                    <div class="action-card-icon">{insight['icon']}</div>
                    <div class="action-card-title">{insight['finding']}</div>
                    <div class="action-card-impact">
                        <strong>⚡ Impact:</strong> {insight['impact']}
                    </div>
                    <div class="action-card-action">
                        {insight['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with assistant_col:
            # ============ AI HEALTH ASSISTANT WITH MULTI-AGENT SUPPORT ============
            if not st.session_state.agent_manager:
                st.error("⚠️ AI agents failed to initialize.")
            else:
                st.markdown("## Healthcare AI Agents")

                # Agent selector - IMPROVED with visual cards
                st.markdown("### Select Your AI Assistant")
                agent_options = st.session_state.agent_manager.get_available_agents()

                # Create visual agent cards
                cols = st.columns(len(agent_options))
                selected_agent = st.session_state.active_chat_model

                for idx, agent_key in enumerate(agent_options):
                    with cols[idx]:
                        agent_info = CHAT_MODEL_INFO[agent_key]
                        is_active = agent_key == st.session_state.active_chat_model
                        is_ready = st.session_state.agent_manager.is_agent_ready(agent_key)

                        # Agent card button
                        if st.button(
                            f"{agent_info['icon']} {agent_info['name']}",
                            key=f"agent_btn_{agent_key}",
                            type="primary" if is_active else "secondary",
                            use_container_width=True,
                            disabled=not is_ready
                        ):
                            if agent_key != st.session_state.active_chat_model:
                                st.session_state.agent_manager.switch_agent(agent_key)
                                st.session_state.active_chat_model = agent_key
                                # Rerun to update UI immediately with the new agent
                                st.rerun()

                        # Status indicator
                        status_text = "🟢 Ready" if is_ready else "⚠️ Loading"
                        st.caption(status_text)

                        # Show description for active agent
                        if is_active:
                            st.info(agent_info['description'], icon=agent_info['icon'])

                # Build context for agent
                risk_level_current, _, _ = classify_risk(st.session_state.prediction_prob)
                context = {
                    "probability": st.session_state.prediction_prob,
                    "risk_level": risk_level_current,
                    "profile_summary": build_profile_summary(st.session_state.user_data),
                    "user_data": st.session_state.user_data
                }

                # Auto-initialize ALL agents on first assessment completion
                # This ensures both Gemini and RAG agents have welcome messages ready
                all_agents = st.session_state.agent_manager.get_available_agents()

                for agent_key in all_agents:
                    # Only initialize if this specific agent hasn't been initialized yet
                    if not st.session_state.chatbot_initialized.get(agent_key, False):
                        # Check if conversation already exists (from agent switching)
                        existing_history = st.session_state.agent_manager.conversation_histories.get(agent_key, [])

                        if existing_history:
                            # Agent was already initialized, just mark it
                            st.session_state.chatbot_initialized[agent_key] = True
                        else:
                            # Generate new welcome message for this agent
                            if agent_key == CHAT_MODEL_GEMINI:
                                intro_message = f"""Please welcome me and provide a personalized overview of my diabetes risk assessment.

My risk probability is {st.session_state.prediction_prob:.1f}% ({risk_level_current} risk).

Review my health profile and provide:
1. A brief interpretation of my risk level
2. The top 3 most important factors contributing to my risk
3. Three specific actionable steps I should take this week

Keep it conversational, supportive, and personalized to MY specific numbers. Start with a greeting like "I've reviewed your diabetes risk assessment..."."""
                            else:  # Lightweight RAG Agent - IMPROVED clinical prompt
                                intro_message = f"""You are a clinical research assistant specializing in diabetes and metabolic health. Review this patient's diabetes risk assessment and provide evidence-based clinical insights.

**Patient Assessment:**
- Risk Probability: {st.session_state.prediction_prob:.1f}% ({risk_level_current} risk)
- Health Profile:
{context.get('profile_summary', 'Profile data not available')}

**Instructions:**
Please provide a comprehensive clinical overview that includes:

1. **Clinical Interpretation**: Analyze their risk level in the context of current diabetes screening guidelines (ADA, USPSTF). Reference specific thresholds or criteria.

2. **Evidence-Based Risk Factors**: Identify their top 3-4 risk factors and cite relevant research or clinical guidelines that explain why these factors matter.

3. **Clinical Recommendations**: Provide specific, actionable recommendations backed by:
   - Current clinical practice guidelines (ADA Standards of Care, etc.)
   - Relevant research studies or meta-analyses
   - Screening recommendations (HbA1c, fasting glucose, OGTT)

4. **Next Steps**: What should they discuss with their physician? What tests should they request? What monitoring is appropriate?

**Response Format:**
Start with a greeting, then structure your response with clear sections. Include citations to guidelines or studies when making clinical recommendations. Be specific and evidence-based throughout.

Begin your assessment now."""

                            # Only show spinner for the currently selected agent
                            if agent_key == selected_agent:
                                with st.spinner(f"🤖 {CHAT_MODEL_INFO[agent_key]['icon']} Preparing personalized insights..."):
                                    welcome_response = st.session_state.agent_manager.send_message(intro_message, agent_type=agent_key)
                            else:
                                # Initialize other agents silently in the background
                                welcome_response = st.session_state.agent_manager.send_message(intro_message, agent_type=agent_key)

                            # Remove the programmatic user message from history
                            if st.session_state.agent_manager.conversation_histories[agent_key]:
                                history = st.session_state.agent_manager.conversation_histories[agent_key]
                                if len(history) >= 2 and history[-2].get('role') == 'user':
                                    # Keep only the assistant's response
                                    st.session_state.agent_manager.conversation_histories[agent_key] = [history[-1]]

                            # Mark this agent as initialized
                            st.session_state.chatbot_initialized[agent_key] = True

                # Display conversation history
                st.markdown("### 💬 Chat")
                conversation = st.session_state.agent_manager.get_conversation_history()

                # Create scrollable chat container
                chat_container = st.container(height=500)
                with chat_container:
                    if conversation:
                        for msg in conversation:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")

                            if role == "user":
                                with st.chat_message("user"):
                                    st.markdown(content)
                            else:
                                agent_icon = CHAT_MODEL_INFO[st.session_state.active_chat_model]['icon']
                                with st.chat_message("assistant", avatar=agent_icon):
                                    st.markdown(content)

                # Chat input
                st.markdown("""
                <div class="info-box">
                    <p>
                        💡 <strong>Try asking:</strong><br>
                        • What lifestyle changes would have the biggest impact?<br>
                        • Can you explain my risk factors?<br>
                        • What should I discuss with my doctor?
                    </p>
                </div>
                """, unsafe_allow_html=True)

                user_message = st.chat_input("Ask about your health assessment...")
                if user_message:
                    # Send message to agent
                    with st.spinner("Thinking..."):
                        response = st.session_state.agent_manager.send_message(user_message)
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9em;">
        <p>Multi-Agent AI System: Gemini Agent (Generic Insights) + Lightweight RAG Agent (Clinical Insights)</p>
        <p>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
