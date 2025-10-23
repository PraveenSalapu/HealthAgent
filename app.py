"""
StreamLit Diabetes Predictor + Gemini Health Chatbot
=====================================================
A comprehensive health prediction application with AI-powered assistance.

Requirements:
    pip install streamlit pandas numpy plotly google-generativeai joblib xgboost imbalanced-learn

Environment Variables:
    GEMINI_API_KEY: Your Google Gemini API key
    GEMINI_MODEL: Model name (default: gemini-pro)

Model Files Required:
    - model_output/ml_xgboost_smoteenn_pipeline.pkl (your trained model)
    - diabetic_averages.json (feature averages for diabetic population)

Important Notes:
    - sklearn version warnings are SAFE to ignore - your pipeline will work correctly
    - The model was trained with sklearn 1.5.1, but works with 1.7.2
    - Your ImbPipeline handles all preprocessing automatically
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
import plotly.graph_objects as go
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
import joblib
import sklearn

# Suppress sklearn version warnings for better UX
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (adjust as needed)
MODEL_PATH = "model_output/ml_xgboost_smoteenn_pipeline.pkl"
THRESHOLD_PATH = "model_output/optimal_threshold.json"
AVERAGES_PATH = "model_output/diabetic_averages.json"

# PREPROCESSING NOTES:
# ====================
# Your model is an imblearn Pipeline with:
#   1. ColumnTransformer (scales numeric: BMI, GenHlth, PhysHlth, Age, Education, Income)
#   2. SMOTEENN sampler (only active during training, skipped during prediction)
#   3. XGBoost classifier
#
# The app passes RAW user input → Pipeline handles scaling automatically.
# Feature order MUST match training: GenHlth, HighBP, DiffWalk, BMI, HighChol,
#                                     Age, HeartDiseaseorAttack, PhysHlth, Income,
#                                     Education, PhysActivity

# Feature definitions - MUST MATCH TRAINING ORDER
FEATURE_CONFIGS = {
    "GenHlth": {"type": "select", "options": [1, 2, 3, 4, 5], 
                "labels": ["Excellent", "Very Good", "Good", "Fair", "Poor"]},
    "HighBP": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "DiffWalk": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "BMI": {"type": "number", "min": 10.0, "max": 70.0, "default": 25.0, "step": 0.1},
    "HighChol": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "Age": {"type": "number", "min": 1, "max": 13, "default": 7, "step": 1, 
            "help": "Age category (1=18-24, 2=25-29, ..., 13=80+)"},
    "HeartDiseaseorAttack": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "PhysHlth": {"type": "number", "min": 0, "max": 30, "default": 0, "step": 1,
                 "help": "Days of poor physical health in past 30 days"},
    "Income": {"type": "number", "min": 1, "max": 8, "default": 4, "step": 1,
               "help": "Income category (1=<$10k, 8=>$75k)"},
    "Education": {"type": "number", "min": 1, "max": 6, "default": 4, "step": 1,
                  "help": "Education level (1=Never, 6=College grad)"},
    "PhysActivity": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]}
}

# Feature display names
FEATURE_NAMES = {
    "GenHlth": "General Health",
    "HighBP": "High Blood Pressure",
    "DiffWalk": "Difficulty Walking",
    "BMI": "Body Mass Index",
    "HighChol": "High Cholesterol",
    "Age": "Age Category",
    "HeartDiseaseorAttack": "Heart Disease or Attack History",
    "PhysHlth": "Physical Health (poor days/month)",
    "Income": "Income Level",
    "Education": "Education Level",
    "PhysActivity": "Physical Activity (last 30 days)"
}

# Default averages (fallback if file not found or missing features)
# Based on BRFSS 2015 diabetic population averages
DEFAULT_DIABETIC_AVERAGES = {
    "GenHlth": 3.29,
    "HighBP": 0.75,
    "DiffWalk": 0.37,
    "BMI": 31.94,
    "HighChol": 0.67,
    "Age": 9.38,
    "HeartDiseaseorAttack": 0.22,
    "PhysHlth": 7.95,
    "Income": 5.21,
    "Education": 4.75,
    "PhysActivity": 0.63
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path: str):
    """Load the trained model from disk with version compatibility checks."""
    try:
        # Suppress sklearn warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try joblib first, then pickle
            try:
                model = joblib.load(model_path)
            except:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        
        # Debug: Show what was loaded
        st.info(f"🔍 Loaded object type: {type(model)}")
        st.info(f"🔍 Object has predict method: {hasattr(model, 'predict')}")
        
        # Check if it's a dict or other container
        if isinstance(model, dict):
            st.warning("⚠️ Loaded a dictionary. Showing keys:")
            st.code(str(list(model.keys())))
            # Try to find the pipeline in the dict
            if 'pipeline' in model:
                model = model['pipeline']
                st.info("✅ Found 'pipeline' key, extracting it")
            elif 'model' in model:
                model = model['model']
                st.info("✅ Found 'model' key, extracting it")
            else:
                st.error("❌ Could not find pipeline in dictionary")
                return None
        
        # Check model type
        from sklearn.pipeline import Pipeline
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            is_imblearn_pipeline = isinstance(model, ImbPipeline)
        except:
            is_imblearn_pipeline = False
        
        is_sklearn_pipeline = isinstance(model, Pipeline)
        is_pipeline = is_sklearn_pipeline or is_imblearn_pipeline
        
        # Verify it has predict method
        if not hasattr(model, 'predict'):
            st.error(f"❌ Loaded object is {type(model)} but has no 'predict' method")
            st.error("This might not be a trained model. Check how the model was saved.")
            return None
        
        # Verify model can make predictions
        try:
            # Test prediction with dummy data in CORRECT ORDER
            test_features = list(FEATURE_CONFIGS.keys())
            test_data = pd.DataFrame([{f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
                                       else FEATURE_CONFIGS[f]["default"] 
                                       for f in test_features}])
            
            st.info(f"🧪 Testing prediction with data shape: {test_data.shape}")
            _ = model.predict(test_data)
            
            pipeline_type = "imblearn Pipeline" if is_imblearn_pipeline else "sklearn Pipeline" if is_sklearn_pipeline else "Bare estimator"
            st.success(f"✅ Model loaded successfully (sklearn v{sklearn.__version__})")
            st.info(f"📊 Model type: {pipeline_type}")
            
            # Show pipeline steps if applicable
            if is_pipeline:
                with st.expander("🔍 View Pipeline Steps", expanded=False):
                    steps_info = "\n".join([f"{i+1}. {name}: {type(step).__name__}" 
                                           for i, (name, step) in enumerate(model.steps)])
                    st.code(steps_info)
                    if is_imblearn_pipeline:
                        st.info("ℹ️ SMOTEENN sampler is only active during training, not prediction")
            
            return model
        except Exception as pred_error:
            st.error(f"⚠️ Model loaded but prediction test failed: {pred_error}")
            st.error(f"Error type: {type(pred_error).__name__}")
            import traceback
            st.code(traceback.format_exc())
            st.warning("The model may be incompatible with the current scikit-learn version or feature order is wrong.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model from {model_path}: {e}")
        st.info("💡 Tip: Ensure the model file exists and was created with a compatible scikit-learn version.")
        import traceback
        st.code(traceback.format_exc())
        return None

def load_diabetic_averages(averages_path: str) -> Dict[str, float]:
    """Load average feature values for diabetic population."""
    try:
        with open(averages_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats:
        # Format 1: Simple dict {"feature": value, ...}
        # Format 2: Array of objects [{"feature": "name", "average_value": value}, ...]
        
        if isinstance(data, list):
            # Array format - convert to dict
            averages = {}
            for item in data:
                feature_name = item.get("feature")
                avg_value = item.get("average_value")
                if feature_name and avg_value is not None:
                    averages[feature_name] = float(avg_value)
            
            # Log what we found
            found_features = set(averages.keys())
            required_features = set(FEATURE_CONFIGS.keys())
            missing = required_features - found_features
            
            if missing:
                st.warning(f"⚠️ Missing features in averages file: {missing}. Using defaults for missing values.")
                # Fill in missing with defaults
                for feature in missing:
                    averages[feature] = DEFAULT_DIABETIC_AVERAGES.get(feature, 0)
            
            return averages
        else:
            # Simple dict format
            return data
            
    except FileNotFoundError:
        st.warning(f"Averages file not found at {averages_path}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES
    except Exception as e:
        st.warning(f"Error loading averages: {e}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES

def sanitize_input(value, feature_name: str) -> float:
    """Sanitize and validate user input."""
    config = FEATURE_CONFIGS[feature_name]
    
    if config["type"] == "number":
        try:
            val = float(value)
            return max(config["min"], min(config["max"], val))
        except:
            return config["default"]
    else:  # select
        return int(value) if value in config["options"] else config["options"][0]

def initialize_gemini(api_key: str, model_name: str = "gemini-pro"):
    """Initialize Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        return None

def get_system_prompt(prediction_prob: float, user_data: Dict[str, float]) -> str:
    """Generate system prompt with prediction context."""
    return f"""You are a compassionate healthcare assistant helping users understand their diabetes risk assessment results. 

CONTEXT:
- The user has completed a diabetes risk assessment
- Model prediction probability: {prediction_prob:.1f}%
- User's health metrics: {json.dumps(user_data, indent=2)}

YOUR ROLE:
- Provide empathetic, non-alarmist guidance based on the risk assessment
- Suggest lifestyle modifications (diet, exercise, sleep, stress management)
- Recommend appropriate follow-up actions
- Answer health-related questions in accessible language
- NEVER provide definitive medical diagnoses or prescribe treatments

CRITICAL SAFETY RULES:
1. Always clarify this is a risk estimation tool, not a diagnosis
2. Use language like "the model estimates" or "based on these factors"
3. Include this disclaimer in your first message: "This assessment is for informational purposes only. Please consult a healthcare provider for proper diagnosis and treatment."
4. Encourage professional medical consultation for any health concerns
5. Avoid creating alarm - focus on actionable, positive steps
6. If asked about medications or treatments, defer to healthcare providers

CONVERSATION STYLE:
- Warm and supportive, but scientifically accurate
- Use simple language, avoid excessive medical jargon
- Provide specific, actionable suggestions
- Ask clarifying questions when helpful
- Acknowledge emotions and concerns

Begin by providing a gentle, contextual interpretation of their {prediction_prob:.1f}% risk probability and offer to answer any questions."""

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_chart(user_data: Dict[str, float], 
                           avg_data: Dict[str, float]) -> go.Figure:
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
        marker_color='#FF6B6B',
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside'
    ))
    
    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Avg. Diabetic Population',
        x=features,
        y=avg_values,
        marker_color='#4ECDC4',
        text=[f'{v:.1f}' for v in avg_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Your Health Metrics vs. Average Diabetic Population',
        xaxis_title='Health Factors',
        yaxis_title='Value',
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig, differences

def create_risk_gauge(probability: float) -> go.Figure:
    """Create risk level gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Probability", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
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

# ============================================================================
# CHATBOT FUNCTIONS
# ============================================================================

def send_message_to_gemini(model, chat_history: List[Dict[str, str]], 
                          user_message: str) -> str:
    """Send message to Gemini and get response."""
    try:
        # Build conversation history
        chat = model.start_chat(history=[])
        
        # Send system prompt as first message if this is the start
        if len(chat_history) == 0:
            system_msg = st.session_state.get('system_prompt', '')
            if system_msg:
                chat.send_message(system_msg)
        
        # Send previous messages
        for msg in chat_history:
            if msg['role'] == 'user':
                chat.send_message(msg['content'])
            # Assistant messages are already in history
        
        # Send current message
        response = chat.send_message(user_message)
        return response.text
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'prediction_prob' not in st.session_state:
        st.session_state.prediction_prob = 0.0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = ""

def reset_app():
    """Reset application state."""
    st.session_state.prediction_made = False
    st.session_state.user_data = {}
    st.session_state.prediction_prob = 0.0
    st.session_state.chat_history = []
    st.session_state.chatbot_initialized = False
    st.session_state.system_prompt = ""

def main():
    st.set_page_config(
        page_title="Diabetes Risk Assessment + AI Health Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🏥 Diabetes Risk Assessment + AI Health Assistant")
    
    # Show environment info in expander
    with st.expander("ℹ️ System Information", expanded=False):
        st.code(f"""
Python: {os.sys.version.split()[0]}
Scikit-learn: {sklearn.__version__}
Streamlit: {st.__version__}
Model Path: {MODEL_PATH}
Averages Path: {AVERAGES_PATH}
        """)
    
    st.markdown("---")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY environment variable not set. Please set it to use the chatbot feature.")
        st.stop()
    
    # Load model and averages
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"⚠️ Could not load model from {MODEL_PATH}. Please check the file path.")
        st.stop()
    
    diabetic_averages = load_diabetic_averages(AVERAGES_PATH)
    
    # Initialize Gemini
    gemini_model = initialize_gemini(api_key, model_name)
    if gemini_model is None:
        st.error("⚠️ Could not initialize Gemini API. Please check your API key.")
        st.stop()
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("🔄 New Assessment", use_container_width=True):
            reset_app()
            st.rerun()
    
    # Main layout
    left_col, right_col = st.columns([1, 1])
    
    # LEFT COLUMN
    with left_col:
        if not st.session_state.prediction_made:
            # FORM VIEW
            st.subheader("📋 Health Assessment Form")
            st.markdown("Please provide your health information below:")
            
            with st.form("health_form"):
                user_inputs = {}
                
                for feature, config in FEATURE_CONFIGS.items():
                    st.markdown(f"**{FEATURE_NAMES[feature]}**")
                    
                    if config["type"] == "number":
                        help_text = config.get("help", "")
                        user_inputs[feature] = st.number_input(
                            f"Enter {FEATURE_NAMES[feature]}",
                            min_value=config["min"],
                            max_value=config["max"],
                            value=config["default"],
                            step=config["step"],
                            help=help_text if help_text else None,
                            key=f"input_{feature}",
                            label_visibility="collapsed"
                        )
                    else:  # select
                        idx = st.selectbox(
                            f"Select {FEATURE_NAMES[feature]}",
                            range(len(config["labels"])),
                            format_func=lambda i: config["labels"][i],
                            key=f"input_{feature}",
                            label_visibility="collapsed"
                        )
                        user_inputs[feature] = config["options"][idx]
                
                submitted = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)
                
                if submitted:
                    # Sanitize inputs
                    clean_data = {k: sanitize_input(v, k) for k, v in user_inputs.items()}
                    
                    # Create dataframe for prediction
                    # IMPORTANT: Maintain the exact feature order expected by the model
                    feature_order = list(FEATURE_CONFIGS.keys())
                    input_df = pd.DataFrame([clean_data])[feature_order]
                    
                    # Debug: Show what's being sent to model
                    with st.expander("🔬 Debug: Input Data", expanded=False):
                        st.write("**Raw User Input (before model):**")
                        st.dataframe(input_df)
                        st.write("**Data types:**")
                        st.write(input_df.dtypes)
                    
                    # Make prediction
                    try:
                        # If model is a Pipeline, it will handle scaling automatically
                        # If model is bare estimator, user must ensure data is on correct scale
                        prediction = model.predict(input_df)[0]
                        prediction_proba = model.predict_proba(input_df)[0]
                        probability = prediction_proba[1] * 100  # Probability of diabetes
                        
                        # Debug: Show model output
                        with st.expander("🔬 Debug: Model Output", expanded=False):
                            st.write(f"**Prediction:** {prediction}")
                            st.write(f"**Probabilities:** {prediction_proba}")
                            st.write(f"**Diabetes Probability:** {probability:.2f}%")
                        
                        # Store in session state
                        st.session_state.user_data = clean_data
                        st.session_state.prediction_prob = probability
                        st.session_state.prediction_made = True
                        
                        # Generate system prompt for chatbot
                        st.session_state.system_prompt = get_system_prompt(probability, clean_data)
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {e}")
                        st.warning("This might be a data preprocessing issue. Check that your model expects the correct feature format.")
        
        else:
            # VISUALIZATION VIEW
            st.subheader("📊 Your Risk Assessment Results")
            
            # Risk gauge
            gauge_fig = create_risk_gauge(st.session_state.prediction_prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Risk interpretation
            prob = st.session_state.prediction_prob
            if prob < 30:
                risk_level = "Low"
                risk_color = "green"
                risk_message = "Your risk factors suggest a lower probability of diabetes."
            elif prob < 60:
                risk_level = "Moderate"
                risk_color = "orange"
                risk_message = "Your risk factors suggest moderate attention to lifestyle factors may be beneficial."
            else:
                risk_level = "Elevated"
                risk_color = "red"
                risk_message = "Your risk factors suggest consulting with a healthcare provider is recommended."
            
            st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            st.info(risk_message)
            
            # Comparison chart
            st.markdown("---")
            st.subheader("📈 Comparison with Diabetic Population")
            comparison_fig, differences = create_comparison_chart(
                st.session_state.user_data, 
                diabetic_averages
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            insights = generate_insights(
                st.session_state.user_data, 
                diabetic_averages, 
                differences
            )
            
            for insight in insights:
                st.markdown(f"• {insight}")
    
    # RIGHT COLUMN - CHATBOT
    with right_col:
        st.subheader("💬 AI Health Assistant")
        
        if not st.session_state.prediction_made:
            st.info("👈 Complete the health assessment form to start a conversation with your AI health assistant.")
        else:
            # Initialize chatbot with first message
            if not st.session_state.chatbot_initialized:
                initial_response = send_message_to_gemini(
                    gemini_model, 
                    [], 
                    st.session_state.system_prompt
                )
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': initial_response
                })
                st.session_state.chatbot_initialized = True
            
            # Chat container
            chat_container = st.container(height=500)
            
            with chat_container:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        with st.chat_message("user"):
                            st.markdown(message['content'])
                    else:
                        with st.chat_message("assistant", avatar="🏥"):
                            st.markdown(message['content'])
            
            # Chat input
            user_message = st.chat_input("Ask me anything about your results or health...")
            
            if user_message:
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_message
                })
                
                # Get assistant response
                with st.spinner("Thinking..."):
                    assistant_response = send_message_to_gemini(
                        gemini_model,
                        st.session_state.chat_history[:-1],  # Exclude the message we just added
                        user_message
                    )
                
                # Add assistant response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': assistant_response
                })
                
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool is for informational purposes only and does not constitute medical advice. "
        "Always consult with a qualified healthcare provider for proper diagnosis and treatment."
    )

if __name__ == "__main__":
    main()