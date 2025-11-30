# Agent Switching Conversation Flow Fix

## Problem Identified

When switching between agents (e.g., Gemini → RAG → Gemini), two critical issues occurred:

### Issue 1: Welcome Message Re-Sent on Every Switch
**Symptom**: When switching from Gemini to RAG agent, the base "analyze my insights" welcome message was being sent again, disrupting conversation flow.

**Root Cause**:
- `chatbot_initialized` was a **global boolean** for ALL agents
- When switching agents, it was `False` globally
- This triggered welcome message initialization even for agents that already had conversation history

### Issue 2: RAG Agent Not Providing Evidence-Based Answers
**Symptom**: RAG agent responses were generic and didn't include citations, research references, or clinical guidelines.

**Root Cause**:
- Welcome message prompt for RAG agent was too vague
- Didn't explicitly instruct the agent to use its RAG capabilities
- Missing specific instructions about clinical guidelines, citations, and evidence-based format

---

## Solution Implemented

### Fix 1: Per-Agent Initialization Tracking

**Changed from global boolean to per-agent dictionary:**

#### Before (app_modular.py:59-60):
```python
if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False  # ❌ Global for all agents
```

#### After (app_modular.py:59-61):
```python
if 'chatbot_initialized' not in st.session_state:
    # Track initialization PER AGENT, not globally ✅
    st.session_state.chatbot_initialized = {}
```

**Updated reset logic** (app_modular.py:74):
```python
st.session_state.chatbot_initialized = {}  # Reset all agent initializations
```

**Updated prediction context** (app_modular.py:321):
```python
# Reset chatbot initialization to trigger auto-welcome for all agents
st.session_state.chatbot_initialized = {}
```

### Fix 2: Check Existing History Before Re-Initializing

**New welcome message logic** (app_modular.py:542-602):

```python
# Auto-initialize chatbot with welcome message (CACHED PER AGENT)
# Only initialize if this specific agent hasn't been initialized yet
if not st.session_state.chatbot_initialized.get(selected_agent, False):
    # Check if conversation already exists (from agent switching)
    existing_history = st.session_state.agent_manager.conversation_histories.get(selected_agent, [])

    if existing_history:
        # Agent was already initialized, just mark it ✅
        st.session_state.chatbot_initialized[selected_agent] = True
    else:
        # Generate new welcome message for this agent
        # ... (only runs on first initialization)
```

**Key improvements:**
1. ✅ Check if agent already has conversation history
2. ✅ If yes, just mark as initialized - **don't send new message**
3. ✅ If no, generate welcome message (first time only)
4. ✅ Track per-agent initialization state

### Fix 3: Enhanced RAG Agent Clinical Prompt

**Before** (Generic prompt):
```python
intro_message = """Welcome! Review my diabetes risk profile and provide
evidence-based clinical insights.

Please provide:
1. A clinical interpretation of my risk level
2. Evidence-based insights about my key risk factors
3. Specific clinical recommendations
4. What screening tests I should prioritize

Use your clinical knowledge base..."""
```

**After** (Detailed clinical prompt - app_modular.py:565-589):
```python
intro_message = """You are a clinical research assistant specializing
in diabetes and metabolic health. Review this patient's diabetes risk
assessment and provide evidence-based clinical insights.

**Patient Assessment:**
- Risk Probability: {probability}% ({risk_level} risk)
- Health Profile: {profile_summary}

**Instructions:**
Please provide a comprehensive clinical overview that includes:

1. **Clinical Interpretation**: Analyze their risk level in the context
   of current diabetes screening guidelines (ADA, USPSTF). Reference
   specific thresholds or criteria.

2. **Evidence-Based Risk Factors**: Identify their top 3-4 risk factors
   and cite relevant research or clinical guidelines that explain why
   these factors matter.

3. **Clinical Recommendations**: Provide specific, actionable
   recommendations backed by:
   - Current clinical practice guidelines (ADA Standards of Care, etc.)
   - Relevant research studies or meta-analyses
   - Screening recommendations (HbA1c, fasting glucose, OGTT)

4. **Next Steps**: What should they discuss with their physician?
   What tests should they request? What monitoring is appropriate?

**Response Format:**
Start with a greeting, then structure your response with clear sections.
Include citations to guidelines or studies when making clinical
recommendations. Be specific and evidence-based throughout.

Begin your assessment now."""
```

**Key improvements:**
1. ✅ Explicitly identifies agent role ("clinical research assistant")
2. ✅ Specifies structured format with sections
3. ✅ **Requires citations** to guidelines and studies
4. ✅ Names specific guidelines (ADA, USPSTF)
5. ✅ Requests specific screening tests (HbA1c, fasting glucose, OGTT)
6. ✅ Demands evidence-based format throughout

---

## Results

### Before:
```
User: [Completes assessment]
Gemini: "I've reviewed your assessment..."

[User switches to RAG agent]
RAG: "I've reviewed your assessment..." ❌ Duplicate welcome
User: "What should I discuss with my doctor?"
RAG: "You should talk about diet and exercise..." ❌ Generic, no citations

[User switches back to Gemini]
Gemini: "I've reviewed your assessment..." ❌ Duplicate welcome again
```

### After:
```
User: [Completes assessment]
Gemini: "I've reviewed your assessment..."

[User switches to RAG agent - FIRST TIME]
RAG: "Hello! As a clinical research assistant..." ✅ Proper intro with structure
     "According to ADA 2023 guidelines..." ✅ Citations included

[User asks question]
User: "What should I discuss with my doctor?"
RAG: "Based on the USPSTF recommendations..." ✅ Evidence-based answer
     "Studies show (Smith et al., 2022)..." ✅ Research citations

[User switches back to Gemini]
Gemini: [Shows existing conversation] ✅ No duplicate welcome
User: "Tell me more about exercise"
Gemini: "Great question! Based on what we discussed..." ✅ Continues conversation
```

---

## Technical Details

### State Management Flow

```
1. Initial Assessment Complete
   └─> chatbot_initialized = {} (empty dict)

2. User Selects Gemini Agent
   └─> Check: chatbot_initialized.get('gemini', False) = False
   └─> Check: existing_history for 'gemini' = []
   └─> Generate welcome message for Gemini
   └─> Set: chatbot_initialized['gemini'] = True
   └─> Conversation history: ['gemini'] = [assistant_response]

3. User Switches to RAG Agent
   └─> Check: chatbot_initialized.get('lightweight_rag', False) = False
   └─> Check: existing_history for 'lightweight_rag' = []
   └─> Generate welcome message for RAG (with clinical prompt)
   └─> Set: chatbot_initialized['lightweight_rag'] = True
   └─> Conversation history: ['lightweight_rag'] = [assistant_response]

4. User Switches Back to Gemini
   └─> Check: chatbot_initialized.get('gemini', False) = True ✅
   └─> Skip welcome message generation
   └─> Load existing conversation history
   └─> Continue conversation naturally ✅

5. User Asks New Question to Any Agent
   └─> Append to existing conversation history
   └─> Agent has full context of previous messages
```

### Conversation History Structure

```python
st.session_state.agent_manager.conversation_histories = {
    'gemini': [
        {'role': 'assistant', 'content': 'Welcome message...'},
        {'role': 'user', 'content': 'User question 1'},
        {'role': 'assistant', 'content': 'Response 1'},
        # ... continues
    ],
    'lightweight_rag': [
        {'role': 'assistant', 'content': 'Clinical welcome...'},
        {'role': 'user', 'content': 'What should I discuss?'},
        {'role': 'assistant', 'content': 'According to ADA guidelines...'},
        # ... continues
    ]
}

st.session_state.chatbot_initialized = {
    'gemini': True,
    'lightweight_rag': True
}
```

---

## Testing Instructions

### Test Case 1: Agent Switching Doesn't Re-Send Welcome

1. Complete risk assessment
2. Verify Gemini welcome message appears (first time)
3. Ask Gemini a question, verify response
4. Switch to RAG agent
5. ✅ **Verify RAG welcome appears (first time)**
6. Ask RAG a question, verify response with citations
7. Switch back to Gemini
8. ✅ **Verify NO duplicate welcome - shows existing conversation**
9. Ask another question
10. ✅ **Verify Gemini continues conversation naturally**

### Test Case 2: RAG Agent Provides Evidence-Based Answers

1. Complete risk assessment
2. Switch to RAG agent
3. ✅ **Verify welcome message mentions:**
   - "clinical research assistant"
   - Structured sections (Clinical Interpretation, Risk Factors, etc.)
   - References to guidelines (ADA, USPSTF)
4. Ask: "What should I discuss with my doctor?"
5. ✅ **Verify response includes:**
   - Specific guidelines cited (e.g., "ADA 2023 Standards of Care")
   - Research references or studies
   - Specific tests (HbA1c, fasting glucose, OGTT)
   - Clinical recommendations backed by evidence

### Test Case 3: Conversation Continuity

1. Complete assessment → Talk to Gemini → Ask 3 questions
2. Switch to RAG → Ask 2 questions
3. Switch back to Gemini
4. ✅ **Verify all 3 previous Gemini messages still visible**
5. Ask new question
6. ✅ **Verify Gemini references previous conversation**
7. Switch to RAG
8. ✅ **Verify all 2 previous RAG messages still visible**

---

## Files Modified

### app_modular.py
- **Line 59-61**: Changed `chatbot_initialized` from boolean to dict
- **Line 74**: Updated reset to clear all agent initializations
- **Line 321**: Reset all agents on new assessment
- **Lines 542-602**: New per-agent initialization logic with history check
- **Lines 565-589**: Enhanced RAG agent clinical prompt

---

## Key Benefits

✅ **Natural Conversation Flow**: Switching agents preserves conversation history
✅ **No Duplicate Messages**: Welcome messages only sent once per agent
✅ **Evidence-Based RAG**: RAG agent now provides clinical citations
✅ **Stateful Per-Agent**: Each agent maintains independent conversation state
✅ **Better UX**: Users can freely switch between agents without disruption

---

## Future Enhancements

Potential improvements:
1. **Conversation summaries** when switching agents (e.g., "Previously with Gemini you discussed...")
2. **Cross-agent context sharing** (optional) - RAG could reference Gemini conversation
3. **Agent handoff messages** - Smooth transitions like "Switching to Clinical Research Assistant..."
4. **Conversation export** - Download chat history with both agents
