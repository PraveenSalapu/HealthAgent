# Streamlit Cloud Deployment Guide
## Health AI Chatbot - Multi-Agent System

This guide will walk you through deploying your Health AI Chatbot to Streamlit Cloud.

---

## ✅ Prerequisites Checklist

Before deploying, ensure you have:

- [x] GitHub repository: `https://github.com/PraveenSalapu/HealthAgentDiabetic.git`
- [x] `requirements.txt` with all dependencies
- [x] Main application file: `app_modular.py`
- [x] Model files in `model_output2/` directory
- [ ] Gemini API key (get one from https://makersuite.google.com/app/apikey)
- [ ] Streamlit Cloud account (sign up at https://share.streamlit.io)

---

## 📋 Step 1: Verify Your Repository

Make sure all required files are committed and pushed to GitHub:

```bash
# Check git status
git status

# If you have uncommitted changes, commit them
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Required Files/Directories:
- ✅ `app_modular.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `config/` - Configuration module
- ✅ `agents/` - Multi-agent system
- ✅ `models/` - ML model loading
- ✅ `ui/` - UI components
- ✅ `utils/` - Utilities
- ✅ `model_output2/` - Trained model files
- ✅ `data/` - Data directory for RAG agent

---

## 🚀 Step 2: Create Streamlit Cloud Account

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub repositories
4. You may be asked to authorize access to specific repositories

---

## 📦 Step 3: Deploy Your Application

### 3.1 Create New App

1. Click the **"New app"** button in Streamlit Cloud dashboard
2. Fill in the deployment form:

   **Repository:** `PraveenSalapu/HealthAgentDiabetic`

   **Branch:** `main`

   **Main file path:** `app_modular.py`

   **App URL (optional):** Choose a custom URL like `health-ai-chatbot`

### 3.2 Advanced Settings

Click **"Advanced settings"** to configure:

**Python version:** `3.11` (recommended) or `3.10`

---

## 🔐 Step 4: Configure Secrets (CRITICAL!)

This is the **most important step** for your Gemini agent to work!

### In Streamlit Cloud Deployment Settings:

1. In the **"Advanced settings"** section, find **"Secrets"**
2. Add your secrets in **TOML format**:

```toml
# Google Gemini API Key
GEMINI_API_KEY = "your_actual_gemini_api_key_here"

# Optional: Override default model
# GEMINI_MODEL = "gemini-2.5-flash"
```

3. Replace `your_actual_gemini_api_key_here` with your **actual API key**

### Getting a Gemini API Key:

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key and paste it into the secrets configuration above

---

## 🎯 Step 5: Deploy!

1. Click **"Deploy!"** button
2. Wait for the build process (typically 3-5 minutes)
3. Watch the deployment logs for any errors

---

## ✅ Step 6: Verify Deployment

Once deployed, test your app:

### Check These Features:
- [ ] App loads without errors
- [ ] Model files load successfully
- [ ] Health assessment form works
- [ ] Prediction generates results
- [ ] Gemini Agent responds to questions
- [ ] RAG Agent initializes (will create sample knowledge base)
- [ ] All visualizations render correctly
- [ ] Agent switching works

### Test Questions:
- "What lifestyle changes would help reduce my risk?"
- "Can you explain my risk factors?"
- "What should I discuss with my doctor?"

---

## 🔧 Troubleshooting Common Issues

### Issue 1: "GEMINI_API_KEY not found"

**Cause:** Secrets not configured properly

**Solution:**
1. Go to your app settings in Streamlit Cloud
2. Click "⋮" menu → "Settings"
3. Go to "Secrets" section
4. Verify your `GEMINI_API_KEY` is set correctly
5. Redeploy the app

### Issue 2: Module Import Errors

**Cause:** Missing dependencies in requirements.txt

**Solution:**
1. Check deployment logs for the missing module
2. Add it to `requirements.txt`
3. Commit and push changes
4. Streamlit Cloud will auto-redeploy

### Issue 3: Model Files Not Found

**Cause:** Model files not in repository or wrong path

**Solution:**
1. Verify `model_output2/` directory exists in your repo
2. Check these files exist:
   - `xgboost_model.json`
   - `preprocessor.pkl`
   - `optimal_threshold.json`
   - `diabetic_averages.json`
3. Ensure they're not in `.gitignore`

### Issue 4: RAG Agent Initialization Errors

**Cause:** ChromaDB or sentence-transformers installation issues

**Solution:**
- The RAG agent will automatically create a sample knowledge base if it can't load external documents
- Check deployment logs for specific errors
- Verify all RAG dependencies are in `requirements.txt`:
  ```
  langchain
  chromadb
  sentence-transformers
  ```

### Issue 5: App Crashes or Runs Slowly

**Cause:** Memory limits or large dependencies

**Solution:**
- Streamlit Cloud free tier has memory limits (~1GB)
- Consider using smaller embedding models
- Check if you can optimize model file sizes
- Monitor resource usage in app settings

---

## 🔄 Updating Your Deployed App

Streamlit Cloud automatically redeploys when you push to your GitHub repository:

```bash
# Make your changes
git add .
git commit -m "Update: description of changes"
git push origin main

# Streamlit Cloud will automatically detect and redeploy
```

You can also manually trigger a reboot from the Streamlit Cloud dashboard.

---

## 📊 Monitoring Your App

### View Logs:
1. Go to your app in Streamlit Cloud
2. Click "⋮" menu → "Logs"
3. Monitor real-time logs for errors or warnings

### Analytics:
- Streamlit Cloud provides basic analytics
- View app uptime, visitor count, and resource usage

---

## 🧪 Local Testing with Streamlit Secrets

To test locally with the same configuration as Streamlit Cloud:

1. Copy the secrets template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml` with your API key:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```

3. Run locally:
   ```bash
   streamlit run app_modular.py
   ```

**Note:** `.streamlit/secrets.toml` is in `.gitignore` and will NOT be committed to GitHub.

---

## 🎉 You're All Set!

Your Health AI Chatbot should now be live on Streamlit Cloud!

### Share Your App:
- Your app URL: `https://[your-app-name].streamlit.app`
- Share this URL with users
- You can customize the URL in app settings

### Next Steps:
- Add more clinical documents to `data/clinical_docs/` for better RAG responses
- Monitor user feedback and improve the app
- Consider adding authentication for production use

---

## 📚 Additional Resources

- **Streamlit Cloud Documentation:** https://docs.streamlit.io/streamlit-community-cloud
- **Streamlit Secrets Management:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Gemini API Documentation:** https://ai.google.dev/docs
- **Troubleshooting Guide:** https://docs.streamlit.io/knowledge-base

---

## ⚠️ Important Notes

1. **Free Tier Limits:**
   - Streamlit Cloud free tier has resource limits
   - Apps may go to sleep after inactivity
   - Limited to 1GB memory

2. **API Key Security:**
   - NEVER commit API keys to GitHub
   - Always use Streamlit secrets
   - Rotate keys regularly

3. **Medical Disclaimer:**
   - This app is for educational purposes only
   - Not a substitute for professional medical advice
   - Users should consult healthcare providers

---

## 🆘 Need Help?

If you encounter issues:

1. Check the deployment logs in Streamlit Cloud
2. Review this troubleshooting guide
3. Visit Streamlit Community Forum: https://discuss.streamlit.io
4. Check GitHub issues: https://github.com/PraveenSalapu/HealthAgentDiabetic/issues

Good luck with your deployment! 🚀
