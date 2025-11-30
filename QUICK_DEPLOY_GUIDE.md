# ⚡ Quick Deployment Guide
**Get your app live in 5 minutes**

---

## 🎯 Prerequisites (Get These First!)

### 1. Gemini API Key
- Visit: https://makersuite.google.com/app/apikey
- Sign in with Google
- Click "Create API Key"
- **Copy and save it!**

### 2. Streamlit Cloud Account
- Visit: https://share.streamlit.io
- Click "Sign in with GitHub"
- Authorize Streamlit

---

## 🚀 Deployment Steps

### Step 1: Push to GitHub ✅
```bash
# Your code is already ready!
# Just make sure it's pushed
git status
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to:** https://share.streamlit.io
2. **Click:** "New app" button
3. **Fill in:**
   - Repository: `PraveenSalapu/HealthAgentDiabetic`
   - Branch: `main`
   - Main file path: `app_modular.py`
   - App URL: `health-ai-chatbot` (or your choice)

### Step 3: Configure Secrets 🔐

4. **Click:** "Advanced settings"
5. **Find:** "Secrets" section
6. **Paste this** (replace with your actual API key):

```toml
GEMINI_API_KEY = "paste_your_gemini_api_key_here"
```

7. **Click:** "Deploy!"

### Step 4: Wait ⏳
- Build takes ~3-5 minutes
- Watch the logs for any errors
- Once done, your app will open automatically

---

## ✅ What's Already Done

- ✅ **Dependencies optimized** (~150MB, cloud-ready)
- ✅ **No Windows-only packages** (works on Linux cloud)
- ✅ **Model files included** (in `model_output/`)
- ✅ **Relative paths** (no hard-coded paths)
- ✅ **Preprocessing configured** (StandardScaler ready)
- ✅ **Multi-agent system** (Gemini + RAG agents)
- ✅ **Admin page** (document upload for RAG)

---

## 🔧 If Something Goes Wrong

### Common Issues:

**1. "Module not found" error**
- Check Streamlit Cloud logs
- Verify `requirements.txt` has the package
- Restart the app

**2. "API key not found" error**
- Double-check secrets configuration
- Make sure you pasted the actual key (not placeholder)
- Format must be: `GEMINI_API_KEY = "your_key"`

**3. "Model file not found" error**
- Ensure `model_output/` folder is committed to GitHub
- Check file names match: `xgb_model.json`, `preprocessing_config.json`, etc.

**4. Build takes too long**
- Normal for first deployment (~5 minutes)
- Check logs for progress
- If stuck >10 minutes, restart deployment

---

## 📊 Expected Results

After successful deployment:

✅ **Homepage:** Diabetes risk assessment form
✅ **Prediction:** Risk probability with gauge chart
✅ **Analysis:** Feature importance & insights
✅ **Chatbot:** Switch between Gemini and RAG agents
✅ **Admin:** Document upload page (sidebar)

---

## 🎉 Testing Your Deployed App

### Test Form Submission:
1. Fill out health assessment form
2. Click "Assess Risk"
3. Verify prediction appears
4. Check gauge chart displays

### Test Chatbot:
1. Select "Gemini Agent"
2. Ask: "What are risk factors for diabetes?"
3. Verify response appears
4. Switch to "RAG Agent"
5. Ask: "What do the guidelines say about diabetes prevention?"

### Test Admin Page:
1. Click "Admin Document Upload" in sidebar
2. Upload a PDF medical document
3. Verify indexing completes
4. Test RAG agent retrieves information from it

---

## 🔗 Useful Links

- **Streamlit Cloud Dashboard:** https://share.streamlit.io
- **App Logs:** Available in Streamlit Cloud UI
- **Gemini API Console:** https://makersuite.google.com
- **Streamlit Docs:** https://docs.streamlit.io

---

## 💡 Pro Tips

1. **Monitor API usage:** Check Gemini API quotas regularly
2. **Test locally first:** Run `streamlit run app_modular.py` before deploying
3. **Use secrets for all keys:** Never hard-code API keys
4. **Check logs regularly:** Streamlit Cloud provides detailed logs
5. **Update dependencies carefully:** Test locally after updates

---

## 📞 Need Help?

- **Streamlit Community:** https://discuss.streamlit.io
- **GitHub Issues:** Create issue in your repo
- **Deployment Guide:** See `DEPLOYMENT_READINESS_REPORT.md` for details

---

**You're all set! Click that Deploy button! 🚀**
