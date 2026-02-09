# 🚀 Deploy MACD Scanner to Streamlit Cloud

## Quick Deployment Guide

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io)

### Step 1: Push to GitHub

1. **Initialize Git** (if not already done):
```bash
cd "c:\Users\Ryan\OneDrive\Desktop\Coding\MACD Model"
git init
git add .
git commit -m "Initial commit: MACD Scanner v2.0 with AI"
```

2. **Create a GitHub repository:**
   - Go to https://github.com/new
   - Name it: `macd-scanner` or `stock-signal-scanner`
   - Keep it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (we already have one)

3. **Push your code:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/macd-scanner.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Sign up for Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account

2. **Create a new app:**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/macd-scanner`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Wait for deployment:**
   - Usually takes 2-5 minutes
   - You'll get a public URL like: `https://your-app.streamlit.app`

### Step 3: Share Your App 🎉

Your app will be live at:
```
https://YOUR_USERNAME-macd-scanner.streamlit.app
```

## Important Notes

### ⚠️ Machine Learning Models
- The ML models (`xgboost_model.pkl`, `lstm_model.pth`) need to be:
  - Either committed to the repo (if < 100MB each)
  - Or trained on first run (slower initial startup)
  - Or stored externally (S3, Google Drive, etc.)

### 📦 Requirements
- All dependencies are in `requirements.txt`
- Streamlit Cloud will automatically install them

### 🎯 Performance Tips
- First scan will be slower (loading models)
- Consider reducing scan scope if hitting memory limits
- Streamlit Cloud free tier has resource limits

### 🔒 Security
- Don't commit any API keys or sensitive data
- Use Streamlit secrets for sensitive configuration

## Alternative: Local Network Deployment

If you want to keep it local but accessible on your network:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

Then access from any device on your network at:
```
http://YOUR_LOCAL_IP:8501
```

## Troubleshooting

### Out of Memory
- Reduce the number of stocks scanned
- Disable ML filtering
- Use lighter models

### Slow Startup
- Models are being loaded (first time)
- Data is being downloaded from Yahoo Finance

### Can't Connect
- Check firewall settings
- Ensure port 8501 is available
- Try different browser

## Support

For issues or questions:
- Streamlit Docs: https://docs.streamlit.io
- Streamlit Community: https://discuss.streamlit.io
