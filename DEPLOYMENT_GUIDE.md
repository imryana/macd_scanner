# 🚀 Deploying Your MACD Scanner to Streamlit Cloud

## Quick Start (5 Minutes)

### Step 1: Test Locally First

1. **Install Streamlit** (if not already installed):
   ```powershell
   pip install streamlit
   ```

2. **Run the app locally**:
   ```powershell
   cd "c:\Users\Ryan\OneDrive\Desktop\Coding\MACD Model"
   streamlit run streamlit_app.py
   ```

3. Your browser should open automatically to `http://localhost:8501`

### Step 2: Create GitHub Repository

1. **Go to GitHub.com** and create a new repository:
   - Name it something like `macd-trading-scanner`
   - Make it **Public** (required for free Streamlit hosting)
   - Don't initialize with README (we already have files)

2. **Push your code to GitHub**:
   ```powershell
   cd "c:\Users\Ryan\OneDrive\Desktop\Coding\MACD Model"
   git init
   git add .
   git commit -m "Initial commit - MACD Scanner"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/macd-trading-scanner.git
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud (FREE!)

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure deployment**:
   - **Repository**: Select your `macd-trading-scanner` repo
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom URL (e.g., `macd-scanner`)

5. **Click "Deploy"**

6. **Wait 2-3 minutes** for deployment

7. **Done!** Your app is now live at `https://YOUR_APP_NAME.streamlit.app`

---

## 📝 Important Files

Make sure these files are in your repository:
- ✅ `streamlit_app.py` - Main Streamlit application
- ✅ `macd_scanner.py` - Your MACD scanner logic
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Files to exclude from Git

---

## 🎨 Customization Options

### Change App Theme

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### Add Custom Domain (Optional)

In Streamlit Cloud settings, you can connect a custom domain like `scanner.yoursite.com`

---

## 🔄 Updating Your App

1. Make changes to your code locally
2. Test with `streamlit run streamlit_app.py`
3. Commit and push to GitHub:
   ```powershell
   git add .
   git commit -m "Update: description of changes"
   git push
   ```
4. Streamlit Cloud automatically redeploys within 1-2 minutes!

---

## ⚠️ Important Notes

### Rate Limits
- Yahoo Finance has rate limits (~2000 requests/hour)
- Full S&P 500 scan requires ~500 requests
- Limit users to 2-3 full scans per hour
- Consider adding caching for better performance

### Performance Tips
1. **Use caching** - Add `@st.cache_data` to expensive functions
2. **Limit initial scan** - Start with sector scans, not full S&P 500
3. **Add progress indicators** - Users need to know scanning is working

### Cost
- **Streamlit Cloud Free Tier**:
  - ✅ 1 public app
  - ✅ Unlimited viewers
  - ✅ Auto-updates from GitHub
  - ✅ 1GB storage
  - ⚠️ App sleeps after 7 days of inactivity

---

## 🐛 Troubleshooting

### App won't start
- Check `requirements.txt` has all dependencies
- Verify `streamlit_app.py` is in the root directory
- Check Streamlit Cloud logs for errors

### Scanner is slow
- Yahoo Finance rate limiting
- Too many stocks being scanned at once
- Use sector scans or smaller batches

### Module not found error
- Add missing package to `requirements.txt`
- Redeploy app

---

## 📚 Next Steps

### Enhancements You Can Add:
1. **Scheduled scans** - Run automatically at market close
2. **Email alerts** - Notify when strong signals appear
3. **Historical tracking** - Store signals over time
4. **Backtesting** - Test signal accuracy
5. **Watchlist** - Let users save favorite stocks
6. **Mobile-friendly** - Optimize for phone access

---

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery) - Get inspiration
- [Streamlit Forum](https://discuss.streamlit.io) - Get help

---

## 📱 Share Your App

Once deployed, share your app URL:
- `https://YOUR_APP_NAME.streamlit.app`

Anyone can access it without needing to install anything!

---

**Need help?** Check the Streamlit documentation or ask in their community forum.

Good luck! 🚀
