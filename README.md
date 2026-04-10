# PSA 10 Profit Scanner — Elite Edition v2.0

A mobile-first Streamlit app that uses Computer Vision to pre-grade raw
Pokémon cards for PSA 10 potential and calculates profit opportunity.

---

## Files

```
psa10_scanner_v2/
├── app.py              ← Full Streamlit application (~700 lines)
├── requirements.txt    ← All Python dependencies
├── config.yaml         ← API keys + CV thresholds (DO NOT commit with real keys)
└── README.md
```

---

## Features

| Feature | Details |
|---|---|
| **eBay Live Scan** | Official Browse API — no ToS risk |
| **Batch Leaderboard** | Paste up to 10 URLs — ranked by PSA 10 confidence |
| **Single Card Analyser** | Upload photo or paste URL — full 10-check pipeline |
| **Gap Finder** | Pre-built dashboard of highest-ROI card targets |
| **Alert Feed** | Fires when a card hits ≥70% PSA 10 confidence |
| **Mock Mode** | Works immediately with no API keys (demo data) |

---

## CV Check Pipeline (10 Checks)

| # | Check | Method | Fail = |
|---|---|---|---|
| 1 | Resolution | Dimension check | Hard Reject |
| 2 | Stock Photo Detection | Aspect ratio + border σ | Hard Reject |
| 3 | Front Centering L/R | Canny edge → border ratio >55% | Hard Reject |
| 4 | Front Centering T/B | Canny edge → border ratio >55% | Hard Reject |
| 5 | Back Centering | Same, 75/25 standard | Hard Reject |
| 6 | Corner Whitening (TL/TR/BL/BR) | White pixel cluster ≥25px | Hard Reject |
| 7 | Glint / Glare | >2% holo area overexposed | Warning only |
| 8 | Scratch Detection | Laplacian in glare zone >0.3% | Hard Reject |
| 9 | Clone Stamp (AI Edit) | NCC block matching across image | Hard Reject |
| 10 | Rosette / Fake Detection | FFT frequency analysis (CMYK) | Forensic Flag |

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` — runs in Demo Mode without API keys.

---

## Deploy to Streamlit Cloud (Free — Use on Phone)

### Step 1: Get eBay API Keys
1. Go to https://developer.ebay.com → sign in with eBay account
2. **My Account → Application Access Keys → Create Keyset → Production**
3. Copy **App ID**, **Cert ID**, **Dev ID**

### Step 2: Push to GitHub
```bash
git init
git add app.py requirements.txt README.md
# Do NOT add config.yaml with real keys
git commit -m "PSA 10 Scanner v2"
git remote add origin https://github.com/YOUR_USERNAME/psa10-scanner
git push -u origin main
```

### Step 3: Deploy
1. Go to https://share.streamlit.io → **New App**
2. Connect GitHub → select your repo → set `app.py` as main file
3. **Advanced Settings → Secrets** → paste:
```toml
[ebay]
app_id  = "YOUR-EBAY-APP-ID-HERE"
cert_id = "YOUR-EBAY-CERT-ID-HERE"
dev_id  = "YOUR-EBAY-DEV-ID-HERE"
```
4. Click **Deploy** — live in ~90 seconds at:
   `https://your-username-psa10-scanner.streamlit.app`

5. On iPhone/Android: open URL → Share → **Add to Home Screen**
   (behaves like a native app)

---

## Multi-Source Strategy

| Source | Method | Status |
|---|---|---|
| **eBay** | Official Browse API | ✅ Fully integrated |
| **Mercari / TCGplayer / Cardmarket** | Paste direct image URLs into Batch tab | ✅ Works via manual URL |
| **Playwright bot-mimicry** | Randomized delays + mouse events to evade detection | ❌ Not included — violates ToS |

For Mercari/Cardmarket: open the listing in your browser, right-click
the card image → "Copy Image Address" → paste into the Batch tab.
The scanner will pull and grade it just like an eBay image.

---

## Note on Strictness

The scanner is tuned to **reject aggressively**. A card with zero flags
still only scores ~80% PSA 10 confidence — real grading has subjectivity
that no CV model can fully capture. Always do a final manual visual
review before purchasing, and verify PSA 10 values on:
- https://www.psacard.com/smr
- https://www.130point.com

---

## Disclaimer

For educational and research purposes only.
Not financial advice. Always verify manually before purchasing.
PSA 10 value estimates are based on historical sales — markets fluctuate.
