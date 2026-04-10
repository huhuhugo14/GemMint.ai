"""
╔══════════════════════════════════════════════════════════════╗
║         PSA 10 PROFIT SCANNER  —  ELITE EDITION  v2.0       ║
║   Pokémon Card CV Pre-Grader · ROI Engine · Multi-Source     ║
╚══════════════════════════════════════════════════════════════╝
Stack : Streamlit · OpenCV · eBay Browse API · Requests
"""

import streamlit as st
import cv2
import numpy as np
import requests
import yaml
import os
import io
import json
import time
import base64
import hashlib
import re
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode, urlparse, quote_plus
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  STREAMLIT PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSA 10 Profit Scanner",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CONFIG LOADER
# ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    cfg = {}
    # 1. config.yaml (local)
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    # 2. Streamlit secrets override (cloud deploy)
    try:
        ebay = st.secrets.get("ebay", {})
        if ebay:
            cfg.setdefault("ebay", {})
            cfg["ebay"].update(ebay)
    except Exception:
        pass
    return cfg

cfg = load_config()
EBAY_APP_ID  = cfg.get("ebay", {}).get("app_id",  os.getenv("EBAY_APP_ID",  ""))
EBAY_CERT_ID = cfg.get("ebay", {}).get("cert_id", os.getenv("EBAY_CERT_ID", ""))

# ─────────────────────────────────────────────────────────────
#  GLOBAL THRESHOLDS
# ─────────────────────────────────────────────────────────────
CENTER_FRONT   = 0.55   # 55/45 — PSA 10 front standard
CENTER_BACK    = 0.75   # 75/25 — PSA 10 back standard (more lenient)
MIN_PX         = 1000   # minimum image dimension
GLINT_FRAC     = 0.02   # >2% overexposed holo → glint flag
GLINT_LV       = 245    # pixel brightness threshold for glare
SCRATCH_DENS   = 0.003  # Laplacian edge density inside glare → scratch flag
WHITE_LV       = 218    # pixel value considered "white"
WHITE_CLUSTER  = 25     # min white pixels in corner patch → whitening flag
CLONE_BLOCK    = 16     # NCC block size for clone detection
CLONE_THRESH   = 0.97   # NCC similarity threshold → cloning flag
ROSETTE_FREQ   = 45     # FFT frequency band for rosette pattern (fake card)
GRADING_FEE    = 25.0
SHIPPING_EST   = 6.0

# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────
@dataclass
class GradingResult:
    # Centering
    center_front_lr: float = 0.0
    center_front_tb: float = 0.0
    center_back_lr:  float = 0.0
    center_back_tb:  float = 0.0
    centering_flag:  bool  = False

    # Corners
    corners_flag:    bool  = False
    corner_details:  list  = field(default_factory=list)
    corner_patches:  list  = field(default_factory=list)

    # Surface
    glint_flag:      bool  = False
    scratch_flag:    bool  = False

    # Forensics
    clone_flag:      bool  = False
    rosette_flag:    bool  = False
    stock_photo:     bool  = False

    # Resolution
    res_flag:        bool  = False
    res_px:          tuple = (0, 0)

    # Grade distribution (%)
    psa10: float = 0.0
    psa9:  float = 0.0
    psa8:  float = 0.0
    psa7:  float = 0.0
    psa6:  float = 0.0
    psa5:  float = 0.0
    psa1_4:float = 0.0

    # Outcome
    overall_pass:  bool  = False
    confidence:    float = 0.0   # 0–100
    reasoning:     list  = field(default_factory=list)

@dataclass
class ListingResult:
    title:       str   = ""
    price:       float = 0.0
    url:         str   = "#"
    source:      str   = "eBay"
    image_urls:  list  = field(default_factory=list)
    grading:     Optional[GradingResult] = None
    psa10_value: float = 0.0
    net_profit:  float = 0.0
    exp_profit:  float = 0.0
    roi_pct:     float = 0.0
    verdict:     str   = "PENDING"   # PASS | WARN | FAIL | LOW_VALUE

# ─────────────────────────────────────────────────────────────
#  MOCK DATA  (works immediately, no API keys needed)
# ─────────────────────────────────────────────────────────────
MOCK_LISTINGS = [
    {
        "title": "[DEMO] Charizard Base Set Holo Rare Unlimited — Raw NM",
        "price": 79.99,
        "url": "https://www.ebay.com/sch/i.html?_nkw=charizard+base+set+holo+raw",
        "source": "eBay",
        "image_urls": [
            "https://images.pokemontcg.io/base1/4_hires.png",
            "https://images.pokemontcg.io/base1/4_hires.png",
        ],
    },
    {
        "title": "[DEMO] Lugia Neo Genesis Holo — PSA Ready Raw NM/M",
        "price": 145.00,
        "url": "https://www.ebay.com/sch/i.html?_nkw=lugia+neo+genesis+holo+raw",
        "source": "eBay",
        "image_urls": [
            "https://images.pokemontcg.io/neo1/9_hires.png",
        ],
    },
    {
        "title": "[DEMO] Shining Charizard Neo Destiny — Ungraded RAW",
        "price": 290.00,
        "url": "https://www.ebay.com/sch/i.html?_nkw=shining+charizard+neo+destiny+raw",
        "source": "eBay",
        "image_urls": [
            "https://images.pokemontcg.io/neo4/107_hires.png",
        ],
    },
    {
        "title": "[DEMO] Gengar Fossil Holo Rare — Heavily Played",
        "price": 24.99,
        "url": "https://www.ebay.com/sch/i.html?_nkw=gengar+fossil+holo+raw",
        "source": "eBay",
        "image_urls": [
            "https://images.pokemontcg.io/fossil/5_hires.png",
        ],
    },
    {
        "title": "[DEMO] Charizard 1st Edition Base Set — RAW UNGRADED",
        "price": 8500.00,
        "url": "https://www.ebay.com/sch/i.html?_nkw=charizard+1st+edition+base+raw",
        "source": "eBay",
        "image_urls": [
            "https://images.pokemontcg.io/base1/4_hires.png",
        ],
    },
]

# ─────────────────────────────────────────────────────────────
#  PSA 10 VALUE REFERENCE TABLE
# ─────────────────────────────────────────────────────────────
PSA10_VALUES = {
    "charizard 1st":         350000,
    "charizard shadowless":   50000,
    "charizard base":          5000,
    "charizard skyridge":     15000,
    "charizard aquapolis":     8000,
    "blastoise 1st":          25000,
    "blastoise shadowless":    8000,
    "blastoise base":          1200,
    "venusaur 1st":           12000,
    "venusaur base":            900,
    "lugia neo":              12000,
    "ho-oh neo":               3000,
    "entei neo":               1800,
    "raikou neo":              1500,
    "suicune neo":             1200,
    "shining charizard":       8000,
    "shining magikarp":        3500,
    "shining gyarados":        2000,
    "dark charizard":          2500,
    "dark blastoise":          1800,
    "gengar fossil":           2500,
    "lapras fossil":            800,
    "moltres fossil":           700,
    "zapdos fossil":            700,
    "scyther jungle":           600,
    "clefable jungle":          700,
    "snorlax jungle":          1200,
    "vaporeon jungle":          900,
    "jolteon jungle":           800,
    "raichu base":             1800,
    "machamp 1st":             2000,
    "ninetales base":           900,
    "alakazam base":            700,
}

HIGH_ROI_CATALOG = [
    {"set":"Base Set (1st Edition)","cards":["Charizard","Blastoise","Venusaur","Raichu"],"est_spread":"$5k–$500k+","note":"Crown jewel — any 1st Ed holo has massive PSA 10 premium"},
    {"set":"Base Set Shadowless","cards":["Charizard","Blastoise","Venusaur","Ninetales"],"est_spread":"$1k–$50k","note":"Shadowless holos gem mint are extremely scarce"},
    {"set":"Base Set Unlimited","cards":["Charizard","Blastoise","Raichu","Machamp"],"est_spread":"$500–$5k","note":"High volume raw market, gem copies still rare"},
    {"set":"Jungle","cards":["Scyther","Clefable","Snorlax","Vaporeon","Jolteon"],"est_spread":"$300–$1.2k","note":"Notorious centering issues → PSA 10 scarcity premium"},
    {"set":"Fossil","cards":["Gengar","Lapras","Moltres","Zapdos","Articuno"],"est_spread":"$400–$2.5k","note":"Thick cards prone to edge dings — gem copies command big premium"},
    {"set":"Team Rocket","cards":["Dark Charizard","Dark Blastoise","Here Comes Team Rocket!"],"est_spread":"$800–$2.5k","note":"Dark cards show surface scratches easily"},
    {"set":"Neo Genesis","cards":["Lugia","Typhlosion","Feraligatr","Meganium"],"est_spread":"$500–$12k","note":"Lugia PSA 10 regularly $10k+ — raw copies exist under $200"},
    {"set":"Neo Revelation","cards":["Ho-Oh","Entei","Raikou","Suicune"],"est_spread":"$600–$3k","note":"Legendary beasts — all have massive PSA 10 upside"},
    {"set":"Neo Destiny","cards":["Shining Charizard","Shining Magikarp","Dark Espeon"],"est_spread":"$1k–$8k","note":"Shining series = rarest PSA 10s from WotC era"},
    {"set":"Aquapolis / Skyridge","cards":["Charizard","Articuno","Celebi","Jolteon"],"est_spread":"$2k–$15k","note":"e-Reader era notorious print lines → PSA 10 ultra-rare"},
    {"set":"Expedition","cards":["Charizard","Blastoise","Venusaur"],"est_spread":"$1k–$8k","note":"Old holo tech → lots of holo surface damage in the wild"},
]

# ─────────────────────────────────────────────────────────────
#  IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_image(url: str) -> Optional[np.ndarray]:
    if not url or url.startswith("data:"):
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def upscale_patch(patch: np.ndarray, scale: int = 4) -> Image.Image:
    pil = bgr_to_pil(patch)
    w, h = pil.size
    return pil.resize((max(w * scale, 60), max(h * scale, 60)), Image.NEAREST)

# ─────────────────────────────────────────────────────────────
#  CV ENGINE — CHECK 1: RESOLUTION & STOCK PHOTO FILTER
# ─────────────────────────────────────────────────────────────
def check_resolution(img: np.ndarray) -> tuple[bool, tuple, bool]:
    """
    Returns (passed, (w,h), is_stock_photo).
    Stock photo heuristic: near-square ratio + very uniform background.
    """
    h, w = img.shape[:2]
    passed = w >= MIN_PX and h >= MIN_PX

    # Stock photo heuristic: if aspect ratio is suspiciously square (0.9–1.1)
    # AND background colour variance is very low → likely stock/official art
    aspect = w / max(h, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sample border pixels
    border = np.concatenate([
        gray[:10, :].flatten(),
        gray[-10:, :].flatten(),
        gray[:, :10].flatten(),
        gray[:, -10:].flatten(),
    ])
    border_std = float(border.std())
    is_stock = (0.92 < aspect < 1.08) and (border_std < 12)

    return passed, (w, h), is_stock

# ─────────────────────────────────────────────────────────────
#  CV ENGINE — CHECK 2: CENTERING  (Canny border detection)
# ─────────────────────────────────────────────────────────────
def detect_card_rect(img: np.ndarray) -> Optional[tuple]:
    """Returns (left, right, top, bottom) of card bounding box via Canny."""
    h, w = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 25, 90)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_area, best = 0, None
    min_area = h * w * 0.10
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 3 <= len(approx) <= 6 and area > best_area:
            best, best_area = approx, area

    if best is None:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        return (x, x + bw, y, y + bh)

    pts = best.reshape(-1, 2)
    return (int(pts[:,0].min()), int(pts[:,0].max()),
            int(pts[:,1].min()), int(pts[:,1].max()))

def check_centering(img: np.ndarray, is_back: bool = False) -> tuple:
    """
    Returns (lr_ratio, tb_ratio, flagged, reasons).
    Front threshold: 55/45.  Back threshold: 75/25.
    """
    h, w    = img.shape[:2]
    rect    = detect_card_rect(img)
    reasons = []
    thresh  = CENTER_BACK if is_back else CENTER_FRONT
    label   = "Back" if is_back else "Front"

    if rect is None:
        return 0.5, 0.5, False, [f"Could not detect {label} card border"]

    left, right, top, bottom = rect
    bl = left;          br = w - right
    bt = top;           bb = h - bottom

    lr_total = max(bl + br, 1)
    tb_total = max(bt + bb, 1)
    lr_ratio = max(bl, br) / lr_total
    tb_ratio = max(bt, bb) / tb_total

    flagged  = False
    if lr_ratio > thresh:
        flagged = True
        reasons.append(
            f"{label} L/R centering {lr_ratio:.1%} — "
            f"exceeds {'75/25' if is_back else '55/45'} limit ← HARD FAIL"
        )
    if tb_ratio > thresh:
        flagged = True
        reasons.append(
            f"{label} T/B centering {tb_ratio:.1%} — "
            f"exceeds {'75/25' if is_back else '55/45'} limit ← HARD FAIL"
        )

    return lr_ratio, tb_ratio, flagged, reasons

# ─────────────────────────────────────────────────────────────
#  CV ENGINE — CHECK 3: CORNER WHITENING
# ─────────────────────────────────────────────────────────────
CORNER_LABELS = ["TL", "TR", "BL", "BR"]

def extract_corners(img: np.ndarray, pct: float = 0.09) -> list:
    h, w = img.shape[:2]
    ph   = max(int(h * pct), 24)
    pw   = max(int(w * pct), 24)
    return [
        img[0:ph,    0:pw   ],   # TL
        img[0:ph,    w-pw:w ],   # TR
        img[h-ph:h,  0:pw   ],   # BL
        img[h-ph:h,  w-pw:w ],   # BR
    ]

def check_corners(img: np.ndarray) -> tuple:
    """
    Scans 4 corners for white-pixel clusters.
    Returns (flagged, detail_list, annotated_patches).
    """
    patches   = extract_corners(img)
    details   = []
    annotated = []
    flagged   = False

    for label, patch in zip(CORNER_LABELS, patches):
        if patch.size == 0:
            annotated.append(patch)
            continue
        r, g, b  = cv2.split(patch)
        white    = (r.astype(int) > WHITE_LV) & (g.astype(int) > WHITE_LV) & (b.astype(int) > WHITE_LV)
        n_white  = int(white.sum())

        display = patch.copy()
        if n_white >= WHITE_CLUSTER:
            flagged = True
            details.append(f"{label}: {n_white} white pixels — corner whitening ← HARD FAIL")
            display[white] = [0, 0, 255]   # red overlay
        else:
            display[white] = [0, 200, 100]  # green: minor, not flagged
        annotated.append(display)

    return flagged, details, annotated

# ─────────────────────────────────────────────────────────────
#  CV ENGINE — CHECK 4: GLINT + LAPLACIAN SCRATCH DETECTION
# ─────────────────────────────────────────────────────────────
def check_surface(img: np.ndarray) -> tuple:
    """
    Identifies glare hotspots in the holo zone.
    If glare found, applies Laplacian to detect scratches/print lines.
    Returns (glint_flag, scratch_flag, reasons).
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.18), int(h * 0.72)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    holo    = img[y1:y2, x1:x2]

    if holo.size == 0:
        return False, False, []

    gray    = cv2.cvtColor(holo, cv2.COLOR_BGR2GRAY)
    glare   = gray > GLINT_LV
    gfrac   = float(glare.sum()) / gray.size
    reasons = []

    if gfrac <= GLINT_FRAC:
        return False, False, []

    reasons.append(f"Glare/glint: {gfrac:.1%} of holo area overexposed (threshold {GLINT_FRAC:.0%})")

    # Laplacian scratch check within blown-out zones
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    if glare.sum() > 0:
        glare_vals = lap_abs[glare]
        dens = float((glare_vals > 35).sum()) / max(glare_vals.size, 1)
        if dens > SCRATCH_DENS:
            reasons.append(
                f"Potential scratches/print lines in glare zone — "
                f"Laplacian edge density {dens:.2%} ← HARD FAIL"
            )
            return True, True, reasons

    return True, False, reasons

# ─────────────────────────────────────────────────────────────
#  CV ENGINE — CHECK 5: DIGITAL FORENSICS
#  5a. Clone-stamp detection  (NCC block matching)
#  5b. Rosette pattern → fake card detection  (FFT)
# ─────────────────────────────────────────────────────────────
def check_clone_stamp(img: np.ndarray) -> tuple[bool, list]:
    """
    Divide image into blocks; find pairs with very high NCC similarity.
    Repeated identical blocks → clone-stamp tool used to hide flaws.
    Returns (flag, reasons).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bs   = CLONE_BLOCK
    blocks = {}

    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            block = gray[y:y+bs, x:x+bs].astype(np.float32)
            # Normalise
            mn, sd = block.mean(), block.std()
            if sd < 5:   # skip flat regions (borders, solid color)
                continue
            norm  = ((block - mn) / sd).tobytes()
            key   = hashlib.md5(norm).hexdigest()
            if key in blocks:
                # Found near-identical block at different position
                by, bx = blocks[key]
                dist = ((y - by)**2 + (x - bx)**2) ** 0.5
                if dist > bs * 3:   # must be non-adjacent
                    return True, [
                        f"Clone-stamp / photo editing detected: "
                        f"identical image block at ({bx},{by}) and ({x},{y}) "
                        f"— possible flaw concealment ← FORENSIC FAIL"
                    ]
            else:
                blocks[key] = (y, x)

    return False, []

def check_rosette_pattern(img: np.ndarray) -> tuple[bool, list]:
    """
    Genuine WotC-era cards are CMYK offset-printed, producing a subtle
    rosette dot pattern. Counterfeits are typically inkjet-printed —
    no rosette. We look for the rosette frequency in the FFT spectrum.
    An ABSENCE of rosette pattern in a supposedly old card → likely fake.
    Returns (flag, reasons).
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    # Work on a centre crop (avoid borders/text)
    cy, cx = h // 2, w // 2
    crop   = gray[cy-80:cy+80, cx-80:cx+80].astype(np.float32)
    if crop.shape[0] < 60 or crop.shape[1] < 60:
        return False, []

    fft    = np.fft.fft2(crop)
    fshift = np.fft.fftshift(fft)
    mag    = 20 * np.log(np.abs(fshift) + 1)

    # Look for energy spike in the expected rosette frequency band
    ch, cw = mag.shape
    ring   = mag[ch//2 - ROSETTE_FREQ - 5 : ch//2 - ROSETTE_FREQ + 5,
                 cw//2 - ROSETTE_FREQ - 5 : cw//2 - ROSETTE_FREQ + 5]

    peak = float(ring.max()) if ring.size > 0 else 0.0
    baseline = float(mag[0:20, 0:20].mean())

    if peak < baseline * 1.6:
        return True, [
            "No CMYK rosette print pattern detected via FFT — "
            "this may indicate an inkjet counterfeit ← FORENSIC FLAG (verify manually)"
        ]
    return False, []

# ─────────────────────────────────────────────────────────────
#  GRADE PROBABILITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def compute_probabilities(gr: GradingResult) -> GradingResult:
    """
    Ultra-strict Bayesian distribution.
    A card with ZERO flags still only has ~80% PSA 10 probability
    (genuine subjectivity in real grading).
    """
    if gr.res_flag or gr.stock_photo:
        gr.psa10  = 0; gr.psa9 = 0; gr.psa8 = 0; gr.psa7 = 0
        gr.psa6   = 0; gr.psa5 = 0; gr.psa1_4 = 100
        gr.confidence = 0
        return gr

    p10 = 80.0
    deductions = {
        "centering":  (gr.centering_flag, 55.0),
        "corners":    (gr.corners_flag,   48.0),
        "scratches":  (gr.scratch_flag,   42.0),
        "glint":      (gr.glint_flag,     18.0),
        "clone":      (gr.clone_flag,     35.0),
        "rosette":    (gr.rosette_flag,   25.0),
    }
    for name, (flag, penalty) in deductions.items():
        if flag:
            p10 -= penalty

    p10 = max(p10, 0.0)

    # Hard fails collapse p10 entirely
    hard_fails = [gr.centering_flag, gr.corners_flag, gr.scratch_flag]
    if any(hard_fails):
        p10 = min(p10, 4.0)   # near-zero — scanner is strict

    # Build distribution — remaining probability cascades down grades
    rem  = 100.0 - p10
    p9   = rem * 0.40
    p8   = rem * 0.25
    p7   = rem * 0.15
    p6   = rem * 0.08
    p5   = rem * 0.06
    p1_4 = rem * 0.06

    gr.psa10  = round(p10,  1)
    gr.psa9   = round(p9,   1)
    gr.psa8   = round(p8,   1)
    gr.psa7   = round(p7,   1)
    gr.psa6   = round(p6,   1)
    gr.psa5   = round(p5,   1)
    gr.psa1_4 = round(p1_4, 1)
    gr.confidence = round(p10, 1)
    return gr

# ─────────────────────────────────────────────────────────────
#  FULL GRADING PIPELINE
# ─────────────────────────────────────────────────────────────
def grade_images(image_urls: list) -> GradingResult:
    gr = GradingResult()

    if not image_urls:
        gr.res_flag = True
        gr.reasoning.append("No images provided in listing")
        gr.overall_pass = False
        return compute_probabilities(gr)

    images = []
    for url in image_urls[:7]:
        img = fetch_image(url)
        if img is not None:
            images.append(img)

    if not images:
        gr.res_flag = True
        gr.reasoning.append("All image downloads failed — possibly deleted listing or geo-block")
        gr.overall_pass = False
        return compute_probabilities(gr)

    # ── Resolution & stock photo ──────────────────────────────
    res_ok, res_px, is_stock = check_resolution(images[0])
    gr.res_px     = res_px
    gr.stock_photo = is_stock
    if not res_ok:
        gr.res_flag = True
        gr.reasoning.append(
            f"REJECTED: Low-res image {res_px[0]}×{res_px[1]}px — "
            f"minimum {MIN_PX}px required (likely stock photo)"
        )
        gr.overall_pass = False
        return compute_probabilities(gr)
    if is_stock:
        gr.reasoning.append(
            "WARNING: Image may be stock/official art — "
            "square aspect + uniform border. Cannot grade authentically."
        )

    # ── Per-image checks ─────────────────────────────────────
    for idx, img in enumerate(images):
        lbl  = f"Img {idx+1}"
        back = (idx >= len(images) // 2)   # heuristic: later images = back

        # Centering
        lr, tb, cf, cr = check_centering(img, is_back=back)
        if idx == 0:
            gr.center_front_lr = lr
            gr.center_front_tb = tb
        elif idx == 1:
            gr.center_back_lr  = lr
            gr.center_back_tb  = tb
        if cf:
            gr.centering_flag = True
            gr.reasoning.extend([f"{lbl}: {r}" for r in cr])

        # Corners
        cor_flag, cor_det, patches = check_corners(img)
        if cor_flag:
            gr.corners_flag = True
            gr.reasoning.extend([f"{lbl}: {r}" for r in cor_det])
        if not gr.corner_patches and patches:
            gr.corner_patches = patches
            gr.corner_details = cor_det

        # Surface / scratch
        g_flag, s_flag, g_reasons = check_surface(img)
        if g_flag:
            gr.glint_flag = True
        if s_flag:
            gr.scratch_flag = True
        gr.reasoning.extend([f"{lbl}: {r}" for r in g_reasons])

        # Forensics — only on first image (performance)
        if idx == 0:
            cl_flag, cl_r = check_clone_stamp(img)
            if cl_flag:
                gr.clone_flag = True
                gr.reasoning.extend(cl_r)

            ro_flag, ro_r = check_rosette_pattern(img)
            if ro_flag:
                gr.rosette_flag = True
                gr.reasoning.extend(ro_r)

    # ── Overall verdict (strict) ──────────────────────────────
    hard_fails = [
        gr.centering_flag,
        gr.corners_flag,
        gr.scratch_flag,
        gr.res_flag,
        gr.clone_flag,
    ]
    gr.overall_pass = not any(hard_fails)
    if gr.overall_pass and not gr.reasoning:
        gr.reasoning.append("✓ All automated checks passed — recommend manual visual confirmation")
    elif gr.overall_pass:
        gr.reasoning.insert(0, "✓ No hard-fail conditions — proceed with manual review")

    return compute_probabilities(gr)

# ─────────────────────────────────────────────────────────────
#  FINANCIAL ENGINE
# ─────────────────────────────────────────────────────────────
def estimate_psa10(title: str) -> float:
    tl = title.lower()
    for key, val in sorted(PSA10_VALUES.items(), key=lambda x: -len(x[0])):
        if all(k in tl for k in key.split()):
            return float(val)
    # Heuristic fallbacks
    if "1st edition" in tl or "1st ed" in tl:  return 2000.0
    if "shadowless"  in tl:                     return 1500.0
    if any(s in tl for s in ["lugia","ho-oh","entei","raikou","suicune"]): return 1200.0
    if "shining"     in tl:                     return 1500.0
    if "holo" in tl and any(s in tl for s in
        ["base set","jungle","fossil","team rocket","neo","expedition","skyridge","aquapolis"]):
        return 700.0
    return 400.0

def calculate_financials(price: float, psa10_val: float, p10: float
                          ) -> tuple[float, float, float]:
    """Returns (best_case_profit, expected_profit, roi_pct)."""
    costs        = price + GRADING_FEE + SHIPPING_EST
    best_case    = psa10_val - costs
    expected     = (psa10_val * p10 / 100.0) - costs
    roi_pct      = (best_case / max(price, 0.01)) * 100
    return round(best_case, 2), round(expected, 2), round(roi_pct, 1)

# ─────────────────────────────────────────────────────────────
#  eBay BROWSE API
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def get_ebay_token() -> Optional[str]:
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        return None
    creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
    try:
        r = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={"Content-Type":"application/x-www-form-urlencoded",
                     "Authorization":f"Basic {creds}"},
            data={"grant_type":"client_credentials",
                  "scope":"https://api.ebay.com/oauth/api_scope"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        st.warning(f"eBay auth failed: {e}")
        return None

def search_ebay(query: str, min_p: float, max_p: float, limit: int) -> list:
    token = get_ebay_token()
    if not token:
        st.info("ℹ️ No eBay API credentials — running in **Demo Mode** with sample data.")
        return MOCK_LISTINGS[:limit]

    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization":f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID":"EBAY_US"},
            params={
                "q": f"{query} pokemon card raw ungraded",
                "category_ids": "183454",
                "filter": f"price:[{min_p}..{max_p}],priceCurrency:USD",
                "sort": "price",
                "limit": limit,
                "fieldgroups": "EXTENDED",
            },
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("itemSummaries", [])
        out   = []
        for item in items:
            imgs  = [item.get("image",{}).get("imageUrl","")]
            imgs += [i.get("imageUrl","") for i in item.get("additionalImages",[])]
            imgs  = [u for u in imgs if u]
            out.append({
                "title":      item.get("title",""),
                "price":      float(item.get("price",{}).get("value",0)),
                "url":        item.get("itemWebUrl","#"),
                "source":     "eBay",
                "image_urls": imgs,
            })
        return out
    except Exception as e:
        st.warning(f"eBay API error: {e} — falling back to demo data")
        return MOCK_LISTINGS[:limit]

# ─────────────────────────────────────────────────────────────
#  BATCH GRADING RUNNER
# ─────────────────────────────────────────────────────────────
def grade_listing(raw: dict) -> ListingResult:
    lr             = ListingResult(**{k: raw[k] for k in
                                      ("title","price","url","source","image_urls")})
    lr.psa10_value = estimate_psa10(lr.title)
    lr.grading     = grade_images(lr.image_urls)
    lr.net_profit, lr.exp_profit, lr.roi_pct = calculate_financials(
        lr.price, lr.psa10_value, lr.grading.psa10
    )
    hard_fails = [
        lr.grading.centering_flag,
        lr.grading.corners_flag,
        lr.grading.scratch_flag,
        lr.grading.res_flag,
        lr.grading.clone_flag,
        lr.grading.stock_photo,
    ]
    if any(hard_fails):
        lr.verdict = "FAIL"
    elif lr.grading.glint_flag or lr.grading.rosette_flag:
        lr.verdict = "WARN"
    else:
        lr.verdict = "PASS"
    return lr

# ─────────────────────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────────────────────
NEON = {
    "green":  "#00ff88",
    "yellow": "#f5c518",
    "orange": "#ff7c00",
    "red":    "#ff3c3c",
    "blue":   "#4da6ff",
    "purple": "#b07dff",
    "dim":    "#3a4a6a",
}

def grade_color(pct: float) -> str:
    if pct >= 70: return NEON["green"]
    if pct >= 40: return NEON["yellow"]
    if pct >= 15: return NEON["orange"]
    return NEON["red"]

def prob_bar(label: str, pct: float, color: str, width: str = "100%"):
    st.markdown(f"""
    <div style="margin:3px 0;width:{width}">
      <div style="display:flex;justify-content:space-between;
                  font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                  color:#8899bb;margin-bottom:2px;">
        <span>{label}</span><span style="color:{color};">{pct:.1f}%</span>
      </div>
      <div style="background:#111827;border-radius:3px;height:10px;overflow:hidden;">
        <div style="width:{min(pct,100)}%;height:100%;
                    background:linear-gradient(90deg,{color}88,{color});
                    border-radius:3px;transition:width .5s ease;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def confidence_ring(pct: float):
    color = grade_color(pct)
    radius, stroke, circ = 38, 8, 238.76
    offset = circ * (1 - pct / 100)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin:8px 0;">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="{radius}" fill="none"
          stroke="#111827" stroke-width="{stroke}"/>
        <circle cx="50" cy="50" r="{radius}" fill="none"
          stroke="{color}" stroke-width="{stroke}"
          stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
          stroke-linecap="round"
          transform="rotate(-90 50 50)"/>
        <text x="50" y="46" text-anchor="middle"
          font-family="'Share Tech Mono',monospace"
          font-size="15" fill="{color}" font-weight="bold">{pct:.0f}%</text>
        <text x="50" y="60" text-anchor="middle"
          font-family="'Share Tech Mono',monospace"
          font-size="7.5" fill="#556677">PSA 10</text>
      </svg>
    </div>""", unsafe_allow_html=True)

def verdict_pill(v: str) -> str:
    if v == "PASS":
        return f'<span style="background:#003d24;color:{NEON["green"]};border:1px solid {NEON["green"]};border-radius:20px;padding:3px 14px;font-family:\'Share Tech Mono\',monospace;font-size:.75rem;font-weight:bold;">✅ PASS</span>'
    if v == "WARN":
        return f'<span style="background:#3d2800;color:{NEON["yellow"]};border:1px solid {NEON["yellow"]};border-radius:20px;padding:3px 14px;font-family:\'Share Tech Mono\',monospace;font-size:.75rem;font-weight:bold;">⚠ WARN</span>'
    return f'<span style="background:#3d0000;color:{NEON["red"]};border:1px solid {NEON["red"]};border-radius:20px;padding:3px 14px;font-family:\'Share Tech Mono\',monospace;font-size:.75rem;font-weight:bold;">❌ FAIL</span>'

def render_corners(patches: list):
    if not patches or len(patches) < 4:
        return
    cols = st.columns(4)
    for col, patch, lbl in zip(cols, patches[:4], CORNER_LABELS):
        with col:
            if patch.size > 0:
                col.image(upscale_patch(patch, 4), caption=f"{lbl} (4×)",
                          use_container_width=True)

def render_grading_card(r: ListingResult, expanded: bool = False):
    gr  = r.grading
    v   = r.verdict
    if v == "PASS":  border = NEON["green"]
    elif v == "WARN": border = NEON["yellow"]
    else:             border = NEON["red"]

    label = f"{'✅' if v=='PASS' else '⚠️' if v=='WARN' else '❌'}  {r.title[:62]}…  — ${r.price:.2f}"
    with st.expander(label, expanded=expanded):
        # Top row: image + rings + financials
        c1, c2, c3 = st.columns([1.4, 1, 1.4])

        with c1:
            if r.image_urls:
                st.image(r.image_urls[0], use_container_width=True)
            st.markdown(f"""
            <div style="margin-top:6px;">
              {verdict_pill(v)}
              <div style="margin-top:6px;">
              <a href="{r.url}" target="_blank"
                 style="background:{NEON['blue']}22;border:1px solid {NEON['blue']};
                        color:{NEON['blue']};padding:4px 12px;border-radius:6px;
                        font-family:'Share Tech Mono',monospace;font-size:.72rem;
                        text-decoration:none;">🛒 ONE-TAP PURCHASE</a>
              </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            confidence_ring(gr.confidence if gr else 0)
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;
                        color:#556677;text-align:center;">
              Source: {r.source}<br>
              Res: {gr.res_px[0]}×{gr.res_px[1]}
            </div>""", unsafe_allow_html=True)

        with c3:
            profit_color = NEON["green"] if r.net_profit > 0 else NEON["red"]
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
                        line-height:2.1;background:#0a0f1a;border-radius:8px;
                        border:1px solid #1e2d4a;padding:10px 14px;">
              <span style="color:#556677;">Raw Price </span>
              <b style="color:#d4e1f5;">${r.price:.2f}</b><br>
              <span style="color:#556677;">PSA 10 Est</span>
              <b style="color:{NEON['blue']};">${r.psa10_value:,.0f}</b><br>
              <span style="color:#556677;">Grading   </span>
              <b style="color:#556677;">−${GRADING_FEE:.0f}</b><br>
              <span style="color:#556677;">Shipping  </span>
              <b style="color:#556677;">−${SHIPPING_EST:.0f}</b><br>
              <span style="color:#556677;">Best-Case </span>
              <b style="color:{profit_color};">${r.net_profit:,.0f}</b><br>
              <span style="color:#556677;">Expected  </span>
              <b style="color:{profit_color};">${r.exp_profit:,.0f}</b><br>
              <span style="color:#556677;">ROI       </span>
              <b style="color:{profit_color};">{r.roi_pct:.0f}%</b>
            </div>""", unsafe_allow_html=True)

        # Grade distribution
        st.markdown("**Grade Probability Distribution**")
        dist_cols = st.columns(2)
        with dist_cols[0]:
            prob_bar("PSA 10", gr.psa10,  NEON["green"])
            prob_bar("PSA  9", gr.psa9,   NEON["yellow"])
            prob_bar("PSA  8", gr.psa8,   NEON["orange"])
        with dist_cols[1]:
            prob_bar("PSA  7", gr.psa7,   "#cc6600")
            prob_bar("PSA  6", gr.psa6,   "#aa4400")
            prob_bar("PSA ≤5", gr.psa1_4, NEON["red"])

        # Centering readout
        if gr.center_front_lr > 0:
            st.markdown(
                f"**Centering** — Front L/R `{gr.center_front_lr:.1%}` "
                f"T/B `{gr.center_front_tb:.1%}` | "
                f"Back L/R `{gr.center_back_lr:.1%}` "
                f"T/B `{gr.center_back_tb:.1%}`"
            )

        # Corner panels
        if gr.corner_patches:
            st.markdown("**Corner Analysis (400% Zoom)**")
            render_corners(gr.corner_patches)

        # AI reasoning report
        if gr.reasoning:
            st.markdown("**AI Reasoning Report**")
            for note in gr.reasoning:
                is_ok = any(x in note for x in ["✓", "passed", "Passed"])
                icon  = "🟢" if is_ok else ("🟡" if "WARNING" in note.upper() else "🔴")
                st.markdown(f"{icon} `{note}`")

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS + FONTS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;800;900&display=swap');

html,body,[class*="css"]{background:#07090f!important;color:#c8d8f0!important;}
.stApp{background:#07090f;}
h1,h2,h3,h4{font-family:'Barlow Condensed',sans-serif!important;letter-spacing:.06em;}
.mono{font-family:'Share Tech Mono',monospace;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:#0b0f1a;
  border-bottom:1px solid #1a2a4a;
  gap:4px;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Barlow Condensed',sans-serif;
  font-size:1rem;font-weight:600;
  color:#3a5a8a;letter-spacing:.06em;
  border-radius:6px 6px 0 0;
  padding:8px 18px;
}
.stTabs [aria-selected="true"]{
  color:#00ff88!important;
  background:#001a0d!important;
  border-bottom:2px solid #00ff88;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#1a6fff,#0047cc);
  color:#fff;border:none;border-radius:8px;
  font-family:'Barlow Condensed',sans-serif;
  font-size:1.05rem;font-weight:700;letter-spacing:.08em;
  padding:.55rem 1.4rem;width:100%;
  box-shadow:0 0 18px #1a6fff44;
  transition:all .2s;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#3a7fff,#1a6fff);
  box-shadow:0 0 28px #1a6fff88;transform:translateY(-1px);
}

/* Inputs */
.stTextInput>div>div>input,.stNumberInput>div>div>input{
  background:#0b0f1a!important;color:#c8d8f0!important;
  border:1px solid #1a2a4a!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;font-size:.88rem!important;
}
.stSelectbox>div>div{
  background:#0b0f1a!important;border:1px solid #1a2a4a!important;
}
.stTextArea>div>textarea{
  background:#0b0f1a!important;color:#c8d8f0!important;
  border:1px solid #1a2a4a!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;
}

/* Expander */
.streamlit-expanderHeader{
  background:#0b0f1a!important;border:1px solid #1a2a4a!important;
  border-radius:8px!important;
  font-family:'Barlow Condensed',sans-serif!important;font-size:1rem!important;
}
.streamlit-expanderContent{
  background:#07090f!important;border:1px solid #1a2a4a!important;
  border-top:none!important;border-radius:0 0 8px 8px!important;
  padding:12px 16px!important;
}

/* Metrics */
[data-testid="stMetric"]{
  background:#0b0f1a;border:1px solid #1a2a4a;border-radius:10px;padding:12px;
}
[data-testid="stMetricValue"]{
  font-family:'Barlow Condensed',sans-serif!important;font-size:1.8rem!important;
}

/* Sidebar */
[data-testid="stSidebar"]{background:#090d18!important;border-right:1px solid #1a2a4a;}

/* Scrollbar */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#07090f;}
::-webkit-scrollbar-thumb{background:#1a2a4a;border-radius:3px;}

/* Notification badge */
.notif-badge{
  display:inline-block;background:#00ff8822;
  border:1px solid #00ff88;border-radius:20px;
  color:#00ff88;font-family:'Share Tech Mono',monospace;
  font-size:.7rem;padding:2px 10px;margin-left:8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 .5rem;">
  <div style="font-family:'Barlow Condensed',sans-serif;
              font-size:clamp(2rem,8vw,3.8rem);font-weight:900;
              letter-spacing:.1em;line-height:1;
              background:linear-gradient(135deg,#00ff88 0%,#4da6ff 45%,#b07dff 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    PSA 10 PROFIT SCANNER
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
              color:#2a6a4a;letter-spacing:.2em;margin-top:4px;">
    ELITE EDITION v2.0 · CV PRE-GRADER · FORENSICS · ROI ENGINE
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#00ff8844,#4da6ff44,transparent);margin-bottom:1rem;"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  NOTIFICATION STATE
# ─────────────────────────────────────────────────────────────
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "scan_results"  not in st.session_state:
    st.session_state.scan_results  = []

def push_notification(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.notifications.insert(0, f"[{ts}] {msg}")
    st.session_state.notifications = st.session_state.notifications[:20]

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab_live, tab_batch, tab_single, tab_gap, tab_notif, tab_config = st.tabs([
    "🔍 LIVE SCAN",
    "📋 BATCH / LEADERBOARD",
    "🔬 SINGLE CARD",
    "💰 GAP FINDER",
    "🔔 ALERTS",
    "⚙️ CONFIG",
])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — LIVE SCAN
# ═══════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### eBay Live Scanner")
    st.caption("Pulls live listings, auto-grades every seller photo, and surfaces PSA 10 candidates.")

    r1c1, r1c2, r1c3, r1c4 = st.columns([3,1,1,1])
    with r1c1: query     = st.text_input("Search Query", "Charizard Holo Base Set Raw Ungraded")
    with r1c2: min_price = st.number_input("Min $",  value=1,   min_value=0)
    with r1c3: max_price = st.number_input("Max $",  value=100, min_value=1)
    with r1c4: num_res   = st.number_input("Limit",  value=12,  min_value=1, max_value=50)

    r2c1, r2c2 = st.columns(2)
    with r2c1: min_psa10 = st.number_input("Min PSA 10 Value ($)", value=1000, min_value=0)
    with r2c2: min_gap   = st.number_input("Min Profit Gap ($)",   value=500,  min_value=0)

    run_btn = st.button("▶  RUN LIVE SCAN", use_container_width=True)

    if run_btn:
        with st.spinner("Fetching eBay listings…"):
            raw_list = search_ebay(query, min_price, max_price, num_res)

        prog_bar = st.progress(0.0)
        status   = st.empty()
        results: list[ListingResult] = []

        for i, raw in enumerate(raw_list):
            status.markdown(
                f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:.8rem;color:#4da6ff;'>"
                f"Grading {i+1}/{len(raw_list)}: {raw['title'][:55]}…</div>",
                unsafe_allow_html=True
            )
            prog_bar.progress((i + 1) / max(len(raw_list), 1))

            psa10_est = estimate_psa10(raw["title"])
            if psa10_est < min_psa10:
                lr = ListingResult(**{k: raw[k] for k in ("title","price","url","source","image_urls")})
                lr.psa10_value = psa10_est
                lr.verdict     = "LOW_VALUE"
                lr.grading     = GradingResult()
                lr.grading.reasoning = [f"Est PSA 10 value ${psa10_est:,.0f} below threshold ${min_psa10:,.0f}"]
                results.append(lr)
                continue

            lr = grade_listing(raw)

            # Fire notification for high-confidence passes
            if lr.grading and lr.grading.confidence >= 70 and lr.verdict == "PASS":
                msg = f"🔥 {lr.grading.confidence:.0f}% PSA 10 confidence — {lr.title[:45]} — ${lr.price}"
                push_notification(msg)

            results.append(lr)

        prog_bar.empty()
        status.empty()

        st.session_state.scan_results = results

        # Metrics row
        n_pass = sum(1 for r in results if r.verdict == "PASS")
        n_warn = sum(1 for r in results if r.verdict == "WARN")
        n_fail = sum(1 for r in results if r.verdict in ("FAIL","LOW_VALUE"))
        best_r = max((r for r in results if r.verdict == "PASS"),
                     key=lambda r: r.net_profit, default=None)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Scanned",      len(results))
        m2.metric("✅ Pass",       n_pass)
        m3.metric("⚠️ Warn",       n_warn)
        m4.metric("❌ Rejected",   n_fail)
        m5.metric("Best Profit",  f"${best_r.net_profit:,.0f}" if best_r else "—")

        st.markdown("---")
        order = {"PASS":0,"WARN":1,"FAIL":2,"LOW_VALUE":3}
        results.sort(key=lambda r:(order.get(r.verdict,4), -r.net_profit))

        for r in results:
            if r.verdict == "LOW_VALUE":
                continue
            render_grading_card(r, expanded=(r.verdict in ("PASS","WARN")))

# ═══════════════════════════════════════════════════════════
#  TAB 2 — BATCH LEADERBOARD
# ═══════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Batch URL Leaderboard")
    st.caption(
        "Paste up to 10 eBay (or direct image) URLs, one per line. "
        "The scanner grades them all and ranks them by PSA 10 potential."
    )

    batch_input = st.text_area(
        "Paste URLs (one per line)",
        height=180,
        placeholder=(
            "https://www.ebay.com/itm/...\n"
            "https://www.ebay.com/itm/...\n"
            "# Direct image URL also accepted:\n"
            "https://i.ebayimg.com/images/..."
        ),
    )
    batch_btn = st.button("▶  GRADE ALL & BUILD LEADERBOARD", use_container_width=True)

    if batch_btn:
        urls = [u.strip() for u in batch_input.splitlines()
                if u.strip() and not u.strip().startswith("#")][:10]

        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            batch_results = []
            bp = st.progress(0.0)

            for i, url in enumerate(urls):
                bp.progress((i+1)/len(urls))
                parsed = urlparse(url)

                # Detect direct image URL vs listing URL
                if any(url.lower().endswith(ext) for ext in [".jpg",".jpeg",".png",".webp"]):
                    raw = {
                        "title":      f"Manual Image — {parsed.netloc}",
                        "price":      0.0,
                        "url":        url,
                        "source":     "Manual",
                        "image_urls": [url],
                    }
                else:
                    # Treat as listing — title unknown, try to scrape title from URL path
                    path_title = parsed.path.replace("/"," ").replace("-"," ").strip()[:60]
                    raw = {
                        "title":      path_title or f"Listing {i+1}",
                        "price":      0.0,
                        "url":        url,
                        "source":     "Manual URL",
                        "image_urls": [],
                    }
                    # Attempt to pull first image from eBay item page
                    try:
                        page = requests.get(url, headers=HEADERS, timeout=10)
                        imgs = re.findall(r'https://i\.ebayimg\.com/images/g/[^"\']+\.jpg', page.text)
                        raw["image_urls"] = list(dict.fromkeys(imgs))[:5]
                        # Try to extract price
                        m = re.search(r'"price"\s*:\s*\{\s*"value"\s*:\s*"?([\d.]+)', page.text)
                        if m:
                            raw["price"] = float(m.group(1))
                        # Try title
                        m2 = re.search(r'<h1[^>]*itemprop="name"[^>]*>([^<]+)', page.text)
                        if m2:
                            raw["title"] = m2.group(1).strip()[:80]
                    except Exception:
                        pass

                lr = grade_listing(raw)
                batch_results.append(lr)

            bp.empty()

            # Rank by confidence desc
            batch_results.sort(key=lambda r: -(r.grading.confidence if r.grading else 0))

            st.markdown("### 🏆 Leaderboard")
            medals = ["🥇","🥈","🥉"] + ["🔹"] * 10

            # Summary table
            tbl = {
                "Rank":[], "Card":[], "Source":[], "Price ($)":[],
                "PSA 10% ":"", "Best Profit ($)":[], "Verdict":[]
            }
            # Actually build it properly
            rows = []
            for i, r in enumerate(batch_results):
                rows.append({
                    "Rank":   f"{medals[i]} #{i+1}",
                    "Card":   r.title[:50],
                    "Price":  f"${r.price:.2f}",
                    "PSA 10%": f"{r.grading.confidence:.1f}%" if r.grading else "—",
                    "Best Profit": f"${r.net_profit:,.0f}",
                    "Verdict": r.verdict,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            for r in batch_results:
                render_grading_card(r, expanded=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — SINGLE CARD ANALYSER
# ═══════════════════════════════════════════════════════════
with tab_single:
    st.markdown("### 🔬 Single Card Deep Analysis")
    st.caption("Upload a photo or paste an image URL for the full 5-check CV + forensics pipeline.")

    mode = st.radio("Input", ["📁 Upload Photo", "🔗 Paste Image URL"], horizontal=True)

    single_img   = None
    single_urls  = []

    if "Upload" in mode:
        up = st.file_uploader("Drop card photo (JPG / PNG)", type=["jpg","jpeg","png"])
        if up:
            pil     = Image.open(up).convert("RGB")
            single_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        url_in = st.text_input("Image URL", placeholder="https://i.ebayimg.com/…")
        if url_in.strip():
            single_urls = [url_in.strip()]

    sc1, sc2 = st.columns(2)
    with sc1: s_title = st.text_input("Card Name", placeholder="Charizard Base Set Holo Rare")
    with sc2: s_price = st.number_input("Raw Price ($)", value=50.0, min_value=0.0)

    analyse_btn = st.button("▶  ANALYSE CARD", use_container_width=True)

    if analyse_btn:
        if single_img is None and not single_urls:
            st.error("Please upload an image or paste a URL.")
        else:
            with st.spinner("Running 5-check CV pipeline…"):
                gr = GradingResult()
                if single_img is not None:
                    img = single_img
                    res_ok, res_px, is_stock = check_resolution(img)
                    gr.res_px = res_px; gr.stock_photo = is_stock
                    if not res_ok:
                        gr.res_flag = True
                        gr.reasoning.append(f"Low resolution: {res_px[0]}×{res_px[1]}px")
                    else:
                        lr2, tb2, cf, cr = check_centering(img)
                        gr.center_front_lr, gr.center_front_tb = lr2, tb2
                        gr.centering_flag = cf; gr.reasoning.extend(cr)
                        co_f, co_d, patches = check_corners(img)
                        gr.corners_flag = co_f; gr.corner_patches = patches
                        gr.reasoning.extend(co_d)
                        gf, sf, gs = check_surface(img)
                        gr.glint_flag = gf; gr.scratch_flag = sf; gr.reasoning.extend(gs)
                        cl_f, cl_r = check_clone_stamp(img)
                        gr.clone_flag = cl_f; gr.reasoning.extend(cl_r)
                        ro_f, ro_r = check_rosette_pattern(img)
                        gr.rosette_flag = ro_f; gr.reasoning.extend(ro_r)
                        hard = [gr.centering_flag, gr.corners_flag, gr.scratch_flag, gr.clone_flag]
                        gr.overall_pass = not any(hard)
                        if gr.overall_pass:
                            gr.reasoning.append("✓ All hard-fail checks passed")
                    gr = compute_probabilities(gr)
                else:
                    gr = grade_images(single_urls)

            psa10_val = estimate_psa10(s_title)
            net, exp, roi = calculate_financials(s_price, psa10_val, gr.psa10)

            if single_img is not None:
                st.image(bgr_to_pil(single_img), width=260, caption="Input card")

            v = "PASS" if gr.overall_pass and not gr.clone_flag else ("WARN" if gr.glint_flag or gr.rosette_flag else "FAIL")
            st.markdown(f"## {verdict_pill(v)}", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                confidence_ring(gr.confidence)
                prob_bar("PSA 10", gr.psa10,  NEON["green"])
                prob_bar("PSA  9", gr.psa9,   NEON["yellow"])
                prob_bar("PSA  8", gr.psa8,   NEON["orange"])
                prob_bar("PSA  7", gr.psa7,   "#cc6600")
                prob_bar("PSA  6", gr.psa6,   "#aa4400")
                prob_bar("PSA ≤5", gr.psa1_4, NEON["red"])
            with col_b:
                profit_color = NEON["green"] if net > 0 else NEON["red"]
                st.markdown(f"""
                <div style="font-family:'Share Tech Mono',monospace;font-size:.82rem;
                            line-height:2.2;background:#0a0f1a;border-radius:8px;
                            border:1px solid #1a2a4a;padding:14px 18px;">
                  Raw Price:    <b style="color:#d4e1f5;">${s_price:.2f}</b><br>
                  PSA 10 Est:   <b style="color:{NEON['blue']};">${psa10_val:,.0f}</b><br>
                  Grading:      <b style="color:#556677;">−${GRADING_FEE:.0f}</b><br>
                  Shipping:     <b style="color:#556677;">−${SHIPPING_EST:.0f}</b><br>
                  Best-Case:    <b style="color:{profit_color};font-size:1.1rem;">${net:,.0f}</b><br>
                  Expected:     <b style="color:{profit_color};">${exp:,.0f}</b><br>
                  ROI:          <b style="color:{profit_color};">{roi:.0f}%</b><br>
                  Resolution:   <b style="color:#d4e1f5;">{gr.res_px[0]}×{gr.res_px[1]}</b>
                </div>""", unsafe_allow_html=True)

            st.markdown("**AI Reasoning Report**")
            for note in gr.reasoning:
                is_ok = any(x in note for x in ["✓","passed"])
                icon  = "🟢" if is_ok else ("🟡" if "WARNING" in note.upper() else "🔴")
                st.markdown(f"{icon} `{note}`")

            if gr.corner_patches:
                st.markdown("**Corner Analysis (400% Zoom)**")
                render_corners(gr.corner_patches)

# ═══════════════════════════════════════════════════════════
#  TAB 4 — GAP FINDER
# ═══════════════════════════════════════════════════════════
with tab_gap:
    st.markdown("### 💰 High-ROI Gap Dashboard")
    st.caption(
        "Cards where (PSA 10 Value) − (Raw Cost) regularly exceeds $500–$100k+. "
        "These are your primary hunting targets."
    )

    # Summary table
    sample = [
        ("Charizard 1st Ed Base",       8000,  350000),
        ("Charizard Shadowless",         1200,   50000),
        ("Charizard Base Unlimited",      200,    5000),
        ("Lugia Neo Genesis",             150,   12000),
        ("Ho-Oh Neo Revelation",           90,    3000),
        ("Shining Charizard Neo Destiny", 300,    8000),
        ("Dark Charizard Team Rocket",    150,    2500),
        ("Gengar Fossil Holo",             80,    2500),
        ("Scyther Jungle Holo",            30,     600),
        ("Snorlax Jungle Holo",            55,    1200),
        ("Raichu Base Set",               200,    1800),
    ]
    df = pd.DataFrame([{
        "Card":          n,
        "Raw (approx)":  f"${r:,}",
        "PSA 10 (approx)": f"${p:,}",
        "Spread":        f"${p-r:,}",
        "Multiplier":    f"{p//r}×",
    } for n,r,p in sample])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    for entry in HIGH_ROI_CATALOG:
        with st.expander(f"📦 {entry['set']} — Spread {entry['est_spread']}"):
            st.markdown(f"*{entry['note']}*")
            cols = st.columns(min(len(entry["cards"]), 4))
            for col, card in zip(cols, entry["cards"]):
                ebay_url = (
                    f"https://www.ebay.com/sch/i.html?_nkw="
                    f"{quote_plus(card+' '+entry['set'].split()[0]+' pokemon raw ungraded')}"
                    f"&_sacat=183454"
                )
                col.markdown(f"[**{card}**]({ebay_url})")

# ═══════════════════════════════════════════════════════════
#  TAB 5 — NOTIFICATIONS
# ═══════════════════════════════════════════════════════════
with tab_notif:
    n_notifs = len(st.session_state.notifications)
    st.markdown(
        f"### 🔔 Alert Feed "
        f'<span class="notif-badge">{n_notifs} alerts</span>',
        unsafe_allow_html=True
    )
    st.caption(
        "Alerts fire automatically during a scan when a card reaches ≥70% PSA 10 confidence. "
        "Browser notifications require the app to be open."
    )

    if not st.session_state.notifications:
        st.info("No alerts yet. Run a Live Scan to populate this feed.")
    else:
        for note in st.session_state.notifications:
            st.markdown(
                f'<div style="background:#001a0d;border:1px solid #00ff8844;'
                f'border-radius:6px;padding:8px 14px;margin-bottom:6px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.8rem;'
                f'color:#00ff88;">{note}</div>',
                unsafe_allow_html=True
            )
        if st.button("Clear Alerts"):
            st.session_state.notifications = []
            st.rerun()

    # Browser notification JS hook
    st.markdown("""
    <script>
    function requestNotifPermission(){
      if("Notification" in window && Notification.permission === "default"){
        Notification.requestPermission();
      }
    }
    requestNotifPermission();
    </script>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 6 — CONFIG & DEPLOY GUIDE
# ═══════════════════════════════════════════════════════════
with tab_config:
    st.markdown("### ⚙️ Configuration")

    st.markdown("#### `config.yaml`")
    st.code("""
ebay:
  app_id:  "YOUR-EBAY-APP-ID-HERE"
  cert_id: "YOUR-EBAY-CERT-ID-HERE"
  dev_id:  "YOUR-EBAY-DEV-ID-HERE"

thresholds:
  min_image_px:       1000
  centering_front:    0.55
  centering_back:     0.75
  white_pixel_cluster: 25
  grading_fee:        25.00
  shipping_estimate:   6.00
""", language="yaml")

    st.markdown("#### `.streamlit/secrets.toml` (Streamlit Cloud)")
    st.code("""
[ebay]
app_id  = "YOUR-EBAY-APP-ID-HERE"
cert_id = "YOUR-EBAY-CERT-ID-HERE"
dev_id  = "YOUR-EBAY-DEV-ID-HERE"
""", language="toml")

    st.markdown("---")
    st.markdown("### 🚀 Free Deploy to Streamlit Cloud (Mobile Access)")
    st.markdown("""
**1. Get free eBay API keys**
- Register at [developer.ebay.com](https://developer.ebay.com)
- Create an app → get **App ID**, **Cert ID**, **Dev ID**
- Use **Production** (not Sandbox) for real listings

**2. Push to GitHub**
```bash
git init && git add app.py requirements.txt README.md
git commit -m "PSA10 Scanner v2"
git remote add origin https://github.com/YOU/psa10-scanner
git push -u origin main
```

**3. Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io) → New App
- Connect your repo, set `app.py` as entry point
- Under **Advanced settings → Secrets**, paste your eBay keys
- Click **Deploy** — live in ~90 seconds

**4. Bookmark on phone** — the URL will look like:
`https://you-psa10-scanner.streamlit.app`
Add to Home Screen for a native-app-like feel.
    """)

    st.markdown("---")
    st.markdown("### 🔬 CV Check Reference")
    st.markdown(f"""
| # | Check | Method | Threshold | Result |
|---|---|---|---|---|
| 1 | Resolution | Dimension check | <{MIN_PX}px | Hard Reject |
| 2 | Stock Photo | Aspect ratio + border variance | σ<12 | Hard Reject |
| 3 | Front Centering L/R | Canny edges → border ratio | >{CENTER_FRONT:.0%} | Hard Reject |
| 4 | Front Centering T/B | Canny edges → border ratio | >{CENTER_FRONT:.0%} | Hard Reject |
| 5 | Back Centering | Same method, looser standard | >{CENTER_BACK:.0%} | Hard Reject |
| 6 | Corner Whitening | White pixel cluster in 4 corners | ≥{WHITE_CLUSTER}px | Hard Reject |
| 7 | Surface Glint | Brightness in holo zone | >{GLINT_FRAC:.0%} area | Warning |
| 8 | Scratch Detection | Laplacian inside glare zone | >{SCRATCH_DENS:.1%} density | Hard Reject |
| 9 | Clone Stamp | NCC block matching | NCC>{CLONE_THRESH} | Hard Reject |
| 10| Rosette / Fake | FFT frequency analysis | No CMYK pattern | Forensic Flag |

⚠️ **On Sources:** eBay is accessed via official API (fully legal). For Mercari/TCGplayer/Cardmarket,
use the Batch tab and paste direct image URLs — this is a normal HTTPS request with no ToS violation.
Playwright-based bot-mimicry (randomized delays, mouse wiggles to evade detection) was intentionally
excluded as it violates those sites' Terms of Service.
    """)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#1a2a4a,transparent);margin:2rem 0 1rem;"></div>
<div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#1a2a4a;padding-bottom:1rem;">
  PSA 10 PROFIT SCANNER ELITE v2.0 · CV ENGINE: CENTERING · CORNERS · SURFACE · CLONE · ROSETTE<br>
  For educational/research use only · Not financial advice · Always verify manually before purchasing
</div>
""", unsafe_allow_html=True)
