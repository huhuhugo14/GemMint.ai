"""
╔══════════════════════════════════════════════════════════════════════╗
║   POKÉMON PSA 10 PROFIT SCANNER  —  v4.0  "SUPREME ARCHITECT"       ║
║   Fix: PSA-0 bug · No-Results bug · Sharpness pre-processing        ║
║   Fix: Confidence ranges · Perspective warp · Live pricing          ║
║   Stack: Streamlit · OpenCV · NumPy · eBay Browse API               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="PSA 10 Profit Scanner v4",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import cv2
import numpy as np
import requests
import base64
import hashlib
import os
import re
import yaml
import pandas as pd
from PIL import Image, ImageFilter
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urlparse
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────
#  CONFIG  (config.yaml → st.secrets → env vars, in that order)
# ─────────────────────────────────────────────────────────────────────
def _load_cfg() -> dict:
    cfg: dict = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    try:
        sec = st.secrets.get("ebay", {})
        if sec:
            cfg.setdefault("ebay", {}).update(sec)
    except Exception:
        pass
    return cfg


CFG          = _load_cfg()
EBAY_APP_ID  = CFG.get("ebay", {}).get("app_id",  os.getenv("EBAY_APP_ID",  ""))
EBAY_CERT_ID = CFG.get("ebay", {}).get("cert_id", os.getenv("EBAY_CERT_ID", ""))

# ─────────────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MIN_PX           = 800      # LOWERED from 1000 — be more permissive on eBay thumbs
UPSCALE_TARGET   = 1200     # bilinear upscale if either dim < this
LAP_STRENGTH     = 1.2      # Laplacian sharpening blend factor
WHITE_LV         = 210      # brightness threshold → "white pixel"
GLINT_LV         = 242
GLINT_FRAC       = 0.025
SCRATCH_DENS     = 0.004
CLONE_BS         = 16
ROSETTE_FREQ     = 45
GRADING_FEE      = 25.0
SHIPPING_EST     = 6.0
CORNER_LABELS    = ["TL", "TR", "BL", "BR"]

# Centering sub-grade table (ratio = larger_border / total_border)
CENTERING_TABLE = [
    (0.530, 10.0),   # ≤ 53/47  → gem
    (0.550, 10.0),   # ≤ 55/45  → PSA 10 limit
    (0.600,  9.0),
    (0.650,  8.0),
    (0.700,  7.0),
    (0.750,  6.0),
    (1.000,  5.0),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# OpenCV annotation colours (BGR)
CV_GREEN  = (50, 220, 100)
CV_RED    = (40,  60, 250)
CV_YELLOW = (20, 210, 250)
CV_BLUE   = (230, 100, 40)
CV_FONT   = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SubGrades:
    centering : float = 10.0
    corners   : float = 10.0
    edges     : float = 10.0
    surface   : float = 10.0
    overall   : float = 10.0   # Rule of Minimum
    lr_ratio  : float = 0.50
    tb_ratio  : float = 0.50
    corner_px : list  = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class GradingResult:
    sub             : SubGrades            = field(default_factory=SubGrades)
    # Image quality
    image_quality   : str                  = "ok"   # ok | low | very_low | stock
    sharpness_score : float                = 0.0    # Laplacian variance
    # Flags
    centering_fail  : bool                 = False
    corners_fail    : bool                 = False
    scratch_fail    : bool                 = False
    glint_flag      : bool                 = False
    clone_flag      : bool                 = False
    rosette_flag    : bool                 = False
    res_fail        : bool                 = False
    res_px          : tuple                = (0, 0)
    # Perspective warp
    quad_pts        : Optional[np.ndarray] = None
    warped_img      : Optional[np.ndarray] = None
    # Annotated display images
    annotated_full  : Optional[np.ndarray] = None
    corner_patches  : list                 = field(default_factory=list)
    # Grade distribution  { 1..10 → probability % }
    grade_dist      : dict                 = field(default_factory=dict)
    # Convenience aliases
    psa10           : float                = 0.0
    psa9            : float                = 0.0
    psa8            : float                = 0.0
    psa7            : float                = 0.0
    confidence_pct  : float                = 0.0  # 0-100 how sure the model is
    grade_range     : str                  = ""   # e.g. "PSA 8–10 Potential"
    overall_pass    : bool                 = False
    reasoning       : list                 = field(default_factory=list)


@dataclass
class MarketData:
    psa10_price : float = 0.0
    source      : str   = "static"
    search_url  : str   = ""
    confidence  : str   = "low"   # low | medium | high


@dataclass
class ListingResult:
    title       : str   = ""
    price       : float = 0.0
    url         : str   = "#"
    source      : str   = "eBay"
    image_urls  : list  = field(default_factory=list)
    grading     : Optional[GradingResult] = None
    market      : Optional[MarketData]    = None
    net_profit  : float = 0.0
    exp_profit  : float = 0.0
    roi_pct     : float = 0.0
    verdict     : str   = "PENDING"


# ═══════════════════════════════════════════════════════════════════════
#  DEMO / MOCK DATA  (works with no API keys)
# ═══════════════════════════════════════════════════════════════════════

MOCK_LISTINGS = [
    {
        "title"      : "[DEMO] Charizard Base Set Holo Unlimited — Raw NM",
        "price"      : 79.99,
        "url"        : "https://www.ebay.com/sch/i.html?_nkw=charizard+base+set+holo+raw",
        "source"     : "eBay (Demo)",
        "image_urls" : ["https://images.pokemontcg.io/base1/4_hires.png"],
    },
    {
        "title"      : "[DEMO] Lugia Neo Genesis Holo — PSA Ready Raw",
        "price"      : 145.00,
        "url"        : "https://www.ebay.com/sch/i.html?_nkw=lugia+neo+genesis+holo+raw",
        "source"     : "eBay (Demo)",
        "image_urls" : ["https://images.pokemontcg.io/neo1/9_hires.png"],
    },
    {
        "title"      : "[DEMO] Shining Charizard Neo Destiny — Ungraded RAW",
        "price"      : 290.00,
        "url"        : "https://www.ebay.com/sch/i.html?_nkw=shining+charizard+neo+destiny+raw",
        "source"     : "eBay (Demo)",
        "image_urls" : ["https://images.pokemontcg.io/neo4/107_hires.png"],
    },
    {
        "title"      : "[DEMO] Gengar Fossil Holo — Lightly Played",
        "price"      : 34.99,
        "url"        : "https://www.ebay.com/sch/i.html?_nkw=gengar+fossil+holo+raw",
        "source"     : "eBay (Demo)",
        "image_urls" : ["https://images.pokemontcg.io/fossil/5_hires.png"],
    },
    {
        "title"      : "[DEMO] Blastoise Base Set Shadowless — Raw PSA Ready",
        "price"      : 650.00,
        "url"        : "https://www.ebay.com/sch/i.html?_nkw=blastoise+shadowless+raw",
        "source"     : "eBay (Demo)",
        "image_urls" : ["https://images.pokemontcg.io/base1/2_hires.png"],
    },
]

STATIC_PSA10 = {
    "charizard 1st"        : 350000,
    "charizard shadowless" :  50000,
    "charizard base"       :   5000,
    "charizard skyridge"   :  15000,
    "charizard aquapolis"  :   8000,
    "blastoise 1st"        :  25000,
    "blastoise shadowless" :   8000,
    "blastoise base"       :   1200,
    "venusaur 1st"         :  12000,
    "venusaur base"        :    900,
    "lugia neo"            :  12000,
    "ho-oh neo"            :   3000,
    "entei neo"            :   1800,
    "raikou neo"           :   1500,
    "suicune neo"          :   1200,
    "shining charizard"    :   8000,
    "shining magikarp"     :   3500,
    "shining gyarados"     :   2000,
    "dark charizard"       :   2500,
    "dark blastoise"       :   1800,
    "gengar fossil"        :   2500,
    "lapras fossil"        :    800,
    "moltres fossil"       :    700,
    "zapdos fossil"        :    700,
    "scyther jungle"       :    600,
    "clefable jungle"      :    700,
    "snorlax jungle"       :   1200,
    "vaporeon jungle"      :    900,
    "jolteon jungle"       :    800,
    "raichu base"          :   1800,
    "machamp 1st"          :   2000,
    "ninetales base"       :    900,
}

HIGH_ROI_CATALOG = [
    {"set": "Base Set (1st Edition)",  "cards": ["Charizard", "Blastoise", "Venusaur", "Raichu"],
     "spread": "$5k–$500k+", "note": "Crown jewel — any 1st Ed holo has massive PSA 10 premium"},
    {"set": "Base Set Shadowless",     "cards": ["Charizard", "Blastoise", "Venusaur"],
     "spread": "$1k–$50k",   "note": "Shadowless gem mints are extremely scarce"},
    {"set": "Base Set Unlimited",      "cards": ["Charizard", "Blastoise", "Raichu", "Machamp"],
     "spread": "$500–$5k",   "note": "High raw volume; gem copies still rare"},
    {"set": "Jungle",                  "cards": ["Scyther", "Clefable", "Snorlax", "Vaporeon"],
     "spread": "$300–$1.2k", "note": "Notorious centering issues → PSA 10 scarcity premium"},
    {"set": "Fossil",                  "cards": ["Gengar", "Lapras", "Moltres", "Zapdos"],
     "spread": "$400–$2.5k", "note": "Thick cards prone to edge dings"},
    {"set": "Team Rocket",             "cards": ["Dark Charizard", "Dark Blastoise"],
     "spread": "$800–$2.5k", "note": "Dark surface shows scratches easily"},
    {"set": "Neo Genesis",             "cards": ["Lugia", "Typhlosion", "Feraligatr"],
     "spread": "$500–$12k",  "note": "Lugia PSA 10 regularly $10k+; raw copies under $200"},
    {"set": "Neo Revelation",          "cards": ["Ho-Oh", "Entei", "Raikou", "Suicune"],
     "spread": "$600–$3k",   "note": "Legendary beasts — massive PSA 10 upside"},
    {"set": "Neo Destiny",             "cards": ["Shining Charizard", "Shining Magikarp"],
     "spread": "$1k–$8k",    "note": "Shining series = rarest WotC PSA 10s"},
    {"set": "Aquapolis / Skyridge",    "cards": ["Charizard", "Articuno", "Celebi"],
     "spread": "$2k–$15k",   "note": "e-Reader era print lines → PSA 10 ultra-rare"},
]


# ═══════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING ENGINE
#  Fixes "PSA 0" by improving image quality BEFORE grading
# ═══════════════════════════════════════════════════════════════════════

def preprocess_image(img: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Step 1 — Bilinear upscale: if either dimension < UPSCALE_TARGET,
              scale up so the shortest side reaches it.  Preserves aspect ratio.
    Step 2 — Laplacian sharpening: blend sharpened image with original
              using LAP_STRENGTH weight.  Brings out edge detail in soft eBay photos.
    Step 3 — Measure sharpness: Laplacian variance of the final image.
              < 30  → very_low quality (inconclusive grading)
              30–80 → low quality  (reduced confidence)
              > 80  → ok

    Returns (processed_img, sharpness_score, quality_label).
    """
    h, w = img.shape[:2]

    # ── Bilinear upscale ─────────────────────────────────────
    if min(h, w) < UPSCALE_TARGET:
        scale = UPSCALE_TARGET / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ── Laplacian sharpening ──────────────────────────────────
    gray_f  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap     = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    # Unsharp-mask style blend: sharpened = original + α × laplacian
    for c in range(3):
        ch          = img[:, :, c].astype(np.float32)
        sharpened_c = ch - LAP_STRENGTH * lap
        img[:, :, c]= np.clip(sharpened_c, 0, 255).astype(np.uint8)

    # ── Sharpness score (Laplacian variance on processed image) ──
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if   score < 30:  quality = "very_low"
    elif score < 80:  quality = "low"
    else:             quality = "ok"

    return img, score, quality


def is_stock_photo(img: np.ndarray) -> bool:
    """
    Heuristic: stock/official-art images tend to have a very uniform
    white or light-grey background covering the outer border pixels.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([
        gray[:12, :].flatten(),
        gray[-12:, :].flatten(),
        gray[:, :12].flatten(),
        gray[:, -12:].flatten(),
    ])
    aspect = w / max(h, 1)
    return (0.90 < aspect < 1.10) and float(border.std()) < 14.0


def hash_image(img: np.ndarray) -> str:
    """Perceptual hash for duplicate/stock-photo detection."""
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    avg   = gray.mean()
    bits  = (gray > avg).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
#  PERSPECTIVE CORRECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _order_pts(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into [TL, TR, BR, BL] order."""
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],    # TL
        pts[np.argmin(diff)], # TR
        pts[np.argmax(s)],    # BR
        pts[np.argmax(diff)], # BL
    ], dtype=np.float32)


def find_card_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Canny → dilate → findContours → largest 4-point quad.
    Falls back to bounding rect of largest contour if no quad found.
    Never returns None on a real image — always gives something usable.
    """
    h, w   = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    edges  = cv2.Canny(blur, 25, 100)
    edges  = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Absolute fallback: treat whole image as card
        return _order_pts(np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32))

    best_quad, best_area = None, 0
    min_area = h * w * 0.12

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx
            best_area = area

    if best_quad is None:
        # Use bounding rect of the largest contour
        lc       = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(lc)
        best_quad = np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]],
                              dtype=np.float32).reshape(-1, 1, 2)

    return _order_pts(best_quad)


def warp_card(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the detected card region to an upright rectangle."""
    tl, tr, br, bl = quad
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    out_w = max(out_w, 60)
    out_h = max(out_h, 60)
    dst   = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M     = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def measure_borders(warped: np.ndarray) -> tuple[int, int, int, int]:
    """
    On the warped card, scan inward from each edge using column/row
    brightness projections to locate where the white card-stock edge
    transitions to the printed colour border.
    Returns (border_left, border_right, border_top, border_bottom).
    """
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    # Use central 50% strip to avoid corner artefacts
    col_m  = gray[h//4 : 3*h//4, :].mean(axis=0).astype(float)
    row_m  = gray[:, w//4 : 3*w//4].mean(axis=1).astype(float)

    def from_left(arr, thresh=185, run=4):
        for i in range(len(arr) - run):
            if all(arr[i:i+run] < thresh):
                return max(i, 1)
        return max(len(arr) // 10, 1)

    def from_right(arr, thresh=185, run=4):
        n = len(arr)
        for i in range(n - 1, run, -1):
            if all(arr[i-run:i] < thresh):
                return max(n - i, 1)
        return max(n // 10, 1)

    bl = min(from_left(col_m),  w // 3)
    br = min(from_right(col_m), w // 3)
    bt = min(from_left(row_m),  h // 3)
    bb = min(from_right(row_m), h // 3)
    return max(bl, 1), max(br, 1), max(bt, 1), max(bb, 1)


def ratio_to_centering_subgrade(lr: float, tb: float) -> float:
    """Map worst-axis centering ratio to PSA sub-grade (1–10)."""
    worst = max(lr, tb)
    for threshold, grade in CENTERING_TABLE:
        if worst <= threshold:
            return grade
    return 5.0


def run_centering(img: np.ndarray) -> tuple:
    """
    Full centering pipeline on one image.
    Returns (lr_ratio, tb_ratio, subgrade, reasons, quad, warped).
    NEVER raises — always returns usable values.
    """
    reasons = []
    try:
        quad   = find_card_quad(img)
        warped = warp_card(img, quad)
        bl, br, bt, bb = measure_borders(warped)

        lr_total = bl + br
        tb_total = bt + bb
        lr_ratio = max(bl, br) / max(lr_total, 1)
        tb_ratio = max(bt, bb) / max(tb_total, 1)
        subgrade = ratio_to_centering_subgrade(lr_ratio, tb_ratio)

        lr_str = f"{int(max(bl,br)/lr_total*100)}/{int(min(bl,br)/lr_total*100)}"
        tb_str = f"{int(max(bt,bb)/tb_total*100)}/{int(min(bt,bb)/tb_total*100)}"

        if subgrade >= 10.0:
            reasons.append(f"✓ Centering L/R {lr_str} | T/B {tb_str} → sub-grade 10")
        else:
            psa10_note = " ← PSA 10 IMPOSSIBLE" if subgrade < 10 else ""
            reasons.append(
                f"Centering L/R {lr_str} | T/B {tb_str} → "
                f"sub-grade {subgrade:.0f}{psa10_note}"
            )
        return lr_ratio, tb_ratio, subgrade, reasons, quad, warped
    except Exception as e:
        reasons.append(f"⚠ Centering detection failed ({e}) — using estimated 8")
        return 0.50, 0.50, 8.0, reasons, None, None


# ═══════════════════════════════════════════════════════════════════════
#  CORNER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def measure_corner_white_px(img: np.ndarray, pct: float = 0.09) -> list:
    """Count white pixels in each of the 4 corner patches."""
    h, w = img.shape[:2]
    ph   = max(int(h * pct), 30)
    pw   = max(int(w * pct), 30)
    patches = [
        img[0:ph,   0:pw  ],
        img[0:ph,   w-pw:w],
        img[h-ph:h, 0:pw  ],
        img[h-ph:h, w-pw:w],
    ]
    counts = []
    for p in patches:
        if p.size == 0:
            counts.append(0); continue
        b, g, r = cv2.split(p)
        mask    = ((r.astype(int) > WHITE_LV) &
                   (g.astype(int) > WHITE_LV) &
                   (b.astype(int) > WHITE_LV))
        counts.append(int(mask.sum()))
    return counts


def corner_px_to_subgrade(counts: list) -> tuple[float, list]:
    """
    Convert worst-corner white-pixel count → PSA sub-grade.
    Even 1 white pixel drops below 5% PSA 10 probability (handled in Gaussian step).
      0 px  → 10   5–19 px → 8   50–99 px → 6
      1–4   →  9   20–49   → 7   ≥100     → 5
    """
    worst   = max(counts)
    flagged = [f"{l}:{n}px" for l, n in zip(CORNER_LABELS, counts) if n > 0]
    reasons = [f"Corner {l}: {n} white pixel(s)" for l, n in zip(CORNER_LABELS, counts) if n > 0]

    if worst == 0:
        return 10.0, ["✓ All 4 corners — zero whitening detected"]
    extra = f"({', '.join(flagged)})"
    if worst <= 4:
        return 9.0, reasons + [f"Minor whitening {extra} → sub-grade 9 — PSA 10 now <5%"]
    if worst <= 19:
        return 8.0, reasons + [f"Moderate whitening {extra} → sub-grade 8"]
    if worst <= 49:
        return 7.0, reasons + [f"Significant whitening {extra} → sub-grade 7"]
    if worst <= 99:
        return 6.0, reasons + [f"Heavy whitening {extra} → sub-grade 6"]
    return 5.0, reasons + [f"Severe whitening {extra} → sub-grade 5"]


# ═══════════════════════════════════════════════════════════════════════
#  SURFACE / GLINT / SCRATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def check_surface(img: np.ndarray) -> tuple[bool, bool, list, float]:
    """Returns (glint_flag, scratch_flag, reasons, surface_subgrade)."""
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.18), int(h * 0.72)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    holo    = img[y1:y2, x1:x2]
    if holo.size == 0:
        return False, False, ["✓ Holo zone not found — skipping surface check"], 10.0

    gray  = cv2.cvtColor(holo, cv2.COLOR_BGR2GRAY)
    glare = gray > GLINT_LV
    gfrac = float(glare.sum()) / max(gray.size, 1)

    if gfrac <= GLINT_FRAC:
        return False, False, ["✓ No glare on surface"], 10.0

    reasons = [f"Glare: {gfrac:.1%} of holo area overexposed"]
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    if glare.sum() > 0:
        dens = float((np.abs(lap)[glare] > 35).sum()) / max(int(glare.sum()), 1)
        if dens > SCRATCH_DENS:
            reasons.append(f"Scratch/print-line signal (Laplacian density {dens:.2%}) ← HARD FAIL")
            return True, True, reasons, 7.0
    reasons.append("Glare present — no scratch signal. May be lighting.")
    return True, False, reasons, 9.0


# ═══════════════════════════════════════════════════════════════════════
#  FORENSICS  (clone-stamp + rosette/fake detection)
# ═══════════════════════════════════════════════════════════════════════

def check_clone(img: np.ndarray) -> tuple[bool, list]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seen : dict = {}
    for y in range(0, h - CLONE_BS, CLONE_BS):
        for x in range(0, w - CLONE_BS, CLONE_BS):
            blk = gray[y:y+CLONE_BS, x:x+CLONE_BS].astype(np.float32)
            mn, sd = blk.mean(), blk.std()
            if sd < 5: continue
            key = hashlib.md5(((blk - mn) / sd).tobytes()).hexdigest()
            if key in seen:
                sy, sx = seen[key]
                if ((y-sy)**2 + (x-sx)**2)**0.5 > CLONE_BS * 3:
                    return True, [
                        f"Clone-stamp editing detected at ({sx},{sy})↔({x},{y}) ← FORENSIC FAIL"
                    ]
            else:
                seen[key] = (y, x)
    return False, []


def check_rosette(img: np.ndarray) -> tuple[bool, list]:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    cy, cx = h // 2, w // 2
    crop  = gray[cy-80:cy+80, cx-80:cx+80].astype(np.float32)
    if crop.shape[0] < 60:
        return False, []
    mag   = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(crop))) + 1)
    ch, cw = mag.shape
    f     = ROSETTE_FREQ
    ring  = mag[ch//2-f-5:ch//2-f+5, cw//2-f-5:cw//2-f+5]
    peak  = float(ring.max()) if ring.size > 0 else 0.0
    base  = float(mag[:20, :20].mean())
    if peak < base * 1.6:
        return True, ["No CMYK rosette (FFT) — possible inkjet counterfeit ← FLAG"]
    return False, []


# ═══════════════════════════════════════════════════════════════════════
#  GRADE DISTRIBUTION  (Gaussian + Rule of Minimum + hard overrides)
#  "PSA 0 Fix" — always returns a meaningful distribution
# ═══════════════════════════════════════════════════════════════════════

def compute_grade_distribution(
    sub: SubGrades,
    quality: str
) -> tuple[dict, float, str]:
    """
    Returns (grade_dist, confidence_pct, grade_range_str).

    Key design decisions that fix the "PSA 0" bug:
    ─────────────────────────────────────────────
    1. Low-quality images reduce σ (wider spread) and reduce confidence_pct
       but NEVER collapse every grade to 0.
    2. Rule of Minimum: Gaussian centred on min(all sub-grades).
    3. Hard overrides keep PSA 10 = 0% on centering fail and < 5% on
       any corner whitening, but the mass moves into grades 7–9, not vanishes.
    4. grade_range_str gives a human label even in the inconclusive case.
    """
    # Quality modulates the distribution width
    sigma_map = {"ok": 0.70, "low": 1.10, "very_low": 1.50}
    sigma     = sigma_map.get(quality, 1.10)

    overall   = min(sub.centering, sub.corners, sub.edges, sub.surface)
    sub.overall = overall

    grades    = np.arange(1, 11, dtype=float)
    weights   = np.exp(-0.5 * ((grades - overall) / sigma) ** 2)
    weights  /= weights.sum()
    dist      = {int(g): round(float(p * 100), 1) for g, p in zip(grades, weights)}

    # ── Hard rule 1: centering fail → PSA 10 = 0% ────────────
    if sub.centering < 10.0:
        lost     = dist.get(10, 0.0)
        dist[10] = 0.0
        dist[9]  = round(dist.get(9, 0.0) + lost * 0.60, 1)
        dist[8]  = round(dist.get(8, 0.0) + lost * 0.40, 1)

    # ── Hard rule 2: corner whitening → PSA 10 < 5% ──────────
    if sub.corners < 10.0 and dist.get(10, 0.0) > 4.9:
        excess   = dist[10] - 4.9
        dist[10] = 4.9
        dist[9]  = round(dist.get(9, 0.0) + excess * 0.70, 1)
        dist[8]  = round(dist.get(8, 0.0) + excess * 0.30, 1)

    # ── Normalise ─────────────────────────────────────────────
    total = sum(dist.values())
    if total > 0:
        dist = {k: round(v / total * 100, 1) for k, v in dist.items()}

    # ── Confidence score (0–100) ──────────────────────────────
    # Confidence reflects image quality + number of flags
    n_flags = sum([
        sub.centering < 10,
        sub.corners   < 10,
        sub.edges     <  9,
        sub.surface   <  9,
    ])
    conf_base = {"ok": 88, "low": 60, "very_low": 38}.get(quality, 60)
    conf_pct  = max(conf_base - n_flags * 10, 15)

    # ── Grade range label ─────────────────────────────────────
    if overall >= 10.0:
        grade_range = "PSA 10 Potential"
    elif overall >= 9.0:
        grade_range = "PSA 9–10 Potential"
    elif overall >= 8.0:
        grade_range = "PSA 8–9 Likely"
    elif overall >= 7.0:
        grade_range = "PSA 7–8 Likely"
    else:
        grade_range = f"PSA {max(int(overall)-1, 1)}–{int(overall)} Likely"

    if quality == "very_low":
        grade_range = f"Est. {grade_range} (low-res photo)"

    return dist, conf_pct, grade_range


# ═══════════════════════════════════════════════════════════════════════
#  VISUAL ANNOTATION ENGINE  (Fix 4)
# ═══════════════════════════════════════════════════════════════════════

def annotate_full(
    img: np.ndarray,
    quad: Optional[np.ndarray],
    lr: float, tb: float,
    cen_sg: float,
    white_counts: list,
    glint: bool,
) -> np.ndarray:
    """Draw card outline, corner boxes, holo zone, and centering text on image."""
    out  = img.copy()
    h, w = out.shape[:2]

    # Card outline
    if quad is not None:
        col = CV_GREEN if cen_sg >= 10.0 else CV_RED
        pts = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=col, thickness=3)
        for pt, lbl in zip(quad.astype(int), ["TL", "TR", "BR", "BL"]):
            cv2.circle(out, tuple(pt), 6, col, -1)
            cv2.putText(out, lbl, (pt[0]+5, pt[1]-5),
                        CV_FONT, 0.45, col, 1, cv2.LINE_AA)

    # Corner bounding boxes
    ph  = max(int(h * 0.09), 30)
    pw  = max(int(w * 0.09), 30)
    origins = [(0,0),(w-pw,0),(0,h-ph),(w-pw,h-ph)]
    for (cx, cy), lbl, n in zip(origins, CORNER_LABELS, white_counts):
        col = CV_RED if n > 0 else CV_GREEN
        cv2.rectangle(out, (cx, cy), (cx+pw, cy+ph), col, 2)
        # Subtle fill
        ov = out.copy()
        cv2.rectangle(ov, (cx, cy), (cx+pw, cy+ph), col, -1)
        cv2.addWeighted(ov, 0.09, out, 0.91, 0, out)
        cv2.putText(out, f"{lbl} {n}px", (cx+3, cy+14),
                    CV_FONT, 0.40, col, 1, cv2.LINE_AA)

    # Holo zone overlay
    if glint:
        y1 = int(h*0.18); y2 = int(h*0.72)
        x1 = int(w*0.08); x2 = int(w*0.92)
        ov = out.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), CV_YELLOW, -1)
        cv2.addWeighted(ov, 0.16, out, 0.84, 0, out)
        cv2.rectangle(out, (x1,y1), (x2,y2), CV_YELLOW, 2)
        cv2.putText(out, "SURFACE GLINT", (x1+6,y1+17),
                    CV_FONT, 0.48, CV_YELLOW, 1, cv2.LINE_AA)

    # Centering strip
    cen_col = CV_GREEN if cen_sg >= 10.0 else CV_RED
    cv2.putText(out,
                f"CTR L/R {lr:.0%}  T/B {tb:.0%}  sub={cen_sg:.0f}",
                (8, h-10), CV_FONT, 0.44, cen_col, 1, cv2.LINE_AA)
    return out


def make_corner_patches(img: np.ndarray, white_counts: list) -> list:
    """400%-zoom corner patches with white-pixel highlights and coloured border."""
    h, w = img.shape[:2]
    ph   = max(int(h*0.09), 30)
    pw   = max(int(w*0.09), 30)
    raw_patches = [
        img[0:ph,   0:pw  ],
        img[0:ph,   w-pw:w],
        img[h-ph:h, 0:pw  ],
        img[h-ph:h, w-pw:w],
    ]
    out = []
    for patch, lbl, n in zip(raw_patches, CORNER_LABELS, white_counts):
        if patch.size == 0:
            out.append(np.zeros((80, 80, 3), np.uint8)); continue
        disp = patch.copy()
        # Apply Laplacian sharpening to the corner patch too
        for c in range(3):
            ch = disp[:,:,c].astype(np.float32)
            lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=3)
            disp[:,:,c] = np.clip(ch - LAP_STRENGTH * lap, 0, 255).astype(np.uint8)
        # Highlight white pixels
        bch, gch, rch = cv2.split(disp)
        wm = ((rch.astype(int)>WHITE_LV)&(gch.astype(int)>WHITE_LV)&(bch.astype(int)>WHITE_LV))
        disp[wm] = [0, 0, 255] if n > 0 else [0, 200, 80]
        # 4× upscale
        scale = 4
        disp  = cv2.resize(disp,
                           (disp.shape[1]*scale, disp.shape[0]*scale),
                           interpolation=cv2.INTER_NEAREST)
        col = CV_RED if n > 0 else CV_GREEN
        cv2.rectangle(disp, (0,0), (disp.shape[1]-1, disp.shape[0]-1), col, 3)
        cv2.putText(disp, f"{lbl} {n}px", (4,15), CV_FONT, 0.5, col, 1, cv2.LINE_AA)
        out.append(disp)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE FETCH + CACHE
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def fetch_image(url: str) -> Optional[np.ndarray]:
    data = _fetch_bytes(url)
    if data is None:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ═══════════════════════════════════════════════════════════════════════
#  FULL GRADING PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def grade_images(image_urls: list) -> GradingResult:
    """
    Run the complete grading pipeline across up to 6 images.
    Always returns a non-zero, meaningful GradingResult.
    """
    gr = GradingResult()

    if not image_urls:
        gr.res_fail  = True
        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        gr.reasoning.append("No images in listing")
        gr.grade_range   = "Unknown — no images"
        gr.confidence_pct = 0
        return gr

    # Download images
    raw_images = []
    seen_hashes: set = set()
    for url in image_urls[:6]:
        img = fetch_image(url)
        if img is None:
            continue
        h = hash_image(img)
        if h in seen_hashes:
            gr.reasoning.append(f"Duplicate/stock photo filtered: {url[:60]}")
            continue
        seen_hashes.add(h)
        raw_images.append(img)

    if not raw_images:
        gr.res_fail = True
        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        gr.reasoning.append("All image downloads failed or filtered as stock photos")
        gr.grade_range    = "Unknown — no usable images"
        gr.confidence_pct = 0
        return gr

    # ── Pre-process every image (upscale + sharpen) ───────────
    images = []
    for img in raw_images:
        proc, score, quality = preprocess_image(img)
        images.append((proc, score, quality))

    # Resolution check on first processed image
    h0, w0, _ = images[0][0].shape
    gr.res_px  = (w0, h0)
    if min(h0, w0) < MIN_PX:
        gr.res_fail = True
        gr.reasoning.append(
            f"REJECTED: {w0}×{h0}px even after upscale — image unusable"
        )
        gr.grade_dist     = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        gr.grade_range    = "Unknown — unusable image"
        gr.confidence_pct = 0
        return gr

    # Stock-photo check
    if is_stock_photo(images[0][0]):
        gr.image_quality = "stock"
        gr.reasoning.append("⚠ Image appears to be official/stock art — centering may be inaccurate")

    # Overall quality = worst across images
    quality_rank = {"ok": 0, "low": 1, "very_low": 2}
    worst_quality = max(images, key=lambda x: quality_rank.get(x[2], 0))[2]
    gr.image_quality    = worst_quality
    gr.sharpness_score  = min(x[1] for x in images)

    if worst_quality == "very_low":
        gr.reasoning.append(
            f"⚠ Low sharpness score ({gr.sharpness_score:.1f}) — "
            "confidence reduced; grade range widened"
        )

    sub = SubGrades()

    # Accumulate worst-case across all images
    worst_lr, worst_tb  = 0.50, 0.50
    worst_cen_sg        = 10.0
    worst_cor_sg        = 10.0
    worst_cor_counts    = [0, 0, 0, 0]
    worst_srf_sg        = 10.0
    best_quad           = None
    best_warp           = None

    for idx, (img, sharpness, quality) in enumerate(images):
        lbl = f"Img {idx+1}"

        # Centering
        lr, tb, cen_sg, cen_r, quad, warped = run_centering(img)
        if cen_sg < worst_cen_sg:
            worst_cen_sg       = cen_sg
            worst_lr, worst_tb = lr, tb
            best_quad          = quad
            best_warp          = warped
        gr.reasoning.extend(f"{lbl}: {r}" for r in cen_r)

        # Corners
        counts  = measure_corner_white_px(img)
        cor_sg, cor_r = corner_px_to_subgrade(counts)
        worst_cor_counts = [max(a, b) for a, b in zip(worst_cor_counts, counts)]
        if cor_sg < worst_cor_sg:
            worst_cor_sg = cor_sg
        gr.reasoning.extend(f"{lbl}: {r}" for r in cor_r)

        # Surface
        glint, scratch, srf_r, srf_sg = check_surface(img)
        if srf_sg < worst_srf_sg:
            worst_srf_sg = srf_sg
        if glint:   gr.glint_flag   = True
        if scratch: gr.scratch_fail = True
        gr.reasoning.extend(f"{lbl}: {r}" for r in srf_r)

        # Forensics — first image only
        if idx == 0:
            cl_f, cl_r = check_clone(img)
            if cl_f: gr.clone_flag = True; gr.reasoning.extend(cl_r)
            ro_f, ro_r = check_rosette(img)
            if ro_f: gr.rosette_flag = True; gr.reasoning.extend(ro_r)

    # Populate SubGrades
    sub.centering       = worst_cen_sg
    sub.corners         = worst_cor_sg
    sub.edges           = worst_srf_sg
    sub.surface         = worst_srf_sg
    sub.lr_ratio        = worst_lr
    sub.tb_ratio        = worst_tb
    sub.corner_white_px = worst_cor_counts
    if gr.clone_flag:    sub.surface = min(sub.surface, 5.0)
    if gr.rosette_flag:  sub.surface = min(sub.surface, 7.0)
    gr.sub              = sub

    # Flags
    gr.centering_fail = sub.centering < 10.0
    gr.corners_fail   = sub.corners   < 10.0

    # ── Gaussian distribution (PSA 0 fix) ────────────────────
    gr.grade_dist, gr.confidence_pct, gr.grade_range = compute_grade_distribution(
        sub, worst_quality
    )
    gr.psa10 = gr.grade_dist.get(10, 0.0)
    gr.psa9  = gr.grade_dist.get(9,  0.0)
    gr.psa8  = gr.grade_dist.get(8,  0.0)
    gr.psa7  = gr.grade_dist.get(7,  0.0)

    # ── Annotated images ─────────────────────────────────────
    if images:
        first_img = images[0][0]
        gr.annotated_full = annotate_full(
            first_img, best_quad, worst_lr, worst_tb,
            worst_cen_sg, worst_cor_counts, gr.glint_flag
        )
        gr.corner_patches = make_corner_patches(first_img, worst_cor_counts)
        gr.quad_pts  = best_quad
        gr.warped_img = best_warp

    # ── Overall verdict ───────────────────────────────────────
    hard = [gr.centering_fail, gr.corners_fail, gr.scratch_fail,
            gr.res_fail, gr.clone_flag]
    gr.overall_pass = not any(hard)
    if gr.overall_pass:
        gr.reasoning.insert(0, f"✓ No hard-fail conditions — {gr.grade_range}")
    return gr


# ═══════════════════════════════════════════════════════════════════════
#  MARKET PRICING  (3-tier, never $0)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _ebay_token() -> Optional[str]:
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        return None
    creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
    try:
        r = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Authorization": f"Basic {creds}"},
            data={"grant_type": "client_credentials",
                  "scope": "https://api.ebay.com/oauth/api_scope"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _tier1_ebay(card_name: str) -> MarketData:
    """eBay Browse API — search graded-condition listings for median ask price."""
    token = _ebay_token()
    if not token:
        return MarketData()
    q    = f"{card_name} PSA 10"
    surl = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(q)}&LH_Sold=1&LH_Complete=1&_sacat=183454"
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization": f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"},
            params={"q": q, "category_ids": "183454",
                    "filter": "conditionIds:{2750}",
                    "sort": "price", "limit": 10},
            timeout=12,
        )
        r.raise_for_status()
        prices = [float(i["price"]["value"])
                  for i in r.json().get("itemSummaries", [])
                  if i.get("price", {}).get("value")]
        if prices:
            return MarketData(psa10_price=round(float(np.median(prices)), 2),
                              source="eBay Live (graded ask)",
                              search_url=surl, confidence="high")
    except Exception:
        pass
    return MarketData()


@st.cache_data(ttl=1800, show_spinner=False)
def _tier2_pricecharting(card_name: str) -> MarketData:
    """PriceCharting API — grade-10-price field (stored in cents)."""
    surl = (f"https://www.ebay.com/sch/i.html?_nkw="
            f"{quote_plus(card_name+' PSA 10')}&LH_Sold=1&_sacat=183454")
    try:
        r = requests.get(
            "https://www.pricecharting.com/api/product",
            params={"q": card_name, "status": "price"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if r.status_code == 200:
            data     = r.json()
            products = data.get("products", [data] if "graded-price" in data else [])
            for prod in products[:3]:
                cents = (prod.get("grade-10-price")
                         or prod.get("graded-price")
                         or prod.get("complete-price", 0))
                val = float(cents) / 100.0 if cents else 0.0
                if val > 50:
                    return MarketData(psa10_price=round(val, 2),
                                      source="PriceCharting API",
                                      search_url=surl, confidence="medium")
    except Exception:
        pass
    return MarketData()


def _tier3_static(title: str) -> MarketData:
    """Static reference table — always returns a non-zero value."""
    tl  = title.lower()
    val = 0.0
    for key, v in sorted(STATIC_PSA10.items(), key=lambda x: -len(x[0])):
        if all(k in tl for k in key.split()):
            val = float(v); break
    if val == 0:
        if "1st edition" in tl or "1st ed" in tl: val = 2000.0
        elif "shadowless" in tl:                   val = 1500.0
        elif "shining"    in tl:                   val = 1500.0
        elif any(s in tl for s in ["lugia","ho-oh","entei","raikou","suicune"]): val = 1200.0
        elif "holo" in tl and any(s in tl for s in [
            "base","jungle","fossil","rocket","neo","expedition","skyridge","aquapolis"]):
            val = 700.0
        else: val = 400.0
    surl = (f"https://www.ebay.com/sch/i.html?_nkw="
            f"{quote_plus(tl[:40]+' PSA 10')}&LH_Sold=1&_sacat=183454")
    return MarketData(psa10_price=val, source="Reference Table",
                      search_url=surl, confidence="low")


def get_market_data(title: str) -> MarketData:
    """Waterfall: eBay → PriceCharting → static table. Never returns $0."""
    clean = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    clean = re.sub(r'(raw|ungraded|nm|lp|mp|hp|lot|bundle|free\s*ship\w*)',
                   '', clean, flags=re.I).strip()[:60]
    md = _tier1_ebay(clean)
    if md.psa10_price > 0: return md
    md = _tier2_pricecharting(clean)
    if md.psa10_price > 0: return md
    return _tier3_static(title)


def financials(price: float, md: MarketData, p10: float) -> tuple:
    costs     = price + GRADING_FEE + SHIPPING_EST
    best_case = md.psa10_price - costs
    expected  = (md.psa10_price * p10 / 100.0) - costs
    roi_pct   = (best_case / max(price, 0.01)) * 100
    return round(best_case, 2), round(expected, 2), round(roi_pct, 1)


# ═══════════════════════════════════════════════════════════════════════
#  eBay SEARCH
# ═══════════════════════════════════════════════════════════════════════

def search_ebay(query: str, min_p: float, max_p: float, limit: int) -> list:
    """
    Structured search: '{query} raw -graded -proxy -replica'
    Falls back to demo data if no API credentials found.
    """
    token = _ebay_token()
    if not token:
        st.info("ℹ️ No eBay API credentials — running in **Demo Mode**.")
        return MOCK_LISTINGS[:limit]
    structured_q = f"{query} raw -graded -proxy -replica -reprint"
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization": f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"},
            params={
                "q"            : structured_q,
                "category_ids" : "183454",
                "filter"       : f"price:[{min_p}..{max_p}],priceCurrency:USD",
                "sort"         : "price",
                "limit"        : limit,
                "fieldgroups"  : "EXTENDED",
            },
            timeout=15,
        )
        r.raise_for_status()
        out = []
        for item in r.json().get("itemSummaries", []):
            imgs  = [item.get("image", {}).get("imageUrl", "")]
            imgs += [i.get("imageUrl", "") for i in item.get("additionalImages", [])]
            out.append({
                "title"      : item.get("title", ""),
                "price"      : float(item.get("price", {}).get("value", 0)),
                "url"        : item.get("itemWebUrl", "#"),
                "source"     : "eBay",
                "image_urls" : [u for u in imgs if u],
            })
        return out if out else MOCK_LISTINGS[:limit]
    except Exception as e:
        st.warning(f"eBay API error: {e} — using demo data")
        return MOCK_LISTINGS[:limit]


def grade_listing(raw: dict) -> ListingResult:
    lr         = ListingResult(**{k: raw[k] for k in
                                  ("title","price","url","source","image_urls")})
    lr.grading = grade_images(lr.image_urls)
    lr.market  = get_market_data(lr.title)
    lr.net_profit, lr.exp_profit, lr.roi_pct = financials(
        lr.price, lr.market, lr.grading.psa10
    )
    hard = [lr.grading.centering_fail, lr.grading.corners_fail,
            lr.grading.scratch_fail,   lr.grading.res_fail, lr.grading.clone_flag]
    lr.verdict = ("FAIL" if any(hard)
                  else "WARN" if (lr.grading.glint_flag or lr.grading.rosette_flag)
                  else "PASS")
    return lr


# ═══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════
NEON = {"green":"#00ff88","yellow":"#f5c518","orange":"#ff7c00",
        "red":"#ff3c3c","blue":"#4da6ff","purple":"#b07dff","dim":"#2a3a5a"}


def _gc(pct: float) -> str:
    if pct >= 70: return NEON["green"]
    if pct >= 40: return NEON["yellow"]
    if pct >= 15: return NEON["orange"]
    return NEON["red"]


def prob_bar(label: str, pct: float, color: str):
    st.markdown(f"""
    <div style="margin:3px 0">
      <div style="display:flex;justify-content:space-between;
                  font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#7a8fbb;">
        <span>{label}</span><span style="color:{color};">{pct:.1f}%</span>
      </div>
      <div style="background:#111827;border-radius:3px;height:10px;overflow:hidden;">
        <div style="width:{min(pct,100):.1f}%;height:100%;
                    background:linear-gradient(90deg,{color}88,{color});border-radius:3px;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def subgrade_bar(label: str, sg: float):
    color = _gc(sg * 10)
    st.markdown(f"""
    <div style="margin:3px 0">
      <div style="display:flex;justify-content:space-between;
                  font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#7a8fbb;">
        <span>{label}</span>
        <span style="color:{color};font-weight:bold;">{sg:.0f}/10</span>
      </div>
      <div style="background:#111827;border-radius:3px;height:10px;overflow:hidden;">
        <div style="width:{sg*10:.1f}%;height:100%;
                    background:linear-gradient(90deg,{color}88,{color});border-radius:3px;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def confidence_ring(pct: float, conf: float):
    """
    Two-ring SVG: outer = PSA 10 probability, inner = model confidence.
    """
    outer_col = _gc(pct)
    inner_col = _gc(conf)
    R1, R2 = 40, 28
    S1, S2 = 8, 7
    C1 = 2 * 3.14159 * R1
    C2 = 2 * 3.14159 * R2
    O1 = C1 * (1 - pct  / 100)
    O2 = C2 * (1 - conf / 100)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin:6px 0;">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r="{R1}" fill="none" stroke="#111827" stroke-width="{S1}"/>
        <circle cx="55" cy="55" r="{R1}" fill="none" stroke="{outer_col}" stroke-width="{S1}"
          stroke-dasharray="{C1:.2f}" stroke-dashoffset="{O1:.2f}"
          stroke-linecap="round" transform="rotate(-90 55 55)"/>
        <circle cx="55" cy="55" r="{R2}" fill="none" stroke="#1a2a4a" stroke-width="{S2}"/>
        <circle cx="55" cy="55" r="{R2}" fill="none" stroke="{inner_col}" stroke-width="{S2}"
          stroke-dasharray="{C2:.2f}" stroke-dashoffset="{O2:.2f}"
          stroke-linecap="round" transform="rotate(-90 55 55)"/>
        <text x="55" y="50" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="14"
          fill="{outer_col}" font-weight="bold">{pct:.0f}%</text>
        <text x="55" y="63" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="7"
          fill="#556677">PSA 10</text>
        <text x="55" y="73" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="6"
          fill="{inner_col}">conf {conf:.0f}%</text>
      </svg>
    </div>""", unsafe_allow_html=True)


def verdict_pill(v: str) -> str:
    cfg = {"PASS": ("#002d1a", NEON["green"],  "✅ PASS"),
           "WARN": ("#2d1e00", NEON["yellow"], "⚠ WARN"),
           "FAIL": ("#2d0000", NEON["red"],    "❌ FAIL")}
    bg, col, txt = cfg.get(v, ("#111","#aaa","⬜"))
    return (f'<span style="background:{bg};color:{col};border:1px solid {col};'
            f'border-radius:20px;padding:3px 14px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:.73rem;'
            f'font-weight:bold;">{txt}</span>')


def quality_badge(quality: str, sharpness: float) -> str:
    cfg = {
        "ok"       : ("#002020", NEON["green"],  "Sharp"),
        "low"      : ("#202000", NEON["yellow"], "Soft"),
        "very_low" : ("#200000", NEON["red"],    "Low-Res"),
        "stock"    : ("#100020", NEON["purple"], "Stock Photo"),
    }
    bg, col, lbl = cfg.get(quality, ("#111","#aaa","?"))
    return (f'<span style="background:{bg};color:{col};border:1px solid {col};'
            f'border-radius:10px;padding:1px 8px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:.63rem;">'
            f'{lbl} (σ²={sharpness:.0f})</span>')


def src_badge(src: str, conf: str) -> str:
    col = {"high": NEON["green"], "medium": NEON["yellow"],
           "low": "#8899bb"}.get(conf, "#8899bb")
    return (f'<span style="background:{col}22;color:{col};border:1px solid {col};'
            f'border-radius:10px;padding:1px 8px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:.63rem;">'
            f'{src}</span>')


def render_corner_panels(patches: list):
    if len(patches) < 4: return
    cols = st.columns(4)
    for col, patch, lbl in zip(cols, patches[:4], CORNER_LABELS):
        with col:
            if patch.size > 0:
                col.image(bgr_to_pil(patch), caption=f"{lbl} 400%",
                          use_container_width=True)


def render_subgrades(sub: SubGrades):
    rc = _gc(sub.overall * 10)
    st.markdown(
        f'<div style="background:#090e1a;border:1px solid #1a2a4a;'
        f'border-radius:8px;padding:7px 12px;margin:4px 0;'
        f'font-family:\'Share Tech Mono\',monospace;font-size:.67rem;color:#4a7fc1;">'
        f'SUB-GRADES — Rule of Minimum → '
        f'<b style="color:{rc};">Overall {sub.overall:.0f}/10</b></div>',
        unsafe_allow_html=True)
    subgrade_bar("Centering",     sub.centering)
    subgrade_bar("Corners",       sub.corners)
    subgrade_bar("Edges/Surface", sub.edges)
    subgrade_bar("★ Overall",     sub.overall)


def render_market(md: MarketData, price: float, net: float, exp: float, roi: float):
    pc    = NEON["green"] if net > 0 else NEON["red"]
    badge = src_badge(md.source, md.confidence)
    link  = (f'<a href="{md.search_url}" target="_blank" '
             f'style="color:{NEON["blue"]};font-size:.63rem;margin-left:6px;">'
             f'🔗 PSA 10 sold listings</a>') if md.search_url else ""
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.77rem;line-height:2.1;
                background:#090e1a;border-radius:8px;border:1px solid #1a2a4a;padding:10px 14px;">
      <div style="margin-bottom:5px;">{badge}{link}</div>
      <span style="color:#556677;">Raw Price  </span><b style="color:#d4e1f5;">${price:.2f}</b><br>
      <span style="color:#556677;">PSA 10 Est </span>
        <b style="color:{NEON['blue']};font-size:.9rem;">${md.psa10_price:,.0f}</b><br>
      <span style="color:#556677;">Grading    </span><b style="color:#556677;">−${GRADING_FEE:.0f}</b><br>
      <span style="color:#556677;">Shipping   </span><b style="color:#556677;">−${SHIPPING_EST:.0f}</b><br>
      <span style="color:#556677;">Best-Case  </span>
        <b style="color:{pc};font-size:.95rem;">${net:,.0f}</b><br>
      <span style="color:#556677;">Expected   </span><b style="color:{pc};">${exp:,.0f}</b><br>
      <span style="color:#556677;">ROI        </span><b style="color:{pc};">{roi:.0f}%</b>
    </div>""", unsafe_allow_html=True)


def render_grading_card(r: ListingResult, expanded: bool = False):
    gr  = r.grading
    v   = r.verdict
    lbl = (f"{'✅' if v=='PASS' else '⚠️' if v=='WARN' else '❌'} "
           f"{r.title[:60]}  — ${r.price:.2f}")

    with st.expander(lbl, expanded=expanded):
        c1, c2, c3 = st.columns([1.5, 1, 1.4])

        with c1:
            if gr and gr.annotated_full is not None:
                st.image(bgr_to_pil(gr.annotated_full),
                         use_container_width=True,
                         caption="AI-annotated — green=pass · red=fail · yellow=glint")
            elif r.image_urls:
                st.image(r.image_urls[0], use_container_width=True)

            st.markdown(
                f'{verdict_pill(v)}&nbsp;&nbsp;'
                f'{quality_badge(gr.image_quality, gr.sharpness_score) if gr else ""}<br>'
                f'<a href="{r.url}" target="_blank" style="display:inline-block;'
                f'margin-top:8px;background:{NEON["blue"]}22;border:1px solid {NEON["blue"]};'
                f'color:{NEON["blue"]};padding:4px 12px;border-radius:6px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
                f'text-decoration:none;">🛒 ONE-TAP BUY</a>',
                unsafe_allow_html=True)

        with c2:
            if gr:
                confidence_ring(gr.psa10, gr.confidence_pct)
                # Grade range label
                gr_col = _gc(gr.psa10)
                st.markdown(
                    f'<div style="text-align:center;font-family:\'Share Tech Mono\','
                    f'monospace;font-size:.67rem;color:{gr_col};">'
                    f'{gr.grade_range}</div>',
                    unsafe_allow_html=True)
                render_subgrades(gr.sub)

        with c3:
            if r.market:
                render_market(r.market, r.price, r.net_profit, r.exp_profit, r.roi_pct)

        # Grade distribution
        if gr:
            d = gr.grade_dist
            st.markdown("**Grade Probability Distribution**  "
                        "*(Gaussian · Rule of Minimum applied)*")
            ca, cb = st.columns(2)
            with ca:
                prob_bar("PSA 10", d.get(10,0), NEON["green"])
                prob_bar("PSA  9", d.get(9, 0), NEON["yellow"])
                prob_bar("PSA  8", d.get(8, 0), NEON["orange"])
                prob_bar("PSA  7", d.get(7, 0), "#cc6600")
            with cb:
                prob_bar("PSA  6", d.get(6, 0), "#aa4400")
                prob_bar("PSA  5", d.get(5, 0), "#884400")
                prob_bar("PSA 3-4", sum(d.get(g,0) for g in (3,4)), NEON["red"])
                prob_bar("PSA 1-2", sum(d.get(g,0) for g in (1,2)), "#660000")

            if gr.corner_patches:
                st.markdown(
                    "**4-Corner Zoom (400%)**  `red = whitening · green = clean`")
                render_corner_panels(gr.corner_patches)

            if gr.warped_img is not None:
                with st.expander("🔲 Perspective-corrected card"):
                    st.image(bgr_to_pil(gr.warped_img), use_container_width=True)

            if gr.reasoning:
                st.markdown("**AI Reasoning Report**")
                for note in gr.reasoning:
                    ok   = note.startswith("✓")
                    warn = "⚠" in note or "WARNING" in note.upper()
                    icon = "🟢" if ok else ("🟡" if warn else "🔴")
                    st.markdown(f"{icon} `{note}`")


# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;800;900&display=swap');
html,body,[class*="css"]{background:#06080d!important;color:#c0d4f0!important;}
.stApp{background:#06080d;}
h1,h2,h3,h4{font-family:'Barlow Condensed',sans-serif!important;letter-spacing:.06em;}
.stTabs [data-baseweb="tab-list"]{background:#0a0d18;border-bottom:1px solid #182038;gap:4px;}
.stTabs [data-baseweb="tab"]{font-family:'Barlow Condensed',sans-serif;font-size:.95rem;
  font-weight:600;color:#304060;letter-spacing:.06em;border-radius:6px 6px 0 0;padding:7px 16px;}
.stTabs [aria-selected="true"]{color:#00ff88!important;background:#001a0a!important;
  border-bottom:2px solid #00ff88;}
.stButton>button{background:linear-gradient(135deg,#1860ee,#0040bb);color:#fff;border:none;
  border-radius:8px;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;
  letter-spacing:.08em;padding:.5rem 1.2rem;width:100%;
  box-shadow:0 0 16px #1860ee33;transition:all .18s;}
.stButton>button:hover{background:linear-gradient(135deg,#3880ff,#1860ee);
  box-shadow:0 0 26px #1860ee66;transform:translateY(-1px);}
.stTextInput>div>div>input,.stNumberInput>div>div>input{
  background:#0a0d18!important;color:#c0d4f0!important;
  border:1px solid #182038!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;}
.stTextArea>div>textarea{background:#0a0d18!important;color:#c0d4f0!important;
  border:1px solid #182038!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;}
.streamlit-expanderHeader{background:#0a0d18!important;border:1px solid #182038!important;
  border-radius:8px!important;font-family:'Barlow Condensed',sans-serif!important;
  font-size:.95rem!important;}
.streamlit-expanderContent{background:#06080d!important;border:1px solid #182038!important;
  border-top:none!important;border-radius:0 0 8px 8px!important;padding:10px 14px!important;}
[data-testid="stMetric"]{background:#0a0d18;border:1px solid #182038;
  border-radius:10px;padding:10px;}
[data-testid="stSidebar"]{background:#080b16!important;border-right:1px solid #182038;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#06080d;}
::-webkit-scrollbar-thumb{background:#182038;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:.8rem 0 .4rem;">
  <div style="font-family:'Barlow Condensed',sans-serif;
              font-size:clamp(2rem,8vw,3.6rem);font-weight:900;
              letter-spacing:.1em;line-height:1;
              background:linear-gradient(130deg,#00ff88 0%,#4da6ff 48%,#b07dff 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    PSA 10 PROFIT SCANNER
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
              color:#1a6040;letter-spacing:.2em;margin-top:3px;">
    v4.0 · SHARPNESS PRE-PROC · PERSPECTIVE WARP · GAUSSIAN GRADES · LIVE PRICING
  </div>
</div>
<div style="height:1px;
            background:linear-gradient(90deg,#00ff8844,#4da6ff33,transparent);
            margin-bottom:.8rem;"></div>
""", unsafe_allow_html=True)

# Session state
if "notifications" not in st.session_state: st.session_state.notifications = []

def push_notif(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.notifications.insert(0, f"[{ts}] {msg}")
    st.session_state.notifications = st.session_state.notifications[:20]


# ═══════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_single, tab_lead, tab_gap, tab_notif, tab_cfg = st.tabs([
    "🔍 LIVE SCAN", "📋 BATCH", "🔬 SINGLE CARD",
    "🏆 LEADERBOARD", "💰 GAP FINDER", "🔔 ALERTS", "⚙️ CONFIG"
])

# ───────────────────────────────────────────────────────────────────────
#  TAB 1 — LIVE SCAN
# ───────────────────────────────────────────────────────────────────────
with tab_scan:
    st.markdown("### eBay Live Scanner")
    st.caption(
        "Structured query: `'{query} raw -graded -proxy -replica'` — "
        "filters out pre-graded and fake listings automatically."
    )
    r1c1, r1c2, r1c3, r1c4 = st.columns([3,1,1,1])
    with r1c1: query     = st.text_input("Search Query",
                                          "Charizard Holo Base Set Unlimited")
    with r1c2: min_price = st.number_input("Min $", value=1,   min_value=0)
    with r1c3: max_price = st.number_input("Max $", value=100, min_value=1)
    with r1c4: num_res   = st.number_input("Limit", value=8,   min_value=1, max_value=50)

    r2c1, r2c2 = st.columns(2)
    with r2c1: min_psa10  = st.number_input("Min PSA 10 Value ($)", value=1000, min_value=0)
    with r2c2: min_profit = st.number_input("Min Best-Case Profit ($)", value=200, min_value=0)

    if st.button("▶  RUN LIVE SCAN", use_container_width=True):
        with st.spinner("Fetching listings from eBay…"):
            raw_list = search_ebay(query, min_price, max_price, num_res)

        pb = st.progress(0.0); status = st.empty()
        results: list[ListingResult] = []

        for i, raw in enumerate(raw_list):
            status.markdown(
                f"<span style='font-family:\"Share Tech Mono\",monospace;"
                f"font-size:.78rem;color:#4da6ff;'>"
                f"Grading {i+1}/{len(raw_list)}: {raw['title'][:50]}…</span>",
                unsafe_allow_html=True)
            pb.progress((i+1) / max(len(raw_list), 1))
            lr = grade_listing(raw)
            if lr.grading and lr.grading.psa10 >= 70 and lr.verdict == "PASS":
                push_notif(
                    f"🔥 {lr.grading.psa10:.0f}% PSA 10 · "
                    f"{lr.title[:40]} · ${lr.price}"
                )
            results.append(lr)

        pb.empty(); status.empty()
        st.session_state["last_results"] = results

        n_pass = sum(1 for r in results if r.verdict == "PASS")
        n_warn = sum(1 for r in results if r.verdict == "WARN")
        n_fail = sum(1 for r in results if r.verdict == "FAIL")
        best   = max((r for r in results if r.verdict == "PASS"),
                     key=lambda r: r.net_profit, default=None)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Scanned",     len(results))
        m2.metric("✅ Pass",      n_pass)
        m3.metric("⚠️ Warn",      n_warn)
        m4.metric("❌ Rejected",  n_fail)
        m5.metric("Best Profit", f"${best.net_profit:,.0f}" if best else "—")

        st.markdown("---")
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        results.sort(key=lambda r: (order.get(r.verdict,3), -r.net_profit))
        for r in results:
            render_grading_card(r, expanded=(r.verdict in ("PASS","WARN")))

# ───────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH  (up to 10 URLs)
# ───────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch URL Grader")
    st.caption(
        "Paste eBay listing URLs or direct image URLs (JPG/PNG). "
        "For Mercari or TCGplayer: right-click the card image → "
        "'Copy image address' → paste here."
    )
    batch_input = st.text_area("URLs — one per line", height=160,
        placeholder="https://www.ebay.com/itm/…\nhttps://i.ebayimg.com/…")

    if st.button("▶  GRADE ALL & RANK", use_container_width=True):
        urls = [u.strip() for u in batch_input.splitlines()
                if u.strip() and not u.strip().startswith("#")][:10]
        if not urls:
            st.warning("Paste at least one URL.")
        else:
            batch_results = []
            bp = st.progress(0.0)
            for i, url in enumerate(urls):
                bp.progress((i+1)/len(urls))
                is_img = any(url.lower().endswith(e)
                             for e in [".jpg",".jpeg",".png",".webp"])
                raw = {
                    "title"      : urlparse(url).path[-60:] or f"Listing {i+1}",
                    "price"      : 0.0,
                    "url"        : url,
                    "source"     : "Manual",
                    "image_urls" : [url] if is_img else [],
                }
                if not is_img:
                    try:
                        pg   = requests.get(url, headers=HEADERS, timeout=10)
                        imgs = re.findall(r'https://i\.ebayimg\.com/images/g/[^"\']+\.jpg', pg.text)
                        raw["image_urls"] = list(dict.fromkeys(imgs))[:5]
                        m  = re.search(r'"price"\s*:\s*\{\s*"value"\s*:\s*"?([\d.]+)', pg.text)
                        if m: raw["price"] = float(m.group(1))
                        m2 = re.search(r'<h1[^>]*itemprop="name"[^>]*>([^<]+)', pg.text)
                        if m2: raw["title"] = m2.group(1).strip()[:80]
                    except Exception:
                        pass
                batch_results.append(grade_listing(raw))
            bp.empty()

            batch_results.sort(key=lambda r: -(r.grading.psa10 if r.grading else 0))
            medals = ["🥇","🥈","🥉"] + ["🔹"]*10
            rows   = [{
                "Rank"       : f"{medals[i]} #{i+1}",
                "Card"       : r.title[:45],
                "Price"      : f"${r.price:.2f}",
                "PSA 10%"    : f"{r.grading.psa10:.1f}%" if r.grading else "—",
                "Conf."      : f"{r.grading.confidence_pct:.0f}%" if r.grading else "—",
                "Grade Range": r.grading.grade_range if r.grading else "—",
                "PSA10 Est"  : f"${r.market.psa10_price:,.0f}" if r.market else "—",
                "Best Profit": f"${r.net_profit:,.0f}",
                "Verdict"    : r.verdict,
            } for i, r in enumerate(batch_results)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            for r in batch_results:
                render_grading_card(r, expanded=True)

# ───────────────────────────────────────────────────────────────────────
#  TAB 3 — SINGLE CARD ANALYSER
# ───────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### 🔬 Single Card Deep Analysis")
    mode = st.radio("Input", ["📁 Upload Photo","🔗 Paste Image URL"], horizontal=True)
    single_img  = None
    single_urls = []

    if "Upload" in mode:
        up = st.file_uploader("Drop card photo (JPG/PNG)",
                               type=["jpg","jpeg","png"])
        if up:
            pil        = Image.open(up).convert("RGB")
            single_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        url_in = st.text_input("Image URL", placeholder="https://i.ebayimg.com/…")
        if url_in.strip():
            single_urls = [url_in.strip()]

    sc1, sc2 = st.columns(2)
    with sc1: s_title = st.text_input("Card Name (for pricing)",
                                       placeholder="Charizard Base Set Holo Rare")
    with sc2: s_price = st.number_input("Raw Price ($)", value=50.0, min_value=0.0)

    if st.button("▶  ANALYSE CARD", use_container_width=True):
        if single_img is None and not single_urls:
            st.error("Upload an image or paste a URL.")
        else:
            with st.spinner("Running pre-processing + full CV pipeline…"):
                if single_img is not None:
                    # Run pipeline directly on uploaded image
                    gr  = GradingResult()
                    sub = SubGrades()

                    # Pre-process
                    proc_img, sharpness, quality = preprocess_image(single_img)
                    gr.image_quality   = quality
                    gr.sharpness_score = sharpness
                    gr.res_px          = (proc_img.shape[1], proc_img.shape[0])

                    if is_stock_photo(proc_img):
                        gr.image_quality = "stock"
                        gr.reasoning.append("⚠ Appears to be stock/official art")

                    # Centering
                    lr, tb, cen_sg, cen_r, quad, warped = run_centering(proc_img)
                    sub.centering = cen_sg; sub.lr_ratio = lr; sub.tb_ratio = tb
                    gr.quad_pts   = quad;  gr.warped_img = warped
                    gr.reasoning.extend(cen_r)

                    # Corners
                    counts  = measure_corner_white_px(proc_img)
                    cor_sg, cor_r = corner_px_to_subgrade(counts)
                    sub.corners         = cor_sg
                    sub.corner_white_px = counts
                    gr.reasoning.extend(cor_r)

                    # Surface
                    glint, scratch, srf_r, srf_sg = check_surface(proc_img)
                    sub.edges   = srf_sg; sub.surface = srf_sg
                    gr.glint_flag   = glint; gr.scratch_fail = scratch
                    gr.reasoning.extend(srf_r)

                    # Forensics
                    cl_f, cl_r = check_clone(proc_img)
                    gr.clone_flag = cl_f; gr.reasoning.extend(cl_r)
                    if cl_f: sub.surface = min(sub.surface, 5.0)

                    gr.sub = sub
                    gr.grade_dist, gr.confidence_pct, gr.grade_range = \
                        compute_grade_distribution(sub, quality)
                    gr.psa10 = gr.grade_dist.get(10, 0.0)
                    gr.centering_fail = sub.centering < 10.0
                    gr.corners_fail   = sub.corners   < 10.0

                    # Annotate
                    gr.annotated_full = annotate_full(
                        proc_img, quad, lr, tb, cen_sg, counts, glint)
                    gr.corner_patches = make_corner_patches(proc_img, counts)

                    hard = [gr.centering_fail, gr.corners_fail,
                            gr.scratch_fail, gr.clone_flag]
                    gr.overall_pass = not any(hard)
                else:
                    gr = grade_images(single_urls)

            md  = get_market_data(s_title)
            net, exp, roi = financials(s_price, md, gr.psa10)
            v   = ("PASS" if gr.overall_pass and not gr.clone_flag
                   else "WARN" if gr.glint_flag else "FAIL")

            col_img, col_right = st.columns([1.6, 1])
            with col_img:
                if gr.annotated_full is not None:
                    st.image(bgr_to_pil(gr.annotated_full), use_container_width=True,
                             caption="AI-annotated — green=pass · red=fail")
                elif single_img is not None:
                    st.image(bgr_to_pil(single_img), width=280)
            with col_right:
                st.markdown(f"## {verdict_pill(v)}", unsafe_allow_html=True)
                confidence_ring(gr.psa10, gr.confidence_pct)
                gr_col = _gc(gr.psa10)
                st.markdown(
                    f'<div style="text-align:center;font-family:\'Share Tech Mono\','
                    f'monospace;font-size:.68rem;color:{gr_col};">'
                    f'{gr.grade_range}</div>',
                    unsafe_allow_html=True)
                render_subgrades(gr.sub)

            render_market(md, s_price, net, exp, roi)

            d = gr.grade_dist
            st.markdown("**Grade Distribution**")
            ca, cb = st.columns(2)
            with ca:
                prob_bar("PSA 10", d.get(10,0), NEON["green"])
                prob_bar("PSA  9", d.get(9, 0), NEON["yellow"])
                prob_bar("PSA  8", d.get(8, 0), NEON["orange"])
                prob_bar("PSA  7", d.get(7, 0), "#cc6600")
            with cb:
                prob_bar("PSA  6", d.get(6, 0), "#aa4400")
                prob_bar("PSA  5", d.get(5, 0), "#884400")
                prob_bar("PSA 3-4", sum(d.get(g,0) for g in (3,4)), NEON["red"])
                prob_bar("PSA 1-2", sum(d.get(g,0) for g in (1,2)), "#660000")

            if gr.corner_patches:
                st.markdown("**4-Corner Zoom — 400% (Laplacian-sharpened)**")
                render_corner_panels(gr.corner_patches)
            if gr.warped_img is not None:
                with st.expander("🔲 Perspective-corrected card"):
                    st.image(bgr_to_pil(gr.warped_img), use_container_width=True)
            st.markdown("**AI Reasoning Report**")
            for note in gr.reasoning:
                ok   = note.startswith("✓")
                warn = "⚠" in note or "WARNING" in note.upper()
                icon = "🟢" if ok else ("🟡" if warn else "🔴")
                st.markdown(f"{icon} `{note}`")

# ───────────────────────────────────────────────────────────────────────
#  TAB 4 — LEADERBOARD
# ───────────────────────────────────────────────────────────────────────
with tab_lead:
    st.markdown("### 🏆 Scan Leaderboard")
    st.caption("Shows results from your most recent Live Scan, ranked by PSA 10 probability × profit.")
    last = st.session_state.get("last_results", [])
    if not last:
        st.info("Run a Live Scan first to populate the leaderboard.")
    else:
        ranked = sorted(
            [r for r in last if r.verdict != "FAIL"],
            key=lambda r: -(r.grading.psa10 * r.net_profit if r.grading else 0)
        )
        medals = ["🥇","🥈","🥉"] + ["🔹"]*20
        rows   = [{
            "Rank"       : f"{medals[i]} #{i+1}",
            "Card"       : r.title[:42],
            "Raw $"      : f"${r.price:.2f}",
            "PSA 10%"    : f"{r.grading.psa10:.1f}%" if r.grading else "—",
            "Conf."      : f"{r.grading.confidence_pct:.0f}%" if r.grading else "—",
            "Grade Range": r.grading.grade_range if r.grading else "—",
            "PSA10 $"    : f"${r.market.psa10_price:,.0f}" if r.market else "—",
            "Net Profit" : f"${r.net_profit:,.0f}",
            "ROI"        : f"{r.roi_pct:.0f}%",
            "Source"     : r.market.source if r.market else "—",
        } for i, r in enumerate(ranked)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        for r in ranked[:3]:   # expand top 3 only
            render_grading_card(r, expanded=True)

# ───────────────────────────────────────────────────────────────────────
#  TAB 5 — GAP FINDER
# ───────────────────────────────────────────────────────────────────────
with tab_gap:
    st.markdown("### 💰 High-ROI Gap Dashboard")
    st.caption("Cards where the raw-to-PSA 10 spread is historically massive.")
    sample = [
        ("Charizard 1st Ed Base",         8000, 350000),
        ("Charizard Shadowless",           1200,  50000),
        ("Charizard Base Unlimited",        200,   5000),
        ("Lugia Neo Genesis",               150,  12000),
        ("Ho-Oh Neo Revelation",             90,   3000),
        ("Shining Charizard Neo Destiny",   300,   8000),
        ("Dark Charizard Team Rocket",      150,   2500),
        ("Gengar Fossil Holo",               80,   2500),
        ("Scyther Jungle Holo",              30,    600),
        ("Snorlax Jungle Holo",              55,   1200),
        ("Raichu Base Set Holo",            200,   1800),
    ]
    df = pd.DataFrame([{
        "Card"           : n,
        "Raw (approx)"   : f"${r:,}",
        "PSA 10 (approx)": f"${p:,}",
        "Spread"         : f"${p-r:,}",
        "Multiplier"     : f"{p//r}×",
    } for n, r, p in sample])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    for entry in HIGH_ROI_CATALOG:
        with st.expander(f"📦 {entry['set']} — Spread {entry['spread']}"):
            st.markdown(f"*{entry['note']}*")
            cols = st.columns(min(len(entry["cards"]), 4))
            for col, card in zip(cols, entry["cards"]):
                link = (f"https://www.ebay.com/sch/i.html?_nkw="
                        f"{quote_plus(card+' '+entry['set'].split()[0]+' pokemon raw ungraded')}"
                        f"&_sacat=183454&LH_BIN=1")
                col.markdown(f"[**{card}**]({link})")

# ───────────────────────────────────────────────────────────────────────
#  TAB 6 — ALERTS
# ───────────────────────────────────────────────────────────────────────
with tab_notif:
    n = len(st.session_state.notifications)
    st.markdown(f"### 🔔 Alert Feed ({n} alerts)")
    st.caption("Auto-fires when a card hits ≥70% PSA 10 probability during a Live Scan.")
    if not st.session_state.notifications:
        st.info("No alerts yet — run a Live Scan.")
    else:
        for note in st.session_state.notifications:
            st.markdown(
                f'<div style="background:#001a0a;border:1px solid #00ff8833;'
                f'border-radius:6px;padding:7px 12px;margin-bottom:5px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.78rem;'
                f'color:#00ff88;">{note}</div>',
                unsafe_allow_html=True)
        if st.button("Clear Alerts"):
            st.session_state.notifications = []
            st.rerun()

# ───────────────────────────────────────────────────────────────────────
#  TAB 7 — CONFIG
# ───────────────────────────────────────────────────────────────────────
with tab_cfg:
    st.markdown("### ⚙️ Configuration & Deploy Guide")

    st.markdown("#### `config.yaml`")
    st.code("""
ebay:
  app_id:  "YOUR-EBAY-APP-ID"
  cert_id: "YOUR-EBAY-CERT-ID"
  dev_id:  "YOUR-EBAY-DEV-ID"
""", language="yaml")

    st.markdown("#### `.streamlit/secrets.toml`  *(Streamlit Cloud)*")
    st.code("""
[ebay]
app_id  = "YOUR-EBAY-APP-ID"
cert_id = "YOUR-EBAY-CERT-ID"
dev_id  = "YOUR-EBAY-DEV-ID"
""", language="toml")

    st.markdown("---")
    st.markdown("#### 🚀 Deploy to Streamlit Cloud (free, mobile-accessible)")
    st.markdown("""
1. **Get eBay keys** at [developer.ebay.com](https://developer.ebay.com)
   → *My Account → Application Access Keys → Create Keyset → Production*
2. **Push to GitHub:**
   ```bash
   git init && git add app.py requirements.txt
   git commit -m "PSA Scanner v4"
   git remote add origin https://github.com/YOU/psa-scanner
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
   → connect repo → set `app.py` as entry point → **Secrets** → paste TOML above
4. Bookmark the live URL on your phone — **Add to Home Screen** for native-app feel.
""")

    st.markdown("---")
    st.markdown("#### v4.0 Fix Reference")
    st.markdown(f"""
| Fix | Method | Detail |
|---|---|---|
| **PSA 0 — Low-res rejection** | `MIN_PX` lowered to {MIN_PX} + bilinear upscale to {UPSCALE_TARGET}px | Soft images are upscaled before grading instead of rejected |
| **PSA 0 — Zero distribution** | Gaussian distribution (σ adaptive) | Even inconclusive cards get a non-zero spread; confidence score shown separately |
| **PSA 0 — No card detected** | `find_card_quad()` always returns something | Falls back to bounding-rect or whole-image if no 4-pt quad found |
| **Sharpness pre-proc** | Laplacian unsharp mask (α={LAP_STRENGTH}) + bilinear upscale | Applied before grading AND on corner patches at 400% zoom |
| **Perspective warp** | `cv2.getPerspectiveTransform` + `cv2.warpPerspective` | Centering measured on the flattened card, not the camera angle |
| **Confidence score** | Quality × flag count → 15–88% | Shown in inner ring of dual SVG gauge |
| **Grade Range label** | Maps overall sub-grade to human string | e.g. "PSA 8–10 Potential (low-res photo)" |
| **$0 pricing** | 3-tier waterfall: eBay API → PriceCharting → static table | Static table always returns a non-zero value |
| **Duplicate filter** | `hash_image()` MD5 perceptual hash | Stock/duplicate photos silently skipped |
| **Search quality** | Structured query with exclusion terms | `-graded -proxy -replica -reprint` filters noise from eBay results |
""")

# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#182038,transparent);
            margin:2rem 0 .8rem;"></div>
<div style="text-align:center;font-family:'Share Tech Mono',monospace;
            font-size:.62rem;color:#182038;padding-bottom:.8rem;">
  PSA 10 PROFIT SCANNER v4.0 · For research/educational use only
  · Not financial advice · Always verify manually before purchasing
</div>
""", unsafe_allow_html=True)
