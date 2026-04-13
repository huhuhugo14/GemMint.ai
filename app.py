"""
╔══════════════════════════════════════════════════════════════════════╗
║   POKÉMON PSA 10 PROFIT SCANNER  —  v5.0  "CARD ISOLATION"          ║
║                                                                      ║
║  PATCH 1 — Card Isolation: all grading (corners, surface) now runs  ║
║            on the perspective-warped card, NOT the background.       ║
║  PATCH 2 — Dynamic Corner Mapping: corner panels snap to the 4       ║
║            actual corners of the detected card contour.              ║
║  PATCH 3 — Query Logic: structured eBay query with set-number        ║
║            extraction + PriceCharting raw-price fallback.            ║
║  PATCH 4 — UI Cleanup: dynamic green/red bounding boxes that track   ║
║            the card edges; Dark Background Warning when the card     ║
║            cannot be isolated; ROI gated on price > $0.             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="PSA 10 Profit Scanner v5",
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
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urlparse
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────
#  CONFIG  (config.yaml → st.secrets → env vars)
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
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MIN_PX         = 800        # minimum usable dimension after upscale
UPSCALE_TARGET = 1200       # shortest side target before grading
LAP_STRENGTH   = 1.2        # Laplacian unsharp-mask alpha
WHITE_LV       = 210        # pixel brightness → "white"
GLINT_LV       = 242
GLINT_FRAC     = 0.025
SCRATCH_DENS   = 0.004
CLONE_BS       = 16
ROSETTE_FREQ   = 45
GRADING_FEE    = 25.0
SHIPPING_EST   = 6.0
CORNER_LABELS  = ["TL", "TR", "BL", "BR"]

# PSA centering table: ratio = larger_border / (L+R total)
CENTERING_TABLE = [
    (0.530, 10.0),
    (0.550, 10.0),
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

# OpenCV BGR colour palette
CV_GREEN  = (50,  220, 100)
CV_RED    = (40,   60, 250)
CV_YELLOW = (20,  210, 250)
CV_ORANGE = (30,  130, 255)
CV_WHITE  = (230, 230, 230)
CV_FONT   = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────────────

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
class IsolationResult:
    """
    PATCH 1 & 2: Output of the card-isolation step.
    Carries the warped card, quad coordinates, and a quality score.
    All downstream grading operates on `warped`, never on the original.
    """
    warped          : Optional[np.ndarray] = None   # perspective-corrected card
    quad_on_orig    : Optional[np.ndarray] = None   # 4×2 float32 in original image space
    # ─ Confidence of the isolation step (0–1)
    # 1.0 = clean 4-point quad found
    # 0.6 = bounding-rect fallback used
    # 0.3 = whole-image fallback used (no card found at all)
    isolation_conf  : float                = 0.0
    warning         : Optional[str]        = None   # e.g. "dark background" message
    # Corner coordinates in warped-card space (for dynamic corner mapping)
    # These are simply the four corners of warped: (0,0), (W,0), (W,H), (0,H)
    corner_coords   : list                 = field(default_factory=list)


@dataclass
class GradingResult:
    sub            : SubGrades             = field(default_factory=SubGrades)
    isolation      : Optional[IsolationResult] = None
    # Flags
    centering_fail : bool                  = False
    corners_fail   : bool                  = False
    scratch_fail   : bool                  = False
    glint_flag     : bool                  = False
    clone_flag     : bool                  = False
    rosette_flag   : bool                  = False
    res_fail       : bool                  = False
    res_px         : tuple                 = (0, 0)
    # Images for display
    annotated_orig  : Optional[np.ndarray] = None   # original with dynamic boxes
    corner_patches  : list                 = field(default_factory=list)  # from warped
    warped_display  : Optional[np.ndarray] = None   # warped card with holo overlay
    # Quality
    image_quality   : str                  = "ok"
    sharpness_score : float                = 0.0
    # Grades
    grade_dist      : dict                 = field(default_factory=dict)
    psa10           : float                = 0.0
    confidence_pct  : float                = 0.0
    grade_range     : str                  = ""
    overall_pass    : bool                 = False
    reasoning       : list                 = field(default_factory=list)


@dataclass
class MarketData:
    psa10_price     : float = 0.0
    raw_price_ref   : float = 0.0   # PATCH 3: raw avg price from PriceCharting
    source          : str   = "static"
    search_url      : str   = ""
    confidence      : str   = "low"


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


# ─────────────────────────────────────────────────────────────────────
#  STATIC DATA
# ─────────────────────────────────────────────────────────────────────
STATIC_PSA10 = {
    "charizard 1st":        350000, "charizard shadowless":  50000,
    "charizard base":          5000, "charizard skyridge":   15000,
    "charizard aquapolis":    8000, "blastoise 1st":         25000,
    "blastoise shadowless":   8000, "blastoise base":          1200,
    "venusaur 1st":          12000, "venusaur base":             900,
    "lugia neo":             12000, "ho-oh neo":               3000,
    "entei neo":              1800, "raikou neo":              1500,
    "suicune neo":            1200, "shining charizard":       8000,
    "shining magikarp":       3500, "shining gyarados":        2000,
    "dark charizard":         2500, "dark blastoise":          1800,
    "gengar fossil":          2500, "lapras fossil":             800,
    "moltres fossil":           700, "zapdos fossil":            700,
    "scyther jungle":           600, "clefable jungle":          700,
    "snorlax jungle":          1200, "vaporeon jungle":           900,
    "jolteon jungle":           800, "raichu base":             1800,
    "machamp 1st":             2000, "ninetales base":            900,
}

MOCK_LISTINGS = [
    {"title":"[DEMO] Charizard Base Set Holo Unlimited — Raw NM", "price":79.99,
     "url":"https://www.ebay.com/sch/i.html?_nkw=charizard+base+set+holo+raw",
     "source":"eBay (Demo)", "image_urls":["https://images.pokemontcg.io/base1/4_hires.png"]},
    {"title":"[DEMO] Lugia Neo Genesis Holo — PSA Ready Raw", "price":145.00,
     "url":"https://www.ebay.com/sch/i.html?_nkw=lugia+neo+genesis+holo+raw",
     "source":"eBay (Demo)", "image_urls":["https://images.pokemontcg.io/neo1/9_hires.png"]},
    {"title":"[DEMO] Shining Charizard Neo Destiny — Ungraded RAW", "price":290.00,
     "url":"https://www.ebay.com/sch/i.html?_nkw=shining+charizard+neo+destiny+raw",
     "source":"eBay (Demo)", "image_urls":["https://images.pokemontcg.io/neo4/107_hires.png"]},
    {"title":"[DEMO] Gengar Fossil Holo — LP", "price":34.99,
     "url":"https://www.ebay.com/sch/i.html?_nkw=gengar+fossil+holo+raw",
     "source":"eBay (Demo)", "image_urls":["https://images.pokemontcg.io/fossil/5_hires.png"]},
    {"title":"[DEMO] Blastoise Base Set Shadowless — PSA Ready", "price":650.00,
     "url":"https://www.ebay.com/sch/i.html?_nkw=blastoise+shadowless+raw",
     "source":"eBay (Demo)", "image_urls":["https://images.pokemontcg.io/base1/2_hires.png"]},
]

HIGH_ROI_CATALOG = [
    {"set":"Base Set (1st Edition)", "cards":["Charizard","Blastoise","Venusaur","Raichu"],
     "spread":"$5k–$500k+", "note":"Crown jewel — any 1st Ed holo has massive PSA 10 premium"},
    {"set":"Base Set Shadowless",    "cards":["Charizard","Blastoise","Venusaur"],
     "spread":"$1k–$50k",   "note":"Shadowless gem mints are extremely scarce"},
    {"set":"Base Set Unlimited",     "cards":["Charizard","Blastoise","Raichu","Machamp"],
     "spread":"$500–$5k",   "note":"High raw volume; gem copies still rare"},
    {"set":"Jungle",                 "cards":["Scyther","Clefable","Snorlax","Vaporeon"],
     "spread":"$300–$1.2k", "note":"Notorious centering issues → PSA 10 premium"},
    {"set":"Fossil",                 "cards":["Gengar","Lapras","Moltres","Zapdos"],
     "spread":"$400–$2.5k", "note":"Thick cards prone to edge dings"},
    {"set":"Team Rocket",            "cards":["Dark Charizard","Dark Blastoise"],
     "spread":"$800–$2.5k", "note":"Dark surface shows scratches easily"},
    {"set":"Neo Genesis",            "cards":["Lugia","Typhlosion","Feraligatr"],
     "spread":"$500–$12k",  "note":"Lugia PSA 10 regularly $10k+; raw copies under $200"},
    {"set":"Neo Revelation",         "cards":["Ho-Oh","Entei","Raikou","Suicune"],
     "spread":"$600–$3k",   "note":"Legendary beasts — massive PSA 10 upside"},
    {"set":"Neo Destiny",            "cards":["Shining Charizard","Shining Magikarp"],
     "spread":"$1k–$8k",    "note":"Shining series = rarest WotC PSA 10s"},
    {"set":"Aquapolis / Skyridge",   "cards":["Charizard","Articuno","Celebi"],
     "spread":"$2k–$15k",   "note":"e-Reader era print lines → PSA 10 ultra-rare"},
]


# ═══════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING  (upscale + Laplacian sharpen)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_image(img: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Bilinear upscale to UPSCALE_TARGET, then Laplacian unsharp-mask.
    Returns (processed, sharpness_score, quality_label)."""
    h, w = img.shape[:2]
    if min(h, w) < UPSCALE_TARGET:
        scale = UPSCALE_TARGET / min(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_LINEAR)
    gray_f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap    = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    for c in range(3):
        ch = img[:,:,c].astype(np.float32)
        img[:,:,c] = np.clip(ch - LAP_STRENGTH * lap, 0, 255).astype(np.uint8)
    score = float(cv2.Laplacian(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    quality = "very_low" if score < 30 else "low" if score < 80 else "ok"
    return img, score, quality


def is_stock_photo(img: np.ndarray) -> bool:
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    border = np.concatenate([gray[:12,:].flatten(), gray[-12:,:].flatten(),
                              gray[:,:12].flatten(), gray[:,-12:].flatten()])
    return (0.90 < w/max(h,1) < 1.10) and float(border.std()) < 14.0


def hash_image(img: np.ndarray) -> str:
    small = cv2.resize(img, (16,16), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bits  = (gray > gray.mean()).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
#  PATCH 1 — CARD ISOLATION ENGINE
#  Produces a clean, perspective-corrected card image (IsolationResult).
#  ALL subsequent grading runs on .warped, not the original background.
# ═══════════════════════════════════════════════════════════════════════

def _order_pts(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into [TL, TR, BR, BL]."""
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],    # TL
        pts[np.argmin(diff)], # TR
        pts[np.argmax(s)],    # BR
        pts[np.argmax(diff)], # BL
    ], dtype=np.float32)


def isolate_card(img: np.ndarray) -> IsolationResult:
    """
    PATCH 1 core function.

    Strategy (three-tier, tried in order):
    ──────────────────────────────────────
    Tier A — Clean 4-point quadrilateral (conf=1.0)
      Canny on luminance with adaptive thresholds → dilate → findContours.
      Pick the largest contour whose approxPolyDP gives exactly 4 points
      AND whose area is ≥ 12% of the frame.

    Tier B — Bounding rect of largest contour (conf=0.6)
      If no 4-pt quad, fall back to the bounding rect of the largest
      contour that's ≥ 12% of the frame.

    Tier C — Whole-image fallback (conf=0.3)
      No significant contour found at all.  Use the full image as-is and
      emit the "Dark Background Warning" so the user knows to re-shoot.

    Output: IsolationResult with warped card, quad coords in original
    image space, per-corner pixel coordinates in warped space,
    isolation confidence, and optional warning string.
    """
    result = IsolationResult()
    h, w   = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Adaptive Canny: use Otsu to find automatic thresholds ─
    blur   = cv2.GaussianBlur(gray, (9, 9), 0)
    otsu_thresh, _ = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = otsu_thresh * 0.4, otsu_thresh
    edges  = cv2.Canny(blur, lo, hi)
    edges  = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.12
    cnts     = [c for c in cnts if cv2.contourArea(c) >= min_area]

    quad          = None
    isolation_conf = 0.0

    # Tier A — 4-point quad
    if cnts:
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:
                quad           = _order_pts(approx)
                isolation_conf = 1.0
                break

    # Tier B — bounding rect fallback
    if quad is None and cnts:
        lc        = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(lc)
        quad          = _order_pts(np.array(
            [[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]], dtype=np.float32
        ).reshape(-1,1,2))
        isolation_conf = 0.6
        result.warning = (
            "⚠ Bounding-rect fallback used — card edges not precisely detected. "
            "Centering measurement may be less accurate."
        )

    # Tier C — full-image fallback + Dark Background Warning
    if quad is None:
        quad           = _order_pts(np.array(
            [[0,0],[w,0],[w,h],[0,h]], dtype=np.float32
        ))
        isolation_conf = 0.3
        result.warning = (
            "🌑 DARK BACKGROUND WARNING: No card outline detected. "
            "Please place the card on a plain dark surface (e.g. a dark cloth or desk) "
            "and re-photograph for accurate grading."
        )

    # Perspective-warp the card to an upright rectangle
    tl, tr, br, bl = quad
    out_w = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    out_h = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    out_w, out_h = max(out_w, 80), max(out_h, 80)
    dst   = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]],
                     dtype=np.float32)
    M     = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    # Corner coordinates in warped space (patch 2: dynamic snap)
    result.warped         = warped
    result.quad_on_orig   = quad
    result.isolation_conf = isolation_conf
    result.corner_coords  = [(0,0),(out_w-1,0),(0,out_h-1),(out_w-1,out_h-1)]
    return result


# ═══════════════════════════════════════════════════════════════════════
#  CENTERING  (runs on warped card, not background)
# ═══════════════════════════════════════════════════════════════════════

def measure_borders_on_warped(warped: np.ndarray) -> tuple[int,int,int,int]:
    """Scan brightness projections on the WARPED card to find white border widths."""
    gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    col_m = gray[h//4:3*h//4, :].mean(axis=0).astype(float)
    row_m = gray[:, w//4:3*w//4].mean(axis=1).astype(float)

    def from_left(arr, thresh=185, run=4):
        for i in range(len(arr)-run):
            if all(arr[i:i+run] < thresh): return max(i,1)
        return max(len(arr)//10, 1)

    def from_right(arr, thresh=185, run=4):
        n = len(arr)
        for i in range(n-1, run, -1):
            if all(arr[i-run:i] < thresh): return max(n-i, 1)
        return max(n//10, 1)

    bl = min(from_left(col_m),  w//3)
    br = min(from_right(col_m), w//3)
    bt = min(from_left(row_m),  h//3)
    bb = min(from_right(row_m), h//3)
    return max(bl,1), max(br,1), max(bt,1), max(bb,1)


def ratio_to_centering_sg(lr: float, tb: float) -> float:
    worst = max(lr, tb)
    for thresh, grade in CENTERING_TABLE:
        if worst <= thresh: return grade
    return 5.0


def run_centering_on_warped(warped: np.ndarray) -> tuple:
    """Compute centering sub-grade from the already-warped card image."""
    reasons = []
    try:
        bl, br, bt, bb = measure_borders_on_warped(warped)
        lr_total = bl + br
        tb_total = bt + bb
        lr_ratio = max(bl, br) / max(lr_total, 1)
        tb_ratio = max(bt, bb) / max(tb_total, 1)
        sg       = ratio_to_centering_sg(lr_ratio, tb_ratio)
        lr_str   = f"{int(max(bl,br)/lr_total*100)}/{int(min(bl,br)/lr_total*100)}"
        tb_str   = f"{int(max(bt,bb)/tb_total*100)}/{int(min(bt,bb)/tb_total*100)}"
        if sg >= 10.0:
            reasons.append(f"✓ Centering L/R {lr_str} | T/B {tb_str} → sub-grade 10")
        else:
            reasons.append(
                f"Centering L/R {lr_str} | T/B {tb_str} → "
                f"sub-grade {sg:.0f} ← PSA 10 IMPOSSIBLE"
            )
        return lr_ratio, tb_ratio, sg, reasons
    except Exception as e:
        reasons.append(f"⚠ Centering measurement failed ({e}) — estimated 8")
        return 0.50, 0.50, 8.0, reasons


# ═══════════════════════════════════════════════════════════════════════
#  PATCH 2 — CORNER ANALYSIS ON WARPED CARD
#  White-pixel counting and patch extraction happen on the isolated card.
# ═══════════════════════════════════════════════════════════════════════

def measure_corner_white_px_on_card(warped: np.ndarray, pct: float = 0.09) -> list:
    """
    Extract corner patches from the WARPED (background-free) card.
    The 4 patch regions snap exactly to the 4 corners of the warp output.
    Returns [TL, TR, BL, BR] white-pixel counts.
    """
    h, w = warped.shape[:2]
    ph   = max(int(h * pct), 30)
    pw   = max(int(w * pct), 30)
    # Patches snap to the 4 corners of the warped rectangle
    patches = [
        warped[0:ph,    0:pw   ],   # TL
        warped[0:ph,    w-pw:w ],   # TR
        warped[h-ph:h,  0:pw   ],   # BL
        warped[h-ph:h,  w-pw:w ],   # BR
    ]
    counts = []
    for p in patches:
        if p.size == 0: counts.append(0); continue
        b, g, r = cv2.split(p)
        mask    = ((r.astype(int) > WHITE_LV) &
                   (g.astype(int) > WHITE_LV) &
                   (b.astype(int) > WHITE_LV))
        counts.append(int(mask.sum()))
    return counts


def corner_px_to_subgrade(counts: list) -> tuple[float, list]:
    worst   = max(counts)
    flagged = [f"{l}:{n}px" for l,n in zip(CORNER_LABELS,counts) if n > 0]
    reasons = [f"Corner {l}: {n} white pixel(s)" for l,n in zip(CORNER_LABELS,counts) if n > 0]
    if worst == 0:
        return 10.0, ["✓ All 4 corners — zero whitening on isolated card"]
    extra = f"({', '.join(flagged)})"
    if worst <=  4: return  9.0, reasons+[f"Minor whitening {extra} → sub-grade 9, PSA 10 <5%"]
    if worst <= 19: return  8.0, reasons+[f"Moderate whitening {extra} → sub-grade 8"]
    if worst <= 49: return  7.0, reasons+[f"Significant whitening {extra} → sub-grade 7"]
    if worst <= 99: return  6.0, reasons+[f"Heavy whitening {extra} → sub-grade 6"]
    return 5.0, reasons+[f"Severe whitening {extra} → sub-grade 5"]


# ═══════════════════════════════════════════════════════════════════════
#  SURFACE / GLINT / SCRATCH  (runs on warped card)
# ═══════════════════════════════════════════════════════════════════════

def check_surface_on_card(warped: np.ndarray) -> tuple[bool, bool, list, float]:
    """Surface analysis on the WARPED card only — no background interference."""
    h, w  = warped.shape[:2]
    y1, y2 = int(h*0.18), int(h*0.72)
    x1, x2 = int(w*0.08), int(w*0.92)
    holo  = warped[y1:y2, x1:x2]
    if holo.size == 0:
        return False, False, ["✓ Holo zone not measurable"], 10.0
    gray  = cv2.cvtColor(holo, cv2.COLOR_BGR2GRAY)
    glare = gray > GLINT_LV
    gfrac = float(glare.sum()) / max(gray.size, 1)
    if gfrac <= GLINT_FRAC:
        return False, False, ["✓ No glare on card surface"], 10.0
    reasons = [f"Glare: {gfrac:.1%} of holo area overexposed"]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if glare.sum() > 0:
        dens = float((np.abs(lap)[glare] > 35).sum()) / max(int(glare.sum()),1)
        if dens > SCRATCH_DENS:
            reasons.append(f"Scratch/print-line signal ({dens:.2%} Laplacian density) ← HARD FAIL")
            return True, True, reasons, 7.0
    reasons.append("Glare present — no scratch signal. Likely lighting artefact.")
    return True, False, reasons, 9.0


# ═══════════════════════════════════════════════════════════════════════
#  FORENSICS  (runs on warped card)
# ═══════════════════════════════════════════════════════════════════════

def check_clone(warped: np.ndarray) -> tuple[bool, list]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seen : dict = {}
    for y in range(0, h-CLONE_BS, CLONE_BS):
        for x in range(0, w-CLONE_BS, CLONE_BS):
            blk = gray[y:y+CLONE_BS, x:x+CLONE_BS].astype(np.float32)
            mn, sd = blk.mean(), blk.std()
            if sd < 5: continue
            key = hashlib.md5(((blk-mn)/sd).tobytes()).hexdigest()
            if key in seen:
                sy, sx = seen[key]
                if ((y-sy)**2+(x-sx)**2)**0.5 > CLONE_BS*3:
                    return True, [f"Clone-stamp editing ({sx},{sy})↔({x},{y}) ← FORENSIC FAIL"]
            else:
                seen[key] = (y, x)
    return False, []


def check_rosette(warped: np.ndarray) -> tuple[bool, list]:
    gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    cy, cx = h//2, w//2
    crop  = gray[cy-80:cy+80, cx-80:cx+80].astype(np.float32)
    if crop.shape[0] < 60: return False, []
    mag   = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(crop)))+1)
    ch,cw = mag.shape; f = ROSETTE_FREQ
    ring  = mag[ch//2-f-5:ch//2-f+5, cw//2-f-5:cw//2-f+5]
    peak  = float(ring.max()) if ring.size>0 else 0.0
    base  = float(mag[:20,:20].mean())
    if peak < base*1.6:
        return True, ["No CMYK rosette (FFT) — possible inkjet counterfeit ← FLAG"]
    return False, []


# ═══════════════════════════════════════════════════════════════════════
#  GRADE DISTRIBUTION  (Gaussian + Rule of Minimum)
# ═══════════════════════════════════════════════════════════════════════

def compute_grade_distribution(sub: SubGrades, quality: str) -> tuple[dict, float, str]:
    sigma_map = {"ok":0.70, "low":1.10, "very_low":1.50}
    sigma     = sigma_map.get(quality, 1.10)
    overall   = min(sub.centering, sub.corners, sub.edges, sub.surface)
    sub.overall = overall
    grades    = np.arange(1, 11, dtype=float)
    weights   = np.exp(-0.5 * ((grades - overall) / sigma)**2)
    weights  /= weights.sum()
    dist      = {int(g): round(float(p*100),1) for g,p in zip(grades, weights)}

    if sub.centering < 10.0:
        lost = dist.get(10,0.0); dist[10] = 0.0
        dist[9] = round(dist.get(9,0.0)+lost*0.60,1)
        dist[8] = round(dist.get(8,0.0)+lost*0.40,1)

    if sub.corners < 10.0 and dist.get(10,0.0) > 4.9:
        excess = dist[10]-4.9; dist[10] = 4.9
        dist[9] = round(dist.get(9,0.0)+excess*0.70,1)
        dist[8] = round(dist.get(8,0.0)+excess*0.30,1)

    total = sum(dist.values())
    if total > 0:
        dist = {k: round(v/total*100,1) for k,v in dist.items()}

    n_flags   = sum([sub.centering<10, sub.corners<10, sub.edges<9, sub.surface<9])
    conf_base = {"ok":88, "low":60, "very_low":38}.get(quality, 60)
    conf_pct  = max(conf_base - n_flags*10, 15)

    if overall >= 10.0: gr = "PSA 10 Potential"
    elif overall >= 9.0: gr = "PSA 9–10 Potential"
    elif overall >= 8.0: gr = "PSA 8–9 Likely"
    elif overall >= 7.0: gr = "PSA 7–8 Likely"
    else:                gr = f"PSA {max(int(overall)-1,1)}–{int(overall)} Likely"
    if quality == "very_low": gr = f"Est. {gr} (low-res)"
    return dist, conf_pct, gr


# ═══════════════════════════════════════════════════════════════════════
#  PATCH 4 — VISUAL ANNOTATION  (dynamic boxes, no fixed coordinates)
# ═══════════════════════════════════════════════════════════════════════

def _sharpen_patch(patch: np.ndarray) -> np.ndarray:
    """Apply Laplacian sharpening to a corner patch for display."""
    out = patch.copy()
    for c in range(3):
        ch = out[:,:,c].astype(np.float32)
        lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=3)
        out[:,:,c] = np.clip(ch - LAP_STRENGTH*lap, 0, 255).astype(np.uint8)
    return out


def build_annotated_original(
    orig: np.ndarray,
    iso: IsolationResult,
    white_counts: list,
    cen_sg: float,
    lr: float,
    tb: float,
    glint: bool,
) -> np.ndarray:
    """
    PATCH 4 — draws on the ORIGINAL image:
    • Card outline quad (green if centering ok, red if centering fail)
      — moves dynamically with the detected card contour
    • Corner bounding boxes that SNAP to the 4 corners of the detected quad
      — green if 0 white pixels, red if any whitening found
    • Holo-zone overlay (yellow) on the WARPED card geometry, projected back
    • Centering text strip at bottom
    • Dark Background Warning banner if isolation confidence is low
    """
    out  = orig.copy()
    h, w = out.shape[:2]
    quad = iso.quad_on_orig

    # ── 1. Card outline ──────────────────────────────────────
    if quad is not None:
        cen_col = CV_GREEN if cen_sg >= 10.0 else CV_RED
        pts = quad.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(out, [pts], isClosed=True, color=cen_col, thickness=3)
        for pt, lbl in zip(quad.astype(int), ["TL","TR","BR","BL"]):
            cv2.circle(out, tuple(pt), 6, cen_col, -1)
            cv2.putText(out, lbl, (pt[0]+5, pt[1]-6),
                        CV_FONT, 0.45, cen_col, 1, cv2.LINE_AA)

    # ── 2. Dynamic corner boxes — snap to detected quad corners ──
    # Box size = 9% of the warped card dimension, then mapped back
    # In practice we draw boxes in original space at each quad corner.
    if quad is not None:
        warped_h = iso.warped.shape[0] if iso.warped is not None else h
        warped_w = iso.warped.shape[1] if iso.warped is not None else w
        ph = max(int(warped_h * 0.09), 30)
        pw = max(int(warped_w * 0.09), 30)

        # quad order: TL, TR, BR, BL
        # We draw boxes from each corner outward
        corner_pt_map = [
            (int(quad[0][0]), int(quad[0][1]), "TL"),   # TL
            (int(quad[1][0]), int(quad[1][1]), "TR"),   # TR
            (int(quad[3][0]), int(quad[3][1]), "BL"),   # BL (quad[3]=BL)
            (int(quad[2][0]), int(quad[2][1]), "BR"),   # BR (quad[2]=BR)
        ]
        offsets = [
            (0,  0,  1,  1),   # TL: box grows right-down
            (-1, 0,  0,  1),   # TR: box grows left-down
            (0, -1,  1,  0),   # BL: box grows right-up
            (-1,-1,  0,  0),   # BR: box grows left-up
        ]
        for (px, py, lbl), (dx0, dy0, dx1, dy1), n in zip(
                corner_pt_map, offsets, white_counts):
            x0 = px + dx0 * pw
            y0 = py + dy0 * ph
            x1 = px + dx1 * pw
            y1 = py + dy1 * ph
            x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
            x0 = max(x0,0); y0 = max(y0,0)
            x1 = min(x1,w-1); y1 = min(y1,h-1)
            col = CV_RED if n > 0 else CV_GREEN
            cv2.rectangle(out, (x0,y0), (x1,y1), col, 2)
            ov = out.copy()
            cv2.rectangle(ov, (x0,y0), (x1,y1), col, -1)
            cv2.addWeighted(ov, 0.10, out, 0.90, 0, out)
            cv2.putText(out, f"{lbl} {n}px", (x0+3, y0+14),
                        CV_FONT, 0.38, col, 1, cv2.LINE_AA)

    # ── 3. Holo zone on warped then drawn as quad overlay ────
    if glint and quad is not None and iso.warped is not None:
        wh, ww = iso.warped.shape[:2]
        # The holo rect in warped space
        wy1,wy2 = int(wh*0.18), int(wh*0.72)
        wx1,wx2 = int(ww*0.08), int(ww*0.92)
        # Map those 4 corners back through inverse perspective transform
        dst_pts = np.array([[0,0],[ww-1,0],[ww-1,wh-1],[0,wh-1]], dtype=np.float32)
        M_inv   = cv2.getPerspectiveTransform(dst_pts, quad)
        holo_corners = np.array([[wx1,wy1],[wx2,wy1],[wx2,wy2],[wx1,wy2]],
                                 dtype=np.float32).reshape(-1,1,2)
        mapped = cv2.perspectiveTransform(holo_corners, M_inv).astype(np.int32)
        ov = out.copy()
        cv2.fillPoly(ov, [mapped], CV_YELLOW)
        cv2.addWeighted(ov, 0.16, out, 0.84, 0, out)
        cv2.polylines(out, [mapped], isClosed=True, color=CV_YELLOW, thickness=2)
        cx = int(mapped[:,0,0].mean()); cy = int(mapped[:,0,1].mean())
        cv2.putText(out, "GLINT", (cx-20, cy), CV_FONT, 0.5, CV_YELLOW, 1, cv2.LINE_AA)

    # ── 4. Centering strip ───────────────────────────────────
    cen_col = CV_GREEN if cen_sg >= 10.0 else CV_RED
    cv2.putText(out, f"CTR L/R {lr:.0%} T/B {tb:.0%} sub={cen_sg:.0f}",
                (8, h-10), CV_FONT, 0.44, cen_col, 1, cv2.LINE_AA)

    # ── 5. Dark Background Warning banner ─────────────────────
    if iso.isolation_conf < 0.5:
        banner_h = 38
        ov = out.copy()
        cv2.rectangle(ov, (0,0), (w, banner_h), CV_ORANGE, -1)
        cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
        cv2.putText(out,
                    "LOW CONTRAST — place card on dark surface for better detection",
                    (6, 24), CV_FONT, 0.46, (0,0,0), 1, cv2.LINE_AA)

    return out


def make_corner_patches_from_card(
    warped: np.ndarray,
    white_counts: list,
    pct: float = 0.09,
) -> list:
    """
    PATCH 2 — Extract corner patches from the WARPED card image.
    The patches snap to the actual card corners (not arbitrary image offsets).
    Apply Laplacian sharpening before display.
    """
    h, w = warped.shape[:2]
    ph   = max(int(h * pct), 30)
    pw   = max(int(w * pct), 30)
    raw  = [
        warped[0:ph,   0:pw  ],   # TL
        warped[0:ph,   w-pw:w],   # TR
        warped[h-ph:h, 0:pw  ],   # BL
        warped[h-ph:h, w-pw:w],   # BR
    ]
    out = []
    for patch, lbl, n in zip(raw, CORNER_LABELS, white_counts):
        if patch.size == 0:
            out.append(np.zeros((80,80,3), np.uint8)); continue
        # Sharpen
        disp = _sharpen_patch(patch)
        # Highlight white pixels
        bch, gch, rch = cv2.split(disp)
        wm = ((rch.astype(int)>WHITE_LV) &
              (gch.astype(int)>WHITE_LV) &
              (bch.astype(int)>WHITE_LV))
        disp[wm] = [0, 0, 255] if n > 0 else [0, 200, 80]
        # 4× upscale (nearest to preserve pixel evidence)
        scale = 4
        disp  = cv2.resize(disp, (disp.shape[1]*scale, disp.shape[0]*scale),
                           interpolation=cv2.INTER_NEAREST)
        col = CV_RED if n > 0 else CV_GREEN
        cv2.rectangle(disp, (0,0), (disp.shape[1]-1, disp.shape[0]-1), col, 3)
        cv2.putText(disp, f"{lbl} {n}px", (4,15), CV_FONT, 0.5, col, 1, cv2.LINE_AA)
        out.append(disp)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE FETCH
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
    if data is None: return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ═══════════════════════════════════════════════════════════════════════
#  FULL GRADING PIPELINE
#  Critical change: every check now runs on iso.warped, not the original
# ═══════════════════════════════════════════════════════════════════════

def grade_single_image(img_orig: np.ndarray, lbl: str) -> dict:
    """
    Run the full pipeline on ONE image.
    Returns a dict of worst-case accumulators for this image.
    """
    result = {
        "lr": 0.50, "tb": 0.50, "cen_sg": 10.0,
        "cor_sg": 10.0, "cor_counts": [0,0,0,0],
        "srf_sg": 10.0, "glint": False, "scratch": False,
        "clone": False, "rosette": False,
        "iso": None, "reasons": [],
    }

    # ── Step 1: Isolate card (PATCH 1) ───────────────────────
    iso = isolate_card(img_orig)
    result["iso"] = iso
    if iso.warning:
        result["reasons"].append(f"{lbl}: {iso.warning}")

    warped = iso.warped
    if warped is None or warped.size == 0:
        result["reasons"].append(f"{lbl}: ⚠ Warp failed — skipping this image")
        return result

    # ── Step 2: Centering on warped card ─────────────────────
    lr, tb, cen_sg, cen_r = run_centering_on_warped(warped)
    result["lr"], result["tb"], result["cen_sg"] = lr, tb, cen_sg
    result["reasons"].extend(f"{lbl}: {r}" for r in cen_r)

    # ── Step 3: Corners on warped card (PATCH 1+2) ───────────
    cor_counts = measure_corner_white_px_on_card(warped)
    cor_sg, cor_r = corner_px_to_subgrade(cor_counts)
    result["cor_sg"]     = cor_sg
    result["cor_counts"] = cor_counts
    result["reasons"].extend(f"{lbl}: {r}" for r in cor_r)

    # ── Step 4: Surface on warped card ───────────────────────
    glint, scratch, srf_r, srf_sg = check_surface_on_card(warped)
    result["srf_sg"] = srf_sg
    result["glint"]  = glint
    result["scratch"]= scratch
    result["reasons"].extend(f"{lbl}: {r}" for r in srf_r)

    # ── Step 5: Forensics on warped card ─────────────────────
    cl_f, cl_r = check_clone(warped)
    if cl_f: result["clone"] = True; result["reasons"].extend(cl_r)
    ro_f, ro_r = check_rosette(warped)
    if ro_f: result["rosette"] = True; result["reasons"].extend(ro_r)

    return result


def grade_images(image_urls: list) -> GradingResult:
    """Run pipeline across up to 6 images. Always returns a meaningful result."""
    gr = GradingResult()

    if not image_urls:
        gr.res_fail  = True
        gr.grade_dist = {g:(100.0 if g==1 else 0.0) for g in range(1,11)}
        gr.reasoning.append("No images in listing")
        gr.grade_range = "Unknown — no images"; gr.confidence_pct = 0
        return gr

    raw_images = []
    seen: set  = set()
    for url in image_urls[:6]:
        img = fetch_image(url)
        if img is None: continue
        h = hash_image(img)
        if h in seen:
            gr.reasoning.append(f"Duplicate/stock photo skipped")
            continue
        seen.add(h)
        raw_images.append(img)

    if not raw_images:
        gr.res_fail = True
        gr.grade_dist = {g:(100.0 if g==1 else 0.0) for g in range(1,11)}
        gr.reasoning.append("All image downloads failed or filtered")
        gr.grade_range = "Unknown — no usable images"; gr.confidence_pct = 0
        return gr

    # Pre-process
    proc_images = []
    for img in raw_images:
        p, score, quality = preprocess_image(img)
        proc_images.append((p, score, quality))

    # Resolution gate
    h0, w0 = proc_images[0][0].shape[:2]
    gr.res_px = (w0, h0)
    if min(h0, w0) < MIN_PX:
        gr.res_fail = True
        gr.reasoning.append(f"REJECTED: {w0}×{h0}px even after upscale")
        gr.grade_dist = {g:(100.0 if g==1 else 0.0) for g in range(1,11)}
        gr.grade_range = "Unknown — unusable image"; gr.confidence_pct = 0
        return gr

    if is_stock_photo(proc_images[0][0]):
        gr.image_quality = "stock"
        gr.reasoning.append("⚠ Image appears to be official/stock art")

    quality_rank  = {"ok":0,"low":1,"very_low":2}
    worst_quality = max(proc_images, key=lambda x: quality_rank.get(x[2],0))[2]
    gr.image_quality   = worst_quality
    gr.sharpness_score = min(x[1] for x in proc_images)

    if worst_quality == "very_low":
        gr.reasoning.append(
            f"⚠ Low sharpness (σ²={gr.sharpness_score:.1f}) — confidence reduced")

    # ── Accumulate worst-case across all images ───────────────
    worst = {
        "lr":0.50, "tb":0.50, "cen_sg":10.0,
        "cor_sg":10.0, "cor_counts":[0,0,0,0],
        "srf_sg":10.0, "glint":False, "scratch":False,
        "clone":False, "rosette":False,
        "iso":None,
    }

    for idx, (img, sharpness, quality) in enumerate(proc_images):
        r = grade_single_image(img, f"Img {idx+1}")
        gr.reasoning.extend(r["reasons"])

        if r["cen_sg"] < worst["cen_sg"]:
            worst["cen_sg"] = r["cen_sg"]
            worst["lr"]     = r["lr"]
            worst["tb"]     = r["tb"]
            worst["iso"]    = r["iso"]

        worst["cor_counts"] = [max(a,b) for a,b in
                               zip(worst["cor_counts"], r["cor_counts"])]
        if r["cor_sg"] < worst["cor_sg"]: worst["cor_sg"] = r["cor_sg"]
        if r["srf_sg"] < worst["srf_sg"]: worst["srf_sg"] = r["srf_sg"]
        if r["glint"]:   worst["glint"]   = True
        if r["scratch"]: worst["scratch"] = True
        if r["clone"]:   worst["clone"]   = True
        if r["rosette"]: worst["rosette"] = True

    # Build SubGrades
    sub = SubGrades(
        centering   = worst["cen_sg"],
        corners     = worst["cor_sg"],
        edges       = worst["srf_sg"],
        surface     = worst["srf_sg"],
        lr_ratio    = worst["lr"],
        tb_ratio    = worst["tb"],
        corner_px   = worst["cor_counts"],
    )
    if worst["clone"]:   sub.surface = min(sub.surface, 5.0)
    if worst["rosette"]: sub.surface = min(sub.surface, 7.0)
    gr.sub           = sub
    gr.centering_fail = sub.centering < 10.0
    gr.corners_fail   = sub.corners   < 10.0
    gr.glint_flag     = worst["glint"]
    gr.scratch_fail   = worst["scratch"]
    gr.clone_flag     = worst["clone"]
    gr.rosette_flag   = worst["rosette"]

    # Grade distribution
    gr.grade_dist, gr.confidence_pct, gr.grade_range = compute_grade_distribution(
        sub, worst_quality)
    gr.psa10 = gr.grade_dist.get(10, 0.0)

    # ── Build display images ──────────────────────────────────
    best_iso = worst["iso"]
    if best_iso and best_iso.warped is not None:
        gr.isolation = best_iso

        # Corner patches from warped card (PATCH 2)
        gr.corner_patches = make_corner_patches_from_card(
            best_iso.warped, worst["cor_counts"]
        )

        # Annotated original with dynamic boxes (PATCH 4)
        orig_img = proc_images[0][0]
        gr.annotated_orig = build_annotated_original(
            orig_img, best_iso,
            worst["cor_counts"], worst["cen_sg"],
            worst["lr"], worst["tb"], worst["glint"]
        )

        # Warped card with surface holo overlay
        warped_disp = best_iso.warped.copy()
        if worst["glint"]:
            wh, ww = warped_disp.shape[:2]
            wy1,wy2 = int(wh*0.18),int(wh*0.72)
            wx1,wx2 = int(ww*0.08),int(ww*0.92)
            ov = warped_disp.copy()
            cv2.rectangle(ov,(wx1,wy1),(wx2,wy2),CV_YELLOW,-1)
            cv2.addWeighted(ov,0.18,warped_disp,0.82,0,warped_disp)
            cv2.rectangle(warped_disp,(wx1,wy1),(wx2,wy2),CV_YELLOW,2)
        gr.warped_display = warped_disp

    # Overall verdict
    hard = [gr.centering_fail, gr.corners_fail, gr.scratch_fail,
            gr.res_fail, gr.clone_flag]
    gr.overall_pass = not any(hard)
    if gr.overall_pass:
        gr.reasoning.insert(0, f"✓ No hard-fail conditions — {gr.grade_range}")
    return gr


# ═══════════════════════════════════════════════════════════════════════
#  PATCH 3 — MARKET PRICING  (structured query + raw-price fallback)
# ═══════════════════════════════════════════════════════════════════════

def _extract_card_query(title: str) -> tuple[str, str]:
    """
    PATCH 3: Extract a clean '{CardName} {SetNumber}' from a messy listing title.
    Returns (clean_name, set_hint) where set_hint may be empty.
    """
    tl = title.lower()
    # Remove listing noise
    noise = r'(raw|ungraded|nm|near mint|lp|mp|hp|\bloc\b|lot|bundle|free\s*ship\w*'
    noise += r'|\bpsa\b|\bbgs\b|\bcgc\b|graded|proxy|replica|reprint|\blt\b|\[.*?\]|\(.*?\))'
    clean  = re.sub(noise, '', tl, flags=re.I).strip()
    clean  = re.sub(r'\s{2,}', ' ', clean).strip()[:65]

    # Extract set hint (look for known set keywords)
    set_keywords = [
        "base set","shadowless","jungle","fossil","team rocket",
        "neo genesis","neo revelation","neo destiny","neo discovery",
        "expedition","aquapolis","skyridge","1st edition","1st ed",
    ]
    set_hint = next((kw for kw in set_keywords if kw in tl), "")
    return clean, set_hint


@st.cache_data(ttl=600, show_spinner=False)
def _ebay_token() -> Optional[str]:
    if not EBAY_APP_ID or not EBAY_CERT_ID: return None
    creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
    try:
        r = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={"Content-Type":"application/x-www-form-urlencoded",
                     "Authorization":f"Basic {creds}"},
            data={"grant_type":"client_credentials",
                  "scope":"https://api.ebay.com/oauth/api_scope"},
            timeout=10)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _tier1_ebay_psa10(card_name: str, set_hint: str) -> MarketData:
    """
    PATCH 3: Structured query '{CardName} {SetHint} PSA 10 -raw -proxy'
    on graded-condition eBay listings.
    """
    token = _ebay_token()
    if not token: return MarketData()

    # Build structured query as specified
    base  = f"{card_name} {set_hint}".strip()
    query = f"{base} PSA 10 -raw -proxy -replica"
    surl  = (f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(base+' PSA 10')}"
             f"&LH_Sold=1&LH_Complete=1&_sacat=183454")
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization":f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID":"EBAY_US"},
            params={"q":query, "category_ids":"183454",
                    "filter":"conditionIds:{2750}",
                    "sort":"price", "limit":10},
            timeout=12)
        r.raise_for_status()
        prices = [float(i["price"]["value"])
                  for i in r.json().get("itemSummaries",[])
                  if i.get("price",{}).get("value")]
        if prices:
            return MarketData(psa10_price=round(float(np.median(prices)),2),
                              source="eBay Live (PSA 10 ask)",
                              search_url=surl, confidence="high")
    except Exception:
        pass
    return MarketData()


@st.cache_data(ttl=1800, show_spinner=False)
def _tier2_pricecharting(card_name: str, set_hint: str) -> MarketData:
    """
    PATCH 3: PriceCharting API — fetch BOTH grade-10-price AND used-price
    (raw market average). If grade-10-price = 0, derive from used-price × multiplier.
    """
    query = f"{card_name} {set_hint}".strip()
    surl  = (f"https://www.ebay.com/sch/i.html?_nkw="
             f"{quote_plus(query+' PSA 10')}&LH_Sold=1&_sacat=183454")
    try:
        r = requests.get(
            "https://www.pricecharting.com/api/product",
            params={"q": query, "status": "price"},
            headers={"User-Agent":"Mozilla/5.0"},
            timeout=10)
        if r.status_code == 200:
            data     = r.json()
            products = data.get("products", [data] if "graded-price" in data else [])
            for prod in products[:3]:
                # Try grade-10 price first
                g10 = prod.get("grade-10-price") or prod.get("graded-price") or 0
                used = prod.get("used-price") or 0
                try:    psa10_val = float(g10)  / 100.0
                except: psa10_val = 0.0
                try:    raw_val   = float(used) / 100.0
                except: raw_val   = 0.0

                # $0 fix: if PSA10 price missing, estimate from raw × set multiplier
                if psa10_val < 50 and raw_val > 0:
                    psa10_val = raw_val * 8.0   # conservative 8× multiplier

                if psa10_val > 50:
                    return MarketData(
                        psa10_price   = round(psa10_val, 2),
                        raw_price_ref = round(raw_val, 2),
                        source        = "PriceCharting API",
                        search_url    = surl,
                        confidence    = "medium",
                    )
    except Exception:
        pass
    return MarketData()


def _tier3_static(title: str) -> MarketData:
    """Static reference table — guaranteed non-zero."""
    tl  = title.lower()
    val = 0.0
    for key, v in sorted(STATIC_PSA10.items(), key=lambda x: -len(x[0])):
        if all(k in tl for k in key.split()):
            val = float(v); break
    if val == 0:
        if "1st edition" in tl or "1st ed" in tl: val = 2000.0
        elif "shadowless" in tl:                   val = 1500.0
        elif "shining"    in tl:                   val = 1500.0
        elif any(s in tl for s in ["lugia","ho-oh","entei","raikou","suicune"]): val=1200.0
        elif "holo" in tl and any(s in tl for s in [
            "base","jungle","fossil","rocket","neo","expedition","skyridge","aquapolis"]):
            val = 700.0
        else: val = 400.0
    surl = (f"https://www.ebay.com/sch/i.html?_nkw="
            f"{quote_plus(tl[:40]+' PSA 10')}&LH_Sold=1&_sacat=183454")
    return MarketData(psa10_price=val, source="Reference Table",
                      search_url=surl, confidence="low")


def get_market_data(title: str) -> MarketData:
    """3-tier waterfall using structured PATCH 3 queries. Never returns $0."""
    clean, set_hint = _extract_card_query(title)
    md = _tier1_ebay_psa10(clean, set_hint)
    if md.psa10_price > 0: return md
    md = _tier2_pricecharting(clean, set_hint)
    if md.psa10_price > 0: return md
    return _tier3_static(title)


def financials(price: float, md: MarketData, p10: float) -> tuple:
    """
    PATCH 4: ROI only calculated when BOTH listing price AND PSA10 value > $0.
    Returns (best_case, expected, roi_pct) — all 0.0 if price guard not met.
    """
    # Guard: skip ROI if price is zero (manual URL input) or PSA10 unknown
    if price <= 0 or md.psa10_price <= 0:
        return 0.0, 0.0, 0.0
    costs     = price + GRADING_FEE + SHIPPING_EST
    best_case = md.psa10_price - costs
    expected  = (md.psa10_price * p10 / 100.0) - costs
    roi_pct   = (best_case / price) * 100
    return round(best_case,2), round(expected,2), round(roi_pct,1)


# ─────────────────────────────────────────────────────────────────────
#  eBay SEARCH  (PATCH 3: structured query format)
# ─────────────────────────────────────────────────────────────────────

def search_ebay(query: str, min_p: float, max_p: float, limit: int) -> list:
    """
    PATCH 3: Structured search query:
      '{query} raw -graded -proxy -replica -reprint'
    Falls back to demo data when no API credentials are configured.
    """
    token = _ebay_token()
    if not token:
        st.info("ℹ️ No eBay credentials — running in **Demo Mode**.")
        return MOCK_LISTINGS[:limit]

    structured = f"{query} raw -graded -proxy -replica -reprint"
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization":f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID":"EBAY_US"},
            params={"q":structured, "category_ids":"183454",
                    "filter":f"price:[{min_p}..{max_p}],priceCurrency:USD",
                    "sort":"price", "limit":limit, "fieldgroups":"EXTENDED"},
            timeout=15)
        r.raise_for_status()
        out = []
        for item in r.json().get("itemSummaries",[]):
            imgs  = [item.get("image",{}).get("imageUrl","")]
            imgs += [i.get("imageUrl","") for i in item.get("additionalImages",[])]
            out.append({"title":item.get("title",""),
                        "price":float(item.get("price",{}).get("value",0)),
                        "url":item.get("itemWebUrl","#"),
                        "source":"eBay",
                        "image_urls":[u for u in imgs if u]})
        return out if out else MOCK_LISTINGS[:limit]
    except Exception as e:
        st.warning(f"eBay API error: {e} — using demo data")
        return MOCK_LISTINGS[:limit]


def grade_listing(raw: dict) -> ListingResult:
    lr         = ListingResult(**{k:raw[k] for k in
                                  ("title","price","url","source","image_urls")})
    lr.grading = grade_images(lr.image_urls)
    lr.market  = get_market_data(lr.title)
    lr.net_profit, lr.exp_profit, lr.roi_pct = financials(
        lr.price, lr.market, lr.grading.psa10)
    hard = [lr.grading.centering_fail, lr.grading.corners_fail,
            lr.grading.scratch_fail, lr.grading.res_fail, lr.grading.clone_flag]
    lr.verdict = ("FAIL" if any(hard)
                  else "WARN" if (lr.grading.glint_flag or lr.grading.rosette_flag)
                  else "PASS")
    return lr


# ═══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════
NEON = {"green":"#00ff88","yellow":"#f5c518","orange":"#ff7c00",
        "red":"#ff3c3c","blue":"#4da6ff","purple":"#b07dff"}


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
    color = _gc(sg*10)
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


def confidence_ring(psa10_pct: float, conf_pct: float):
    oc = _gc(psa10_pct); ic = _gc(conf_pct)
    R1,R2 = 40,28; S1,S2 = 8,7
    C1,C2 = 2*3.14159*R1, 2*3.14159*R2
    O1,O2 = C1*(1-psa10_pct/100), C2*(1-conf_pct/100)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin:6px 0;">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r="{R1}" fill="none" stroke="#111827" stroke-width="{S1}"/>
        <circle cx="55" cy="55" r="{R1}" fill="none" stroke="{oc}" stroke-width="{S1}"
          stroke-dasharray="{C1:.2f}" stroke-dashoffset="{O1:.2f}"
          stroke-linecap="round" transform="rotate(-90 55 55)"/>
        <circle cx="55" cy="55" r="{R2}" fill="none" stroke="#1a2a4a" stroke-width="{S2}"/>
        <circle cx="55" cy="55" r="{R2}" fill="none" stroke="{ic}" stroke-width="{S2}"
          stroke-dasharray="{C2:.2f}" stroke-dashoffset="{O2:.2f}"
          stroke-linecap="round" transform="rotate(-90 55 55)"/>
        <text x="55" y="50" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="14"
          fill="{oc}" font-weight="bold">{psa10_pct:.0f}%</text>
        <text x="55" y="62" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="7" fill="#556677">PSA 10</text>
        <text x="55" y="72" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="6"
          fill="{ic}">conf {conf_pct:.0f}%</text>
      </svg>
    </div>""", unsafe_allow_html=True)


def verdict_pill(v: str) -> str:
    cfg = {"PASS":("#002d1a",NEON["green"],"✅ PASS"),
           "WARN":("#2d1e00",NEON["yellow"],"⚠ WARN"),
           "FAIL":("#2d0000",NEON["red"],"❌ FAIL")}
    bg,col,txt = cfg.get(v,("#111","#aaa","⬜"))
    return (f'<span style="background:{bg};color:{col};border:1px solid {col};'
            f'border-radius:20px;padding:3px 14px;font-family:\'Share Tech Mono\','
            f'monospace;font-size:.73rem;font-weight:bold;">{txt}</span>')


def isolation_badge(conf: float) -> str:
    if conf >= 0.95: lbl,col = "Card Detected ✓", NEON["green"]
    elif conf >= 0.55: lbl,col = "Approx. Detection", NEON["yellow"]
    else: lbl,col = "No Card Found ⚠", NEON["red"]
    return (f'<span style="background:{col}22;color:{col};border:1px solid {col};'
            f'border-radius:10px;padding:1px 8px;font-family:\'Share Tech Mono\','
            f'monospace;font-size:.63rem;">{lbl} ({conf:.0%})</span>')


def src_badge(src: str, conf: str) -> str:
    col = {"high":NEON["green"],"medium":NEON["yellow"],"low":"#8899bb"}.get(conf,"#8899bb")
    return (f'<span style="background:{col}22;color:{col};border:1px solid {col};'
            f'border-radius:10px;padding:1px 8px;font-family:\'Share Tech Mono\','
            f'monospace;font-size:.63rem;">{src}</span>')


def render_subgrades(sub: SubGrades):
    rc = _gc(sub.overall*10)
    st.markdown(
        f'<div style="background:#090e1a;border:1px solid #1a2a4a;border-radius:8px;'
        f'padding:7px 12px;margin:4px 0;font-family:\'Share Tech Mono\',monospace;'
        f'font-size:.67rem;color:#4a7fc1;">'
        f'SUB-GRADES (Rule of Minimum) → '
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
             f'🔗 PSA 10 sold</a>') if md.search_url else ""
    # Raw price reference (if available from PriceCharting)
    raw_ref_row = ""
    if md.raw_price_ref > 0:
        raw_ref_row = (f'<span style="color:#556677;">Raw Avg Ref  </span>'
                       f'<b style="color:#8899bb;">${md.raw_price_ref:,.0f}</b><br>')
    # PATCH 4: ROI rows only shown when price > 0
    roi_rows = ""
    if price > 0 and md.psa10_price > 0:
        roi_rows = (
            f'<span style="color:#556677;">Best-Case  </span>'
            f'<b style="color:{pc};font-size:.95rem;">${net:,.0f}</b><br>'
            f'<span style="color:#556677;">Expected   </span>'
            f'<b style="color:{pc};">${exp:,.0f}</b><br>'
            f'<span style="color:#556677;">ROI        </span>'
            f'<b style="color:{pc};">{roi:.0f}%</b>'
        )
    else:
        roi_rows = ('<span style="color:#556677;">ROI        </span>'
                    '<b style="color:#556677;">— (enter listing price)</b>')
    price_str = f"${price:.2f}" if price > 0 else "—"
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.77rem;line-height:2.1;
                background:#090e1a;border-radius:8px;border:1px solid #1a2a4a;padding:10px 14px;">
      <div style="margin-bottom:5px;">{badge}{link}</div>
      <span style="color:#556677;">Raw Price  </span><b style="color:#d4e1f5;">{price_str}</b><br>
      {raw_ref_row}
      <span style="color:#556677;">PSA 10 Est </span>
        <b style="color:{NEON['blue']};font-size:.9rem;">${md.psa10_price:,.0f}</b><br>
      <span style="color:#556677;">Grading    </span><b style="color:#556677;">−${GRADING_FEE:.0f}</b><br>
      <span style="color:#556677;">Shipping   </span><b style="color:#556677;">−${SHIPPING_EST:.0f}</b><br>
      {roi_rows}
    </div>""", unsafe_allow_html=True)


def render_corner_panels(patches: list):
    if len(patches) < 4: return
    cols = st.columns(4)
    for col, patch, lbl in zip(cols, patches[:4], CORNER_LABELS):
        with col:
            if patch.size > 0:
                col.image(bgr_to_pil(patch), caption=f"{lbl} 400%",
                          use_container_width=True)


def render_grading_card(r: ListingResult, expanded: bool = False):
    gr  = r.grading
    v   = r.verdict
    lbl = (f"{'✅' if v=='PASS' else '⚠️' if v=='WARN' else '❌'} "
           f"{r.title[:60]}  — ${r.price:.2f}")

    with st.expander(lbl, expanded=expanded):
        # Dark background warning at top of card
        if gr and gr.isolation and gr.isolation.isolation_conf < 0.5:
            st.warning(gr.isolation.warning or
                       "⚠ No card outline detected — place card on a dark surface")

        c1, c2, c3 = st.columns([1.5, 1, 1.4])
        with c1:
            if gr and gr.annotated_orig is not None:
                st.image(bgr_to_pil(gr.annotated_orig), use_container_width=True,
                         caption="Dynamic boxes snap to card edges — green=pass · red=fail")
            elif r.image_urls:
                st.image(r.image_urls[0], use_container_width=True)

            iso_conf = gr.isolation.isolation_conf if (gr and gr.isolation) else 0.0
            st.markdown(
                f'{verdict_pill(v)}&nbsp;{isolation_badge(iso_conf)}<br>'
                f'<a href="{r.url}" target="_blank" style="display:inline-block;'
                f'margin-top:8px;background:{NEON["blue"]}22;border:1px solid {NEON["blue"]};'
                f'color:{NEON["blue"]};padding:4px 12px;border-radius:6px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
                f'text-decoration:none;">🛒 ONE-TAP BUY</a>',
                unsafe_allow_html=True)

        with c2:
            if gr:
                confidence_ring(gr.psa10, gr.confidence_pct)
                gr_col = _gc(gr.psa10)
                st.markdown(
                    f'<div style="text-align:center;font-family:\'Share Tech Mono\','
                    f'monospace;font-size:.67rem;color:{gr_col};">{gr.grade_range}</div>',
                    unsafe_allow_html=True)
                render_subgrades(gr.sub)

        with c3:
            if r.market:
                render_market(r.market, r.price, r.net_profit, r.exp_profit, r.roi_pct)

        if gr:
            d = gr.grade_dist
            st.markdown("**Grade Distribution**  *(Gaussian · Rule of Minimum)*")
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
                    "**4-Corner Zoom (400%) — snapped to isolated card corners**  "
                    "`red = whitening · green = clean`")
                render_corner_panels(gr.corner_patches)

            if gr.warped_display is not None:
                with st.expander("🔲 Isolated card (perspective-corrected)"):
                    st.image(bgr_to_pil(gr.warped_display),
                             caption="All grading was performed on this image — not the background",
                             use_container_width=True)

            if gr.reasoning:
                st.markdown("**AI Reasoning Report**")
                for note in gr.reasoning:
                    ok   = note.startswith("✓")
                    warn = "⚠" in note or "WARNING" in note.upper() or "🌑" in note
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

# ─────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────
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
    v5.0 · CARD ISOLATION · DYNAMIC CORNERS · STRUCTURED SEARCH · DARK BG WARNING
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#00ff8844,#4da6ff33,transparent);
            margin-bottom:.8rem;"></div>
""", unsafe_allow_html=True)

if "notifications" not in st.session_state: st.session_state.notifications = []
if "last_results"  not in st.session_state: st.session_state.last_results  = []

def push_notif(msg):
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

# ─────────────────────────────────────────────────────────────────────
#  TAB 1 — LIVE SCAN
# ─────────────────────────────────────────────────────────────────────
with tab_scan:
    st.markdown("### eBay Live Scanner")
    st.caption("Structured query: `{query} raw -graded -proxy -replica -reprint`")
    r1c1,r1c2,r1c3,r1c4 = st.columns([3,1,1,1])
    with r1c1: query     = st.text_input("Search Query","Charizard Holo Base Set Unlimited")
    with r1c2: min_price = st.number_input("Min $", value=1,   min_value=0)
    with r1c3: max_price = st.number_input("Max $", value=100, min_value=1)
    with r1c4: num_res   = st.number_input("Limit", value=8,   min_value=1, max_value=50)

    r2c1,r2c2 = st.columns(2)
    with r2c1: min_psa10  = st.number_input("Min PSA 10 Value ($)", value=1000, min_value=0)
    with r2c2: min_profit = st.number_input("Min Best-Case Profit ($)", value=200, min_value=0)

    if st.button("▶  RUN LIVE SCAN", use_container_width=True):
        with st.spinner("Fetching listings…"):
            raw_list = search_ebay(query, min_price, max_price, num_res)
        pb = st.progress(0.0); status = st.empty()
        results = []
        for i, raw in enumerate(raw_list):
            status.markdown(
                f"<span style='font-family:\"Share Tech Mono\",monospace;"
                f"font-size:.78rem;color:#4da6ff;'>"
                f"Grading {i+1}/{len(raw_list)}: {raw['title'][:50]}…</span>",
                unsafe_allow_html=True)
            pb.progress((i+1)/max(len(raw_list),1))
            lr = grade_listing(raw)
            if lr.grading and lr.grading.psa10 >= 70 and lr.verdict == "PASS":
                push_notif(f"🔥 {lr.grading.psa10:.0f}% PSA 10 · {lr.title[:40]} · ${lr.price}")
            results.append(lr)
        pb.empty(); status.empty()
        st.session_state.last_results = results

        n_p = sum(1 for r in results if r.verdict=="PASS")
        n_w = sum(1 for r in results if r.verdict=="WARN")
        n_f = sum(1 for r in results if r.verdict=="FAIL")
        best = max((r for r in results if r.verdict=="PASS"),
                   key=lambda r: r.net_profit, default=None)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Scanned",    len(results))
        m2.metric("✅ Pass",     n_p)
        m3.metric("⚠️ Warn",     n_w)
        m4.metric("❌ Rejected", n_f)
        m5.metric("Best Profit",f"${best.net_profit:,.0f}" if best else "—")
        st.markdown("---")
        results.sort(key=lambda r:({"PASS":0,"WARN":1,"FAIL":2}.get(r.verdict,3),-r.net_profit))
        for r in results:
            render_grading_card(r, expanded=(r.verdict in ("PASS","WARN")))

# ─────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch URL Grader")
    st.caption(
        "Paste eBay listing URLs or direct image URLs (JPG/PNG). "
        "For other sites: right-click card image → 'Copy image address' → paste here."
    )
    batch_input = st.text_area("URLs — one per line", height=160,
        placeholder="https://www.ebay.com/itm/…\nhttps://i.ebayimg.com/…")
    if st.button("▶  GRADE ALL & RANK", use_container_width=True):
        urls = [u.strip() for u in batch_input.splitlines()
                if u.strip() and not u.strip().startswith("#")][:10]
        if not urls:
            st.warning("Paste at least one URL.")
        else:
            batch_res = []
            bp = st.progress(0.0)
            for i, url in enumerate(urls):
                bp.progress((i+1)/len(urls))
                is_img = any(url.lower().endswith(e) for e in [".jpg",".jpeg",".png",".webp"])
                raw = {"title":urlparse(url).path[-60:] or f"Listing {i+1}",
                       "price":0.0,"url":url,"source":"Manual",
                       "image_urls":[url] if is_img else []}
                if not is_img:
                    try:
                        pg = requests.get(url,headers=HEADERS,timeout=10)
                        imgs = re.findall(r'https://i\.ebayimg\.com/images/g/[^"\']+\.jpg',pg.text)
                        raw["image_urls"] = list(dict.fromkeys(imgs))[:5]
                        m = re.search(r'"price"\s*:\s*\{\s*"value"\s*:\s*"?([\d.]+)',pg.text)
                        if m: raw["price"] = float(m.group(1))
                        m2 = re.search(r'<h1[^>]*itemprop="name"[^>]*>([^<]+)',pg.text)
                        if m2: raw["title"] = m2.group(1).strip()[:80]
                    except Exception: pass
                batch_res.append(grade_listing(raw))
            bp.empty()
            batch_res.sort(key=lambda r:-(r.grading.psa10 if r.grading else 0))
            medals = ["🥇","🥈","🥉"]+["🔹"]*10
            rows = [{
                "Rank":f"{medals[i]} #{i+1}","Card":r.title[:42],
                "Price":f"${r.price:.2f}" if r.price>0 else "—",
                "PSA 10%":f"{r.grading.psa10:.1f}%" if r.grading else "—",
                "Conf.":f"{r.grading.confidence_pct:.0f}%" if r.grading else "—",
                "Grade Range":r.grading.grade_range if r.grading else "—",
                "PSA10 Est":f"${r.market.psa10_price:,.0f}" if r.market else "—",
                "Best Profit":f"${r.net_profit:,.0f}" if r.net_profit != 0 else "—",
                "Verdict":r.verdict,
            } for i,r in enumerate(batch_res)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            for r in batch_res: render_grading_card(r, expanded=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 3 — SINGLE CARD
# ─────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### 🔬 Single Card Deep Analysis")
    mode = st.radio("Input", ["📁 Upload Photo","🔗 Paste Image URL"], horizontal=True)
    single_img  = None
    single_urls = []

    if "Upload" in mode:
        up = st.file_uploader("Drop card photo (JPG/PNG)", type=["jpg","jpeg","png"])
        if up:
            pil        = Image.open(up).convert("RGB")
            single_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        url_in = st.text_input("Image URL", placeholder="https://i.ebayimg.com/…")
        if url_in.strip(): single_urls = [url_in.strip()]

    sc1,sc2 = st.columns(2)
    with sc1: s_title = st.text_input("Card Name (for pricing)", placeholder="Charizard Base Set Holo Rare")
    with sc2: s_price = st.number_input("Raw Price ($)", value=0.0, min_value=0.0,
                                         help="Leave at $0 to skip ROI calculation")

    if st.button("▶  ANALYSE CARD", use_container_width=True):
        if single_img is None and not single_urls:
            st.error("Upload an image or paste a URL.")
        else:
            with st.spinner("Isolating card + running CV pipeline…"):
                if single_img is not None:
                    # Build result directly from uploaded image
                    gr  = GradingResult()
                    sub = SubGrades()
                    proc, sharpness, quality = preprocess_image(single_img)
                    gr.image_quality   = quality
                    gr.sharpness_score = sharpness
                    gr.res_px          = (proc.shape[1], proc.shape[0])

                    if is_stock_photo(proc):
                        gr.image_quality = "stock"
                        gr.reasoning.append("⚠ Appears to be official/stock art")

                    # Isolate card (PATCH 1)
                    iso = isolate_card(proc)
                    gr.isolation = iso
                    if iso.warning:
                        gr.reasoning.append(iso.warning)

                    warped = iso.warped or proc
                    # Centering on warped
                    lr, tb, cen_sg, cen_r = run_centering_on_warped(warped)
                    sub.centering = cen_sg; sub.lr_ratio = lr; sub.tb_ratio = tb
                    gr.reasoning.extend(cen_r)
                    # Corners on warped (PATCH 2)
                    counts = measure_corner_white_px_on_card(warped)
                    cor_sg, cor_r = corner_px_to_subgrade(counts)
                    sub.corners = cor_sg; sub.corner_px = counts
                    gr.reasoning.extend(cor_r)
                    # Surface on warped
                    glint, scratch, srf_r, srf_sg = check_surface_on_card(warped)
                    sub.edges = srf_sg; sub.surface = srf_sg
                    gr.glint_flag = glint; gr.scratch_fail = scratch
                    gr.reasoning.extend(srf_r)
                    # Forensics
                    cl_f, cl_r = check_clone(warped)
                    gr.clone_flag = cl_f; gr.reasoning.extend(cl_r)
                    if cl_f: sub.surface = min(sub.surface, 5.0)

                    gr.sub = sub
                    gr.centering_fail = sub.centering < 10.0
                    gr.corners_fail   = sub.corners   < 10.0
                    gr.grade_dist, gr.confidence_pct, gr.grade_range = \
                        compute_grade_distribution(sub, quality)
                    gr.psa10 = gr.grade_dist.get(10, 0.0)

                    # Annotate (PATCH 4)
                    gr.annotated_orig = build_annotated_original(
                        proc, iso, counts, cen_sg, lr, tb, glint)
                    gr.corner_patches = make_corner_patches_from_card(warped, counts)

                    # Warped display
                    wd = warped.copy()
                    if glint:
                        wh,ww = wd.shape[:2]
                        wy1,wy2 = int(wh*0.18),int(wh*0.72)
                        wx1,wx2 = int(ww*0.08),int(ww*0.92)
                        ov=wd.copy(); cv2.rectangle(ov,(wx1,wy1),(wx2,wy2),CV_YELLOW,-1)
                        cv2.addWeighted(ov,0.18,wd,0.82,0,wd)
                        cv2.rectangle(wd,(wx1,wy1),(wx2,wy2),CV_YELLOW,2)
                    gr.warped_display = wd
                    hard = [gr.centering_fail, gr.corners_fail, gr.scratch_fail, gr.clone_flag]
                    gr.overall_pass = not any(hard)
                else:
                    gr = grade_images(single_urls)

            md  = get_market_data(s_title)
            net, exp, roi = financials(s_price, md, gr.psa10)
            v   = ("PASS" if gr.overall_pass and not gr.clone_flag
                   else "WARN" if gr.glint_flag else "FAIL")

            # Dark Background Warning (PATCH 4)
            if gr.isolation and gr.isolation.isolation_conf < 0.5:
                st.warning(gr.isolation.warning or
                           "⚠ No card outline detected — place card on a dark surface")

            col_img, col_right = st.columns([1.6, 1])
            with col_img:
                if gr.annotated_orig is not None:
                    st.image(bgr_to_pil(gr.annotated_orig), use_container_width=True,
                             caption="Dynamic boxes track card edges — green=pass · red=fail")
                elif single_img is not None:
                    st.image(bgr_to_pil(single_img), width=280)

            with col_right:
                st.markdown(f"## {verdict_pill(v)}", unsafe_allow_html=True)
                iso_c = gr.isolation.isolation_conf if gr.isolation else 0.0
                st.markdown(isolation_badge(iso_c), unsafe_allow_html=True)
                confidence_ring(gr.psa10, gr.confidence_pct)
                gr_col = _gc(gr.psa10)
                st.markdown(
                    f'<div style="text-align:center;font-family:\'Share Tech Mono\','
                    f'monospace;font-size:.68rem;color:{gr_col};">{gr.grade_range}</div>',
                    unsafe_allow_html=True)
                render_subgrades(gr.sub)

            render_market(md, s_price, net, exp, roi)

            d = gr.grade_dist
            st.markdown("**Grade Distribution**")
            ca,cb = st.columns(2)
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
                st.markdown("**4-Corner Zoom (400%) — snapped to card corners, Laplacian-sharpened**")
                render_corner_panels(gr.corner_patches)

            if gr.warped_display is not None:
                with st.expander("🔲 Isolated card (perspective-corrected, graded on this)"):
                    st.image(bgr_to_pil(gr.warped_display), use_container_width=True)

            st.markdown("**AI Reasoning Report**")
            for note in gr.reasoning:
                ok   = note.startswith("✓")
                warn = "⚠" in note or "WARNING" in note.upper() or "🌑" in note
                icon = "🟢" if ok else ("🟡" if warn else "🔴")
                st.markdown(f"{icon} `{note}`")

# ─────────────────────────────────────────────────────────────────────
#  TAB 4 — LEADERBOARD
# ─────────────────────────────────────────────────────────────────────
with tab_lead:
    st.markdown("### 🏆 Scan Leaderboard")
    last = st.session_state.last_results
    if not last:
        st.info("Run a Live Scan first to populate the leaderboard.")
    else:
        ranked = sorted(
            [r for r in last if r.verdict != "FAIL"],
            key=lambda r: -(r.grading.psa10*(r.net_profit if r.net_profit>0 else 1)
                            if r.grading else 0)
        )
        medals = ["🥇","🥈","🥉"]+["🔹"]*20
        rows   = [{
            "Rank":f"{medals[i]} #{i+1}", "Card":r.title[:42],
            "Raw $":f"${r.price:.2f}",
            "PSA 10%":f"{r.grading.psa10:.1f}%" if r.grading else "—",
            "Conf.":f"{r.grading.confidence_pct:.0f}%" if r.grading else "—",
            "Grade Range":r.grading.grade_range if r.grading else "—",
            "PSA10 $":f"${r.market.psa10_price:,.0f}" if r.market else "—",
            "Net Profit":f"${r.net_profit:,.0f}" if r.net_profit!=0 else "—",
            "ROI":f"{r.roi_pct:.0f}%" if r.roi_pct!=0 else "—",
            "Source":r.market.source if r.market else "—",
        } for i,r in enumerate(ranked)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        for r in ranked[:3]: render_grading_card(r, expanded=True)

# ─────────────────────────────────────────────────────────────────────
#  TAB 5 — GAP FINDER
# ─────────────────────────────────────────────────────────────────────
with tab_gap:
    st.markdown("### 💰 High-ROI Gap Dashboard")
    sample = [
        ("Charizard 1st Ed Base",       8000,350000),("Charizard Shadowless",1200,50000),
        ("Charizard Base Unlimited",      200, 5000),("Lugia Neo Genesis",   150,12000),
        ("Ho-Oh Neo Revelation",           90, 3000),("Shining Charizard",   300, 8000),
        ("Dark Charizard Team Rocket",    150, 2500),("Gengar Fossil Holo",   80, 2500),
        ("Scyther Jungle Holo",            30,  600),("Snorlax Jungle Holo",  55, 1200),
        ("Raichu Base Set Holo",          200, 1800),
    ]
    df = pd.DataFrame([{"Card":n,"Raw":f"${r:,}","PSA10":f"${p:,}",
                         "Spread":f"${p-r:,}","×":f"{p//r}×"} for n,r,p in sample])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    for entry in HIGH_ROI_CATALOG:
        with st.expander(f"📦 {entry['set']} — {entry['spread']}"):
            st.markdown(f"*{entry['note']}*")
            cols = st.columns(min(len(entry["cards"]),4))
            for col,card in zip(cols, entry["cards"]):
                link = (f"https://www.ebay.com/sch/i.html?_nkw="
                        f"{quote_plus(card+' '+entry['set'].split()[0]+' pokemon raw ungraded')}"
                        f"&_sacat=183454&LH_BIN=1")
                col.markdown(f"[**{card}**]({link})")

# ─────────────────────────────────────────────────────────────────────
#  TAB 6 — ALERTS
# ─────────────────────────────────────────────────────────────────────
with tab_notif:
    n = len(st.session_state.notifications)
    st.markdown(f"### 🔔 Alert Feed ({n})")
    if not st.session_state.notifications:
        st.info("No alerts — run a Live Scan.")
    else:
        for note in st.session_state.notifications:
            st.markdown(
                f'<div style="background:#001a0a;border:1px solid #00ff8833;'
                f'border-radius:6px;padding:7px 12px;margin-bottom:5px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.78rem;'
                f'color:#00ff88;">{note}</div>', unsafe_allow_html=True)
        if st.button("Clear Alerts"):
            st.session_state.notifications = []; st.rerun()

# ─────────────────────────────────────────────────────────────────────
#  TAB 7 — CONFIG
# ─────────────────────────────────────────────────────────────────────
with tab_cfg:
    st.markdown("### ⚙️ Configuration")
    st.code("""
# config.yaml
ebay:
  app_id:  "YOUR-EBAY-APP-ID"
  cert_id: "YOUR-EBAY-CERT-ID"
  dev_id:  "YOUR-EBAY-DEV-ID"
""", language="yaml")
    st.code("""
# .streamlit/secrets.toml  (Streamlit Cloud)
[ebay]
app_id  = "YOUR-EBAY-APP-ID"
cert_id = "YOUR-EBAY-CERT-ID"
dev_id  = "YOUR-EBAY-DEV-ID"
""", language="toml")

    st.markdown("---")
    st.markdown("### v5.0 Patch Reference")
    st.markdown("""
**PATCH 1 — Card Isolation (fixes red boxes scanning background)**

`isolate_card()` introduces a 3-tier detection strategy:
- *Tier A (conf=1.0)*: Adaptive Canny (Otsu thresholds) → dilate → largest 4-point `approxPolyDP` quad
- *Tier B (conf=0.6)*: Bounding-rect of largest contour if no 4-pt quad found
- *Tier C (conf=0.3)*: Whole-image fallback + **Dark Background Warning** displayed in UI

The warped card is stored in `IsolationResult.warped`. Every downstream check
(`measure_corner_white_px_on_card`, `check_surface_on_card`, `check_clone`,
`check_rosette`) now receives `warped`, never the original background image.

---

**PATCH 2 — Dynamic Corner Mapping**

`make_corner_patches_from_card()` extracts patches from `iso.warped` directly —
the 4 patches snap to the exact corners of the isolated card rectangle, not a
fixed 9% of the full image. Laplacian sharpening is applied to every patch before
400% upscale and display.

`build_annotated_original()` maps corner box positions back from the warped card
to the original image by projecting from each `quad_on_orig` corner point, so the
boxes move with the card regardless of angle or position in frame.

---

**PATCH 3 — Search Query Logic**

`_extract_card_query()` strips listing noise (raw, ungraded, NM, LP, etc.) and
extracts a set-keyword hint. The eBay query is built as:
`"{CardName} {SetHint} PSA 10 -raw -proxy -replica"`

`_tier2_pricecharting()` now reads both `grade-10-price` AND `used-price`.
If PSA 10 price is missing or $0, it derives an estimate from `used-price × 8`
(conservative multiplier). This eliminates the $0.00 pricing bug.

---

**PATCH 4 — UI Cleanup**

- Fixed-coordinate red boxes removed entirely from `annotate_full()`.
- `build_annotated_original()` draws corner boxes that project from the detected
  `quad_on_orig` corners, so they always track the card's actual edges.
- `financials()` now returns `(0, 0, 0)` if `price ≤ 0` or `psa10_price ≤ 0`.
  ROI rows in `render_market()` display "— (enter listing price)" instead of
  a misleading $0 calculation.
- `isolation_badge()` displays the detection confidence in the UI header of every card.
- Dark Background Warning shown as a Streamlit `st.warning()` banner on the card
  when `isolation_conf < 0.5`.
""")

# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#182038,transparent);
            margin:2rem 0 .8rem;"></div>
<div style="text-align:center;font-family:'Share Tech Mono',monospace;
            font-size:.62rem;color:#182038;padding-bottom:.8rem;">
  PSA 10 PROFIT SCANNER v5.0 · For research/educational use only
  · Not financial advice · Always verify manually
</div>
""", unsafe_allow_html=True)
