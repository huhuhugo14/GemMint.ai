"""
╔══════════════════════════════════════════════════════════════════╗
║  PSA 10 PROFIT SCANNER  —  ELITE EDITION  v3.0                  ║
║  Fix 1: Perspective-Transform Centering                          ║
║  Fix 2: Sub-Grade System + Rule of Minimum + Gaussian Dist.     ║
║  Fix 3: Live Market Pricing (3-tier, never $0)                  ║
║  Fix 4: Visual Bounding Boxes on full annotated image           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import requests
import yaml, os, re, base64, hashlib
import pandas as pd
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urlparse
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSA 10 Profit Scanner v3",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CONFIG LOADER
# ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    cfg = {}
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

cfg          = load_config()
EBAY_APP_ID  = cfg.get("ebay", {}).get("app_id",  os.getenv("EBAY_APP_ID",  ""))
EBAY_CERT_ID = cfg.get("ebay", {}).get("cert_id", os.getenv("EBAY_CERT_ID", ""))

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
MIN_PX         = 1000
WHITE_LV       = 215     # pixel brightness = "white"
GLINT_LV       = 245     # pixel brightness = glare
GLINT_FRAC     = 0.02    # fraction of holo area overexposed → glint
SCRATCH_DENS   = 0.003   # Laplacian density in glare zone → scratch
CLONE_BS       = 16      # block size for clone-stamp detection
ROSETTE_FREQ   = 45      # FFT band for CMYK rosette check
GRADING_FEE    = 25.0
SHIPPING_EST   = 6.0
CORNER_LABELS  = ["TL", "TR", "BL", "BR"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# OpenCV drawing colours (BGR)
CV_GREEN  = (0, 220, 100)
CV_RED    = (0, 60,  255)
CV_YELLOW = (0, 200, 255)
CV_WHITE  = (230, 230, 230)
CV_FONT   = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class SubGrades:
    """PSA 4-category sub-grades, each 1.0–10.0."""
    centering: float = 10.0
    corners:   float = 10.0
    edges:     float = 10.0
    surface:   float = 10.0
    overall:   float = 10.0   # Rule of Minimum: min of above four
    # Raw measurements for display
    lr_ratio:  float = 0.50
    tb_ratio:  float = 0.50
    corner_white_px: list = field(default_factory=lambda: [0, 0, 0, 0])

@dataclass
class GradingResult:
    sub:            SubGrades = field(default_factory=SubGrades)
    # Flags
    centering_flag: bool  = False
    corners_flag:   bool  = False
    scratch_flag:   bool  = False
    glint_flag:     bool  = False
    clone_flag:     bool  = False
    rosette_flag:   bool  = False
    stock_photo:    bool  = False
    res_flag:       bool  = False
    res_px:         tuple = (0, 0)
    # Perspective transform outputs
    quad_pts:       Optional[np.ndarray] = None
    warped_img:     Optional[np.ndarray] = None
    # Annotated images for display
    annotated_full: Optional[np.ndarray] = None
    corner_patches: list = field(default_factory=list)
    # Grade distribution grades 1–10 → probability %
    grade_dist:     dict = field(default_factory=dict)
    # Convenience aliases
    psa10: float = 0.0
    psa9:  float = 0.0
    psa8:  float = 0.0
    psa7:  float = 0.0
    psa6:  float = 0.0
    psa5:  float = 0.0
    psa1_4:float = 0.0
    # Outcome
    overall_pass: bool  = False
    confidence:   float = 0.0
    reasoning:    list  = field(default_factory=list)

@dataclass
class MarketData:
    psa10_price:  float = 0.0
    source:       str   = "static_table"
    search_url:   str   = ""
    confidence:   str   = "low"   # low | medium | high

@dataclass
class ListingResult:
    title:      str   = ""
    price:      float = 0.0
    url:        str   = "#"
    source:     str   = "eBay"
    image_urls: list  = field(default_factory=list)
    grading:    Optional[GradingResult] = None
    market:     Optional[MarketData]    = None
    net_profit: float = 0.0
    exp_profit: float = 0.0
    roi_pct:    float = 0.0
    verdict:    str   = "PENDING"

# ─────────────────────────────────────────────────────────────
#  STATIC PRICE TABLE  (Tier-3 fallback — never $0)
# ─────────────────────────────────────────────────────────────
STATIC_PSA10 = {
    "charizard 1st":       350000, "charizard shadowless":  50000,
    "charizard base":        5000, "charizard skyridge":    15000,
    "charizard aquapolis":   8000, "blastoise 1st":         25000,
    "blastoise shadowless":  8000, "blastoise base":          1200,
    "venusaur 1st":         12000, "venusaur base":             900,
    "lugia neo":            12000, "ho-oh neo":               3000,
    "entei neo":             1800, "raikou neo":              1500,
    "suicune neo":           1200, "shining charizard":       8000,
    "shining magikarp":      3500, "shining gyarados":        2000,
    "dark charizard":        2500, "dark blastoise":          1800,
    "gengar fossil":         2500, "lapras fossil":             800,
    "moltres fossil":          700, "zapdos fossil":             700,
    "scyther jungle":          600, "clefable jungle":           700,
    "snorlax jungle":         1200, "vaporeon jungle":            900,
    "jolteon jungle":           800, "raichu base":              1800,
    "machamp 1st":            2000, "ninetales base":             900,
}

HIGH_ROI_CATALOG = [
    {"set":"Base Set (1st Edition)","cards":["Charizard","Blastoise","Venusaur","Raichu"],
     "est_spread":"$5k–$500k+","note":"Crown jewel — any 1st Ed holo has massive PSA 10 premium"},
    {"set":"Base Set Shadowless","cards":["Charizard","Blastoise","Venusaur","Ninetales"],
     "est_spread":"$1k–$50k","note":"Shadowless holos gem mint are extremely scarce"},
    {"set":"Base Set Unlimited","cards":["Charizard","Blastoise","Raichu","Machamp"],
     "est_spread":"$500–$5k","note":"High volume raw market, gem copies still rare"},
    {"set":"Jungle","cards":["Scyther","Clefable","Snorlax","Vaporeon","Jolteon"],
     "est_spread":"$300–$1.2k","note":"Notorious centering issues → PSA 10 scarcity premium"},
    {"set":"Fossil","cards":["Gengar","Lapras","Moltres","Zapdos","Articuno"],
     "est_spread":"$400–$2.5k","note":"Thick cards prone to edge dings — gem copies command premium"},
    {"set":"Team Rocket","cards":["Dark Charizard","Dark Blastoise","Here Comes Team Rocket!"],
     "est_spread":"$800–$2.5k","note":"Dark surface shows scratches easily"},
    {"set":"Neo Genesis","cards":["Lugia","Typhlosion","Feraligatr","Meganium"],
     "est_spread":"$500–$12k","note":"Lugia PSA 10 regularly $10k+ — raw copies under $200"},
    {"set":"Neo Revelation","cards":["Ho-Oh","Entei","Raikou","Suicune"],
     "est_spread":"$600–$3k","note":"Legendary beasts — massive PSA 10 upside"},
    {"set":"Neo Destiny","cards":["Shining Charizard","Shining Magikarp","Dark Espeon"],
     "est_spread":"$1k–$8k","note":"Shining series = rarest WotC PSA 10s"},
    {"set":"Aquapolis / Skyridge","cards":["Charizard","Articuno","Celebi","Jolteon"],
     "est_spread":"$2k–$15k","note":"e-Reader era print lines → PSA 10 ultra-rare"},
]

MOCK_LISTINGS = [
    {"title":"[DEMO] Charizard Base Set Holo Unlimited — Raw NM","price":79.99,
     "url":"https://www.ebay.com/sch/i.html?_nkw=charizard+base+set+holo+raw","source":"eBay (Demo)",
     "image_urls":["https://images.pokemontcg.io/base1/4_hires.png"]},
    {"title":"[DEMO] Lugia Neo Genesis Holo — PSA Ready Raw","price":145.00,
     "url":"https://www.ebay.com/sch/i.html?_nkw=lugia+neo+genesis+holo+raw","source":"eBay (Demo)",
     "image_urls":["https://images.pokemontcg.io/neo1/9_hires.png"]},
    {"title":"[DEMO] Shining Charizard Neo Destiny — Ungraded RAW","price":290.00,
     "url":"https://www.ebay.com/sch/i.html?_nkw=shining+charizard+neo+destiny+raw","source":"eBay (Demo)",
     "image_urls":["https://images.pokemontcg.io/neo4/107_hires.png"]},
    {"title":"[DEMO] Gengar Fossil Holo Rare — HP","price":24.99,
     "url":"https://www.ebay.com/sch/i.html?_nkw=gengar+fossil+holo+raw","source":"eBay (Demo)",
     "image_urls":["https://images.pokemontcg.io/fossil/5_hires.png"]},
]

# ═══════════════════════════════════════════════════════════════════
#  FIX 1 — PERSPECTIVE-TRANSFORM CENTERING
# ═══════════════════════════════════════════════════════════════════

def order_quad_pts(pts: np.ndarray) -> np.ndarray:
    """
    Return 4 points in [TL, TR, BR, BL] order.
    TL = smallest x+y sum; BR = largest; TR = smallest diff; BL = largest diff.
    """
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],     # TL
        pts[np.argmin(diff)],  # TR
        pts[np.argmax(s)],     # BR
        pts[np.argmax(diff)],  # BL
    ], dtype=np.float32)

def find_card_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Canny edge detection → dilate to bridge gaps → find largest 4-point
    quadrilateral contour → return ordered [TL, TR, BR, BL] or None.
    """
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_quad, best_area = None, 0
    min_area = h * w * 0.15   # card must occupy ≥15% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad, best_area = approx, area

    if best_quad is None:
        # Fallback: use bounding rect of largest contour
        lc = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(lc)
        best_quad = np.array(
            [[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]],
            dtype=np.float32
        ).reshape(-1, 1, 2)

    return order_quad_pts(best_quad)

def warp_perspective_card(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Perspective-warp the card to an upright rectangle whose dimensions
    match the measured width and height of the detected quadrilateral.
    """
    tl, tr, br, bl = quad
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    out_w = max(out_w, 50)
    out_h = max(out_h, 50)
    dst   = np.array(
        [[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]],
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))

def measure_border_widths_on_warped(warped: np.ndarray) -> tuple:
    """
    On the perspective-corrected card, scan inward from each of the 4 edges
    using column/row mean projections to find where the white outer border
    transitions into the printed coloured border.

    Returns (border_left, border_right, border_top, border_bottom) in pixels.
    A Pokémon card's printed centering border is the coloured frame that
    separates the white card stock edge from the artwork.
    """
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    # Use the middle 50% strip to avoid corner artefacts
    mid_y1, mid_y2 = h // 4, 3 * h // 4
    mid_x1, mid_x2 = w // 4, 3 * w // 4

    col_means = gray[mid_y1:mid_y2, :].mean(axis=0).astype(float)   # shape (w,)
    row_means = gray[:, mid_x1:mid_x2].mean(axis=1).astype(float)   # shape (h,)

    def first_dark_run(arr: np.ndarray, thresh: float = 185, run: int = 4) -> int:
        """Scan left→right (or top→bottom) and return position of first dark run."""
        for i in range(len(arr) - run):
            if all(arr[i:i+run] < thresh):
                return i
        return max(len(arr) // 8, 1)  # fallback: 12.5% of dimension

    def first_dark_run_rev(arr: np.ndarray, thresh: float = 185, run: int = 4) -> int:
        """Scan right→left and return distance from right edge."""
        n = len(arr)
        for i in range(n - 1, run, -1):
            if all(arr[i-run:i] < thresh):
                return n - i
        return max(n // 8, 1)

    bl = first_dark_run(col_means)
    br = first_dark_run_rev(col_means)
    bt = first_dark_run(row_means)
    bb = first_dark_run_rev(row_means)

    # Sanity clamp: border can't be > 30% of dimension
    bl = min(bl, w // 3); br = min(br, w // 3)
    bt = min(bt, h // 3); bb = min(bb, h // 3)
    return max(bl, 1), max(br, 1), max(bt, 1), max(bb, 1)

def centering_ratio_to_subgrade(lr: float, tb: float) -> float:
    """
    Convert worst-axis centering ratio to PSA sub-grade.

    Ratio format: larger_border / (left + right), so 0.50 = perfect,
    0.55 = 55/45,  0.60 = 60/40, etc.

    PSA grading standard:
      ≤ 0.55  (55/45)  → sub-grade 10  (PSA 10 eligible)
      ≤ 0.60  (60/40)  → sub-grade 9
      ≤ 0.65  (65/35)  → sub-grade 8
      ≤ 0.70  (70/30)  → sub-grade 7
      ≤ 0.75  (75/25)  → sub-grade 6
        > 0.75          → sub-grade 5
    """
    worst = max(lr, tb)
    if   worst <= 0.55: return 10.0
    elif worst <= 0.60: return  9.0
    elif worst <= 0.65: return  8.0
    elif worst <= 0.70: return  7.0
    elif worst <= 0.75: return  6.0
    else:               return  5.0

def run_centering_check(img: np.ndarray) -> tuple:
    """
    Full pipeline for one image:
      1. find_card_quad  (Canny → contour → 4 pts)
      2. warp_perspective_card
      3. measure_border_widths_on_warped
      4. compute LR/TB ratios and sub-grade

    Returns:
      (lr_ratio, tb_ratio, subgrade, reasons, quad_pts, warped_img)
    """
    reasons = []
    quad    = find_card_quad(img)

    if quad is None:
        reasons.append("⚠ Card outline not detected — centering unmeasurable")
        return 0.50, 0.50, 8.0, reasons, None, None

    warped = warp_perspective_card(img, quad)
    bl, br, bt, bb = measure_border_widths_on_warped(warped)

    lr_total = bl + br
    tb_total = bt + bb
    lr_ratio = max(bl, br) / max(lr_total, 1)
    tb_ratio = max(bt, bb) / max(tb_total, 1)
    subgrade = centering_ratio_to_subgrade(lr_ratio, tb_ratio)

    # Human-readable ratio strings  e.g. "57/43"
    lr_str = f"{int(max(bl,br)/lr_total*100)}/{int(min(bl,br)/lr_total*100)}"
    tb_str = f"{int(max(bt,bb)/tb_total*100)}/{int(min(bt,bb)/tb_total*100)}"

    if subgrade >= 10.0:
        reasons.append(f"✓ Centering L/R {lr_str} | T/B {tb_str} → sub-grade 10 (PSA 10 eligible)")
    else:
        reasons.append(
            f"Centering L/R {lr_str} | T/B {tb_str} → sub-grade {subgrade:.0f}"
            f" {'← PSA 10 IMPOSSIBLE' if subgrade < 10 else ''}"
        )

    return lr_ratio, tb_ratio, subgrade, reasons, quad, warped

# ═══════════════════════════════════════════════════════════════════
#  FIX 4 — VISUAL BOUNDING BOXES & CORNER ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════

def annotate_full_image(
    img: np.ndarray,
    quad: Optional[np.ndarray],
    lr_ratio: float,
    tb_ratio: float,
    cen_sg: float,
    white_counts: list,
    corner_sg: float,
    glint_flag: bool,
) -> np.ndarray:
    """
    Draw ALL visual evidence on the card image:

    • Card outline quadrilateral  → green if centering ≥ 10, red if < 10
    • One bounding box per corner → green (0 white px) or red (any white px)
      with a white-pixel count label on each box
    • Yellow semi-transparent overlay on the holo zone if glint detected
    • Centering measurement text at the bottom
    """
    out = img.copy()
    h, w = out.shape[:2]

    # ── 1. Card outline quad ─────────────────────────────────
    if quad is not None:
        color = CV_GREEN if cen_sg >= 10.0 else CV_RED
        pts   = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
        for pt, lbl in zip(quad.astype(int), ["TL","TR","BR","BL"]):
            cv2.circle(out, tuple(pt), 5, color, -1)
            cv2.putText(out, lbl, (pt[0]+4, pt[1]-4), CV_FONT, 0.5, color, 1, cv2.LINE_AA)

    # ── 2. Corner bounding boxes ─────────────────────────────
    ph = max(int(h * 0.09), 28)
    pw = max(int(w * 0.09), 28)
    corner_origins = [(0, 0), (w-pw, 0), (0, h-ph), (w-pw, h-ph)]

    for (cx, cy), lbl, n_white in zip(corner_origins, CORNER_LABELS, white_counts):
        color = CV_RED if n_white > 0 else CV_GREEN
        # Outer box (2px)
        cv2.rectangle(out, (cx, cy), (cx+pw, cy+ph), color, 2)
        # Small fill to make the corner pop
        overlay = out.copy()
        cv2.rectangle(overlay, (cx, cy), (cx+pw, cy+ph), color, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
        # Label: corner name + white pixel count
        badge = f"{lbl} {n_white}px"
        cv2.putText(out, badge, (cx+3, cy+14), CV_FONT, 0.42, color, 1, cv2.LINE_AA)

    # ── 3. Holo / surface zone overlay ───────────────────────
    if glint_flag:
        y1 = int(h * 0.18); y2 = int(h * 0.72)
        x1 = int(w * 0.08); x2 = int(w * 0.92)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), CV_YELLOW, -1)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), CV_YELLOW, 2)
        cv2.putText(out, "SURFACE GLINT DETECTED",
                    (x1+6, y1+18), CV_FONT, 0.50, CV_YELLOW, 1, cv2.LINE_AA)

    # ── 4. Centering text strip at bottom ────────────────────
    cen_color = CV_GREEN if cen_sg >= 10.0 else CV_RED
    cen_label = (f"CTR L/R {lr_ratio:.0%}  T/B {tb_ratio:.0%}  "
                 f"sub={cen_sg:.0f}  {'✓ OK' if cen_sg>=10 else '✗ FAIL'}")
    cv2.putText(out, cen_label, (8, h-10), CV_FONT, 0.46, cen_color, 1, cv2.LINE_AA)

    return out

def make_corner_patches(img: np.ndarray, white_counts: list) -> list:
    """
    Extract each corner patch, upscale 4×, draw a red/green border,
    overlay white-pixel highlights in red, and label with pixel count.
    Returns a list of 4 BGR numpy arrays ready to display.
    """
    h, w = img.shape[:2]
    ph   = max(int(h * 0.09), 28)
    pw   = max(int(w * 0.09), 28)
    slices = [
        img[0:ph,    0:pw   ],   # TL
        img[0:ph,    w-pw:w ],   # TR
        img[h-ph:h,  0:pw   ],   # BL
        img[h-ph:h,  w-pw:w ],   # BR
    ]
    patches = []
    for patch, lbl, n_white in zip(slices, CORNER_LABELS, white_counts):
        if patch.size == 0:
            patches.append(np.zeros((60, 60, 3), np.uint8))
            continue
        disp = patch.copy()
        # Highlight white pixels
        bch, gch, rch = cv2.split(disp)
        wmask = (rch.astype(int) > WHITE_LV) & \
                (gch.astype(int) > WHITE_LV) & \
                (bch.astype(int) > WHITE_LV)
        if n_white > 0:
            disp[wmask] = [0, 0, 255]      # red = fail
        else:
            disp[wmask] = [0, 200, 80]     # green = clean

        # 4× upscale (nearest-neighbour to preserve pixel evidence)
        scale = 4
        disp  = cv2.resize(disp,
                           (disp.shape[1] * scale, disp.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
        # Coloured border + label
        color = CV_RED if n_white > 0 else CV_GREEN
        cv2.rectangle(disp, (0, 0), (disp.shape[1]-1, disp.shape[0]-1), color, 3)
        cv2.putText(disp, f"{lbl} {n_white}px",
                    (4, 15), CV_FONT, 0.50, color, 1, cv2.LINE_AA)
        patches.append(disp)
    return patches

# ═══════════════════════════════════════════════════════════════════
#  CORNER / EDGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def measure_corner_white_pixels(img: np.ndarray) -> list:
    """Return [TL, TR, BL, BR] white-pixel counts in each corner patch."""
    h, w = img.shape[:2]
    ph   = max(int(h * 0.09), 28)
    pw   = max(int(w * 0.09), 28)
    patches = [
        img[0:ph,    0:pw   ],
        img[0:ph,    w-pw:w ],
        img[h-ph:h,  0:pw   ],
        img[h-ph:h,  w-pw:w ],
    ]
    counts = []
    for s in patches:
        if s.size == 0:
            counts.append(0); continue
        bch, gch, rch = cv2.split(s)
        mask = ((rch.astype(int) > WHITE_LV) &
                (gch.astype(int) > WHITE_LV) &
                (bch.astype(int) > WHITE_LV))
        counts.append(int(mask.sum()))
    return counts

def corner_counts_to_subgrade(counts: list) -> tuple:
    """
    Convert worst-corner white-pixel count to PSA sub-grade.

    PSA is extremely strict on corners:
      0 white px              → 10  (gem mint corners)
      1–4 white px (any corner)→  9  (PSA 10 probability drops below 5%)
      5–19 px                 → 8
      20–49 px                → 7
      50–99 px                → 6
      ≥ 100 px                → 5

    Returns (subgrade, reasons_list).
    """
    worst   = max(counts)
    flagged = [f"{lbl}:{n}px" for lbl, n in zip(CORNER_LABELS, counts) if n > 0]
    reasons = []

    for lbl, n in zip(CORNER_LABELS, counts):
        if n > 0:
            reasons.append(f"Corner {lbl}: {n} white pixel(s) — whitening detected")

    if worst == 0:
        reasons.append("✓ All 4 corners clean — zero whitening")
        return 10.0, reasons

    extra = f"({', '.join(flagged)})"
    if worst <= 4:
        reasons.append(f"Minor whitening {extra} → sub-grade 9 — PSA 10 probability now <5%")
        return 9.0, reasons
    if worst <= 19:
        reasons.append(f"Moderate whitening {extra} → sub-grade 8")
        return 8.0, reasons
    if worst <= 49:
        reasons.append(f"Significant whitening {extra} → sub-grade 7")
        return 7.0, reasons
    if worst <= 99:
        reasons.append(f"Heavy whitening {extra} → sub-grade 6")
        return 6.0, reasons
    reasons.append(f"Severe whitening {extra} → sub-grade 5")
    return 5.0, reasons

# ─────────────────────────────────────────────────────────────
#  SURFACE / GLINT / SCRATCH
# ─────────────────────────────────────────────────────────────

def check_surface(img: np.ndarray) -> tuple:
    """
    Detect glare in the holo zone.  If glare found, apply Laplacian
    inside the blown-out region to detect scratches/print lines.
    Returns (glint_flag, scratch_flag, reasons, subgrade).
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.18), int(h * 0.72)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    holo = img[y1:y2, x1:x2]
    if holo.size == 0:
        return False, False, ["✓ Holo zone not found — skipping surface check"], 10.0

    gray  = cv2.cvtColor(holo, cv2.COLOR_BGR2GRAY)
    glare = gray > GLINT_LV
    gfrac = float(glare.sum()) / max(gray.size, 1)

    if gfrac <= GLINT_FRAC:
        return False, False, ["✓ No glare detected on surface"], 10.0

    reasons = [f"Glare: {gfrac:.1%} of holo area overexposed (threshold {GLINT_FRAC:.0%})"]
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    if glare.sum() > 0:
        dens = float((lap_abs[glare] > 35).sum()) / max(int(glare.sum()), 1)
        if dens > SCRATCH_DENS:
            reasons.append(
                f"Scratch/print-line signal in glare zone "
                f"(Laplacian density {dens:.2%}) ← HARD FAIL"
            )
            return True, True, reasons, 7.0
    reasons.append("Glare present but no scratch signal — may be lighting artefact")
    return True, False, reasons, 9.0

# ─────────────────────────────────────────────────────────────
#  FORENSICS
# ─────────────────────────────────────────────────────────────

def check_clone_stamp(img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seen = {}
    for y in range(0, h - CLONE_BS, CLONE_BS):
        for x in range(0, w - CLONE_BS, CLONE_BS):
            blk = gray[y:y+CLONE_BS, x:x+CLONE_BS].astype(np.float32)
            mn, sd = blk.mean(), blk.std()
            if sd < 5: continue
            key = hashlib.md5(((blk - mn) / sd).tobytes()).hexdigest()
            if key in seen:
                sy, sx = seen[key]
                if ((y-sy)**2 + (x-sx)**2)**0.5 > CLONE_BS * 3:
                    return True, [f"Clone-stamp editing detected at ({sx},{sy})↔({x},{y}) ← FORENSIC FAIL"]
            else:
                seen[key] = (y, x)
    return False, []

def check_rosette(img: np.ndarray) -> tuple:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    cy, cx = h//2, w//2
    crop  = gray[cy-80:cy+80, cx-80:cx+80].astype(np.float32)
    if crop.shape[0] < 60:
        return False, []
    mag   = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(crop))) + 1)
    ch, cw = mag.shape
    f     = ROSETTE_FREQ
    ring  = mag[ch//2-f-5:ch//2-f+5, cw//2-f-5:cw//2-f+5]
    peak  = float(ring.max()) if ring.size > 0 else 0.0
    base  = float(mag[0:20, 0:20].mean())
    if peak < base * 1.6:
        return True, ["No CMYK rosette pattern (FFT) — possible inkjet counterfeit ← FLAG"]
    return False, []

def check_resolution_stock(img: np.ndarray) -> tuple:
    h, w    = img.shape[:2]
    passed  = w >= MIN_PX and h >= MIN_PX
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aspect  = w / max(h, 1)
    border  = np.concatenate([gray[:10,:].flatten(), gray[-10:,:].flatten(),
                               gray[:,:10].flatten(), gray[:,-10:].flatten()])
    is_stock = (0.92 < aspect < 1.08) and (float(border.std()) < 12)
    return passed, (w, h), is_stock

# ═══════════════════════════════════════════════════════════════════
#  FIX 2 — SUB-GRADE SYSTEM + RULE OF MINIMUM + GAUSSIAN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════

def compute_grade_distribution(sub: SubGrades) -> dict:
    """
    Step 1 — Rule of Minimum:
        overall = min(centering, corners, edges, surface)
        The overall grade can NEVER exceed the lowest sub-grade.

    Step 2 — Gaussian distribution centred on 'overall' (σ = 0.75).
        This models the real-world variability in PSA grading results.

    Step 3 — Hard rules override:
        • centering sub-grade < 10  → PSA 10 probability = 0% immediately
        • any corner white pixel (corners < 10) → PSA 10 < 5%
        • Redistribute lost probability proportionally down to grades 8–9.

    Returns dict {grade_int: probability_float} for grades 1–10, summing to 100%.
    """
    overall   = min(sub.centering, sub.corners, sub.edges, sub.surface)
    sub.overall = overall

    grades  = np.arange(1, 11, dtype=float)
    sigma   = 0.75
    weights = np.exp(-0.5 * ((grades - overall) / sigma) ** 2)
    weights = weights / weights.sum()
    dist    = {int(g): round(float(p * 100), 1) for g, p in zip(grades, weights)}

    # ── Hard rule 1: centering < 10 → PSA 10 = 0% ────────────
    if sub.centering < 10.0:
        lost        = dist.get(10, 0.0)
        dist[10]    = 0.0
        # Redistibute lost probability to grades 9 and 8
        dist[9]     = round(dist.get(9,  0.0) + lost * 0.65, 1)
        dist[8]     = round(dist.get(8,  0.0) + lost * 0.35, 1)

    # ── Hard rule 2: any corner white px → PSA 10 < 5% ───────
    if sub.corners < 10.0 and dist.get(10, 0.0) > 4.9:
        excess      = dist[10] - 4.9
        dist[10]    = 4.9
        dist[9]     = round(dist.get(9,  0.0) + excess * 0.70, 1)
        dist[8]     = round(dist.get(8,  0.0) + excess * 0.30, 1)

    # Normalise so everything sums to exactly 100%
    total = sum(dist.values())
    if total > 0:
        dist = {k: round(v / total * 100, 1) for k, v in dist.items()}

    return dist

# ═══════════════════════════════════════════════════════════════════
#  FIX 3 — LIVE MARKET PRICING  (3-tier waterfall, never returns $0)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _ebay_oauth_token() -> Optional[str]:
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
def _tier1_ebay_graded_search(card_name: str) -> MarketData:
    """
    Tier 1: Search eBay Browse API for '{card_name} PSA 10' in the
    graded-card condition bucket (conditionIds 2750).  Extract median
    ask price as a proxy for market value.
    """
    token = _ebay_oauth_token()
    if not token:
        return MarketData()

    search_q = f"{card_name} PSA 10"
    sold_url  = (f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(search_q)}"
                 f"&LH_Sold=1&LH_Complete=1&_sacat=183454")
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization": f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"},
            params={"q": search_q, "category_ids": "183454",
                    "filter": "conditionIds:{2750}", "sort": "price", "limit": 10},
            timeout=12,
        )
        r.raise_for_status()
        prices = []
        for item in r.json().get("itemSummaries", []):
            try:
                p = float(item.get("price", {}).get("value", 0))
                if p > 0: prices.append(p)
            except Exception:
                continue
        if prices:
            val = round(float(np.median(prices)), 2)
            return MarketData(psa10_price=val, source="eBay Live (graded ask)",
                              search_url=sold_url, confidence="high")
    except Exception:
        pass
    return MarketData()

@st.cache_data(ttl=1800, show_spinner=False)
def _tier2_pricecharting(card_name: str) -> MarketData:
    """
    Tier 2: PriceCharting public API.
    Returns 'grade-10-price' (stored in cents) for the best-matching product.
    """
    sold_url = (f"https://www.ebay.com/sch/i.html?_nkw="
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
                try:
                    val = float(cents) / 100.0
                except Exception:
                    val = 0.0
                if val > 50:
                    return MarketData(psa10_price=round(val, 2),
                                      source="PriceCharting API",
                                      search_url=sold_url, confidence="medium")
    except Exception:
        pass
    return MarketData()

def _tier3_static(title: str) -> MarketData:
    """
    Tier 3: Static lookup table + heuristics.
    ALWAYS returns a non-zero value — eliminates the $0 bug.
    """
    tl   = title.lower()
    val  = 0.0
    # Try longest-match key first
    for key, v in sorted(STATIC_PSA10.items(), key=lambda x: -len(x[0])):
        if all(k in tl for k in key.split()):
            val = float(v); break

    if val == 0:
        if "1st edition" in tl or "1st ed" in tl: val = 2000.0
        elif "shadowless" in tl:                   val = 1500.0
        elif "shining"    in tl:                   val = 1500.0
        elif any(s in tl for s in ["lugia","ho-oh","entei","raikou","suicune"]): val = 1200.0
        elif ("holo" in tl and any(s in tl for s in
              ["base set","jungle","fossil","team rocket","neo",
               "expedition","skyridge","aquapolis"])): val = 700.0
        else: val = 400.0

    sold_url = (f"https://www.ebay.com/sch/i.html?_nkw="
                f"{quote_plus(tl[:40]+' PSA 10')}&LH_Sold=1&_sacat=183454")
    return MarketData(psa10_price=val, source="Reference Table",
                      search_url=sold_url, confidence="low")

def get_market_data(title: str) -> MarketData:
    """
    3-tier waterfall — stops at first tier that returns a price > 0.
    Guarantees psa10_price is never $0.
    """
    # Clean listing noise for better API queries
    clean = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    clean = re.sub(r'(raw|ungraded|nm|lp|mp|hp|lot|bundle|free\s*ship\w*)',
                   '', clean, flags=re.I).strip()[:60]

    md = _tier1_ebay_graded_search(clean)
    if md.psa10_price > 0:
        return md

    md = _tier2_pricecharting(clean)
    if md.psa10_price > 0:
        return md

    return _tier3_static(title)   # always non-zero

def calculate_financials(price: float, md: MarketData, p10: float) -> tuple:
    """
    best_case = PSA10_value − (raw_price + grading_fee + shipping)
    expected  = (PSA10_value × P(PSA10)) − costs
    roi_pct   = best_case / raw_price × 100
    """
    costs     = price + GRADING_FEE + SHIPPING_EST
    best_case = md.psa10_price - costs
    expected  = (md.psa10_price * p10 / 100.0) - costs
    roi_pct   = (best_case / max(price, 0.01)) * 100
    return round(best_case, 2), round(expected, 2), round(roi_pct, 1)

# ═══════════════════════════════════════════════════════════════════
#  IMAGE FETCH
# ═══════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════
#  FULL GRADING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def grade_images(image_urls: list) -> GradingResult:
    """
    Run all checks across up to 6 images, accumulate worst-case sub-grades,
    produce Gaussian grade distribution, annotate images.
    """
    gr = GradingResult()

    if not image_urls:
        gr.res_flag = True
        gr.reasoning.append("No images in listing")
        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        return gr

    images = [img for url in image_urls[:6]
              if (img := fetch_image(url)) is not None]

    if not images:
        gr.res_flag = True
        gr.reasoning.append("All image downloads failed")
        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        return gr

    # Resolution + stock-photo check (first image)
    res_ok, res_px, is_stock = check_resolution_stock(images[0])
    gr.res_px      = res_px
    gr.stock_photo = is_stock
    if not res_ok:
        gr.res_flag = True
        gr.reasoning.append(
            f"REJECTED: {res_px[0]}×{res_px[1]}px below {MIN_PX}px minimum — likely stock photo"
        )
        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
        return gr
    if is_stock:
        gr.reasoning.append("⚠ Image appears to be stock/official art — centering may be inaccurate")

    # Accumulators — we take the worst-case across all images
    sub             = SubGrades()
    best_quad       = None
    best_warp       = None
    worst_lr        = 0.50
    worst_tb        = 0.50
    worst_cen_sg    = 10.0
    worst_cor_sg    = 10.0
    worst_cor_counts= [0, 0, 0, 0]
    worst_srf_sg    = 10.0
    do_clone        = True   # only first image

    for idx, img in enumerate(images):
        lbl = f"Img {idx+1}"

        # ── Fix 1: Perspective-transform centering ────────────
        lr, tb, cen_sg, cen_r, quad, warped = run_centering_check(img)
        if cen_sg < worst_cen_sg:
            worst_cen_sg = cen_sg
            worst_lr, worst_tb = lr, tb
            best_quad = quad
            best_warp = warped
        gr.reasoning.extend(f"{lbl}: {r}" for r in cen_r)

        # ── Corners ───────────────────────────────────────────
        counts  = measure_corner_white_pixels(img)
        cor_sg, cor_r = corner_counts_to_subgrade(counts)
        worst_cor_counts = [max(a, b) for a, b in zip(worst_cor_counts, counts)]
        if cor_sg < worst_cor_sg:
            worst_cor_sg = cor_sg
        gr.reasoning.extend(f"{lbl}: {r}" for r in cor_r)

        # ── Surface ───────────────────────────────────────────
        glint, scratch, srf_r, srf_sg = check_surface(img)
        if srf_sg < worst_srf_sg:
            worst_srf_sg = srf_sg
        if glint:   gr.glint_flag   = True
        if scratch: gr.scratch_flag = True
        gr.reasoning.extend(f"{lbl}: {r}" for r in srf_r)

        # ── Forensics (first image only — CPU intensive) ──────
        if do_clone:
            cl_f, cl_r = check_clone_stamp(img)
            if cl_f: gr.clone_flag = True; gr.reasoning.extend(cl_r)
            ro_f, ro_r = check_rosette(img)
            if ro_f: gr.rosette_flag = True; gr.reasoning.extend(ro_r)
            do_clone = False

    # Populate SubGrades
    sub.centering        = worst_cen_sg
    sub.corners          = worst_cor_sg
    sub.edges            = worst_srf_sg
    sub.surface          = worst_srf_sg
    sub.lr_ratio         = worst_lr
    sub.tb_ratio         = worst_tb
    sub.corner_white_px  = worst_cor_counts
    if gr.clone_flag:   sub.surface = min(sub.surface, 5.0)
    if gr.rosette_flag: sub.surface = min(sub.surface, 7.0)

    gr.sub           = sub
    gr.quad_pts      = best_quad
    gr.warped_img    = best_warp
    gr.centering_flag = sub.centering < 10.0
    gr.corners_flag   = sub.corners   < 10.0

    # ── Fix 2: Gaussian distribution + Rule of Minimum ───────
    gr.grade_dist = compute_grade_distribution(sub)
    gr.psa10   = gr.grade_dist.get(10, 0.0)
    gr.psa9    = gr.grade_dist.get(9,  0.0)
    gr.psa8    = gr.grade_dist.get(8,  0.0)
    gr.psa7    = gr.grade_dist.get(7,  0.0)
    gr.psa6    = gr.grade_dist.get(6,  0.0)
    gr.psa5    = gr.grade_dist.get(5,  0.0)
    gr.psa1_4  = sum(gr.grade_dist.get(g, 0.0) for g in range(1, 5))
    gr.confidence = gr.psa10

    # ── Fix 4: Annotated full image + corner patches ──────────
    if images:
        gr.annotated_full = annotate_full_image(
            images[0], best_quad, worst_lr, worst_tb,
            worst_cen_sg, worst_cor_counts, worst_cor_sg, gr.glint_flag
        )
        gr.corner_patches = make_corner_patches(images[0], worst_cor_counts)

    # Overall verdict
    hard_fails = [gr.res_flag, gr.centering_flag, gr.corners_flag,
                  gr.scratch_flag, gr.clone_flag]
    gr.overall_pass = not any(hard_fails)
    if gr.overall_pass:
        gr.reasoning.insert(0,
            f"✓ No hard-fail conditions — PSA 10 probability {gr.psa10:.1f}%")
    return gr

# ─────────────────────────────────────────────────────────────
#  eBay SEARCH
# ─────────────────────────────────────────────────────────────

def search_ebay(query: str, min_p: float, max_p: float, limit: int) -> list:
    token = _ebay_oauth_token()
    if not token:
        st.info("ℹ️ No eBay credentials found — running in **Demo Mode**.")
        return MOCK_LISTINGS[:limit]
    try:
        r = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers={"Authorization": f"Bearer {token}",
                     "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"},
            params={"q": f"{query} pokemon card raw ungraded",
                    "category_ids": "183454",
                    "filter": f"price:[{min_p}..{max_p}],priceCurrency:USD",
                    "sort": "price", "limit": limit, "fieldgroups": "EXTENDED"},
            timeout=15,
        )
        r.raise_for_status()
        out = []
        for item in r.json().get("itemSummaries", []):
            imgs  = [item.get("image", {}).get("imageUrl", "")]
            imgs += [i.get("imageUrl", "") for i in item.get("additionalImages", [])]
            out.append({
                "title":      item.get("title", ""),
                "price":      float(item.get("price", {}).get("value", 0)),
                "url":        item.get("itemWebUrl", "#"),
                "source":     "eBay",
                "image_urls": [u for u in imgs if u],
            })
        return out
    except Exception as e:
        st.warning(f"eBay error: {e} — using demo data")
        return MOCK_LISTINGS[:limit]

def grade_listing(raw: dict) -> ListingResult:
    lr         = ListingResult(**{k: raw[k] for k in
                                  ("title", "price", "url", "source", "image_urls")})
    lr.grading = grade_images(lr.image_urls)
    lr.market  = get_market_data(lr.title)
    lr.net_profit, lr.exp_profit, lr.roi_pct = calculate_financials(
        lr.price, lr.market, lr.grading.psa10)
    hard = [lr.grading.centering_flag, lr.grading.corners_flag,
            lr.grading.scratch_flag,   lr.grading.res_flag, lr.grading.clone_flag]
    lr.verdict = ("FAIL" if any(hard)
                  else "WARN" if (lr.grading.glint_flag or lr.grading.rosette_flag)
                  else "PASS")
    return lr

# ═══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════
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
                  font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#8899bb;">
        <span>{label}</span><span style="color:{color};">{pct:.1f}%</span>
      </div>
      <div style="background:#111827;border-radius:3px;height:10px;overflow:hidden;">
        <div style="width:{min(pct,100):.1f}%;height:100%;
                    background:linear-gradient(90deg,{color}88,{color});border-radius:3px;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def subgrade_bar(label: str, sg: float):
    pct   = sg * 10.0
    color = _gc(pct)
    st.markdown(f"""
    <div style="margin:3px 0">
      <div style="display:flex;justify-content:space-between;
                  font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#8899bb;">
        <span>{label}</span>
        <span style="color:{color};font-weight:bold;">{sg:.0f} / 10</span>
      </div>
      <div style="background:#111827;border-radius:3px;height:10px;overflow:hidden;">
        <div style="width:{pct:.1f}%;height:100%;
                    background:linear-gradient(90deg,{color}88,{color});border-radius:3px;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def confidence_ring(pct: float):
    color = _gc(pct)
    r, stroke, circ = 38, 8, 238.76
    offset = circ * (1 - pct / 100)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin:8px 0;">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="#111827" stroke-width="{stroke}"/>
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
          stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
          stroke-linecap="round" transform="rotate(-90 50 50)"/>
        <text x="50" y="46" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="15"
          fill="{color}" font-weight="bold">{pct:.0f}%</text>
        <text x="50" y="60" text-anchor="middle"
          font-family="'Share Tech Mono',monospace" font-size="7.5"
          fill="#556677">PSA 10</text>
      </svg>
    </div>""", unsafe_allow_html=True)

def verdict_pill(v: str) -> str:
    cfg = {"PASS": ("#003d24", NEON["green"],  "✅ PASS"),
           "WARN": ("#3d2800", NEON["yellow"], "⚠ WARN"),
           "FAIL": ("#3d0000", NEON["red"],    "❌ FAIL")}
    bg, col, txt = cfg.get(v, ("#111", "#aaa", "⬜"))
    return (f'<span style="background:{bg};color:{col};border:1px solid {col};'
            f'border-radius:20px;padding:3px 14px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:.75rem;font-weight:bold;">'
            f'{txt}</span>')

def src_badge(src: str, conf: str) -> str:
    col = {"high": NEON["green"], "medium": NEON["yellow"],
           "low": "#8899bb"}.get(conf, "#8899bb")
    return (f'<span style="background:{col}22;color:{col};border:1px solid {col};'
            f'border-radius:10px;padding:1px 8px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:.65rem;">'
            f'{src}</span>')

def render_corner_panels(patches: list):
    if not patches or len(patches) < 4: return
    cols = st.columns(4)
    for col, patch, lbl in zip(cols, patches[:4], CORNER_LABELS):
        with col:
            if patch.size > 0:
                col.image(bgr_to_pil(patch), caption=f"{lbl} (400%)",
                          use_container_width=True)

def render_subgrade_panel(sub: SubGrades):
    rule_color = _gc(sub.overall * 10)
    st.markdown(
        f'<div style="background:#0a0f1a;border:1px solid #1e2d4a;border-radius:8px;'
        f'padding:8px 12px;margin:4px 0;font-family:\'Share Tech Mono\',monospace;'
        f'font-size:.68rem;color:#4a7fc1;letter-spacing:.08em;">'
        f'PSA SUB-GRADES — Rule of Minimum → Overall: '
        f'<b style="color:{rule_color};">{sub.overall:.0f}/10</b></div>',
        unsafe_allow_html=True)
    subgrade_bar("Centering",    sub.centering)
    subgrade_bar("Corners",      sub.corners)
    subgrade_bar("Edges/Surface",sub.edges)
    subgrade_bar("★ Overall",    sub.overall)

def render_market_panel(md: MarketData, price: float, net: float, exp: float, roi: float):
    pc    = _gc(net / max(abs(net), 1) * 70 + 30) if net > 0 else NEON["red"]
    badge = src_badge(md.source, md.confidence)
    link  = (f'<a href="{md.search_url}" target="_blank" '
             f'style="color:{NEON["blue"]};font-size:.65rem;margin-left:8px;">'
             f'🔗 PSA 10 sold listings</a>') if md.search_url else ""
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;line-height:2.1;
                background:#0a0f1a;border-radius:8px;border:1px solid #1e2d4a;padding:10px 14px;">
      <div style="margin-bottom:5px;">{badge}{link}</div>
      <span style="color:#556677;">Raw Price  </span><b style="color:#d4e1f5;">${price:.2f}</b><br>
      <span style="color:#556677;">PSA 10 Est </span><b style="color:{NEON['blue']};">${md.psa10_price:,.0f}</b><br>
      <span style="color:#556677;">Grading    </span><b style="color:#556677;">−${GRADING_FEE:.0f}</b><br>
      <span style="color:#556677;">Shipping   </span><b style="color:#556677;">−${SHIPPING_EST:.0f}</b><br>
      <span style="color:#556677;">Best-Case  </span><b style="color:{pc};font-size:1rem;">${net:,.0f}</b><br>
      <span style="color:#556677;">Expected   </span><b style="color:{pc};">${exp:,.0f}</b><br>
      <span style="color:#556677;">ROI        </span><b style="color:{pc};">{roi:.0f}%</b>
    </div>""", unsafe_allow_html=True)

def render_grading_card(r: ListingResult, expanded: bool = False):
    gr  = r.grading
    v   = r.verdict
    lbl = f"{'✅' if v=='PASS' else '⚠️' if v=='WARN' else '❌'}  {r.title[:62]}  — ${r.price:.2f}"

    with st.expander(lbl, expanded=expanded):
        c1, c2, c3 = st.columns([1.4, 1, 1.4])

        with c1:
            # Show annotated image (Fix 4)
            if gr and gr.annotated_full is not None:
                st.image(bgr_to_pil(gr.annotated_full), use_container_width=True,
                         caption="AI-annotated — green=pass, red=fail")
            elif r.image_urls:
                st.image(r.image_urls[0], use_container_width=True)
            st.markdown(
                f'{verdict_pill(v)}<br><br>'
                f'<a href="{r.url}" target="_blank" '
                f'style="background:{NEON["blue"]}22;border:1px solid {NEON["blue"]};'
                f'color:{NEON["blue"]};padding:4px 12px;border-radius:6px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.72rem;'
                f'text-decoration:none;">🛒 ONE-TAP PURCHASE</a>',
                unsafe_allow_html=True)

        with c2:
            if gr:
                confidence_ring(gr.confidence)
                render_subgrade_panel(gr.sub)
                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.68rem;'
                    f'color:#556677;text-align:center;margin-top:4px;">'
                    f'{r.source} · {gr.res_px[0]}×{gr.res_px[1]}px</div>',
                    unsafe_allow_html=True)

        with c3:
            if r.market:
                render_market_panel(r.market, r.price, r.net_profit, r.exp_profit, r.roi_pct)

        # Grade distribution (Fix 2)
        if gr:
            st.markdown("**Full Grade Distribution** — Gaussian, Rule of Minimum applied")
            d = gr.grade_dist
            ca, cb = st.columns(2)
            with ca:
                prob_bar("PSA 10", d.get(10, 0), NEON["green"])
                prob_bar("PSA  9", d.get(9,  0), NEON["yellow"])
                prob_bar("PSA  8", d.get(8,  0), NEON["orange"])
                prob_bar("PSA  7", d.get(7,  0), "#cc6600")
            with cb:
                prob_bar("PSA  6", d.get(6,  0), "#aa4400")
                prob_bar("PSA  5", d.get(5,  0), "#884400")
                prob_bar("PSA 3-4", sum(d.get(g, 0) for g in (3, 4)), NEON["red"])
                prob_bar("PSA 1-2", sum(d.get(g, 0) for g in (1, 2)), "#880000")

            # Corner panels (Fix 4)
            if gr.corner_patches:
                st.markdown("**Corner Analysis — 400% Zoom**  `red = white pixel(s)  green = clean`")
                render_corner_panels(gr.corner_patches)

            # Perspective-corrected card
            if gr.warped_img is not None:
                with st.expander("🔲 Perspective-corrected card (centering measured on this)"):
                    st.image(bgr_to_pil(gr.warped_img), use_container_width=True)

            # Reasoning report
            if gr.reasoning:
                st.markdown("**AI Reasoning Report**")
                for note in gr.reasoning:
                    ok   = note.startswith("✓")
                    warn = "⚠" in note or "WARNING" in note.upper()
                    icon = "🟢" if ok else ("🟡" if warn else "🔴")
                    st.markdown(f"{icon} `{note}`")

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;800;900&display=swap');
html,body,[class*="css"]{background:#07090f!important;color:#c8d8f0!important;}
.stApp{background:#07090f;}
h1,h2,h3,h4{font-family:'Barlow Condensed',sans-serif!important;letter-spacing:.06em;}
.stTabs [data-baseweb="tab-list"]{background:#0b0f1a;border-bottom:1px solid #1a2a4a;gap:4px;}
.stTabs [data-baseweb="tab"]{font-family:'Barlow Condensed',sans-serif;font-size:1rem;
  font-weight:600;color:#3a5a8a;letter-spacing:.06em;border-radius:6px 6px 0 0;padding:8px 18px;}
.stTabs [aria-selected="true"]{color:#00ff88!important;background:#001a0d!important;border-bottom:2px solid #00ff88;}
.stButton>button{background:linear-gradient(135deg,#1a6fff,#0047cc);color:#fff;border:none;
  border-radius:8px;font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;font-weight:700;
  letter-spacing:.08em;padding:.55rem 1.4rem;width:100%;box-shadow:0 0 18px #1a6fff44;transition:all .2s;}
.stButton>button:hover{background:linear-gradient(135deg,#3a7fff,#1a6fff);
  box-shadow:0 0 28px #1a6fff88;transform:translateY(-1px);}
.stTextInput>div>div>input,.stNumberInput>div>div>input{background:#0b0f1a!important;
  color:#c8d8f0!important;border:1px solid #1a2a4a!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;}
.stTextArea>div>textarea{background:#0b0f1a!important;color:#c8d8f0!important;
  border:1px solid #1a2a4a!important;border-radius:7px!important;
  font-family:'Share Tech Mono',monospace!important;}
.streamlit-expanderHeader{background:#0b0f1a!important;border:1px solid #1a2a4a!important;
  border-radius:8px!important;font-family:'Barlow Condensed',sans-serif!important;font-size:1rem!important;}
.streamlit-expanderContent{background:#07090f!important;border:1px solid #1a2a4a!important;
  border-top:none!important;border-radius:0 0 8px 8px!important;padding:12px 16px!important;}
[data-testid="stMetric"]{background:#0b0f1a;border:1px solid #1a2a4a;border-radius:10px;padding:12px;}
[data-testid="stSidebar"]{background:#090d18!important;border-right:1px solid #1a2a4a;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:#07090f;}
::-webkit-scrollbar-thumb{background:#1a2a4a;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
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
    v3.0 · PERSPECTIVE CENTERING · GAUSSIAN SUB-GRADES · LIVE PRICING · VISUAL BOXES
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#00ff8844,#4da6ff44,transparent);
            margin-bottom:1rem;"></div>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────
if "notifications" not in st.session_state: st.session_state.notifications = []

def push_notif(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.notifications.insert(0, f"[{ts}] {msg}")
    st.session_state.notifications = st.session_state.notifications[:20]

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab_live, tab_batch, tab_single, tab_gap, tab_notif, tab_cfg = st.tabs([
    "🔍 LIVE SCAN", "📋 BATCH", "🔬 SINGLE CARD",
    "💰 GAP FINDER", "🔔 ALERTS", "⚙️ CONFIG"
])

# ───────────────────────────────────────────────────────────────
#  TAB 1 — LIVE SCAN
# ───────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("### eBay Live Scanner")
    r1c1, r1c2, r1c3, r1c4 = st.columns([3, 1, 1, 1])
    with r1c1: query     = st.text_input("Search Query",
                                          "Charizard Holo Base Set Raw Ungraded")
    with r1c2: min_price = st.number_input("Min $", value=1, min_value=0)
    with r1c3: max_price = st.number_input("Max $", value=100, min_value=1)
    with r1c4: num_res   = st.number_input("Limit", value=8, min_value=1, max_value=50)

    r2c1, r2c2 = st.columns(2)
    with r2c1: min_psa10 = st.number_input("Min PSA 10 Value ($)", value=1000, min_value=0)
    with r2c2: min_gap   = st.number_input("Min Best-Case Profit ($)", value=500, min_value=0)

    if st.button("▶  RUN LIVE SCAN", use_container_width=True):
        with st.spinner("Fetching listings…"):
            raw_list = search_ebay(query, min_price, max_price, num_res)

        pb = st.progress(0.0); status = st.empty()
        results: list[ListingResult] = []

        for i, raw in enumerate(raw_list):
            status.markdown(
                f"<span style='font-family:\"Share Tech Mono\",monospace;font-size:.8rem;"
                f"color:#4da6ff;'>Grading {i+1}/{len(raw_list)}: {raw['title'][:55]}…</span>",
                unsafe_allow_html=True)
            pb.progress((i + 1) / max(len(raw_list), 1))
            lr = grade_listing(raw)
            if lr.grading and lr.grading.psa10 >= 70 and lr.verdict == "PASS":
                push_notif(f"🔥 {lr.grading.psa10:.0f}% PSA 10 — {lr.title[:45]} — ${lr.price}")
            results.append(lr)

        pb.empty(); status.empty()

        n_pass = sum(1 for r in results if r.verdict == "PASS")
        n_warn = sum(1 for r in results if r.verdict == "WARN")
        n_fail = sum(1 for r in results if r.verdict == "FAIL")
        best   = max((r for r in results if r.verdict == "PASS"),
                     key=lambda r: r.net_profit, default=None)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Scanned",     len(results))
        m2.metric("✅ Pass",      n_pass)
        m3.metric("⚠️ Warn",      n_warn)
        m4.metric("❌ Rejected",  n_fail)
        m5.metric("Best Profit", f"${best.net_profit:,.0f}" if best else "—")

        st.markdown("---")
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        results.sort(key=lambda r: (order.get(r.verdict, 3), -r.net_profit))
        for r in results:
            render_grading_card(r, expanded=(r.verdict in ("PASS", "WARN")))

# ───────────────────────────────────────────────────────────────
#  TAB 2 — BATCH / LEADERBOARD
# ───────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch URL Leaderboard")
    st.caption("Paste up to 10 eBay or direct image URLs — ranked by PSA 10 confidence.")
    batch_input = st.text_area("URLs (one per line)", height=150,
                               placeholder="https://www.ebay.com/itm/…\nhttps://i.ebayimg.com/…")

    if st.button("▶  GRADE ALL & BUILD LEADERBOARD", use_container_width=True):
        urls = [u.strip() for u in batch_input.splitlines()
                if u.strip() and not u.strip().startswith("#")][:10]
        if not urls:
            st.warning("Enter at least one URL.")
        else:
            batch_results = []
            bp = st.progress(0.0)
            for i, url in enumerate(urls):
                bp.progress((i + 1) / len(urls))
                is_img = any(url.lower().endswith(e)
                             for e in [".jpg", ".jpeg", ".png", ".webp"])
                raw = {"title": urlparse(url).path[:60] or f"Listing {i+1}",
                       "price": 0.0, "url": url, "source": "Manual",
                       "image_urls": [url] if is_img else []}
                if not is_img:
                    try:
                        pg = requests.get(url, headers=HEADERS, timeout=10)
                        imgs = re.findall(r'https://i\.ebayimg\.com/images/g/[^"\']+\.jpg',
                                          pg.text)
                        raw["image_urls"] = list(dict.fromkeys(imgs))[:5]
                        m  = re.search(r'"price"\s*:\s*\{\s*"value"\s*:\s*"?([\d.]+)',
                                       pg.text)
                        if m: raw["price"] = float(m.group(1))
                        m2 = re.search(r'<h1[^>]*itemprop="name"[^>]*>([^<]+)', pg.text)
                        if m2: raw["title"] = m2.group(1).strip()[:80]
                    except Exception:
                        pass
                batch_results.append(grade_listing(raw))
            bp.empty()

            batch_results.sort(key=lambda r: -(r.grading.psa10 if r.grading else 0))
            medals = ["🥇", "🥈", "🥉"] + ["🔹"] * 10
            rows = [{
                "Rank": f"{medals[i]} #{i+1}", "Card": r.title[:50],
                "Price": f"${r.price:.2f}",
                "PSA 10%": f"{r.grading.psa10:.1f}%" if r.grading else "—",
                "Overall Sub": f"{r.grading.sub.overall:.0f}/10" if r.grading else "—",
                "PSA10 Value": f"${r.market.psa10_price:,.0f}" if r.market else "—",
                "Best Profit": f"${r.net_profit:,.0f}",
                "Verdict": r.verdict,
            } for i, r in enumerate(batch_results)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            for r in batch_results:
                render_grading_card(r, expanded=True)

# ───────────────────────────────────────────────────────────────
#  TAB 3 — SINGLE CARD
# ───────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### 🔬 Single Card Deep Analysis")
    mode = st.radio("Input", ["📁 Upload Photo", "🔗 Paste Image URL"], horizontal=True)
    single_img  = None
    single_urls = []

    if "Upload" in mode:
        up = st.file_uploader("Drop card photo (JPG/PNG)",
                               type=["jpg", "jpeg", "png"])
        if up:
            pil = Image.open(up).convert("RGB")
            single_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        url_in = st.text_input("Image URL",
                               placeholder="https://i.ebayimg.com/…")
        if url_in.strip():
            single_urls = [url_in.strip()]

    sc1, sc2 = st.columns(2)
    with sc1: s_title = st.text_input("Card Name",
                                       placeholder="Charizard Base Set Holo Rare")
    with sc2: s_price = st.number_input("Raw Price ($)", value=50.0, min_value=0.0)

    if st.button("▶  ANALYSE CARD", use_container_width=True):
        if single_img is None and not single_urls:
            st.error("Upload an image or paste a URL.")
        else:
            with st.spinner("Running full CV pipeline…"):
                if single_img is not None:
                    # Build GradingResult directly from the uploaded image
                    gr  = GradingResult()
                    sub = SubGrades()
                    res_ok, res_px, is_stock = check_resolution_stock(single_img)
                    gr.res_px = res_px; gr.stock_photo = is_stock
                    if not res_ok:
                        gr.res_flag = True
                        gr.reasoning.append(f"Low-res {res_px[0]}×{res_px[1]}px")
                        gr.grade_dist = {g: (100.0 if g == 1 else 0.0) for g in range(1, 11)}
                    else:
                        # Fix 1: perspective centering
                        lr, tb, cen_sg, cen_r, quad, warped = run_centering_check(single_img)
                        sub.centering = cen_sg; sub.lr_ratio = lr; sub.tb_ratio = tb
                        gr.quad_pts = quad; gr.warped_img = warped
                        gr.reasoning.extend(cen_r)

                        # Corners
                        counts  = measure_corner_white_pixels(single_img)
                        cor_sg, cor_r = corner_counts_to_subgrade(counts)
                        sub.corners = cor_sg
                        sub.corner_white_px = counts
                        gr.reasoning.extend(cor_r)

                        # Surface
                        glint, scratch, srf_r, srf_sg = check_surface(single_img)
                        sub.edges   = srf_sg
                        sub.surface = srf_sg
                        gr.glint_flag   = glint
                        gr.scratch_flag = scratch
                        gr.reasoning.extend(srf_r)

                        # Forensics
                        cl_f, cl_r = check_clone_stamp(single_img)
                        gr.clone_flag = cl_f
                        gr.reasoning.extend(cl_r)
                        if cl_f: sub.surface = min(sub.surface, 5.0)

                        gr.sub = sub
                        # Fix 2: Gaussian distribution
                        gr.grade_dist = compute_grade_distribution(sub)
                        gr.psa10  = gr.grade_dist.get(10, 0.0)
                        gr.confidence = gr.psa10
                        gr.centering_flag = sub.centering < 10.0
                        gr.corners_flag   = sub.corners   < 10.0

                        # Fix 4: annotations
                        gr.annotated_full = annotate_full_image(
                            single_img, quad, lr, tb, cen_sg, counts, cor_sg, glint)
                        gr.corner_patches = make_corner_patches(single_img, counts)

                        hard = [gr.centering_flag, gr.corners_flag,
                                gr.scratch_flag, gr.clone_flag]
                        gr.overall_pass = not any(hard)
                else:
                    gr = grade_images(single_urls)

            # Fix 3: market data
            md  = get_market_data(s_title)
            net, exp, roi = calculate_financials(s_price, md, gr.psa10)
            v   = ("PASS" if gr.overall_pass and not gr.clone_flag
                   else "WARN" if gr.glint_flag else "FAIL")

            col_img, col_right = st.columns([1.5, 1])
            with col_img:
                if gr.annotated_full is not None:
                    st.image(bgr_to_pil(gr.annotated_full), use_container_width=True,
                             caption="AI-annotated — green=pass, red=fail")
                elif single_img is not None:
                    st.image(bgr_to_pil(single_img), width=280)
            with col_right:
                st.markdown(f"## {verdict_pill(v)}", unsafe_allow_html=True)
                confidence_ring(gr.confidence)
                render_subgrade_panel(gr.sub)

            render_market_panel(md, s_price, net, exp, roi)

            st.markdown("**Full Grade Distribution**")
            d = gr.grade_dist
            ca, cb = st.columns(2)
            with ca:
                prob_bar("PSA 10", d.get(10, 0), NEON["green"])
                prob_bar("PSA  9", d.get(9,  0), NEON["yellow"])
                prob_bar("PSA  8", d.get(8,  0), NEON["orange"])
                prob_bar("PSA  7", d.get(7,  0), "#cc6600")
            with cb:
                prob_bar("PSA  6", d.get(6,  0), "#aa4400")
                prob_bar("PSA  5", d.get(5,  0), "#884400")
                prob_bar("PSA 3-4", sum(d.get(g, 0) for g in (3, 4)), NEON["red"])
                prob_bar("PSA 1-2", sum(d.get(g, 0) for g in (1, 2)), "#880000")

            if gr.corner_patches:
                st.markdown("**Corner Analysis — 400% Zoom**")
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

# ───────────────────────────────────────────────────────────────
#  TAB 4 — GAP FINDER
# ───────────────────────────────────────────────────────────────
with tab_gap:
    st.markdown("### 💰 High-ROI Gap Dashboard")
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
        "Card": n,
        "Raw (approx)":   f"${r:,}",
        "PSA 10 (approx)": f"${p:,}",
        "Spread":          f"${p-r:,}",
        "Multiplier":      f"{p//r}×",
    } for n, r, p in sample])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    for entry in HIGH_ROI_CATALOG:
        with st.expander(f"📦 {entry['set']} — Spread {entry['est_spread']}"):
            st.markdown(f"*{entry['note']}*")
            cols = st.columns(min(len(entry["cards"]), 4))
            for col, card in zip(cols, entry["cards"]):
                link = (f"https://www.ebay.com/sch/i.html?_nkw="
                        f"{quote_plus(card+' '+entry['set'].split()[0]+' pokemon raw ungraded')}"
                        f"&_sacat=183454")
                col.markdown(f"[**{card}**]({link})")

# ───────────────────────────────────────────────────────────────
#  TAB 5 — ALERTS
# ───────────────────────────────────────────────────────────────
with tab_notif:
    n = len(st.session_state.notifications)
    st.markdown(f"### 🔔 Alert Feed ({n} alerts)")
    if not st.session_state.notifications:
        st.info("No alerts yet. Run a Live Scan.")
    else:
        for note in st.session_state.notifications:
            st.markdown(
                f'<div style="background:#001a0d;border:1px solid #00ff8844;'
                f'border-radius:6px;padding:8px 14px;margin-bottom:6px;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:.8rem;'
                f'color:#00ff88;">{note}</div>',
                unsafe_allow_html=True)
        if st.button("Clear Alerts"):
            st.session_state.notifications = []
            st.rerun()

# ───────────────────────────────────────────────────────────────
#  TAB 6 — CONFIG & FIX REFERENCE
# ───────────────────────────────────────────────────────────────
with tab_cfg:
    st.markdown("### ⚙️ Configuration")
    st.code("""
# config.yaml
ebay:
  app_id:  "YOUR-EBAY-APP-ID-HERE"
  cert_id: "YOUR-EBAY-CERT-ID-HERE"
  dev_id:  "YOUR-EBAY-DEV-ID-HERE"
""", language="yaml")
    st.code("""
# .streamlit/secrets.toml  (Streamlit Cloud)
[ebay]
app_id  = "YOUR-EBAY-APP-ID-HERE"
cert_id = "YOUR-EBAY-CERT-ID-HERE"
dev_id  = "YOUR-EBAY-DEV-ID-HERE"
""", language="toml")

    st.markdown("---")
    st.markdown("### v3.0 Technical Fix Reference")
    st.markdown(f"""
**Fix 1 — Perspective-Transform Centering**
- `find_card_quad()`: Canny (thresh 30/110) → dilate → largest 4-point contour → `order_quad_pts()`
- `warp_perspective_card()`: `cv2.getPerspectiveTransform` + `cv2.warpPerspective`
- `measure_border_widths_on_warped()`: column/row mean projections on the central 50% strip,
  scans inward until brightness drops below 185 (white→coloured border transition)
- Centering ratio → sub-grade: ≤55% → 10, ≤60% → 9, ≤65% → 8, ≤70% → 7, ≤75% → 6, >75% → 5
- **Hard rule**: centering sub-grade < 10 → PSA 10 probability = **0% immediately**

**Fix 2 — Sub-Grade System + Gaussian Distribution**
- `SubGrades` dataclass: Centering, Corners, Edges, Surface each scored 1–10
- `corner_counts_to_subgrade()`: 0px → 10; 1–4px → 9 (PSA 10 < 5%); 5–19 → 8; 20–49 → 7; 50–99 → 6; ≥100 → 5
- `compute_grade_distribution()`:
  1. **Rule of Minimum**: `overall = min(centering, corners, edges, surface)`
  2. **Gaussian** centred on `overall`, σ = 0.75 → models real PSA variance
  3. Hard overrides: centering fail → P(10) = 0%; corner whitening → P(10) < 5%
  4. Normalise to 100%

**Fix 3 — Live Market Pricing (never $0)**
| Tier | Source | Method | Confidence |
|---|---|---|---|
| 1 | eBay Browse API | Graded-card search (conditionId 2750) → median price | High |
| 2 | PriceCharting API | `grade-10-price` field (stored in cents ÷ 100) | Medium |
| 3 | Static reference table | Longest-key match + heuristic fallbacks | Low |

Each listing card shows the source badge, confidence level, and a direct
link to eBay sold PSA 10 results so you can verify the estimate instantly.

**Fix 4 — Visual Bounding Boxes**
- `annotate_full_image()` draws on every listing's primary photo:
  - **Card outline quad** (green if centering ≥ 10, red if < 10)
  - **4 corner boxes** with per-corner pixel counts (green = 0px, red = any px)
  - **Yellow holo overlay** if glint flag is set
  - **Centering text** at bottom with L/R, T/B ratios and sub-grade
- `make_corner_patches()`: 400% nearest-neighbour upscale, red pixel highlights,
  coloured border, pixel count label — same colour code as the full image
- Warped (perspective-corrected) card shown in a collapsible expander
""")

# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#1a2a4a,transparent);
            margin:2rem 0 1rem;"></div>
<div style="text-align:center;font-family:'Share Tech Mono',monospace;
            font-size:.65rem;color:#1a2a4a;padding-bottom:1rem;">
  PSA 10 PROFIT SCANNER v3.0 · FIX 1: PERSPECTIVE CENTERING · FIX 2: GAUSSIAN SUB-GRADES
  · FIX 3: LIVE PRICING · FIX 4: VISUAL CV BOXES<br>
  For research/educational use only · Not financial advice · Always verify manually
</div>
""", unsafe_allow_html=True)
