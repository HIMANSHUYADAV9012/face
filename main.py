

import io
import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple
from contextlib import asynccontextmanager

import cv2
import numpy as np
import aiohttp
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

# Try to load .env file for Telegram credentials only
try:
    from dotenv import load_dotenv
    load_dotenv()  # looks for .env in current directory
except ImportError:
    pass  # dotenv not installed – will use system env or None

# ─── HARDCODED CONFIGURATION (all except Telegram) ──────────────────────
DATA_FOLDER       = "data"                     # folder containing .pkl files
INSIGHTFACE_MODEL = "buffalo_l"                # InsightFace model name
DET_SIZE          = (640, 640)                 # detection input size
MAX_IMAGE_BYTES   = 5 * 1024 * 1024            # 5 MB
SUPPORTED_TYPES   = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}

BLUR_THRESHOLD    = 15.0                       # lower = more tolerant (0 to disable)
MIN_FACE_SIZE     = 60                         # minimum face width/height in pixels
MATCH_THRESHOLD   = 0.40                       # cosine similarity threshold

CORS_ORIGINS      = ["*"]                      # allow all origins (change if needed)

# Telegram settings (read from environment – can be set in .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Lifespan (startup / shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app, persons
    log.info("Loading InsightFace model ...")
    face_app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=DET_SIZE)
    log.info("InsightFace ready.")

    persons = load_all_persons(DATA_FOLDER)
    if not persons:
        log.warning(f"No .pkl files found in '{DATA_FOLDER}'. Verification will always fail.")
    else:
        total_embs = sum(len(embs) for _, embs in persons)
        log.info(f"Loaded {len(persons)} person(s) with {total_embs} embedding(s).")
    yield
    # shutdown: nothing special

app = FastAPI(title="Face Verification API", version="2.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Helper functions ───────────────────────────────────────────────────
def load_all_persons(folder: str) -> List[Tuple[str, List[np.ndarray]]]:
    """Scan 'folder' for .pkl files, load each and return list of (person_id, embeddings)."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        log.warning(f"Folder '{folder}' does not exist. No persons loaded.")
        return []

    persons = []
    for pkl_path in folder_path.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            embeddings = data.get("embeddings")
            person_id  = data.get("person_id", pkl_path.stem)
            if not embeddings:
                log.warning(f"{pkl_path}: missing 'embeddings' list. Skipped.")
                continue
            emb_list = [np.array(e, dtype=np.float32) for e in embeddings]
            persons.append((person_id, emb_list))
            log.info(f"Loaded {len(emb_list)} embedding(s) for '{person_id}' from {pkl_path.name}")
        except Exception as e:
            log.error(f"Failed to load {pkl_path}: {e}")
    return persons

def is_blurry(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    log.debug(f"Blur score: {var:.2f} (threshold={BLUR_THRESHOLD})")
    return var < BLUR_THRESHOLD

def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors are L2-normalised by InsightFace."""
    return float(np.dot(a, b))

def similarity_to_confidence(sim: float) -> float:
    """Map cosine similarity [-1, 1] to percentage [0, 100]."""
    return round(max(0.0, min(100.0, (sim + 1) / 2 * 100)), 1)

async def send_to_telegram(image_bytes: bytes, filename: str, matched: bool, person_id: str, similarity: float, confidence: float):
    """
    Send image to Telegram along with verification details.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram credentials missing – skipping forward.")
        return

    # Prepare detailed caption
    if matched:
        status_text = "✅ MATCH SUCCESS"
        match_line = f"Person ID: {person_id}\nSimilarity: {similarity:.4f}\nConfidence: {confidence}%"
    else:
        status_text = "❌ MATCH FAILED"
        match_line = f"Best similarity: {similarity:.4f}\nConfidence: {confidence}%\nNo registered person matched."

    caption = f"""{status_text}
━━━━━━━━━━━━━━━━━━━
File: {filename}
{match_line}
━━━━━━━━━━━━━━━━━━━
Threshold: {MATCH_THRESHOLD} (cosine similarity)"""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    # Create multipart form data
    form = aiohttp.FormData()
    form.add_field('chat_id', TELEGRAM_CHAT_ID)
    form.add_field('caption', caption)
    form.add_field('photo', io.BytesIO(image_bytes), filename=filename, content_type='image/jpeg')

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=10) as resp:
                if resp.status == 200:
                    log.info(f"Image + details forwarded to Telegram (chat {TELEGRAM_CHAT_ID})")
                else:
                    text = await resp.text()
                    log.error(f"Telegram error {resp.status}: {text[:200]}")
    except Exception as e:
        log.error(f"Failed to send to Telegram: {e}")

# ─── Main endpoint ──────────────────────────────────────────────────────
@app.post("/verify-face")
async def verify_face(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()
    if content_type not in SUPPORTED_TYPES and ext not in SUPPORTED_EXTS:
        raise HTTPException(415, "Unsupported file type. Use JPG, JPEG, PNG, or WEBP.")

    # 2. Read and check size
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File too large. Maximum {MAX_IMAGE_BYTES // (1024*1024)} MB.")

    # 3. Decode image
    img_bgr = bytes_to_bgr(raw)
    if img_bgr is None:
        raise HTTPException(400, "Could not decode image.")

    # 4. Blur check
    if BLUR_THRESHOLD > 0 and is_blurry(img_bgr):
        raise HTTPException(400, "Image is too blurry. Please provide a clearer photo.")

    # 5. Detect faces
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    if len(faces) == 0:
        raise HTTPException(400, "No face detected in the image.")
    if len(faces) > 1:
        raise HTTPException(400, f"{len(faces)} faces detected. Provide exactly one face.")

    face = faces[0]

    # 6. Face size check
    x1, y1, x2, y2 = face.bbox.astype(int)
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        raise HTTPException(400, f"Face too small (minimum {MIN_FACE_SIZE}px). Move closer.")

    # 7. Compare with all loaded persons
    live_emb = face.normed_embedding
    best_match_person = None
    best_sim = -1.0

    for person_id, embeddings in persons:
        for emb in embeddings:
            sim = cosine_similarity(live_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_match_person = person_id

    matched = best_sim >= MATCH_THRESHOLD
    confidence = similarity_to_confidence(best_sim)

    log.info(f"Best match: person='{best_match_person}', similarity={best_sim:.4f}, confidence={confidence}, matched={matched}")

    # 8. Schedule Telegram forwarding with full details (after response)
    background_tasks.add_task(
        send_to_telegram,
        raw,
        file.filename or "upload.jpg",
        matched,
        best_match_person if matched else None,
        best_sim,
        confidence
    )

    # 9. Return JSON response
    return {
        "success": matched,
        "match": matched,
        "person_id": best_match_person if matched else None,
        "similarity": round(best_sim, 4),
        "confidence": confidence,
        "message": "Face matched successfully" if matched else "Face does not match any registered person"
    }

# ─── Health check ──────────────────────────────────────────────────────
@app.get("/")
def health():
    person_list = [pid for pid, _ in persons] if persons else []
    return {
        "status": "ok",
        "loaded_persons": person_list,
        "total_persons": len(persons),
        "total_embeddings": sum(len(embs) for _, embs in (persons or [])),
        "blur_threshold": BLUR_THRESHOLD,
        "min_face_size": MIN_FACE_SIZE,
        "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }