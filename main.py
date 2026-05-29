import io
import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
import aiohttp
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from uniface import RetinaFace, ArcFace, compute_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ========== CONFIGURATION ==========
DATA_FOLDER       = "data"
MAX_IMAGE_BYTES   = 5 * 1024 * 1024
SUPPORTED_TYPES   = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}

BLUR_THRESHOLD    = 15.0
MIN_FACE_SIZE     = 60
MATCH_THRESHOLD   = 0.40

CORS_ORIGINS      = ["*"]
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== GLOBALS ==========
detector = None
recognizer = None
persons = []   # list of (person_id, List[embedding_array])

# ========== HELPER FUNCTIONS ==========
def load_all_persons(folder: str) -> List[Tuple[str, List[np.ndarray]]]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        log.warning(f"Folder '{folder}' does not exist.")
        return []
    persons_list = []
    for pkl_path in folder_path.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            embeddings = data.get("embeddings")
            person_id = data.get("person_id", pkl_path.stem)
            if not embeddings:
                log.warning(f"{pkl_path.name} missing 'embeddings' – skip")
                continue
            emb_list = []
            for e in embeddings:
                arr = np.array(e, dtype=np.float32)
                if arr.shape[0] != 512:
                    log.error(f"{pkl_path.name} embedding is {arr.shape[0]}D, need 512D")
                    raise ValueError("Invalid embedding dimension")
                emb_list.append(arr)
            persons_list.append((person_id, emb_list))
            log.info(f"Loaded {person_id}: {len(emb_list)} embedding(s)")
        except Exception as e:
            log.error(f"Failed to load {pkl_path.name}: {e}")
    return persons_list

def is_blurry(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < BLUR_THRESHOLD

def bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def similarity_to_confidence(sim: float) -> float:
    return round(max(0.0, min(100.0, (sim + 1) / 2 * 100)), 1)

async def send_to_telegram(image_bytes: bytes, filename: str, matched: bool,
                           person_id: Optional[str], similarity: float, confidence: float):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

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
    form = aiohttp.FormData()
    form.add_field('chat_id', TELEGRAM_CHAT_ID)
    form.add_field('caption', caption)
    form.add_field('photo', io.BytesIO(image_bytes), filename=filename, content_type='image/jpeg')

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=10) as resp:
                if resp.status == 200:
                    log.info(f"Forwarded to Telegram (chat {TELEGRAM_CHAT_ID})")
                else:
                    text = await resp.text()
                    log.error(f"Telegram error {resp.status}: {text[:200]}")
    except Exception as e:
        log.error(f"Failed to send to Telegram: {e}")

# ========== LIFESPAN ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, recognizer, persons
    log.info("Initializing UniFace models...")
    detector = RetinaFace()
    recognizer = ArcFace()
    log.info("UniFace models ready.")

    persons = load_all_persons(DATA_FOLDER)
    if not persons:
        log.warning(f"No valid .pkl files found in '{DATA_FOLDER}'")
    else:
        total = sum(len(embs) for _, embs in persons)
        log.info(f"Loaded {len(persons)} person(s) with {total} embedding(s)")
    yield

app = FastAPI(title="Face Verification API (UniFace)", version="5.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ========== MAIN ENDPOINT ==========
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

    # 5. Detect faces with UniFace
    faces = detector.detect(img_bgr)
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

    # 7. Extract live embedding
    live_emb = recognizer.get_normalized_embedding(img_bgr, face.landmarks)

    # 8. Compare with stored embeddings
    best_match_person = None
    best_sim = -1.0

    for person_id, emb_list in persons:
        for stored_emb in emb_list:
            sim = compute_similarity(live_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_match_person = person_id

    # --- FIX: convert numpy types to native Python types ---
    matched = (best_sim >= MATCH_THRESHOLD)
    matched = bool(matched)                     # convert numpy.bool -> bool
    similarity_val = float(best_sim)            # convert numpy.float32 -> float
    confidence = similarity_to_confidence(similarity_val)

    log.info(f"Best match: person='{best_match_person}', similarity={similarity_val:.4f}, confidence={confidence}, matched={matched}")

    # 9. Send to Telegram in background
    background_tasks.add_task(
        send_to_telegram,
        raw,
        file.filename or "upload.jpg",
        matched,
        best_match_person if matched else None,
        similarity_val,
        confidence
    )

    # 10. Return JSON
    return {
        "success": matched,
        "match": matched,
        "person_id": best_match_person if matched else None,
        "similarity": round(similarity_val, 4),
        "confidence": confidence,
        "message": "Face matched successfully" if matched else "Face does not match any registered person"
    }

# ========== HEALTH CHECK ==========
@app.get("/")
def health():
    person_list = [pid for pid, _ in persons] if persons else []
    return {
        "status": "ok",
        "service": "UniFace (ONNX Runtime)",
        "loaded_persons": person_list,
        "total_persons": len(persons),
        "total_embeddings": sum(len(embs) for _, embs in (persons or [])),
        "blur_threshold": BLUR_THRESHOLD,
        "min_face_size": MIN_FACE_SIZE,
        "match_threshold": MATCH_THRESHOLD,
        "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }

