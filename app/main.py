import io
import json
import os
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.explain import ECGExplainer, confidence_level
from src.predict import ECGPredictor

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_PATH = ROOT / "outputs" / "feedback.jsonl"

MAX_FILE_MB = 10 # tiny images cannot be readable ECGs
MAX_SIDE = 10_000  # reject absurdly large images
ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SESSIONS = 200  # keep memory bounded

state = {}  # holds the predictor
sessions = OrderedDict()  # session_id -> ECGExplainer (in memory: lost on restart)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: stop at startup instead of failing on the first request
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    state["predictor"] = ECGPredictor()  # load the model ONCE at startup
    yield
    state.clear()


app = FastAPI(title="CardioLens AI", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    question: Optional[str] = Field(
        default=None, max_length=500
    )  # None -> first explanation
    regenerate: bool = False  # True -> answer the last message again


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int = Field(ge=0, le=1000)
    rating: Literal["up", "down"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


# Plain `def` (not `async def`): model inference is blocking CPU work.
# FastAPI runs plain `def` endpoints in a thread pool, so the server stays responsive.
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    language: Literal["English", "Bangla"] = Form("English"),
):
    # 1) Basic input validation
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only JPEG or PNG images are allowed.")

    # Read at most limit + 1 byte, so a huge upload never fills memory
    data = file.file.read(MAX_FILE_MB * 1024 * 1024 + 1)
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB).")

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size

    # Reject excessively large images
        if width > MAX_SIDE or height > MAX_SIDE:
            raise HTTPException(
                status_code=422,
                detail="Image dimensions are too large.",
            )

    # Decode image and detect corrupted files
        image.load()

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        ValueError,
        SyntaxError,
):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
    )

    # 2) CNN prediction
    result = state["predictor"].predict(image)

    # 3) Start a chat session for this prediction
    session_id = uuid.uuid4().hex
    sessions[session_id] = ECGExplainer(result, language)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)  # drop the oldest session

    return {
        "session_id": session_id,
        "confidence_level": confidence_level(result["confidence"]),
        **result,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    explainer = sessions.get(req.session_id)
    if explainer is None:
        raise HTTPException(404, "Unknown session. Upload an ECG first.")
    try:
        if req.regenerate:
            answer = explainer.regenerate()
        elif req.question:
            answer = explainer.ask(req.question)
        else:
            answer = explainer.explain()
    except ValueError:
        raise HTTPException(409, "Nothing to regenerate yet.")
    except Exception:
        # Do not leak internal errors to the client
        raise HTTPException(
            503, "The explanation service is unavailable. Please try again."
        )
    return {"answer": answer}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Stores a thumbs up/down. NO chat text and NO image is saved, only metadata."""
    explainer = sessions.get(req.session_id)
    if explainer is None:
        raise HTTPException(404, "Unknown session.")

    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id,
        "message_index": req.message_index,
        "rating": req.rating,
        "predicted_class": explainer.result["predicted_class"],
        "confidence_level": confidence_level(explainer.result["confidence"]),
    }
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "saved"}
