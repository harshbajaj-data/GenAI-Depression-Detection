

import os
import time
import logging
import threading
import random
import re
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, Response

# Transformers (for text model and conversational generator)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tensorflow.keras.models import load_model

# Try MediaPipe for better face detection; fallback to Haar cascade
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

# -------------------------
# Configuration
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# odel paths here
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "fer2013_model", "face_fer2013_cnn_plus.h5")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "goemotions_model")  #  4-class HuggingFace model

IMG_SIZE = (48, 48)
FINAL_CLASSES = ["depression", "anxiety", "frustration", "calmness"]

# Stream settings
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_FPS = 20
FRAME_SLEEP = 1.0 / STREAM_FPS

# Fusion weight clamps
MIN_TEXT_WEIGHT = 0.2
MAX_TEXT_WEIGHT = 0.8

# Conversation memory size
CONVERSATION_MEMORY_SIZE = 6

# Crisis intervention settings
CRISIS_LOG_FILE = os.path.join(BASE_DIR, "crisis_logs.txt")
CRISIS_ESCALATION_THRESHOLD = 3  # Number of crisis indicators before escalation

# Flask app
app = Flask(__name__, template_folder="../frontend/structure", static_folder="../frontend/static")

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("EmotionBackend")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(_formatter)
logger.addHandler(_stdout_handler)
_file_handler = logging.FileHandler(os.path.join(BASE_DIR, "backend.log"))
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)

# Crisis logging
crisis_logger = logging.getLogger("CrisisIntervention")
crisis_logger.setLevel(logging.WARNING)
_crisis_handler = logging.FileHandler(CRISIS_LOG_FILE)
_crisis_handler.setFormatter(_formatter)
crisis_logger.addHandler(_crisis_handler)

# -------------------------
# Globals
# -------------------------
face_model = None
text_tokenizer = None
text_model = None
face_detector_mp = None
face_cascade = None
chat_generator = None

camera = None
is_detecting = False

# Queues for threaded architecture
frame_queue = Queue(maxsize=2)
processed_frame_queue = Queue(maxsize=2)
stop_event = threading.Event()

current_text = ""
emotion_history: List[np.ndarray] = []
conversation_history: List[str] = []

# Crisis tracking
crisis_indicators_session = []
last_crisis_time = None

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.frame_times: List[float] = []
        self.prediction_times: List[float] = []

    def log_frame_time(self, d: float):
        self.frame_times.append(d)
        if len(self.frame_times) > 300:
            self.frame_times.pop(0)

    def log_prediction_time(self, d: float):
        self.prediction_times.append(d)
        if len(self.prediction_times) > 300:
            self.prediction_times.pop(0)

    def get_stats(self) -> Dict[str, float]:
        if not self.frame_times:
            return {"fps": 0.0, "avg_prediction_time_ms": 0.0}
        avg_frame = float(np.mean(self.frame_times))
        fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
        avg_pred = float(np.mean(self.prediction_times)) if self.prediction_times else 0.0
        return {"fps": round(fps, 2), "avg_prediction_time_ms": round(avg_pred * 1000, 2)}

perf_monitor = PerformanceMonitor()

# -------------------------
# Safe decorator
# -------------------------
from functools import wraps

def safe_call(default=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Error in %s: %s", fn.__name__, e)
                return default
        return wrapper
    return deco

# -------------------------
# Model loading
# -------------------------
@safe_call()
def load_models():
    global face_model, text_tokenizer, text_model, face_detector_mp, face_cascade, chat_generator
    logger.info("Loading models...")

    # Face model (Keras)
    try:
        if os.path.exists(FACE_MODEL_PATH):
            face_model = load_model(FACE_MODEL_PATH)
            logger.info("Loaded face model from %s", FACE_MODEL_PATH)
        else:
            logger.warning("Face model not found at %s - face model disabled", FACE_MODEL_PATH)
            face_model = None
    except Exception as e:
        logger.exception("Failed to load face model: %s", e)
        face_model = None

    # Text model (HF-format expected)
    try:
        if os.path.exists(TEXT_MODEL_PATH):
            text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
            text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
            text_model.eval()
            logger.info("Loaded text model from %s", TEXT_MODEL_PATH)
        else:
            logger.warning("Text model path not found: %s - text model disabled", TEXT_MODEL_PATH)
            text_tokenizer, text_model = None, None
    except Exception as e:
        logger.exception("Failed to load text model: %s", e)
        text_tokenizer, text_model = None, None

    # Face detector: MediaPipe preferred
    try:
        if _HAS_MEDIAPIPE:
            mp_face = mp.solutions.face_detection
            face_detector_mp = mp_face.FaceDetection(min_detection_confidence=0.5)
            logger.info("MediaPipe face detector initialized")
        else:
            face_detector_mp = None
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            logger.info("Haar cascade initialized as fallback")
    except Exception as e:
        logger.exception("Face detector init failed: %s", e)
        face_detector_mp = None
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Chat generator: try BlenderBot distilled or DialoGPT fallback
    try:
        device = 0 if torch.cuda.is_available() else -1
        # prefer BlenderBot for instruction-like prompts
        chat_generator = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill", device=device)
        logger.info("Chat generator ready (BlenderBot) on %s", 'cuda' if device == 0 else 'cpu')
    except Exception as e:
        logger.warning("BlenderBot init failed, trying DialoGPT: %s", e)
        try:
            chat_generator = pipeline("text-generation", model="microsoft/DialoGPT-small", device=device)
            logger.info("Fallback chat generator ready (DialoGPT)")
        except Exception as e2:
            logger.warning("Could not load any chat generator model: %s", e2)
            chat_generator = None

    logger.info("Model loading complete")

# -------------------------
# Face detection helpers
# -------------------------
@safe_call(default=[])
def detect_faces_mediapipe(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes = []
    if face_detector_mp is None:
        return boxes
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector_mp.process(rgb)
    if results and getattr(results, 'detections', None):
        h, w, _ = frame.shape
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(max(0, bbox.xmin * w))
            y = int(max(0, bbox.ymin * h))
            bw = int(min(w - x, bbox.width * w))
            bh = int(min(h - y, bbox.height * h))
            boxes.append((x, y, bw, bh))
    return boxes

@safe_call(default=[])
def detect_faces_haar(gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if face_cascade is None:
        return []
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    out = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            out.append((x, y, w, h))
    return out

# -------------------------
# Prediction helpers
# -------------------------
@safe_call(default=np.array([0.25, 0.25, 0.25, 0.25]))
def predict_face_emotion(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    t0 = time.time()
    try:
        roi = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, IMG_SIZE)
        roi_norm = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(roi_norm, axis=(0, -1))
        if face_model is None:
            perf_monitor.log_prediction_time(time.time() - t0)
            return np.array([0.25, 0.25, 0.25, 0.25])
        preds = face_model.predict(roi_input, verbose=0)
        preds = np.asarray(preds[0]).astype(float)
        s = np.sum(preds)
        perf_monitor.log_prediction_time(time.time() - t0)
        return preds / s if s > 0 else np.array([0.25, 0.25, 0.25, 0.25])
    except Exception as e:
        logger.exception("Face prediction error: %s", e)
        perf_monitor.log_prediction_time(time.time() - t0)
        return np.array([0.25, 0.25, 0.25, 0.25])

@safe_call(default=np.array([0.25, 0.25, 0.25, 0.25]))
def predict_text_emotion(text: str) -> np.ndarray:
    t0 = time.time()
    if not text or text_model is None or text_tokenizer is None:
        perf_monitor.log_prediction_time(time.time() - t0)
        return np.array([0.25, 0.25, 0.25, 0.25])
    try:
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            perf_monitor.log_prediction_time(time.time() - t0)
            if len(probs) == 4:
                return probs
            else:
                logger.warning("Text model returned %d classes, expected 4. Returning uniform.", len(probs))
                return np.array([0.25, 0.25, 0.25, 0.25])
    except Exception as e:
        logger.exception("Text prediction error: %s", e)
        perf_monitor.log_prediction_time(time.time() - t0)
        return np.array([0.25, 0.25, 0.25, 0.25])

# -------------------------
# Fusion & smoothing
# -------------------------
@safe_call(default=np.array([0.25, 0.25, 0.25, 0.25]))
def adaptive_fuse(face_probs: np.ndarray, text_probs: np.ndarray, face_conf_percent: float) -> np.ndarray:
    # face_conf_percent is 0-100
    face_w = min(max(face_conf_percent / 100.0, 0.2), 0.9)
    text_w = 1.0 - face_w
    text_w = min(max(text_w, MIN_TEXT_WEIGHT), MAX_TEXT_WEIGHT)
    fused = face_w * np.array(face_probs) + text_w * np.array(text_probs)
    s = np.sum(fused)
    return fused / s if s > 0 else np.array([0.25, 0.25, 0.25, 0.25])

@safe_call(default=np.array([0.25, 0.25, 0.25, 0.25]))
def smooth_predictions(new_probs: np.ndarray, history_size: int = 7) -> np.ndarray:
    global emotion_history
    emotion_history.append(np.array(new_probs))
    if len(emotion_history) > history_size:
        emotion_history.pop(0)
    weights = np.exp(np.linspace(0, 1, len(emotion_history)))
    weights = weights / np.sum(weights)
    smoothed = np.zeros_like(emotion_history[0], dtype=float)
    for i, p in enumerate(emotion_history):
        smoothed += weights[i] * p
    s = np.sum(smoothed)
    return smoothed / s if s > 0 else np.array([0.25, 0.25, 0.25, 0.25])

# -------------------------
# Drawing utilities
# -------------------------
EMOTION_COLORS = {
    "depression": (255, 107, 107),
    "anxiety": (78, 205, 196),
    "frustration": (254, 202, 87),
    "calmness": (72, 198, 239)
}

def draw_face_box(frame: np.ndarray, x: int, y: int, w: int, h: int, probs: np.ndarray, confidence: float):
    idx = int(np.argmax(probs))
    emotion = FINAL_CLASSES[idx]
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    thickness = max(2, int(confidence / 20))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cs = 16
    cv2.line(frame, (x, y), (x + cs, y), color, thickness + 1)
    cv2.line(frame, (x, y), (x, y + cs), color, thickness + 1)
    cv2.line(frame, (x + w, y), (x + w - cs, y), color, thickness + 1)
    cv2.line(frame, (x + w, y), (x + w, y + cs), color, thickness + 1)
    label = f"{emotion.capitalize()} {confidence:.0f}%"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y - th - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 10, y0 + th + 8), color, -1)
    cv2.putText(frame, label, (x + 5, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# -------------------------
# Camera & processing threads
# -------------------------

def camera_capture_loop(cam_index: int = 0):
    global camera, stop_event
    logger.info("Camera capture thread started")
    try:
        cam = cv2.VideoCapture(cam_index)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, STREAM_FPS)
        if not cam.isOpened():
            logger.error("Unable to open camera %s", cam_index)
            return
        camera = cam
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            try:
                if not frame_queue.full():
                    frame_queue.put(frame, block=False)
            except Exception:
                pass
            time.sleep(FRAME_SLEEP)
    except Exception as e:
        logger.exception("Camera loop error: %s", e)
    finally:
        try:
            cam.release()
        except Exception:
            pass
        logger.info("Camera capture thread exiting")


def processing_loop():
    logger.info("Processing thread started")
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue
        t0 = time.time()
        try:
            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detect_faces_mediapipe(frame) if face_detector_mp else []
            if not faces:
                haar_faces = detect_faces_haar(gray)
                faces = [(x, y, fw, fh) for (x, y, fw, fh) in haar_faces] if haar_faces else []

            final_probs = None
            face_detected = False
            confidence = 0.0

            if faces:
                face_detected = True
                x, y, w_f, h_f = max(faces, key=lambda f: f[2] * f[3])
                x, y, w_f, h_f = max(0, x), max(0, y), max(1, w_f), max(1, h_f)

                face_probs = predict_face_emotion(frame, x, y, w_f, h_f)
                face_conf = float(np.max(face_probs) * 100)
                text_probs = predict_text_emotion(current_text)
                fused = adaptive_fuse(face_probs, text_probs, face_conf)
                final_probs = smooth_predictions(fused)
                confidence = float(np.max(final_probs) * 100)
                draw_face_box(frame, x, y, w_f, h_f, final_probs, confidence)

            perf_monitor.log_frame_time(time.time() - t0)

            meta = {'frame': frame, 'probs': final_probs.tolist() if final_probs is not None else None, 'face_detected': bool(face_detected)}
            try:
                if not processed_frame_queue.full():
                    processed_frame_queue.put(meta, block=False)
            except Exception:
                pass

        except Exception as e:
            logger.exception("Error in processing loop: %s", e)
        finally:
            try:
                frame_queue.task_done()
            except Exception:
                pass
    logger.info("Processing thread exiting")

# -------------------------
# Video stream generator
# -------------------------

def generate_video_stream():
    while True:
        try:
            meta = processed_frame_queue.get(timeout=1.0)
        except Empty:
            frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera inactive", (80, STREAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        frame = meta.get('frame')
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------
# ENHANCED Crisis Detection & Response System
# -------------------------

@dataclass
class ResponseContext:
    primary_context: str
    sub_context: Optional[str] = None
    urgency_level: str = "medium"
    time_context: Optional[str] = None
    relationship_context: Optional[str] = None
    intensity_markers: Optional[List[str]] = None
    cultural_context: Optional[str] = None
    vulnerable_population: Optional[str] = None
    crisis_type: Optional[str] = None

@dataclass
class CrisisIndicator:
    severity: str  # "immediate", "high", "moderate"
    category: str  # "suicidal", "self_harm", "substance", "psychosis", etc.
    confidence: float
    timestamp: datetime
    text_sample: str

def log_crisis_event(indicator: CrisisIndicator, user_context: str = ""):
    """Log crisis events for monitoring and intervention"""
    global crisis_indicators_session, last_crisis_time
    crisis_indicators_session.append(indicator)
    last_crisis_time = datetime.now()
    
    crisis_logger.warning(
        f"CRISIS DETECTED: {indicator.category} | Severity: {indicator.severity} | "
        f"Confidence: {indicator.confidence:.2f} | Context: {user_context} | "
        f"Text: {indicator.text_sample[:100]}..."
    )

class EnhancedPremiumResponseGenerator:
    def __init__(self):
        self.crisis_patterns = self._build_crisis_patterns()
        self.context_patterns = self._build_context_patterns()
        self.response_database = self._build_comprehensive_response_database()
        self.intensity_markers = self._build_intensity_markers()
        self.follow_up_strategies = self._build_follow_up_strategies()
        self.cultural_adaptations = self._build_cultural_adaptations()
        self.vulnerable_population_responses = self._build_vulnerable_population_responses()
        self.crisis_resources = self._build_crisis_resources()

    def _build_crisis_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "immediate": {
                "suicidal_direct": [
                    "want to die", "kill myself", "end it all", "suicide", "take my own life",
                    "not worth living", "better off dead", "end the pain", "can't go on"
                ],
                "suicidal_indirect": [
                    "everyone would be better without me", "i'm a burden", "world without me",
                    "tired of existing", "want to disappear", "nothing matters", "no point",
                    "can't take it anymore", "too much pain", "worthless", "useless"
                ],
                "self_harm_active": [
                    "hurt myself", "cut myself", "harm myself", "punish myself",
                    "deserve pain", "cutting", "burning myself", "hitting myself"
                ],
                "overdose_substance": [
                    "take all the pills", "overdose", "drink myself to death",
                    "too many pills", "end it with drugs", "poison myself"
                ],
                "psychotic_break": [
                    "voices telling me", "they're watching me", "conspiracy against me",
                    "not real", "simulation", "controlling my thoughts", "implanted"
                ]
            },
            "high": {
                "severe_depression": [
                    "completely hopeless", "empty inside", "numb", "hollow",
                    "can't feel anything", "dead inside", "lost everything",
                    "ruined my life", "destroyed", "broken beyond repair"
                ],
                "panic_crisis": [
                    "can't breathe", "heart racing", "dying", "losing control",
                    "going crazy", "panic attack", "can't stop shaking", "terrified"
                ],
                "dissociation": [
                    "not real", "floating", "outside my body", "not myself",
                    "disconnected", "like a dream", "foggy", "detached"
                ],
                "severe_anxiety": [
                    "paralyzed with fear", "can't function", "overwhelming panic",
                    "constant terror", "afraid of everything", "can't leave house"
                ],
                "eating_disorder": [
                    "starving myself", "binge and purge", "hate my body",
                    "fat and disgusting", "can't eat", "obsessed with weight"
                ],
                "substance_abuse": [
                    "drinking too much", "using drugs", "can't stop drinking",
                    "high all the time", "need substances", "addicted"
                ],
                "domestic_violence": [
                    "hitting me", "abusing me", "afraid to go home",
                    "threatening me", "controlling me", "violent partner"
                ]
            },
            "moderate": {
                "self_worth": [
                    "hate myself", "disappointed in myself", "failure",
                    "not good enough", "always mess up", "can't do anything right"
                ],
                "social_isolation": [
                    "completely alone", "no friends", "nobody cares",
                    "isolated", "lonely", "disconnected from everyone"
                ],
                "trauma_response": [
                    "flashbacks", "nightmares", "triggered", "traumatic",
                    "can't forget", "haunted", "reliving", "ptsd"
                ],
                "grief_loss": [
                    "lost someone", "died", "grief", "mourning",
                    "can't cope with loss", "missing them", "bereaved"
                ]
            }
        }

    def _build_intensity_markers(self) -> Dict[str, List[str]]:
        return {
            "crisis": ["want to die", "kill myself", "end it all", "can't go on", "no point", "worthless", "hopeless", "give up", "suicidal", "hurt myself"],
            "severe": ["extremely", "devastated", "destroyed", "ruined", "disaster", "nightmare", "terrible", "awful", "horrible", "worst", "hate", "despise", "furious", "panic", "terrified", "overwhelmed", "breaking down", "falling apart"],
            "high": ["really", "very", "so much", "intense", "strong", "deep", "serious", "major", "significant"],
            "medium": ["worried", "concerned", "upset", "frustrated", "annoyed", "sad", "down", "anxious", "nervous", "stressed", "bothered", "disappointed", "confused"],
            "low": ["slightly", "a bit", "somewhat", "kinda", "maybe", "sort of", "not sure", "wondering", "thinking about", "considering"]
        }

    def _build_context_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "academic": {
                "exam_stress": ["exam", "test", "quiz", "midterm", "final", "assessment", "evaluation", "failing"],
                "grades": ["grade", "gpa", "marks", "score", "result", "academic performance"],
                "homework": ["homework", "assignment", "project", "essay", "deadline"],
                "school_pressure": ["school stress", "academic pressure", "study overload", "college stress"]
            },
            "career": {
                "job_search": ["job hunting", "interview", "resume", "application", "hiring", "unemployed", "job rejection"],
                "workplace": ["work", "office", "boss", "manager", "colleague", "workplace harassment"],
                "career_crisis": ["career change", "job loss", "fired", "laid off", "career confusion"]
            },
            "relationships": {
                "romantic": ["boyfriend", "girlfriend", "partner", "spouse", "dating", "relationship", "breakup", "divorce"],
                "family": ["mother", "father", "parent", "sibling", "family", "family conflict", "toxic family"],
                "friendship": ["friend", "social circle", "peer pressure", "social rejection", "loneliness"],
                "social": ["social anxiety", "social skills", "making friends", "social isolation"]
            },
            "health": {
                "mental": ["depression", "anxiety", "therapy", "counseling", "medication", "mental health"],
                "physical": ["sick", "illness", "doctor", "hospital", "chronic pain", "disability"],
                "addiction": ["alcohol", "drugs", "gambling", "addiction", "substance abuse", "recovery"]
            },
            "financial": {
                "money": ["money", "financial", "broke", "debt", "loan", "budget", "poverty"],
                "housing": ["homeless", "eviction", "rent", "housing crisis", "shelter"]
            },
            "personal": {
                "identity": ["who am i", "identity", "purpose", "meaning", "sexual orientation", "gender identity"],
                "goals": ["goal", "dream", "ambition", "future", "life direction"],
                "trauma": ["abuse", "assault", "trauma", "ptsd", "flashbacks", "triggered"],
                "grief": ["death", "loss", "grief", "mourning", "funeral", "bereaved"]
            },
            "legal": {
                "trouble": ["arrest", "police", "court", "legal trouble", "lawsuit", "criminal charges"]
            },
            "social_issues": {
                "discrimination": ["racism", "sexism", "homophobia", "transphobia", "discrimination", "prejudice"],
                "bullying": ["bullied", "harassment", "cyberbullying", "workplace bullying"]
            }
        }

    def _build_comprehensive_response_database(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        return {
            "academic": {
                "exam_stress": {
                    "depression": [
                        "Exam stress can feel crushing, but remember - this one test doesn't define your intelligence or future. What specific material is making you feel most overwhelmed?",
                        "I hear how defeated you're feeling about this exam. Sometimes when we're this low, everything feels impossible. What's one small topic you could review for just 15 minutes today?",
                        "Feeling hopeless about exams is more common than you think. Even brilliant people struggle with test anxiety. Have you talked to your instructor about your concerns?"
                    ],
                    "anxiety": [
                        "Pre-exam nerves hitting hard? Let's ground you: take 3 slow breaths and name one thing you DO know well about this subject. Build from there.",
                        "Exam anxiety is your brain trying to protect you, but it's overdoing it. What's the worst-case scenario you're imagining, and how likely is it really?",
                        "Those exam butterflies are intense! Try this: study for 25 minutes, then take a 5-minute walk. Your brain needs breaks to absorb information."
                    ],
                    "frustration": [
                        "Exam prep driving you up the wall? What specific concept keeps tripping you up? Sometimes a different explanation or approach clicks better.",
                        "I get it - when the material won't stick, it's infuriating! Have you tried explaining it out loud to someone (or even a mirror)? Teaching helps learning.",
                        "Stuck on exam content? Switch tactics: if you've been reading, try flashcards. If you've been memorizing, try practice problems. Mix it up."
                    ],
                    "calmness": [
                        "Great mindset for exam prep! You seem focused and ready. What's your strategy for tackling the trickiest topics?",
                        "Love the calm energy - that's perfect for effective studying. Are you planning to do a final review or dive into new material?"
                    ]
                }
            },
            "career": {
                "job_search": {
                    "depression": [
                        "Job hunting can feel soul-crushing, especially with rejections piling up. You're not unemployable - the market is tough. How long have you been searching?",
                        "The job search grind can make anyone feel worthless. Remember: rejection often has nothing to do with your qualifications. What type of roles are you targeting?",
                        "I understand how demoralizing constant job rejection feels. Your worth isn't determined by employment status. What's one small step you can take today?"
                    ],
                    "anxiety": [
                        "Job interview nerves? Totally normal! What's the specific part that makes you most anxious - the questions, the pressure, or something else?",
                        "Job search anxiety is exhausting - all that uncertainty and waiting. What's one small step you could take today to feel more in control?",
                        "The fear of job interviews can be paralyzing. Remember, they already liked your resume enough to call you. What questions worry you most?"
                    ],
                    "frustration": [
                        "Job applications disappearing into the void is SO frustrating! Are you getting interviews but no offers, or not hearing back at all?",
                        "The job market can be absolutely maddening. What's bugging you most - the process, the requirements, or the lack of responses?",
                        "Getting no response to applications is incredibly annoying. Have you tried following up or connecting with hiring managers on LinkedIn?"
                    ],
                    "calmness": [
                        "You seem confident about your job search - that's a great energy to have! What's your current strategy or focus?",
                        "Nice to see you're approaching job hunting with a level head. Any particular opportunities you're excited about?"
                    ]
                },
                "workplace": {
                    "depression": [
                        "Workplace toxicity can drain your soul. Remember, this job doesn't define you. What's the most difficult part of your work environment?",
                        "Work-related depression is real and valid. You deserve better treatment. Have you considered speaking with HR or looking for other options?"
                    ],
                    "anxiety": [
                        "Workplace anxiety can make every day feel like torture. What specific situations trigger your anxiety most - meetings, deadlines, or interactions?",
                        "Office stress can be overwhelming. Are there particular people or tasks that spike your anxiety levels?"
                    ],
                    "frustration": [
                        "Work frustration is the worst because you're stuck there daily. What's the main source - management, workload, or colleagues?",
                        "Workplace irritation builds up over time. Is it one big issue or lots of small annoyances that are getting to you?"
                    ]
                }
            },
            "relationships": {
                "romantic": {
                    "depression": [
                        "Relationship pain cuts so deep - it touches our need for connection and love. What's happening that's making you feel this way?",
                        "When love hurts, everything else feels meaningless. You don't have to go through this alone. What's been the hardest part?",
                        "Romantic relationships can bring the highest highs and lowest lows. What's making you feel so down about your relationship situation?"
                    ],
                    "anxiety": [
                        "Relationship anxiety is torture - all those 'what if' scenarios playing in your head. What's the main thing you're worried about?",
                        "Love can make us so vulnerable and anxious. Are you worried about the future of the relationship or something specific that happened?",
                        "The fear in relationships is often worse than reality. What specific scenarios keep playing in your mind?"
                    ],
                    "frustration": [
                        "Relationship frustration is so draining! Are you feeling unheard, unappreciated, or like you're putting in all the effort?",
                        "Partner issues can be incredibly annoying, especially when the same problems keep coming up. What's the recurring theme?",
                        "Being frustrated with someone you love is particularly hard. What behavior or pattern is driving you crazy?"
                    ],
                    "calmness": [
                        "It's beautiful to hear someone who sounds peaceful about their love life. What's working well in your relationship right now?",
                        "You sound grounded about your romantic situation. That's a healthy place to be! Anything you want to explore or appreciate?"
                    ]
                },
                "family": {
                    "depression": [
                        "Family problems hit different because these are supposed to be your safe people. What's going on that's weighing so heavily?",
                        "When family relationships are broken, it affects everything. You can't choose family, but you can choose how to protect yourself. What's happening?",
                        "Family dysfunction can leave deep wounds. Remember, you're not responsible for fixing everyone else. What's the most painful part right now?"
                    ],
                    "anxiety": [
                        "Family anxiety often comes from unpredictable dynamics or fear of conflict. What situations with family make you most nervous?",
                        "Worrying about family can consume your thoughts. Are you anxious about a specific family member or situation?",
                        "Family gatherings or interactions can trigger intense anxiety. What aspect of family relationships stresses you most?"
                    ],
                    "frustration": [
                        "Family can push our buttons like nobody else! What specific family behavior or pattern is driving you up the wall?",
                        "Being frustrated with family is complex because you can't just walk away. What's the recurring issue that keeps coming up?",
                        "Family frustration often builds over years. Is this about old patterns or something new that happened?"
                    ]
                },
                "social": {
                    "depression": [
                        "Social isolation and loneliness can be devastating. Humans need connection. How long have you been feeling disconnected from others?",
                        "When social relationships feel empty or nonexistent, life loses color. What's making it hard to connect with people right now?",
                        "Feeling socially rejected or left out hurts deeply. Our brains process social pain like physical pain. What's been happening socially?"
                    ],
                    "anxiety": [
                        "Social anxiety can be paralyzing - that fear of judgment is intense. What social situations trigger your anxiety most?",
                        "The fear of saying something wrong or being rejected socially is exhausting. What specific social interactions worry you?",
                        "Social anxiety often makes us avoid the very connections we need. What would help you feel more comfortable in social situations?"
                    ]
                }
            },
            "health": {
                "mental": {
                    "depression": [
                        "Mental health struggles are real medical conditions, not personal failings. What kind of support do you have right now?",
                        "Depression is like having a broken leg - it's a health condition that needs treatment. Are you currently getting any professional help?",
                        "Mental health conditions can feel isolating, but you're not alone in this. What's been the most challenging aspect lately?"
                    ],
                    "anxiety": [
                        "Anxiety disorders are incredibly common and treatable. What triggers your anxiety most - specific situations or is it more generalized?",
                        "Living with anxiety is exhausting because your brain is constantly in alarm mode. Have you found anything that helps calm your nervous system?",
                        "Anxiety can make everything feel dangerous when it's not. What physical symptoms do you experience when anxiety hits?"
                    ]
                },
                "addiction": {
                    "depression": [
                        "Addiction often coexists with depression - they feed each other. Recovery is possible, but it's hard work. Are you currently in any treatment program?",
                        "Addiction is a disease, not a moral failing. The shame and depression make it harder to get help. What's your biggest barrier to treatment?",
                        "The cycle of addiction and depression feels hopeless, but people break free every day. What's motivating you to consider change?"
                    ],
                    "anxiety": [
                        "Many people use substances to cope with anxiety, but it usually makes anxiety worse over time. What originally led you to substance use?",
                        "Withdrawal anxiety is real and scary, but it's temporary. Are you getting medical support for detox or withdrawal management?",
                        "The anxiety about quitting substances can be overwhelming. What worries you most about getting clean or sober?"
                    ]
                }
            },
            "personal": {
                "identity": {
                    "depression": [
                        "Identity struggles can feel like losing yourself entirely. It's okay not to have all the answers about who you are. What's making you question your identity?",
                        "Sexual orientation and gender identity exploration can be confusing and scary, especially without support. What's been the hardest part of this journey?",
                        "Not knowing who you are or where you fit can be deeply depressing. Identity develops over time - you don't have to figure it all out at once."
                    ],
                    "anxiety": [
                        "Identity anxiety often comes from fear of rejection or not fitting in. What aspects of your identity feel most uncertain or scary?",
                        "Questioning your sexual orientation or gender identity can create intense anxiety. What support systems do you have during this exploration?",
                        "The pressure to label yourself or fit into categories can be overwhelming. What would happen if you gave yourself permission to be uncertain for now?"
                    ]
                },
                "trauma": {
                    "depression": [
                        "Trauma can leave lasting wounds that affect every part of life. Healing is possible, but it takes time and often professional help. Are you getting trauma-informed care?",
                        "The depression after trauma is your mind's way of trying to protect you, but it can become a prison. What happened, and how long ago?",
                        "Trauma survivors often blame themselves, but it was never your fault. What's been the hardest part about living with these experiences?"
                    ],
                    "anxiety": [
                        "Trauma anxiety and hypervigilance are exhausting - your nervous system is stuck in alarm mode. What triggers your anxiety responses most?",
                        "PTSD symptoms can make you feel like you're reliving trauma over and over. Are you experiencing flashbacks, nightmares, or panic attacks?",
                        "After trauma, the world can feel dangerous even when you're safe. What helps you feel more grounded in the present moment?"
                    ]
                },
                "grief": {
                    "depression": [
                        "Grief is love with nowhere to go. The depression that comes with loss is normal, but when it's severe, you need extra support. Who did you lose?",
                        "Losing someone important changes you forever. There's no timeline for grief. What's been the most difficult part since your loss?",
                        "Grief depression can make you feel disconnected from everything that used to matter. How recent was your loss, and what support do you have?"
                    ],
                    "anxiety": [
                        "Grief can trigger intense anxiety about losing other important people. Are you finding yourself constantly worrying about other loved ones?",
                        "The anxiety after loss often comes from feeling like the world isn't safe anymore. What fears have developed since your loss?",
                        "Sometimes grief includes panic about your own mortality or health. What specific worries keep coming up for you?"
                    ]
                }
            },
            "social_issues": {
                "discrimination": {
                    "depression": [
                        "Experiencing discrimination is traumatic and can lead to deep depression. Your pain is valid. What type of discrimination are you facing?",
                        "Systemic oppression takes a toll on mental health. You're not weak for struggling - you're dealing with injustice. What's been happening?",
                        "Being targeted because of who you are is devastating. You deserve to exist safely and be treated with dignity. What support do you need?"
                    ],
                    "anxiety": [
                        "Living with the constant threat of discrimination creates chronic anxiety. What situations make you feel most unsafe or worried?",
                        "The hypervigilance required to navigate discrimination is exhausting. How is this affecting your daily life and sense of safety?",
                        "Anticipating prejudice and preparing for discrimination is a heavy burden. What would help you feel more secure?"
                    ]
                },
                "bullying": {
                    "depression": [
                        "Bullying can destroy self-worth and create lasting depression. You don't deserve this treatment. Where is the bullying happening?",
                        "Being targeted by bullies makes you question everything about yourself. Remember: their behavior says nothing about your worth. How long has this been going on?",
                        "Workplace or school bullying can make those environments feel like torture chambers. Have you reported this to anyone in authority?"
                    ],
                    "anxiety": [
                        "Living in fear of bullies creates constant anxiety. What situations or places make you feel most anxious about encountering them?",
                        "Bullying anxiety can make you want to avoid school, work, or social situations entirely. How is this affecting your daily functioning?",
                        "The anticipation of bullying is often worse than the actual events. What coping strategies have you tried?"
                    ]
                }
            }
        }

    def _build_cultural_adaptations(self) -> Dict[str, Dict[str, str]]:
        return {
            "collectivist": {
                "family_focus": "Family harmony is important, but your individual wellbeing matters too.",
                "shame_sensitivity": "Cultural expectations can create additional pressure. Your struggles don't bring shame to your family.",
                "authority_respect": "While respecting elders and authority is valued, you still deserve to be treated with dignity."
            },
            "individualist": {
                "self_reliance": "It's okay to ask for help - that's not weakness, that's wisdom.",
                "achievement_pressure": "Your worth isn't determined by your achievements or productivity.",
                "independence": "Being independent doesn't mean you have to handle everything alone."
            },
            "religious": {
                "faith_crisis": "Spiritual struggles are normal parts of faith journeys. Questioning doesn't mean losing faith.",
                "guilt_shame": "Most religious traditions emphasize compassion and forgiveness, including for yourself.",
                "community_support": "Faith communities can be sources of support during difficult times."
            }
        }

    def _build_vulnerable_population_responses(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "lgbtq": {
                "coming_out": [
                    "Coming out is a personal journey that only you can decide the timing and pace of. What's making this feel urgent or scary right now?",
                    "Your identity is valid regardless of others' acceptance. What support systems do you have in place?",
                    "The fear of rejection from loved ones is real. Have you connected with LGBTQ+ support groups or resources?"
                ],
                "discrimination": [
                    "Facing discrimination for who you are is traumatic. You deserve to exist safely and authentically. What's been happening?",
                    "LGBTQ+ discrimination can create lasting mental health impacts. Are you in a safe living situation right now?",
                    "Your identity is not a choice - discrimination against you is wrong. What kind of support would be most helpful?"
                ]
            },
            "teens": {
                "academic_pressure": [
                    "High school pressure can feel overwhelming, but remember - your worth isn't determined by grades or college admissions.",
                    "Teen years involve so much change and pressure. What's feeling most overwhelming right now?",
                    "Social media can make everything feel more intense. How is online pressure affecting your mental health?"
                ],
                "identity_development": [
                    "Figuring out who you are is the main job of adolescence. It's normal to feel confused or uncertain.",
                    "Peer pressure and fitting in can feel life-or-death important, but your authentic self matters most.",
                    "Teen identity struggles are temporary but feel permanent. What aspects of yourself are you questioning?"
                ]
            },
            "elderly": {
                "isolation": [
                    "Loneliness in later life is unfortunately common but not inevitable. What connections have you lost recently?",
                    "Aging can bring losses - friends, family, abilities - and that grief is real. What's been the hardest change?",
                    "Social connections are crucial for mental health at any age. What barriers are keeping you from connecting with others?"
                ],
                "health_decline": [
                    "Watching your body change with age can be frightening and depressing. What health changes are worrying you most?",
                    "Chronic illness or pain can make life feel unbearable. Are you getting adequate pain management and support?",
                    "The loss of independence can feel devastating. What abilities or activities are you most concerned about losing?"
                ]
            },
            "immigrants": {
                "cultural_adjustment": [
                    "Adapting to a new culture while maintaining your identity is incredibly challenging. What's been the hardest part of this transition?",
                    "Immigration stress affects mental health in unique ways. Are you dealing with language barriers, discrimination, or cultural conflicts?",
                    "Missing your home culture and people is natural. How are you maintaining connections to your roots while adapting?"
                ],
                "documentation_stress": [
                    "Immigration status uncertainty creates chronic stress and fear. What aspects of your legal situation are most concerning?",
                    "The fear of deportation or family separation is traumatic. Are you connected with immigration legal services?",
                    "Living in legal limbo affects every aspect of life. What support systems do you have in place?"
                ]
            },
            "parents": {
                "overwhelm": [
                    "Parenting is the hardest job with no training manual. Feeling overwhelmed doesn't make you a bad parent. What's the most challenging part right now?",
                    "Parental burnout is real and common. When was the last time you had time for yourself?",
                    "The pressure to be a perfect parent is impossible and harmful. What support do you need to feel more capable?"
                ],
                "single_parenting": [
                    "Single parenting is doing two full-time jobs alone. You're not failing - the system lacks support. What's your biggest challenge?",
                    "The isolation and responsibility of single parenting can be crushing. What kind of help would make the biggest difference?",
                    "Feeling guilty about not being enough for your kids is common among single parents. Remember: love matters more than perfection."
                ]
            },
            "caregivers": {
                "burnout": [
                    "Caregiver burnout is real and serious. You can't pour from an empty cup. When did you last take care of yourself?",
                    "Caring for someone else while neglecting yourself isn't sustainable. What support do you need to continue caregiving?",
                    "The guilt caregivers feel about needing breaks is common but misplaced. Self-care isn't selfish - it's necessary."
                ]
            }
        }

    def _build_crisis_resources(self) -> Dict[str, Dict[str, str]]:
        return {
            "immediate": {
                "suicidal": "IMMEDIATE ACTION NEEDED: Contact 988 (Suicide & Crisis Lifeline), text 'HOME' to 741741 (Crisis Text Line), or go to your nearest emergency room. You don't have to face this alone.",
                "self_harm": "Please reach out immediately: 988 (Suicide & Crisis Lifeline), Crisis Text Line (text HOME to 741741), or contact emergency services. You deserve care and support.",
                "overdose": "MEDICAL EMERGENCY: Call 911 or go to the emergency room immediately. If you've taken pills or substances, this is a medical crisis that needs immediate attention.",
                "psychosis": "This sounds very distressing. Please contact 988 (Suicide & Crisis Lifeline), go to an emergency room, or call 911. These experiences need professional evaluation.",
                "domestic_violence": "Your safety is the priority. National Domestic Violence Hotline: 1-800-799-7233. They can help with safety planning and resources."
            },
            "high_priority": {
                "severe_depression": "Consider contacting 988 (Suicide & Crisis Lifeline) or a mental health professional today. Depression this severe needs immediate professional support.",
                "panic_attacks": "While panic attacks feel terrifying, they're not dangerous. Try box breathing (4-4-4-4). If they're frequent, see a doctor soon.",
                "substance_abuse": "SAMHSA National Helpline: 1-800-662-4357 provides 24/7 treatment referrals and information for addiction and mental health.",
                "eating_disorder": "National Eating Disorders Association: 1-800-931-2237. Eating disorders are serious medical conditions that need professional treatment."
            },
            "supportive": {
                "general": "Psychology Today and Open Path Collective can help find affordable therapy. Many employers offer Employee Assistance Programs (EAPs) with free counseling.",
                "lgbtq": "Trevor Project (1-866-488-7386) provides 24/7 LGBTQ+ crisis support. PFLAG offers resources for families.",
                "teens": "Teen Line: 1-800-852-8336 (6-10pm PT). Crisis Text Line: text HOME to 741741 anytime.",
                "veterans": "Veterans Crisis Line: 1-800-273-8255 (press 1) or text 838255. VA mental health services are available."
            }
        }

    def _build_follow_up_strategies(self) -> Dict[str, List[str]]:
        return {
            "clarification": [
                "What's the specific part that's bothering you most?",
                "Can you tell me more about what happened?",
                "How long has this been going on?",
                "What triggered this feeling today?"
            ],
            "solution_focused": [
                "What's one small step you could take today?",
                "Have you tried anything that's helped before?",
                "What would make this situation feel more manageable?",
                "Who in your life might be able to offer support?"
            ],
            "emotional_validation": [
                "That sounds really difficult to deal with.",
                "Your feelings about this make complete sense.",
                "Anyone would struggle with this situation.",
                "You're being really hard on yourself."
            ],
            "perspective_shifting": [
                "What would you tell a friend going through this?",
                "How do you think you'll feel about this in a year?",
                "What's the worst that could realistically happen?",
                "What's something good that could come from this challenge?"
            ],
            "safety_focused": [
                "Are you in a safe place right now?",
                "Do you have someone you can call if things get worse?",
                "Have you thought about hurting yourself?",
                "What keeps you going when things feel hopeless?"
            ]
        }

    def detect_crisis_level(self, text: str) -> Optional[CrisisIndicator]:
        """Enhanced crisis detection with multiple severity levels"""
        text_lower = text.lower()
        
        # Check immediate crisis indicators
        for category, patterns in self.crisis_patterns["immediate"].items():
            for pattern in patterns:
                if pattern in text_lower:
                    confidence = min(0.95, len([p for p in patterns if p in text_lower]) * 0.3)
                    return CrisisIndicator(
                        severity="immediate",
                        category=category,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        text_sample=text[:200]
                    )
        
        # Check high priority indicators
        for category, patterns in self.crisis_patterns["high"].items():
            matches = [p for p in patterns if p in text_lower]
            if matches:
                confidence = min(0.85, len(matches) * 0.25)
                return CrisisIndicator(
                    severity="high",
                    category=category,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    text_sample=text[:200]
                )
        
        # Check moderate indicators
        for category, patterns in self.crisis_patterns["moderate"].items():
            matches = [p for p in patterns if p in text_lower]
            if matches:
                confidence = min(0.75, len(matches) * 0.2)
                return CrisisIndicator(
                    severity="moderate",
                    category=category,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    text_sample=text[:200]
                )
        
        return None

    def detect_context(self, text: str) -> ResponseContext:
        text_lower = text.lower()
        primary_context = None
        sub_context = None
        vulnerable_population = None
        cultural_context = None
        
        # Detect primary context
        for context, sub_patterns in self.context_patterns.items():
            for sub, patterns in sub_patterns.items():
                if any(pat in text_lower for pat in patterns):
                    primary_context = context
                    sub_context = sub
                    break
            if primary_context:
                break

        # Detect vulnerable population indicators
        lgbtq_terms = ["gay", "lesbian", "bisexual", "transgender", "queer", "coming out", "closet", "pride"]
        teen_terms = ["high school", "teenager", "teen", "adolescent", "parents won't let me"]
        elderly_terms = ["retirement", "aging", "elderly", "senior", "grandchildren", "late in life"]
        parent_terms = ["my kids", "my children", "parenting", "mom life", "dad life"]
        immigrant_terms = ["immigration", "visa", "deportation", "home country", "cultural differences"]
        
        if any(term in text_lower for term in lgbtq_terms):
            vulnerable_population = "lgbtq"
        elif any(term in text_lower for term in teen_terms):
            vulnerable_population = "teens"
        elif any(term in text_lower for term in elderly_terms):
            vulnerable_population = "elderly"
        elif any(term in text_lower for term in parent_terms):
            vulnerable_population = "parents"
        elif any(term in text_lower for term in immigrant_terms):
            vulnerable_population = "immigrants"

        # Detect cultural context clues
        family_honor_terms = ["family honor", "shame on family", "disappoint parents", "cultural expectations"]
        religious_terms = ["sin", "faith", "prayer", "god", "religious", "church", "mosque", "temple"]
        
        if any(term in text_lower for term in family_honor_terms):
            cultural_context = "collectivist"
        elif any(term in text_lower for term in religious_terms):
            cultural_context = "religious"

        # Determine intensity level
        intensity_level = "medium"
        for level, markers in self.intensity_markers.items():
            if any(marker in text_lower for marker in markers):
                intensity_level = level
                break

        # Time context
        time_context = None
        if any(word in text_lower for word in ["today", "now", "urgent", "asap", "immediately", "right now"]):
            time_context = "immediate"
        elif any(word in text_lower for word in ["tomorrow", "soon", "this week"]):
            time_context = "short_term"

        return ResponseContext(
            primary_context=primary_context or "general",
            sub_context=sub_context,
            urgency_level=intensity_level,
            time_context=time_context,
            cultural_context=cultural_context,
            vulnerable_population=vulnerable_population
        )

    def generate_crisis_response(self, crisis_indicator: CrisisIndicator) -> str:
        """Generate appropriate crisis intervention response"""
        if crisis_indicator.severity == "immediate":
            base_response = self.crisis_resources["immediate"].get(
                crisis_indicator.category,
                self.crisis_resources["immediate"]["suicidal"]
            )
            return f"{base_response}\n\nYour life has value and meaning. This pain you're feeling is temporary, even though it doesn't feel that way right now. Please reach out for help immediately."
        
        elif crisis_indicator.severity == "high":
            base_response = self.crisis_resources["high_priority"].get(
                crisis_indicator.category,
                self.crisis_resources["high_priority"]["severe_depression"]
            )
            return f"I'm very concerned about what you're sharing. {base_response}\n\nYou don't have to handle this alone. What's one person you could reach out to today?"
        
        else:  # moderate
            return f"What you're experiencing sounds really difficult and painful. While this isn't an immediate crisis, these feelings deserve attention and support. {self.crisis_resources['supportive']['general']} Have you been able to talk to anyone about these feelings?"

    def select_response_for_context(self, context: ResponseContext, emotion: str, user_text: str) -> str:
        """Select appropriate response based on context and emotion"""
        
        # Check for vulnerable population specific responses
        if (context.vulnerable_population and 
            context.vulnerable_population in self.vulnerable_population_responses and
            context.sub_context in self.vulnerable_population_responses[context.vulnerable_population]):
            responses = self.vulnerable_population_responses[context.vulnerable_population][context.sub_context]
            return random.choice(responses)
        
        # Check main response database
        if (context.primary_context in self.response_database and
            context.sub_context and context.sub_context in self.response_database[context.primary_context] and
            emotion in self.response_database[context.primary_context][context.sub_context]):
            responses = self.response_database[context.primary_context][context.sub_context][emotion]
            return random.choice(responses)
        
        # Fallback to contextual response
        return self._generate_contextual_fallback(context, emotion, user_text)

    def _generate_contextual_fallback(self, context: ResponseContext, emotion: str, user_text: str) -> str:
        emotion_starters = {
            "depression": "That sounds really heavy and difficult. ",
            "anxiety": "I can sense the worry and stress in what you're sharing. ",
            "frustration": "I can hear how frustrated and annoyed you are about this. ",
            "calmness": "It’s great to see you approaching this with a quiet and reflective mindset. "
        }
        
        context_specific = {
            "academic": "School and academic pressure can be overwhelming. ",
            "career": "Work and career stuff can really impact our wellbeing. ",
            "relationships": "People and relationship issues can be emotionally draining. ",
            "health": "Health concerns affect every aspect of our lives. ",
            "financial": "Money stress touches everything and can feel suffocating. ",
            "personal": "Personal struggles and identity questions are deeply challenging. ",
            "social_issues": "Facing discrimination or social challenges is exhausting and painful. "
        }
        
        starter = emotion_starters.get(emotion, "I hear what you're going through. ")
        context_acknowledgment = context_specific.get(context.primary_context, "")
        
        # Add cultural sensitivity if detected
        if context.cultural_context and context.cultural_context in self.cultural_adaptations:
            cultural_note = self.cultural_adaptations[context.cultural_context].get("shame_sensitivity", "")
            if cultural_note:
                context_acknowledgment += f" {cultural_note} "
        
        follow_ups = [
            "What's the most challenging part for you right now?",
            "How has this been affecting your daily life?",
            "What kind of support would be most helpful?",
            "Have you been dealing with this for a while?",
            "What would help you feel more supported?"
        ]
        
        return f"{starter}{context_acknowledgment}{random.choice(follow_ups)}"

    def _add_follow_up_question(self, context: ResponseContext, emotion: str, crisis_level: Optional[str] = None) -> str:
        """Add appropriate follow-up question based on context and crisis level"""
        if crisis_level == "immediate":
            return random.choice(self.follow_up_strategies["safety_focused"])
        elif crisis_level in ["high", "moderate"]:
            return random.choice(self.follow_up_strategies["safety_focused"] + self.follow_up_strategies["emotional_validation"])
        elif emotion == "depression":
            if context.primary_context == "academic":
                return "What's one small academic goal you could focus on today?"
            elif context.primary_context == "relationships":
                return "Who in your life makes you feel most understood?"
            else:
                return random.choice(self.follow_up_strategies["emotional_validation"])
        elif emotion == "anxiety":
            if context.time_context == "immediate":
                return "What's one thing you can control in this situation right now?"
            else:
                return "What physical sensations are you noticing with this anxiety?"
        elif emotion == "frustration":
            return random.choice([
                "What would need to change for this to improve?",
                "Is there one aspect of this you could tackle first?",
                "What's worked to help you manage frustration before?"
            ])
        else:
            return random.choice(self.follow_up_strategies["clarification"])

    def generate_response(self, user_text: str, detected_emotion: str) -> str:
        """Main response generation method with comprehensive crisis handling"""
        
        # Input validation
        if not user_text or len(user_text.strip()) < 3:
            return "I'd like to help you, but I need a bit more information. What's on your mind?"
        
        # Limit input length for safety
        user_text = user_text[:2000]
        
        # Detect crisis level first
        crisis_indicator = self.detect_crisis_level(user_text)
        if crisis_indicator:
            log_crisis_event(crisis_indicator, f"Emotion: {detected_emotion}")
            
            # For immediate crises, return crisis response immediately
            if crisis_indicator.severity == "immediate":
                return self.generate_crisis_response(crisis_indicator)
        
        # Detect context
        context = self.detect_context(user_text)
        
        # Generate base response
        base_response = self.select_response_for_context(context, detected_emotion, user_text)
        
        # Add crisis response if needed (for high/moderate)
        if crisis_indicator and crisis_indicator.severity in ["high", "moderate"]:
            crisis_response = self.generate_crisis_response(crisis_indicator)
            base_response = f"{base_response}\n\n{crisis_response}"
        
        # Add follow-up question if appropriate
        crisis_level = crisis_indicator.severity if crisis_indicator else None
        follow_up = self._add_follow_up_question(context, detected_emotion, crisis_level)
        
        # Combine response parts
        if follow_up and len(base_response + " " + follow_up) < 400:
            final_response = f"{base_response} {follow_up}"
        else:
            final_response = base_response
        
        return final_response

    def generate_response_with_model(self, user_text: str, detected_emotion: str) -> str:

        # Get base response first
        base_response = self.generate_response(user_text, detected_emotion)
        
        # Don't modify crisis responses
        if any(keyword in base_response for keyword in ["988", "741741", "emergency", "IMMEDIATE ACTION", "crisis"]):
            return base_response
        
        # Try to enhance with chat model
        if chat_generator is None:
            return base_response
        
        try:
            context = self.detect_context(user_text)
            
            # Create context-aware prompt
            prompt = f"""You are an empathetic AI assistant providing emotional support.

Context: User is dealing with {context.primary_context} issues, specifically {context.sub_context or 'general concerns'}.
Emotional state: {detected_emotion} with {context.urgency_level} intensity.
Population context: {context.vulnerable_population or 'general'}
Cultural context: {context.cultural_context or 'none detected'}

User message: "{user_text[:200]}..."

Base supportive response: "{base_response[:150]}..."

Guidelines:
- Be empathetic and validate their feelings
- Avoid giving medical or legal advice
- Keep response conversational and supportive
- Don't repeat the base response exactly
- Focus on emotional support and understanding
- Ask one thoughtful follow-up question if appropriate
- Keep under 200 words

Enhanced response:"""

            outputs = chat_generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
            enhanced = outputs[0].get('generated_text', '').strip()
            
            # Extract just the response part
            if "Enhanced response:" in enhanced:
                enhanced = enhanced.split("Enhanced response:")[-1].strip()
            
            # Clean up and validate
            lines = [line.strip() for line in enhanced.split('\n') if line.strip()]
            if lines:
                candidate = lines[0]
                # Safety checks
                if (50 < len(candidate) < 500 and 
                    not any(bad_phrase in candidate.lower() for bad_phrase in 
                           ["i am an ai", "as an ai", "i cannot", "i'm not qualified"]) and
                    not any(crisis_word in candidate.lower() for crisis_word in 
                           ["kill", "die", "suicide", "harm"])):
                    return candidate
        
        except Exception as e:
            logger.exception("Enhanced response generation failed: %s", e)
        
        return base_response

# -------------------------
# Initialize Response Generator
# -------------------------
enhanced_premium_gen = EnhancedPremiumResponseGenerator()

def generate_ai_response_final(user_text: str, detected_emotion: str) -> str:
    """Main response generation endpoint"""
    try:
        return enhanced_premium_gen.generate_response_with_model(user_text, detected_emotion)
    except Exception as e:
        logger.exception("generate_ai_response_final error: %s", e)
        return enhanced_premium_gen.generate_response(user_text, detected_emotion)

# -------------------------
# Recommendations System
# -------------------------
def get_comprehensive_recommendations(emotion_probs: List[float], context: Optional[str] = None) -> Dict:
    idx = int(np.argmax(emotion_probs))
    emotion = FINAL_CLASSES[idx]
    confidence = float(emotion_probs[idx])
    
    base_recommendations = {
        "depression": {
            "immediate": ["Take a 5-minute walk outside", "Reach out to one trusted person", "Write down one thing you're grateful for"],
            "daily": ["Maintain a sleep schedule", "Engage in light physical activity", "Practice mindfulness or meditation"],
            "professional": ["Consider therapy (CBT, DBT)", "Speak with primary care doctor", "Look into support groups"],
            "crisis": "If having thoughts of self-harm: 988 (Suicide & Crisis Lifeline), Crisis Text Line (text HOME to 741741)"
        },
        "anxiety": {
            "immediate": ["Box breathing (4-4-4-4)", "5-4-3-2-1 grounding technique", "Progressive muscle relaxation"],
            "daily": ["Limit caffeine", "Regular exercise", "Maintain consistent routines"],
            "professional": ["Anxiety therapy (CBT, exposure therapy)", "Medical evaluation for anxiety disorders", "Stress management counseling"],
            "crisis": "For panic attacks or severe anxiety: Focus on breathing, remember it will pass, call 988 if overwhelmed"
        },
        "frustration": {
            "immediate": ["Take 10 deep breaths", "Step away from the situation", "Physical movement or exercise"],
            "daily": ["Identify frustration triggers", "Practice assertiveness skills", "Use problem-solving techniques"],
            "professional": ["Anger management counseling", "Conflict resolution therapy", "Stress management training"],
            "crisis": "If anger feels dangerous: Remove yourself from situation, call someone you trust, seek immediate help"
        },
        "calmness": {
            "immediate": ["Maintain this peaceful state", "Reflect on what's working well", "Plan next positive steps"],
            "daily": ["Continue current coping strategies", "Share positivity with others", "Set realistic goals"],
            "professional": ["Consider this a good time for therapy maintenance", "Wellness check-ins", "Preventive mental health care"],
            "crisis": "Use this stable time to build support systems and coping skills"
        }
    }
    
    recommendations = base_recommendations.get(emotion, base_recommendations["calmness"])
    
    # Add context-specific recommendations
    if context:
        context_additions = {
            "academic": {
                "activities": ["Break study into small chunks", "Form study groups", "Visit professor during office hours"],
                "message": "Academic stress is manageable with the right strategies and support."
            },
            "relationships": {
                "activities": ["Communicate openly about feelings", "Set healthy boundaries", "Spend time with supportive friends"],
                "message": "Healthy relationships require work from both sides and clear communication."
            },
            "career": {
                "activities": ["Update resume", "Network with professionals", "Consider career counseling"],
                "message": "Career challenges are opportunities for growth and new directions."
            }
        }
        if context in context_additions:
            recommendations["contextual"] = context_additions[context]
    
    return {
        "emotion": emotion,
        "confidence": confidence,
        "recommendations": recommendations,
        "message": f"You're experiencing {emotion} with {confidence*100:.0f}% confidence. These suggestions are tailored to help you right now."
    }

# -------------------------
# Session Management for Crisis Tracking
# -------------------------
def check_crisis_escalation() -> Dict[str, any]:
    """Monitor for escalating crisis indicators"""
    global crisis_indicators_session, last_crisis_time
    
    current_time = datetime.now()
    recent_indicators = [
        indicator for indicator in crisis_indicators_session 
        if (current_time - indicator.timestamp).seconds < 1800  # Last 30 minutes
    ]
    
    immediate_count = len([i for i in recent_indicators if i.severity == "immediate"])
    high_count = len([i for i in recent_indicators if i.severity == "high"])
    
    escalation_risk = "low"
    if immediate_count > 0:
        escalation_risk = "critical"
    elif high_count >= 2:
        escalation_risk = "high"
    elif len(recent_indicators) >= 3:
        escalation_risk = "moderate"
    
    return {
        "risk_level": escalation_risk,
        "recent_indicators": len(recent_indicators),
        "immediate_count": immediate_count,
        "high_count": high_count,
        "last_crisis": last_crisis_time.isoformat() if last_crisis_time else None
    }

# -------------------------
# Routes 
# -------------------------
@app.route("/")
def index():
    return render_template("index3.html")

@app.route("/start_detection", methods=["POST"])
def start_detection():
    global is_detecting, stop_event, crisis_indicators_session
    try:
        if is_detecting:
            return jsonify({"success": True, "message": "Detection already running"})
        
        # Clear previous session data
        crisis_indicators_session.clear()
        
        stop_event.clear()
        t1 = threading.Thread(target=camera_capture_loop, daemon=True)
        t2 = threading.Thread(target=processing_loop, daemon=True)
        t1.start()
        t2.start()
        is_detecting = True
        logger.info("Detection started with enhanced crisis monitoring")
        return jsonify({"success": True, "message": "Detection started"})
    except Exception as e:
        logger.exception("Error in start_detection: %s", e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    global is_detecting, stop_event, emotion_history, crisis_indicators_session
    try:
        if not is_detecting:
            return jsonify({"success": True, "message": "Detection already stopped"})
        stop_event.set()
        with frame_queue.mutex:
            frame_queue.queue.clear()
        with processed_frame_queue.mutex:
            processed_frame_queue.queue.clear()
        emotion_history.clear()
        crisis_indicators_session.clear()
        is_detecting = False
        logger.info("Detection stopped and resources released")
        return jsonify({"success": True, "message": "Detection stopped"})
    except Exception as e:
        logger.exception("Error in stop_detection: %s", e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/update_text", methods=["POST"])
def update_text():
    global current_text
    try:
        payload = request.get_json() or {}
        new_text = payload.get("text", "").strip()
        
        # Input validation and sanitization
        if len(new_text) > 2000:
            new_text = new_text[:2000]
        
        # Check for crisis indicators in real-time
        if new_text:
            crisis_indicator = enhanced_premium_gen.detect_crisis_level(new_text)
            if crisis_indicator and crisis_indicator.severity == "immediate":
                log_crisis_event(crisis_indicator, "Real-time text analysis")
        
        current_text = new_text
        return jsonify({"success": True, "message": "Text updated"})
    except Exception as e:
        logger.exception("Error in update_text: %s", e)
        return jsonify({"success": False, "message": str(e)})
    


@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    try:
        payload = request.get_json() or {}
        text = payload.get("text", "").strip()
        
        if not text or len(text) < 3:
            return jsonify({"success": False, "message": "Text too short"})
        
        # Input validation and sanitization
        if len(text) > 2000:
            text = text[:2000]
        
        # Analyze the text using your existing text emotion prediction function
        text_probs = predict_text_emotion(text)
        
        # Determine dominant emotion
        dominant_idx = int(np.argmax(text_probs))
        dominant_emotion = FINAL_CLASSES[dominant_idx]
        
        return jsonify({
            "success": True,
            "probabilities": text_probs.tolist(),
            "dominant_emotion": dominant_emotion,
            "confidence": float(np.max(text_probs))
        })
        
    except Exception as e:
        logger.exception("Error in analyze_text: %s", e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/get_emotion_data")
def get_emotion_data():
    if not is_detecting:
        return jsonify({
            "success": False, 
            "message": "Detection not running", 
            "probabilities": [0.25, 0.25, 0.25, 0.25], 
            "face_detected": False,
            "crisis_status": check_crisis_escalation()
        })
    try:
        try:
            meta = processed_frame_queue.get(timeout=0.5)
        except Empty:
            return jsonify({
                "success": True, 
                "probabilities": [0.25, 0.25, 0.25, 0.25], 
                "face_detected": False, 
                "emotions": FINAL_CLASSES,
                "crisis_status": check_crisis_escalation()
            })
        
        probs = meta.get('probs')
        face_detected = meta.get('face_detected', False)
        
        return jsonify({
            "success": True, 
            "probabilities": probs if probs is not None else [0.25, 0.25, 0.25, 0.25], 
            "face_detected": bool(face_detected), 
            "emotions": FINAL_CLASSES,
            "crisis_status": check_crisis_escalation()
        })
    except Exception as e:
        logger.exception("Error in get_emotion_data: %s", e)
        return jsonify({
            "success": False, 
            "message": str(e), 
            "probabilities": [0.25, 0.25, 0.25, 0.25], 
            "face_detected": False,
            "crisis_status": check_crisis_escalation()
        })

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/generate_response", methods=["POST"])
def generate_response():
    try:
        payload = request.get_json() or {}
        user_text = payload.get("text", "").strip()
        detected_emotion = payload.get("detected_emotion", "calmness")  # Get live emotion from frontend
        confidence = payload.get("confidence", 0)
        
        if not user_text or len(user_text) < 3:
            return jsonify({"success": False, "message": "Please provide more text for analysis"})
        
        # Use the live-detected emotion instead of emotion_history
        # If no live emotion provided, fall back to text analysis
        if detected_emotion == "calmness" and not confidence:
            # Fallback: analyze the text if no live emotion data
            text_probs = predict_text_emotion(user_text)
            detected_emotion = FINAL_CLASSES[int(np.argmax(text_probs))]
        
        # Generate comprehensive response using the CURRENT emotion
        reply = generate_ai_response_final(user_text, detected_emotion)
        
        # Get context for additional insights
        context = enhanced_premium_gen.detect_context(user_text)
        crisis_indicator = enhanced_premium_gen.detect_crisis_level(user_text)
        
        response_data = {
            "success": True,
            "reply": reply,
            "emotion": detected_emotion,  # Return the emotion we used
            "context": {
                "primary": context.primary_context,
                "sub": context.sub_context,
                "urgency": context.urgency_level,
                "vulnerable_population": context.vulnerable_population,
                "cultural_context": context.cultural_context
            },
            "crisis_status": check_crisis_escalation()
        }
        
        if crisis_indicator:
            response_data["crisis_detected"] = {
                "severity": crisis_indicator.severity,
                "category": crisis_indicator.category,
                "confidence": crisis_indicator.confidence
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception("Error in generate_response: %s", e)
        return jsonify({"success": False, "message": str(e)})
    
    
@app.route("/get_recommendations")
def recommendations():
    try:
        # Get current emotion probabilities
        try:
            meta = processed_frame_queue.get(timeout=0.5)
            probabilities = meta.get('probs')
        except Empty:
            probabilities = None
        
        if probabilities is not None:
            # Determine context from current text if available
            context = None
            if current_text:
                detected_context = enhanced_premium_gen.detect_context(current_text)
                context = detected_context.primary_context
            
            rec = get_comprehensive_recommendations(probabilities, context)
            dominant = FINAL_CLASSES[int(np.argmax(probabilities))]
            
            return jsonify({
                "success": True, 
                "recommendations": rec, 
                "dominant_emotion": dominant,
                "crisis_status": check_crisis_escalation()
            })
        
        return jsonify({"success": False, "message": "No emotion data available"})
    except Exception as e:
        logger.exception("Error in recommendations: %s", e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/crisis_status")
def crisis_status():
    """Dedicated endpoint for crisis monitoring"""
    try:
        status = check_crisis_escalation()
        recent_indicators = [
            {
                "severity": indicator.severity,
                "category": indicator.category,
                "timestamp": indicator.timestamp.isoformat(),
                "confidence": indicator.confidence
            }
            for indicator in crisis_indicators_session[-5:]  # Last 5 indicators
        ]
        
        return jsonify({
            "success": True,
            "crisis_status": status,
            "recent_indicators": recent_indicators,
            "total_session_indicators": len(crisis_indicators_session)
        })
    except Exception as e:
        logger.exception("Error in crisis_status: %s", e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/performance")
def performance():
    stats = perf_monitor.get_stats()
    models_loaded = bool(face_model is not None and text_model is not None)
    camera_active = camera is not None and getattr(camera, 'isOpened', lambda: False)()
    
    return jsonify({
        "success": True, 
        "stats": stats, 
        "models_loaded": models_loaded, 
        "camera_active": camera_active,
        "crisis_monitoring": {
            "session_indicators": len(crisis_indicators_session),
            "last_crisis": last_crisis_time.isoformat() if last_crisis_time else None
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "models_loaded": bool(face_model is not None and text_model is not None), 
        "detecting": is_detecting, 
        "crisis_system": "active",
        "timestamp": time.time()
    })

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    logger.info("Starting Enhanced GenAI Emotion Detection backend with Comprehensive Crisis Support")
    logger.info("Crisis logging enabled at: %s", CRISIS_LOG_FILE)
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

# -------------------------
# Non-destructive enhancements (appended) - Rich layered responses & memory wiring
# -------------------------
from dataclasses import asdict
import datetime as dt

class RichResponseEngine:
    def __init__(self):
        self.empathy_variants = [
            "That sounds really difficult and heavy.",
            "I can hear how painful that feels right now.",
            "You're carrying a lot — it's understandable to feel overwhelmed.",
            "I hear how heavy that is for you.",
            "That must be so exhausting to carry."
        ]
        self.validation_variants = [
            "What you're feeling makes sense given what you've described.",
            "Anyone in your situation would likely feel similarly.",
            "It makes sense you'd be reacting strongly to this.",
            "Your feelings are valid and understandable.",
        ]
        self.followups = [
            "What's the most challenging part for you right now?",
            "When did this start feeling like too much?",
            "Who could you call or message if things got worse?",
            "Would you like a short breathing exercise now?"
        ]
        self.coping = {
            "breathing": ["Try a 4-4-4 breathing: breathe 4s, hold 4s, exhale 4s.", "Try 3 slow diaphragmatic breaths right now."],
            "grounding": ["Name 5 things you can see, 4 you can touch, 3 you can hear.", "Hold a comforting object and describe it to yourself."],
            "microtask": ["Break one task into a 10-minute chunk and do only that.", "Set a 10-minute timer and focus on a single small step."]
        }

    def sample(self, arr):
        return random.choice(arr) if arr else ""

    def build_reply(self, user_text: str, detected_emotion: str, conversation_history_local: list):
        empathy = self.sample(self.empathy_variants)
        validation = self.sample(self.validation_variants)
        context_note = ""
        try:
            if 'enhanced_premium_gen' in globals() and hasattr(enhanced_premium_gen, 'detect_context'):
                ctx = enhanced_premium_gen.detect_context(user_text)
                if ctx and getattr(ctx, 'primary_context', None):
                    context_note = f" This seems related to {ctx.primary_context} concerns."
        except Exception:
            context_note = ""
        validation = validation + context_note
        coping_options = []
        if detected_emotion == 'anxiety':
            coping_options.append(self.sample(self.coping['breathing']))
            coping_options.append(self.sample(self.coping['grounding']))
        elif detected_emotion == 'depression':
            coping_options.append(self.sample(self.coping['microtask']))
            coping_options.append("Reach out to one trusted person and say you need a quick chat.")
        elif detected_emotion == 'frustration':
            coping_options.append(self.sample(self.coping['grounding']))
            coping_options.append("Try stepping away for 10 minutes and move your body.")
        else:
            coping_options.append("You're sounding steady — keep doing what helps you.")

        follow_up = self.sample(self.followups)
        if detected_emotion == 'anxiety':
            follow_up = "Would you like to try a brief breathing exercise together now?"
        elif detected_emotion == 'depression':
            follow_up = "Is there one small thing you feel you could try today that might help even a little?"

        escalation = None
        try:
            concat_recent = ' '.join([t for t in conversation_history_local[-CONVERSATION_MEMORY_SIZE:]])
            if 'enhanced_premium_gen' in globals() and hasattr(enhanced_premium_gen, 'detect_crisis_level'):
                ci = enhanced_premium_gen.detect_crisis_level(concat_recent)
                if ci and getattr(ci, 'severity', None) == 'immediate':
                    escalation = ci
        except Exception:
            escalation = None

        parts = [empathy, validation]
        if coping_options:
            parts.append("Here's something that sometimes helps: " + coping_options[0])
        parts.append(follow_up)
        plain = "\n\n".join(parts)
        payload = {
            'empathy': empathy,
            'validation': validation,
            'coping': coping_options,
            'follow_up': follow_up,
            'emotion': detected_emotion,
            'timestamp': dt.datetime.utcnow().isoformat() + 'Z'
        }
        return plain, payload, escalation

_rich_response_engine = RichResponseEngine()
_last_reply_payload = None

def generate_ai_response_enhanced(user_text: str, detected_emotion: str) -> str:
    global _last_reply_payload
    try:
        convo = conversation_history if 'conversation_history' in globals() else []
        # conversation_history in original is list[str]; ensure list[str]
        convo_texts = [t if isinstance(t, str) else t.get('text','') for t in convo]
        plain, payload, escalation = _rich_response_engine.build_reply(user_text, detected_emotion, convo_texts)
        try:
            if escalation is not None:
                if 'enhanced_premium_gen' in globals() and hasattr(enhanced_premium_gen, 'generate_crisis_response'):
                    crisis_text = enhanced_premium_gen.generate_crisis_response(escalation)
                    plain = crisis_text + "\n\n" + plain
                    payload['escalation'] = {'severity': escalation.severity, 'category': escalation.category, 'confidence': escalation.confidence}
        except Exception:
            pass
        _last_reply_payload = payload
        return plain
    except Exception:
        try:
            if 'generate_ai_response_final_original' in globals():
                return generate_ai_response_final_original(user_text, detected_emotion)
            elif 'enhanced_premium_gen' in globals() and hasattr(enhanced_premium_gen, 'generate_response_with_model'):
                return enhanced_premium_gen.generate_response_with_model(user_text, detected_emotion)
        except Exception:
            pass
        return "I hear you. Can you tell me a bit more about what's happening?"

# preserve original if present
try:
    if 'generate_ai_response_final' in globals() and callable(generate_ai_response_final):
        generate_ai_response_final_original = generate_ai_response_final
except Exception:
    generate_ai_response_final_original = None

# rebind name
generate_ai_response_final = generate_ai_response_enhanced

# helper endpoint for frontend to fetch structured payload
try:
    @app.route('/last_reply_payload')
    def last_reply_payload():
        global _last_reply_payload
        if _last_reply_payload is None:
            return jsonify({'success':False, 'message':'No payload available yet'})
        return jsonify({'success':True, 'payload':_last_reply_payload})
except Exception:
    pass


