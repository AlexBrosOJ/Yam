import os
import logging
import sqlite3
import json
import uuid
import zipfile
import io
import threading
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import tempfile
import pytz
import joblib

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cough_server")

app = FastAPI(title="Cough Detection Server", version="3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(HTTPException)
async def http_error_handler(request, error):
    return JSONResponse({"status": "error", "message": str(error.detail)}, status_code=error.status_code)


# ---- Configuration ----
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')
# These are retention HOURS, not a scheduler interval. Keep records as long as audio.
CLEANUP_INTERVAL_HOURS = int(os.getenv("RECORD_RETENTION_HOURS", "120"))
THRESHOLD = float(os.getenv("COUGH_THRESHOLD", "0.50"))
ENABLE_LEGACY_ENHANCEMENT = os.getenv("ENABLE_LEGACY_ENHANCEMENT", "false").lower() == "true"
MIN_RMS = float(os.getenv("MIN_RMS", "0"))
MIN_PEAK = float(os.getenv("MIN_PEAK", "0"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
_MODEL_LOCK = threading.RLock()
_UPLOAD_LOCK = threading.RLock()
_last_cleanup = 0.0
if not 0 <= THRESHOLD <= 1 or MIN_RMS < 0 or MIN_PEAK < 0:
    raise ValueError("Invalid detection thresholds")
if CLEANUP_INTERVAL_HOURS <= 0 or MAX_UPLOAD_BYTES <= 0:
    raise ValueError("Retention and upload limit must be positive")

# Хранилище аудио
AUDIO_STORAGE_PATH = os.getenv("AUDIO_STORAGE_PATH", "cough_audio_storage")
MAX_AUDIO_FILES = int(os.getenv("MAX_AUDIO_FILES", "500"))  # Максимум 500 файлов
AUDIO_RETENTION_DAYS = int(os.getenv("AUDIO_RETENTION_DAYS", "5"))  # Хранить файлы 5 дней

# Создаем папки
if MAX_AUDIO_FILES < 1 or AUDIO_RETENTION_DAYS < 1:
    raise ValueError("Audio storage limits must be positive")
os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)

def get_current_datetime():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Database ----
DB_PATH = os.getenv("DB_PATH", "cough_db.db")

def init_db():
    """Инициализация БД с автоматической очисткой"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cough_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            filename TEXT,
            file_path TEXT,
            probability REAL,
            cough_detected INTEGER,
            message TEXT,
            top_classes TEXT,
            cough_stats TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            audio_saved INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_time ON cough_records(device_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cough_detected ON cough_records(cough_detected)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_saved ON cough_records(audio_saved)')
    
    conn.commit()
    conn.close()
    
    cleanup_old_records()
    cleanup_old_audio_files()
    logger.info("✅ Database initialized with cleanup")

def _owned_audio_path(path):
    root = Path(AUDIO_STORAGE_PATH).resolve()
    candidate = Path(path)
    return (not candidate.is_symlink() and candidate.resolve().parent == root
            and candidate.name.startswith("cough_") and candidate.suffix == ".wav")


def _clear_missing_audio_references():
    with sqlite3.connect(DB_PATH) as conn:
        for record_id, path in conn.execute("SELECT id, file_path FROM cough_records WHERE file_path != ''").fetchall():
            if not Path(path).is_file():
                conn.execute("UPDATE cough_records SET file_path='', audio_saved=0 WHERE id=?", (record_id,))


def cleanup_old_records():
    """Remove expired records together with their owned files, without orphaning audio."""
    cutoff = (datetime.now(SERVER_TIMEZONE) - timedelta(hours=CLEANUP_INTERVAL_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, file_path FROM cough_records WHERE timestamp < ?", (cutoff,)).fetchall()
        for record_id, path in rows:
            if path and _owned_audio_path(path):
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    logger.exception("Cannot remove expired audio %s", path)
                    continue  # Retry later; don't lose its database reference.
            conn.execute("DELETE FROM cough_records WHERE id = ?", (record_id,))
    return len(rows)

def cleanup_old_audio_files(days_old=None):
    """Scan owned WAV files, including historical orphans; retention uses file mtime."""
    days = AUDIO_RETENTION_DAYS if days_old is None else days_old
    cutoff = time.time() - days * 86400
    removed = 0
    for path in Path(AUDIO_STORAGE_PATH).glob("cough_*.wav"):
        if not _owned_audio_path(path):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Cannot clean audio %s", path)
    _clear_missing_audio_references()
    enforce_audio_storage_limit()
    return removed

def enforce_audio_storage_limit():
    files = []
    for path in Path(AUDIO_STORAGE_PATH).glob("cough_*.wav"):
        if _owned_audio_path(path):
            try:
                files.append((path.stat().st_mtime_ns, str(path), path))
            except FileNotFoundError:
                pass
    for _, _, path in sorted(files)[:max(0, len(files) - MAX_AUDIO_FILES)]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Cannot enforce audio limit: %s", path)
    _clear_missing_audio_references()

def save_cough_audio(audio_bytes: bytes, device_id: str, probability: float) -> str:
    """Сохраняет аудио с кашлем в файл"""
    try:
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Очищаем device_id от небезопасных символов
        safe_device_id = "".join(c for c in device_id if c.isalnum() or c in ('-', '_'))[:50]
        safe_device_id = safe_device_id if safe_device_id else "unknown"
        
        # Формируем имя файла
        filename = f"cough_{timestamp}_{safe_device_id}_{unique_id}.wav"
        filepath = os.path.join(AUDIO_STORAGE_PATH, filename)
        
        # Сохраняем файл
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        
        logger.info(f"💾 Сохранено аудио: {filename} (prob={probability:.3f})")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения аудио: {e}")
        return ""

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None
SCALER = None

def load_models():
    global OUR_MODEL, YAMNET_MODEL, SCALER
    
    try:
        OUR_MODEL = tf.keras.models.load_model(
            'cough_detection_final_optimized.keras',
            compile=False
        )
        logger.info("Новая оптимизированная модель загружена (2079 фич)")
        
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("YAMNet загружен")
        
        SCALER = joblib.load('cough_scaler_final_optimized.pkl')
        if getattr(SCALER, "n_features_in_", None) != 2079:
            raise ValueError("Scaler must expect 2079 features")
        if OUR_MODEL.input_shape[-1] != 2079 or OUR_MODEL.output_shape[-1] != 1:
            raise ValueError("Expected model input 2079 and one output")
        logger.info("Scaler загружен")
        
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        raise

# ---- Feature Extraction ----
def extract_features_new(waveform, sr, yamnet_model):
    """Извлекает РОВНО 2079 фич как в обученной модели"""
    try:
        if np.max(np.abs(waveform)) < 0.01:
            return None
        
        max_val = np.max(np.abs(waveform))
        waveform = waveform / (max_val + 1e-8)
        
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform_tf)
        
        avg_pool = tf.reduce_mean(embeddings, axis=0).numpy()
        max_pool = tf.reduce_max(embeddings, axis=0).numpy()
        
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=512)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, hop_length=512)[0]
        
        spectral_features = np.array([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),  
            np.mean(spectral_bandwidth)
        ])
        
        zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=512)[0]
        zcr_mean = np.mean(zcr)
        
        rms = librosa.feature.rms(y=waveform, hop_length=512)[0]
        rms_mean = np.mean(rms)
        
        combined = np.concatenate([
            avg_pool,
            max_pool,  
            mfcc_mean,
            mfcc_std,
            spectral_features,
            [zcr_mean, rms_mean]
        ])
        
        if combined.shape != (2079,):
            raise ValueError(f"Expected 2079 features, got {combined.shape}")

        return combined
        
    except Exception as e:
        logger.error(f"Ошибка извлечения фич: {e}")
        return None
    

# ---- Feature penalty ----

def fast_enhanced_check(waveform, sr, original_prob):
    """Быстрая улучшенная проверка (умеренные штрафы)"""
    
    modified_prob = original_prob
    
    # 1. Проверка соотношения низких/средних частот
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=512)
    low_freq = np.mean(np.abs(mfcc[:4, :]))      # Низкие
    mid_freq = np.mean(np.abs(mfcc[4:8, :]))     # Средние
    low_to_mid = low_freq / (mid_freq + 1e-6)
    
    # Умеренные штрафы: макс -20%
    if low_to_mid < 0.18:  # Слишком мало низких (ложное)
        modified_prob *= 0.78  # -10% (было -35%)
    elif low_to_mid > 0.27:  # Много низких (кашель)
        modified_prob *= 1.25  # +10% (было +20%)
    
    # 2. Проверка резкости атаки
    envelope = np.abs(waveform)
    envelope_smooth = np.convolve(envelope, np.ones(1000)/1000, mode='same')
    
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(envelope_smooth, height=np.max(envelope_smooth)*0.3)
    
    if len(peaks) > 0:
        peak_to_rms_ratio = envelope_smooth[peaks[0]] / (np.sqrt(np.mean(waveform**2)) + 1e-6)
        
        # Умеренные штрафы
        if peak_to_rms_ratio < 7.1:  # Низкая пиковость (ложное)
            modified_prob *= 0.82  # -10% (было -20%)
        elif peak_to_rms_ratio > 7.3:  # Высокая пиковость (кашель)
            modified_prob *= 1.25  # +10% (было +15%)
    else:
        # Нет пиков - легкий штраф
        modified_prob *= 0.75  # -5% (было -30%)
    
    # 3. Проверка спектральной плоскости
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=waveform))
    
    # Умеренные штрафы
    if spectral_flatness < 0.04:  # Слишком тонально (ложное)
        modified_prob *= 0.82  # -5% (было -15%)
    elif spectral_flatness > 0.05:  # Более шумно (кашель)
        modified_prob *= 1.152  # +5% (было +10%)
    
    # 4. Количество пиков
    zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=256)[0]
    zcr_peaks = len(find_peaks(zcr, height=np.mean(zcr)*1.5)[0])
    
    # Умеренные штрафы
    if zcr_peaks < 35:  # Мало изменений (ложное)
        modified_prob *= 0.9  # -5% (было -25%)
    elif zcr_peaks > 45:  # Много изменений (кашель)
        modified_prob *= 1.1  # +5% (было +15%)
    
    # Возвращаем float, а не numpy тип!
    return float(np.clip(modified_prob, 0.0, 1.0))

# ---- Feature penalty ----

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Same one-second/2079-feature model input; no heuristic penalties by default."""
    if OUR_MODEL is None or SCALER is None or YAMNET_MODEL is None:
        raise HTTPException(503, "Models are not loaded")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            waveform, sr = librosa.load(tmp_path, sr=16000, mono=True, duration=1.0)
        except Exception as error:
            raise HTTPException(422, "Cannot decode audio") from error
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
    if not len(waveform) or not np.isfinite(waveform).all():
        raise HTTPException(422, "Audio is empty or contains non-finite samples")
    rms = float(np.sqrt(np.mean(waveform**2)))
    peak = float(np.max(np.abs(waveform)))
    clipping_fraction = float(np.mean(np.abs(waveform) >= 0.999))
    skipped = peak == 0 or rms < MIN_RMS or peak < MIN_PEAK
    original_prob = 0.0
    final_prob = 0.0
    if not skipped:
        waveform = waveform / (peak + 1e-8)
        waveform = np.pad(waveform, (0, max(0, 16000-len(waveform))))[:16000]
        with _MODEL_LOCK:
            features = extract_features_new(waveform, sr, YAMNET_MODEL)
            if features is None or features.shape != (2079,) or not np.isfinite(features).all():
                raise RuntimeError("Feature extraction failed")
            features_scaled = SCALER.transform(features.reshape(1, -1))
            if not np.isfinite(features_scaled).all():
                raise RuntimeError("Scaler produced non-finite values")
            prediction = np.asarray(OUR_MODEL.predict(features_scaled, verbose=0))
        if prediction.shape != (1, 1):
            raise RuntimeError("Unexpected model output shape: %s" % (prediction.shape,))
        original_prob = float(prediction[0, 0])
        if not np.isfinite(original_prob) or not 0 <= original_prob <= 1:
            raise RuntimeError("Model output is not a finite score in [0, 1]")
        final_prob = (fast_enhanced_check(waveform, sr, original_prob)
                      if ENABLE_LEGACY_ENHANCEMENT else original_prob)
    detected = bool(not skipped and final_prob > THRESHOLD)
    logger.info("analysis file=%s rms=%.6f peak=%.6f clipping=%.4f orig=%.4f final=%.4f threshold=%.3f skipped=%s cough=%s",
                filename, rms, peak, clipping_fraction, original_prob, final_prob, THRESHOLD, skipped, detected)
    return {
        "probability": float(final_prob), "original_probability": float(original_prob),
        "cough_detected": detected, "confidence": float(final_prob),
        "message": "Тишина" if skipped else "COUGH_DETECTED" if detected else "NO_COUGH",
        "cough_count": 1 if detected else 0,
        "enhancement_applied": bool(ENABLE_LEGACY_ENHANCEMENT and not skipped),
        "probability_reduction": float(max(0.0, original_prob-final_prob)),
        "threshold": THRESHOLD, "rms": rms, "peak": peak,
        "clipping_fraction": clipping_fraction, "skipped_by_gate": bool(skipped),
        "legacy_enhancement_enabled": ENABLE_LEGACY_ENHANCEMENT
    }

# ---- API Endpoints ----

@app.post("/upload")
def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    """Synchronous route runs in FastAPI's thread pool, not on the asyncio event loop."""
    global _last_cleanup
    raw = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio upload is too large")
    filename = os.path.basename(audio.filename or "audio.wav")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        # Single worker recommended. Serialize DB+inference bookkeeping and network retries.
        with _UPLOAD_LOCK:
            if time.monotonic() - _last_cleanup > 60:
                cleanup_old_audio_files()
                cleanup_old_records()
                _last_cleanup = time.monotonic()
            with sqlite3.connect(DB_PATH) as conn:
                recent = conn.execute("SELECT id, file_path, cough_stats FROM cough_records WHERE device_id=? AND filename=? ORDER BY id DESC LIMIT 1",
                                      (device_id, filename)).fetchone()
            if recent:
                try:
                    saved = json.loads(recent[2] or "{}")
                except (ValueError, TypeError):
                    saved = {}
                if saved.get("sha256") == digest and "analysis" in saved:
                    exists = bool(recent[1] and Path(recent[1]).is_file())
                    return JSONResponse({"status": "success", **saved["analysis"], "record_id": recent[0],
                        "audio_saved": exists, "download_url": f"/download/{recent[0]}" if exists else None,
                        "all_coughs_url": "/coughs/audio/all", "duplicate": True})
            result = analyze_audio(raw, filename)
            path = save_cough_audio(raw, device_id, result["probability"])
            if not path:
                raise RuntimeError("Cannot save audio; retry later")
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.execute("""INSERT INTO cough_records
                        (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp, audio_saved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (device_id, filename, path, result["probability"], int(result["cough_detected"]),
                         result["message"], "[]", json.dumps({"audio_saved": True, "sha256": digest, "analysis": result}),
                         get_current_datetime(), 1))
                    record_id = cursor.lastrowid
            except Exception:
                Path(path).unlink(missing_ok=True)
                raise
            try:
                enforce_audio_storage_limit()
            except Exception:
                # Record has committed: do not turn housekeeping failure into a duplicate upload.
                logger.exception("Storage maintenance failed after successful upload")
            exists = Path(path).is_file()
            return JSONResponse({"status": "success", **result, "record_id": record_id,
                "audio_saved": exists, "download_url": f"/download/{record_id}" if exists else None,
                "all_coughs_url": "/coughs/audio/all"})
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Upload analysis/storage failed: %s", filename)
        raise HTTPException(500, "Audio processing failed; retry later") from error

@app.get("/download/{record_id}")
async def download_audio(record_id: int):
    """Скачать конкретный аудиофайл по ID записи"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT device_id, filename, file_path 
            FROM cough_records 
            WHERE id = ?
        ''', (record_id,))
        
        record = cursor.fetchone()
        conn.close()
        
        if not record:
            raise HTTPException(404, f"Запись с ID {record_id} не найдена")
        
        device_id, filename, file_path = record
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(404, "Аудиофайл не найден или был удален")
        
        return FileResponse(
            path=file_path,
            filename=f"cough_{record_id}_{os.path.basename(filename)}",
            media_type="audio/wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания: {e}")
        raise HTTPException(500, str(e))

@app.get("/coughs/audio/list")
async def list_cough_audio(
    device_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(100, le=500)
):
    """Список всех записей с кашлем и аудио"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, device_id, filename, file_path, probability, timestamp 
            FROM cough_records  
            WHERE file_path != '' 
            AND file_path IS NOT NULL
        '''
        params = []
        
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        
        if date_from:
            query += " AND DATE(timestamp) >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND DATE(timestamp) <= ?"
            params.append(date_to)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record_id, device, filename, file_path, prob, timestamp = row
            
            file_exists = os.path.exists(file_path) if file_path else False
            
            records.append({
                "id": record_id,
                "device_id": device,
                "filename": filename,
                "probability": float(prob),
                "timestamp": timestamp,
                "has_audio": file_exists,
                "file_size": os.path.getsize(file_path) if file_exists else 0,
                "download_url": f"/download/{record_id}" if file_exists else None
            })
        
        return {
            "status": "success",
            "count": len(records),
            "storage_path": AUDIO_STORAGE_PATH,
            "max_files": MAX_AUDIO_FILES,
            "current_files": len([r for r in records if r["has_audio"]]),
            "records": records
        }
        
    except Exception as e:
        logger.exception(f"Ошибка получения списка аудио: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/coughs/audio/all")
async def download_all_cough_audio():
    """Скачать ВСЕ аудиофайлы с кашлем одним ZIP архивом"""
    try:
        # Получаем все файлы с кашлем
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, device_id, filename, file_path, timestamp 
            FROM cough_records 
            WHERE file_path != '' 
            AND file_path IS NOT NULL
            ORDER BY timestamp DESC
        ''')
        
        records = cursor.fetchall()
        conn.close()
        
        # Проверяем есть ли файлы
        valid_files = []
        for record_id, device_id, filename, file_path, timestamp in records:
            if file_path and os.path.exists(file_path):
                valid_files.append({
                    "id": record_id,
                    "device_id": device_id,
                    "filename": filename,
                    "file_path": file_path,
                    "timestamp": timestamp
                })
        
        if not valid_files:
            raise HTTPException(404, "Нет аудиофайлов с кашлем для скачивания")
        
        # Создаем ZIP архив в памяти
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in valid_files:
                try:
                    # Добавляем файл в архив с понятным именем
                    archive_name = f"cough_{file_info['id']}_{file_info['device_id']}_{os.path.basename(file_info['filename'])}"
                    zip_file.write(file_info['file_path'], archive_name)
                except Exception as e:
                    logger.error(f"Ошибка добавления файла {file_info['file_path']} в архив: {e}")
        
        zip_buffer.seek(0)
        
        # Формируем имя архива
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filename = f"all_cough_audio_{timestamp}.zip"
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={archive_filename}",
                "Content-Type": "application/zip"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ошибка создания архива: {e}")
        raise HTTPException(500, f"Ошибка создания архива: {str(e)}")

@app.get("/coughs/audio/stats")
async def get_audio_stats():
    """Статистика по хранилищу аудио"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE cough_detected = 1')
        total_coughs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE audio_saved = 1')
        coughs_with_audio = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE file_path != ""')
        files_in_db = cursor.fetchone()[0]
        
        conn.close()
        
        # Статистика по файловой системе
        audio_files = []
        total_size = 0
        if os.path.exists(AUDIO_STORAGE_PATH):
            for filename in os.listdir(AUDIO_STORAGE_PATH):
                filepath = os.path.join(AUDIO_STORAGE_PATH, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    total_size += size
                    audio_files.append({
                        "name": filename,
                        "size_mb": round(size / (1024*1024), 3),
                        "path": filepath
                    })
        
        # Сортируем по размеру
        audio_files.sort(key=lambda x: x["size_mb"], reverse=True)
        
        return {
            "status": "success",
            "storage": {
                "path": AUDIO_STORAGE_PATH,
                "max_files": MAX_AUDIO_FILES,
                "current_files": len(audio_files),
                "total_size_mb": round(total_size / (1024*1024), 2),
                "retention_days": AUDIO_RETENTION_DAYS
            },
            "database": {
                "total_coughs": total_coughs,
                "coughs_with_audio": coughs_with_audio,
                "files_in_db": files_in_db
            },
            "largest_files": audio_files[:10] if audio_files else []
        }
        
    except Exception as e:
        logger.exception(f"Ошибка получения статистики: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.delete("/coughs/audio/cleanup")
async def cleanup_audio_endpoint(days_old: int = Query(AUDIO_RETENTION_DAYS, ge=1, le=365)):
    """Ручная очистка старых аудиофайлов"""
    try:
        deleted_files = cleanup_old_audio_files(days_old)
        
        return {
            "status": "success",
            "message": f"Удалено {deleted_files} аудиофайлов старше {days_old} дней",
            "deleted_count": deleted_files
        }
    except Exception as e:
        logger.exception(f"Ошибка ручной очистки: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ---- Существующие эндпоинты (оставляем как есть) ----
@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """Статистика за сегодня"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = get_current_date()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(cough_detected) as total_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp)=?
        ''', (device_id, today))
        
        stats = cursor.fetchone()
        total_recordings = int(stats[0] or 0) if stats else 0
        total_coughs = int(stats[1] or 0) if stats else 0
        avg_probability = float(stats[2] or 0.0) if stats and stats[2] is not None else 0.0
        
        hourly_stats = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records
                WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=? 
                AND strftime('%H', timestamp)=?
            ''', (device_id, today, f"{hour:02d}"))
            count_row = cursor.fetchone()
            count = int(count_row[0] or 0) if count_row else 0
            hourly_stats.append({"hour": hour_str, "count": count})
        
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [
            {"time": row[0], "probability": float(row[1])} 
            for row in cursor.fetchall()
        ]
        
        peak_hours = "Нет данных"
        cough_frequency = "0 раз/день"
        
        if total_coughs > 0:
            if hourly_stats:
                max_hour = max(hourly_stats, key=lambda x: x["count"])
                peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
            cough_frequency = f"{total_coughs} раз/день"
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total_recordings,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_probability, 3)
            },
            "hourly_stats": hourly_stats,
            "recent_coughs": recent_coughs,
            "patterns": {
                "peak_hours": peak_hours,
                "cough_frequency": cough_frequency,
                "intensity": "Высокая" if avg_probability > 0.7 else "Средняя" if avg_probability > 0.3 else "Низкая",
                "trend": "📊"
            }
        }
        
        return result
        
    except Exception as e:
        logger.exception(f"Ошибка статистики: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

from fastapi.responses import RedirectResponse

@app.get("/debug/stats/{device_id}")
async def debug_stats_redirect(device_id: str):
    return RedirectResponse(url=f"/stats/{device_id}")

@app.get("/records/all")
async def get_all_records(
    device_id: str = None, 
    limit: int = 100,
    offset: int = 0,
    include_audio: bool = False
):
    """Получить ВСЕ записи (с пагинацией)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if device_id:
            query = '''
                SELECT * FROM cough_records 
                WHERE device_id=? 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            '''
            params = (device_id, limit, offset)
            cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=?', (device_id,))
        else:
            query = '''
                SELECT * FROM cough_records 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            '''
            params = (limit, offset)
            cursor.execute('SELECT COUNT(*) FROM cough_records')
        
        total_count = cursor.fetchone()[0]
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            record = dict(row)
            record['cough_detected'] = bool(record['cough_detected'])
            record['probability'] = float(record['probability'])
            
            if not include_audio:
                record.pop('file_path', None)
            
            records.append(record)
        
        conn.close()
        
        return {
            "status": "success",
            "total_records": total_count,
            "returned_records": len(records),
            "limit": limit,
            "offset": offset,
            "device_id": device_id,
            "records": records
        }
        
    except Exception as e:
        logger.exception(f"Ошибка получения записей: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/records/{device_id}/count")
async def get_records_count(device_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=?', (device_id,))
        total = cursor.fetchone()[0]
        
        today = get_current_date()
        cursor.execute('''
            SELECT COUNT(*) FROM cough_records 
            WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=?
        ''', (device_id, today))
        today_coughs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=? AND cough_detected=1', (device_id,))
        all_coughs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "device_id": device_id,
            "total_records": total,
            "coughs_today": today_coughs,
            "total_coughs": all_coughs,
            "last_cleanup": CLEANUP_INTERVAL_HOURS
        }
        
    except Exception as e:
        logger.exception(f"Ошибка подсчета: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.delete("/records/cleanup")
async def manual_cleanup():
    try:
        deleted_count = cleanup_old_records()
        return {
            "status": "success",
            "message": f"Удалено {deleted_count} записей старше {CLEANUP_INTERVAL_HOURS} часов",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.exception(f"Ошибка ручной очистки: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/records/export/{device_id}")
async def export_records(device_id: str, format: str = "json"):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM cough_records 
            WHERE device_id=? 
            ORDER BY timestamp
        ''', (device_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        conn.close()
        
        records = []
        for row in rows:
            record = dict(zip(columns, row))
            record['cough_detected'] = bool(record['cough_detected'])
            records.append(record)
        
        if format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
            
            return {
                "status": "success",
                "format": "csv",
                "count": len(records),
                "data": output.getvalue()
            }
        
        else:
            return {
                "status": "success",
                "format": "json",
                "count": len(records),
                "records": records
            }
        
    except Exception as e:
        logger.exception(f"Ошибка экспорта: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/health")
async def health_check():
    model_loaded = OUR_MODEL is not None and SCALER is not None and YAMNET_MODEL is not None
    db_exists = os.path.exists(DB_PATH)
    
    db_status = "healthy"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM cough_records')
        db_status = "healthy"
        conn.close()
    except:
        db_status = "unhealthy"
    
    audio_storage_ok = os.path.exists(AUDIO_STORAGE_PATH) and os.path.isdir(AUDIO_STORAGE_PATH)
    
    storage_size = 0
    if audio_storage_ok:
        storage_size = sum(os.path.getsize(os.path.join(AUDIO_STORAGE_PATH, f)) 
                          for f in os.listdir(AUDIO_STORAGE_PATH) 
                          if os.path.isfile(os.path.join(AUDIO_STORAGE_PATH, f)))
    
    return JSONResponse({
        "status": "healthy" if model_loaded and db_status == "healthy" else "degraded",
        "model_loaded": model_loaded,
        "scaler_loaded": SCALER is not None,
        "database": db_status,
        "audio_storage": "healthy" if audio_storage_ok else "unhealthy",
        "database_path": DB_PATH,
        "audio_storage_path": AUDIO_STORAGE_PATH,
        "storage_size_mb": round(storage_size / (1024*1024), 2),
        "max_audio_files": MAX_AUDIO_FILES,
        "current_audio_files": len([f for f in os.listdir(AUDIO_STORAGE_PATH) if os.path.isfile(os.path.join(AUDIO_STORAGE_PATH, f))]),
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "threshold": THRESHOLD,
        "legacy_enhancement_enabled": ENABLE_LEGACY_ENHANCEMENT,
        "min_rms": MIN_RMS, "min_peak": MIN_PEAK,
        "server_version": "3.1",
        "timestamp": datetime.now().isoformat(),
        "features_dimension": 2079 if model_loaded else "unknown"
    })

@app.get("/")
async def root():
    return {
        "message": "🔥 УЛУЧШЕННЫЙ Сервер Детекции Кашля v3.1",
        "version": "3.1",
        "features": "Сохранение аудио, настраиваемый лимит файлов, скачивание архива",
        "endpoints": {
            "POST /upload": "Загрузить аудио (сохраняет все успешные анализы)",
            "GET /download/{id}": "Скачать конкретный файл",
            "GET /coughs/audio/all": "СКАЧАТЬ ВСЕ ФАЙЛЫ (ZIP архив)",
            "GET /coughs/audio/list": "Список всех файлов с кашлем",
            "GET /coughs/audio/stats": "Статистика хранилища",
            "DELETE /coughs/audio/cleanup": "Очистить старые файлы",
            "GET /stats/{device_id}": "Статистика за сегодня",
            "GET /records/all": "Все записи",
            "GET /health": "Проверка здоровья"
        }
    }

# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    """Запуск при старте сервера"""
    logger.info("🚀 Запуск сервера детекции кашля v3.1...")
    init_db()
    load_models()
    logger.info(f"✅ Сервер готов! Порог кашля: {THRESHOLD}")
    logger.info(f"💾 Аудиохранилище: {AUDIO_STORAGE_PATH}")
    logger.info(f"📁 Максимум файлов: {MAX_AUDIO_FILES}")
    logger.info(f"🧹 Хранение записей: {CLEANUP_INTERVAL_HOURS} часов; обслуживание при загрузках")
    logger.info(f"🗑️ Хранение аудио: до {AUDIO_RETENTION_DAYS} дней и {MAX_AUDIO_FILES} файлов")

# ---- Main ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting COUGH DETECTION SERVER v3.1 on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
