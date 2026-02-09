import os
import logging
import sqlite3
import json
import uuid
import zipfile
import io
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

app = FastAPI(title="Cough Detection Server", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Configuration ----
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')
CLEANUP_INTERVAL_HOURS = 5
THRESHOLD = 0.825

# Хранилище аудио
AUDIO_STORAGE_PATH = "cough_audio_storage"
MAX_AUDIO_FILES = 1000  # Максимум 1000 файлов
AUDIO_RETENTION_DAYS = 30  # Хранить файлы 30 дней

# Создаем папки
os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)

def get_current_datetime():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Database ----
DB_PATH = "cough_db.db"

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

def cleanup_old_records():
    """Удаление записей старше CLEANUP_INTERVAL_HOURS часов"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now(SERVER_TIMEZONE) - 
                      timedelta(hours=CLEANUP_INTERVAL_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            DELETE FROM cough_records 
            WHERE timestamp < ?
        ''', (cutoff_time,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"🧹 Удалено {deleted_count} записей старше {CLEANUP_INTERVAL_HOURS} часов")
        
        return deleted_count
    except Exception as e:
        logger.error(f"Ошибка очистки БД: {e}")
        return 0

def cleanup_old_audio_files():
    """Удаление старых аудиофайлов"""
    try:
        cutoff_date = datetime.now() - timedelta(days=AUDIO_RETENTION_DAYS)
        
        # Получаем старые записи с аудио
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, file_path FROM cough_records 
            WHERE file_path != '' 
            AND timestamp < ?
        ''', (cutoff_date.strftime("%Y-%m-%d %H:%M:%S"),))
        
        old_records = cursor.fetchall()
        
        deleted_files = 0
        for record_id, file_path in old_records:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    
                    # Обновляем запись
                    cursor.execute('''
                        UPDATE cough_records 
                        SET file_path = '', audio_saved = 0 
                        WHERE id = ?
                    ''', (record_id,))
                except Exception as e:
                    logger.error(f"Ошибка удаления файла {file_path}: {e}")
        
        conn.commit()
        conn.close()
        
        if deleted_files > 0:
            logger.info(f"🧹 Удалено {deleted_files} старых аудиофайлов")
        
        # Проверяем лимит файлов
        enforce_audio_storage_limit()
        
        return deleted_files
    except Exception as e:
        logger.error(f"Ошибка очистки аудиофайлов: {e}")
        return 0

def enforce_audio_storage_limit():
    """Проверяет и соблюдает лимит файлов"""
    try:
        # Получаем все файлы с информацией о дате
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, file_path, timestamp FROM cough_records 
            WHERE file_path != '' 
            ORDER BY timestamp ASC
        ''')
        
        files = cursor.fetchall()
        
        if len(files) <= MAX_AUDIO_FILES:
            return
        
        # Удаляем самые старые файлы сверх лимита
        files_to_delete = files[:len(files) - MAX_AUDIO_FILES]
        
        deleted_count = 0
        for record_id, file_path, _ in files_to_delete:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    
                    cursor.execute('''
                        UPDATE cough_records 
                        SET file_path = '', audio_saved = 0 
                        WHERE id = ?
                    ''', (record_id,))
                except Exception as e:
                    logger.error(f"Ошибка удаления файла {file_path}: {e}")
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"🧹 Удалено {deleted_count} файлов для соблюдения лимита {MAX_AUDIO_FILES}")
        
    except Exception as e:
        logger.error(f"Ошибка проверки лимита хранилища: {e}")

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
        logger.info("✅ Новая оптимизированная модель загружена (2079 фич)")
        
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ YAMNet загружен")
        
        SCALER = joblib.load('cough_scaler_final_optimized.pkl')
        logger.info("✅ Scaler загружен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки моделей: {e}")
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
        
        expected_dim = 2079
        if len(combined) != expected_dim:
            logger.warning(f"Размерность фич {len(combined)} вместо {expected_dim}")
            if len(combined) > expected_dim:
                combined = combined[:expected_dim]
            else:
                padding = np.zeros(expected_dim - len(combined))
                combined = np.concatenate([combined, padding])
        
        return combined
        
    except Exception as e:
        logger.error(f"Ошибка извлечения фич: {e}")
        return None
    

# ---- Feature penalty ----

def fast_enhanced_check(waveform, sr, original_prob):
    """Быстрая улучшенная проверка (оптимизирована для сервера)"""
    
    # Только 2 самые важные проверки для скорости
    modified_prob = original_prob
    
    # 1. Быстрая проверка спектрального распределения
    # Используем MFCC вместо полного STFT для скорости
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=512)
    
    # Низкие частоты: первые 4 MFCC, средние: следующие 4
    low_freq_energy = np.mean(np.abs(mfcc[:4, :]))
    mid_freq_energy = np.mean(np.abs(mfcc[4:8, :]))
    
    if mid_freq_energy > low_freq_energy * 1.5:  # Доминируют средние частоты (речь)
        modified_prob *= 0.9
    
    # 2. Быстрая проверка резкости атаки через ZCR
    zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=256)[0]
    zcr_std = np.std(zcr)
    
    # Кашель имеет более резкие изменения ZCR
    if zcr_std < 0.05:  # Слишком плавно (не кашель)
        modified_prob *= 0.9
    
    return modified_prob

# ---- Feature penalty ----

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Анализ аудио с новой моделью и улучшенной проверкой"""
    if not OUR_MODEL or not SCALER or not YAMNET_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Модели не загружены"}
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.02:
            return {"probability": 0.0, "cough_detected": False, "message": "Тишина"}
        
        max_val = np.max(np.abs(waveform))
        if max_val < 0.01:
            return {"probability": 0.0, "cough_detected": False, "message": "Слишком тихо"}
        
        waveform = waveform / (max_val + 1e-8)
        
        target_length = 16000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        features = extract_features_new(waveform, sr, YAMNET_MODEL)
        
        if features is None:
            return {"probability": 0.0, "cough_detected": False, "message": "Ошибка извлечения фич"}
    
        if len(features) != 2079:
            logger.error(f"Неверная размерность фич: {len(features)} вместо 2079")
            if len(features) < 2079:
                padding = np.zeros(2079 - len(features))
                features = np.concatenate([features, padding])
            else:
                features = features[:2079]
        
        features_scaled = SCALER.transform(features.reshape(1, -1))
        
        prediction = OUR_MODEL.predict(features_scaled, verbose=0)
        original_prob = float(prediction[0][0])
        
        # === ДОБАВЛЕНА УЛУЧШЕННАЯ ПРОВЕРКА ===
        enhanced_prob = fast_enhanced_check(waveform, sr, original_prob)
        
        # Логируем разницу для отладки
        prob_diff = original_prob - enhanced_prob
        if prob_diff > 0.1:  # Значительное снижение
            logger.warning(f"⚠️ Сильное снижение вероятности: {original_prob:.3f} → {enhanced_prob:.3f} (diff: -{prob_diff:.3f})")
        
        is_cough = enhanced_prob > THRESHOLD
        
        logger.info(f"🎯 Анализ: {filename} | orig={original_prob:.3f} | enhanced={enhanced_prob:.3f} | cough={is_cough}")
        
        return {
            "probability": enhanced_prob,  # Возвращаем улучшенную вероятность
            "original_probability": original_prob,  # Сохраняем оригинал для отладки
            "cough_detected": bool(is_cough),
            "confidence": enhanced_prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "cough_count": 1 if is_cough else 0,
            "enhancement_applied": enhanced_prob != original_prob,  # Флаг применения улучшения
            "probability_reduction": round(prob_diff, 3) if prob_diff > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return {"probability": 0.0, "cough_detected": False, "message": f"Ошибка: {str(e)}"}

# ---- API Endpoints ----

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    """Загрузка аудио и анализ"""
    logger.info(f"📥 Загрузка: {audio.filename}, device_id: {device_id}")
    
    try:
        cleanup_old_records()
        
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Пустой файл")
        
        current_datetime = get_current_datetime()
        
        result = analyze_audio(raw, audio.filename)
        
        # Сохраняем аудио если найден кашель
        audio_path = ""
        audio_saved = 0
        if result["cough_detected"]:
            audio_path = save_cough_audio(raw, device_id, result["probability"])
            audio_saved = 1 if audio_path else 0
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp, audio_saved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, 
            audio.filename,
            audio_path,
            float(result["probability"]),
            int(result["cough_detected"]),
            result["message"],
            "[]",
            json.dumps({"audio_saved": bool(audio_path)}),
            current_datetime,
            audio_saved
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Результат: {result}, ID записи: {record_id}")
        
        return JSONResponse({
            "status": "success", 
            **result,
            "record_id": record_id,
            "audio_saved": bool(audio_path),
            "download_url": f"/download/{record_id}" if audio_path else None,
            "all_coughs_url": "/coughs/audio/all"
        })
        
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(500, str(e))

@app.get("/download/{record_id}")
async def download_audio(record_id: int):
    """Скачать конкретный аудиофайл по ID записи"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT device_id, filename, file_path, cough_detected 
            FROM cough_records 
            WHERE id = ?
        ''', (record_id,))
        
        record = cursor.fetchone()
        conn.close()
        
        if not record:
            raise HTTPException(404, f"Запись с ID {record_id} не найдена")
        
        device_id, filename, file_path, cough_detected = record
        
        if not cough_detected:
            raise HTTPException(400, "Это не запись с кашлем")
        
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
            WHERE cough_detected = 1 
            AND file_path != '' 
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
            WHERE cough_detected = 1 
            AND file_path != '' 
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
async def cleanup_audio_endpoint(days_old: int = Query(AUDIO_RETENTION_DAYS, le=365)):
    """Ручная очистка старых аудиофайлов"""
    try:
        deleted_files = cleanup_old_audio_files()
        
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
    model_loaded = OUR_MODEL is not None and SCALER is not None
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
        "timestamp": datetime.now().isoformat(),
        "features_dimension": 2079 if model_loaded else "unknown"
    })

@app.get("/")
async def root():
    return {
        "message": "🔥 УЛУЧШЕННЫЙ Сервер Детекции Кашля v3.0",
        "version": "3.0",
        "features": "Сохранение аудио, лимит 1000 файлов, скачивание архива",
        "endpoints": {
            "POST /upload": "Загрузить аудио (сохраняет если кашель)",
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
    logger.info("🚀 Запуск сервера детекции кашля v3.0...")
    init_db()
    load_models()
    logger.info(f"✅ Сервер готов! Порог кашля: {THRESHOLD}")
    logger.info(f"💾 Аудиохранилище: {AUDIO_STORAGE_PATH}")
    logger.info(f"📁 Максимум файлов: {MAX_AUDIO_FILES}")
    logger.info(f"🧹 Автоочистка записей: каждые {CLEANUP_INTERVAL_HOURS} часов")
    logger.info(f"🗑️ Автоочистка аудио: каждые {AUDIO_RETENTION_DAYS} дней")

# ---- Main ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting COUGH DETECTION SERVER v3.0 on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")