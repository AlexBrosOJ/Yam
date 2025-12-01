# server_fixed_stats.py
import os
import logging
import sqlite3
from datetime import datetime
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import tempfile
import noisereduce as nr
import scipy.signal as signal
import pytz

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cough_server")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Configuration ----
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')

def get_current_datetime():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Database ----
DB_PATH = "cough_db.db"
def init_db():
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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized")

init_db()

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None

def load_models():
    global OUR_MODEL, YAMNET_MODEL
    try:
        OUR_MODEL = tf.keras.models.load_model('improved_cough_model.keras', compile=False)
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ Improved models loaded")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

load_models()

def preprocess_audio(waveform, sr):
    try:
        waveform = librosa.effects.preemphasis(waveform)
        
        if len(waveform) > 8000:
            try:
                noise_sample = waveform[:4000]
                waveform = nr.reduce_noise(y=waveform, sr=sr, y_noise=noise_sample, prop_decrease=0.7)
            except:
                pass
        
        sos = signal.butter(4, [80, 4000], 'bandpass', fs=sr, output='sos')
        waveform = signal.sosfilt(sos, waveform)
        
        max_amp = np.max(np.abs(waveform))
        if max_amp < 0.01:
            return None
        waveform = waveform / max_amp * 0.91
        
        return waveform
    except:
        return None

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    if not OUR_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Model not loaded"}
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.01:
            return {"probability": 0.0, "cough_detected": False, "message": "Silence"}
        
        waveform = preprocess_audio(waveform, sr)
        if waveform is None:
            return {"probability": 0.0, "cough_detected": False, "message": "Too quiet after processing"}
        
        target_length = 16000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_features = np.mean(mfcc, axis=1)
        
        combined_features = np.concatenate([avg_embedding, mfcc_features]).reshape(1, -1)
        
        prediction = OUR_MODEL.predict(combined_features, verbose=0)
        prob = float(prediction[0][0])
        
        is_cough = prob > 0.546
        
        logger.info(f"🎯 IMPROVED MODEL: {filename} | prob={prob:.3f} | cough={is_cough}")
        
        return {
            "probability": prob,
            "cough_detected": is_cough,
            "confidence": prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "cough_count": 1 if is_cough else 0
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"probability": 0.0, "cough_detected": False, "message": f"Error: {str(e)}"}

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    logger.info(f"📥 Upload: {audio.filename}, device_id: {device_id}")
    
    try:
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Empty file")
        
        current_datetime = get_current_datetime()
        current_date = get_current_date()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{device_id}_{audio.filename}"
        
        result = analyze_audio(raw, audio.filename)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, 
            filename,
            "",
            float(result["probability"]),
            int(result["cough_detected"]),
            result["message"],
            "[]",
            "{}",
            current_datetime
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Analysis result: {result}")
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/debug/stats/{device_id}")
async def debug_stats(device_id: str):
    """ФИКСИРОВАННАЯ СТАТИСТИКА - ТОЧНО КАК ОЖИДАЕТ ПРИЛОЖЕНИЕ"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = get_current_date()
        
        logger.info(f"🔍 DEBUG STATS: device_id={device_id}, today={today}")
        
        # Основная статистика за сегодня - ФИКСИРУЕМ ИМЕНА ПОЛЕЙ
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
        
        logger.info(f"📊 Статистика сегодня: total={total_recordings}, coughs={total_coughs}, avg_prob={avg_probability}")
        
        # Статистика по часам - ФИКСИРУЕМ ФОРМАТ
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
        
        # Последние кашли - ФИКСИРУЕМ ФОРМАТ
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [
            {"time": row[0], "probability": float(row[1])} 
            for row in cursor.fetchall()
        ]
        
        # Паттерны - ФИКСИРУЕМ ФОРМАТ
        peak_hours = "Нет данных"
        cough_frequency = "0 раз/день"
        
        if total_coughs > 0:
            # Находим пиковые часы
            if hourly_stats:
                max_hour = max(hourly_stats, key=lambda x: x["count"])
                peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
            
            # Частота кашля
            cough_frequency = f"{total_coughs} раз/день"
        
        conn.close()
        
        # ФИКСИРУЕМ СТРУКТУРУ ОТВЕТА - ТОЧНО КАК В СТАРОМ СЕРВЕРЕ
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
        
        logger.info(f"📊 Финальный результат статистики: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"DEBUG Stats error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """Основная статистика (для совместимости)"""
    try:
        # Просто вызываем debug_stats для единообразия
        return await debug_stats(device_id)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy", 
        "model_loaded": OUR_MODEL is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    return {"message": "Improved Cough Detection Server", "status": "running"}


# server_fixed_stats.py - ДОБАВЛЯЕМ НОВЫЕ ФУНКЦИИ

@app.get("/stats/{device_id}/range")
async def get_stats_range(device_id: str, start_date: str, end_date: str):
    """Статистика за произвольный период"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        logger.info(f"📊 RANGE STATS: device_id={device_id}, period={start_date} to {end_date}")
        
        # Основная статистика за период
        cursor.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(cough_detected) as total_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp) BETWEEN ? AND ?
        ''', (device_id, start_date, end_date))
        
        stats = cursor.fetchone()
        total_recordings = int(stats[0] or 0) if stats else 0
        total_coughs = int(stats[1] or 0) if stats else 0
        avg_probability = float(stats[2] or 0.0) if stats and stats[2] is not None else 0.0
        
        # Статистика по дням
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_recordings,
                SUM(cough_detected) as daily_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp) BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (device_id, start_date, end_date))
        
        daily_stats = []
        for row in cursor.fetchall():
            daily_stats.append({
                "date": row[0],
                "total_recordings": row[1],
                "total_coughs": row[2],
                "avg_probability": float(row[3] or 0.0)
            })
        
        # Статистика по неделям
        cursor.execute('''
            SELECT 
                strftime('%Y-%W', timestamp) as week,
                COUNT(*) as total_recordings,
                SUM(cough_detected) as weekly_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp) BETWEEN ? AND ?
            GROUP BY strftime('%Y-%W', timestamp)
            ORDER BY week
        ''', (device_id, start_date, end_date))
        
        weekly_stats = []
        for row in cursor.fetchall():
            weekly_stats.append({
                "week": row[0],
                "total_recordings": row[1],
                "total_coughs": row[2],
                "avg_probability": float(row[3] or 0.0)
            })
        
        conn.close()
        
        result = {
            "period_stats": {
                "start_date": start_date,
                "end_date": end_date,
                "total_recordings": total_recordings,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_probability, 3)
            },
            "daily_stats": daily_stats,
            "weekly_stats": weekly_stats
        }
        
        logger.info(f"📊 Range stats result: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Range stats error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/stats/{device_id}/available_dates")
async def get_available_dates(device_id: str):
    """Получить все доступные даты для устройства"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT DATE(timestamp) as date 
            FROM cough_records 
            WHERE device_id=? 
            ORDER BY date DESC
        ''', (device_id,))
        
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {"available_dates": dates}
        
    except Exception as e:
        logger.exception(f"Available dates error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/stats/{device_id}/daily/{date}")
async def get_daily_stats(device_id: str, date: str):
    """Статистика за конкретный день"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Основная статистика за день
        cursor.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(cough_detected) as total_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp)=?
        ''', (device_id, date))
        
        stats = cursor.fetchone()
        total_recordings = int(stats[0] or 0) if stats else 0
        total_coughs = int(stats[1] or 0) if stats else 0
        avg_probability = float(stats[2] or 0.0) if stats and stats[2] is not None else 0.0
        
        # Почасовые данные
        hourly_stats = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records
                WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=? 
                AND strftime('%H', timestamp)=?
            ''', (device_id, date, f"{hour:02d}"))
            count_row = cursor.fetchone()
            count = int(count_row[0] or 0) if count_row else 0
            hourly_stats.append({"hour": hour_str, "count": count})
        
        conn.close()
        
        result = {
            "date": date,
            "stats": {
                "total_recordings": total_recordings,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_probability, 3)
            },
            "hourly_stats": hourly_stats
        }
        
        return result
        
    except Exception as e:
        logger.exception(f"Daily stats error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting FIXED STATS server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")







