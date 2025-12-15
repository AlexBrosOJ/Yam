import os
import logging
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import tempfile
import pytz
import joblib

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cough_server")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Configuration ----
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')
CLEANUP_INTERVAL_HOURS = 24  # Удалять детальные записи старше 1 дня (статистика сохраняется)
THRESHOLD = 0.63  # Порог для кашля

def get_current_datetime():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Database ----
DB_PATH = "cough_db.db"

def init_db():
    """Инициализация БД с агрегированной статистикой"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Основная таблица для детальных записей (будем очищать)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cough_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            filename TEXT,
            probability REAL,
            cough_detected INTEGER,
            cough_count INTEGER DEFAULT 1,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. НОВАЯ таблица для АГРЕГИРОВАННОЙ статистики (НЕ очищается!)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_aggregated_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            date TEXT,  -- Формат: YYYY-MM-DD
            total_recordings INTEGER DEFAULT 0,
            total_coughs INTEGER DEFAULT 0,
            total_cough_episodes INTEGER DEFAULT 0,
            avg_probability REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(device_id, date)  -- Одна запись на устройство на день
        )
    ''')
    
    # 3. Таблица для доступных дат (для истории)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS available_dates (
            device_id TEXT,
            date TEXT,
            PRIMARY KEY (device_id, date)
        )
    ''')
    
    # Индексы для быстрого поиска
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_time ON cough_records(device_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_aggregated ON daily_aggregated_stats(device_id, date)')
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Database initialized with aggregated statistics")

def update_available_dates(device_id: str, date: str):
    """Обновляем список доступных дат для устройства"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO available_dates (device_id, date) 
            VALUES (?, ?)
        ''', (device_id, date))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка обновления доступных дат: {e}")

def cleanup_old_records():
    """Удаление ДЕТАЛЬНЫХ записей старше CLEANUP_INTERVAL_HOURS часов (статистика сохраняется)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now(SERVER_TIMEZONE) - 
                      timedelta(hours=CLEANUP_INTERVAL_HOURS))
        
        # Удаляем ТОЛЬКО детальные записи, статистика остается
        cursor.execute('''
            DELETE FROM cough_records 
            WHERE timestamp < ?
        ''', (cutoff_time.strftime("%Y-%m-%d %H:%M:%S"),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"🧹 Удалено {deleted_count} детальных записей старше {CLEANUP_INTERVAL_HOURS} часов")
        
        return deleted_count
    except Exception as e:
        logger.error(f"Ошибка очистки детальных записей: {e}")
        return 0

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None
SCALER = None

def load_models():
    """Загрузка модели и scaler'а"""
    global OUR_MODEL, YAMNET_MODEL, SCALER
    
    try:
        OUR_MODEL = tf.keras.models.load_model(
            'cough_detection_improved_model.keras', 
            compile=False
        )
        logger.info("✅ Новая модель загружена")
        
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ YAMNet загружен")
        
        SCALER = joblib.load('cough_scaler.pkl')
        logger.info("✅ Scaler загружен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки моделей: {e}")
        raise

# ---- Feature Extraction ----
def extract_features_new(waveform, sr, yamnet_model):
    """Извлекает 2079 фич"""
    try:
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform_tf)
        
        avg_pool = tf.reduce_mean(embeddings, axis=0).numpy()
        max_pool = tf.reduce_max(embeddings, axis=0).numpy()
        
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=13, hop_length=512
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=sr, hop_length=512
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=sr, hop_length=512
        )[0]
        
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
        
        return combined
        
    except Exception as e:
        logger.error(f"Ошибка извлечения фич: {e}")
        return None

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Анализ аудио"""
    if not OUR_MODEL or not SCALER or not YAMNET_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Модели не загружены"}
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.01:
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
        
        features_scaled = SCALER.transform(features.reshape(1, -1))
        prediction = OUR_MODEL.predict(features_scaled, verbose=0)
        prob = float(prediction[0][0])
        is_cough = prob > THRESHOLD
        
        cough_count = 1 if is_cough else 0
        
        logger.info(f"🎯 Анализ: {filename} | prob={prob:.3f} | cough={is_cough} | count={cough_count}")
        
        return {
            "probability": prob,
            "cough_detected": bool(is_cough),
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "cough_count": cough_count
        }
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return {"probability": 0.0, "cough_detected": False, "message": f"Ошибка: {str(e)}", "cough_count": 0}

# ---- API Endpoints ----

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    """Загрузка аудио и анализ с обновлением агрегированной статистики"""
    logger.info(f"📥 Загрузка: {audio.filename}, device_id: {device_id}")
    
    try:
        # Автоочистка старых детальных записей
        cleanup_old_records()
        
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Пустой файл")
        
        current_datetime = get_current_datetime()
        today = get_current_date()
        
        # Анализ аудио
        result = analyze_audio(raw, audio.filename)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Сохраняем детальную запись
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, probability, cough_detected, cough_count, message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, 
            audio.filename,
            float(result["probability"]),
            int(result["cough_detected"]),
            result["cough_count"],
            result["message"],
            current_datetime
        ))
        
        # 2. 🔥 ОБНОВЛЯЕМ агрегированную статистику (НАКОПЛЕНИЕ)
        # Создаем запись для сегодня, если еще нет
        cursor.execute('''
            INSERT OR IGNORE INTO daily_aggregated_stats 
            (device_id, date) VALUES (?, ?)
        ''', (device_id, today))
        
        # Рассчитываем среднюю вероятность кашлей за сегодня
        cursor.execute('''
            SELECT AVG(probability) 
            FROM cough_records 
            WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=?
        ''', (device_id, today))
        avg_prob_row = cursor.fetchone()
        avg_probability = float(avg_prob_row[0]) if avg_prob_row[0] is not None else 0.0
        
        # Обновляем агрегированные данные (УВЕЛИЧИВАЕМ счетчики)
        cursor.execute('''
            UPDATE daily_aggregated_stats 
            SET 
                total_recordings = total_recordings + 1,
                total_coughs = total_coughs + ?,
                total_cough_episodes = total_cough_episodes + ?,
                avg_probability = ?,
                last_updated = ?
            WHERE device_id=? AND date=?
        ''', (
            int(result["cough_detected"]),      # +1 если обнаружен кашель
            result["cough_count"],              # количество кашлевых эпизодов
            avg_probability,
            current_datetime,
            device_id,
            today
        ))
        
        # 3. Обновляем список доступных дат
        update_available_dates(device_id, today)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Статистика обновлена: total_recordings +1, coughs +{int(result['cough_detected'])}")
        
        return JSONResponse({
            "status": "success", 
            **result,
            "stats_updated": True
        })
        
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """Статистика за сегодня из АГРЕГИРОВАННОЙ таблицы"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = get_current_date()
        
        # 🔥 Берем данные из АГРЕГИРОВАННОЙ таблицы (не пересчитываем!)
        cursor.execute('''
            SELECT 
                total_recordings,
                total_coughs,
                total_cough_episodes,
                avg_probability
            FROM daily_aggregated_stats 
            WHERE device_id=? AND date=?
        ''', (device_id, today))
        
        row = cursor.fetchone()
        
        if row:
            total_recordings = int(row[0] or 0)
            total_coughs = int(row[1] or 0)
            total_cough_episodes = int(row[2] or 0)
            avg_probability = float(row[3] or 0.0)
        else:
            # Первый раз за сегодня
            total_recordings = 0
            total_coughs = 0
            total_cough_episodes = 0
            avg_probability = 0.0
        
        # Почасовые данные (из детальных записей, если они еще есть)
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
        
        # Последние кашли (из детальных записей)
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [
            {"time": row[0], "probability": float(row[1])} 
            for row in cursor.fetchall()
        ]
        
        # Паттерны
        peak_hours = "Нет данных"
        cough_frequency = "0 раз/день"
        
        if total_coughs > 0 and hourly_stats:
            max_hour = max(hourly_stats, key=lambda x: x["count"])
            if max_hour["count"] > 0:
                peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
        
        if total_cough_episodes > 0:
            cough_frequency = f"{total_cough_episodes} раз/день"
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total_recordings,      # НЕ уменьшится после очистки!
                "total_coughs": total_coughs,              # НЕ уменьшится после очистки!
                "total_cough_episodes": total_cough_episodes,
                "avg_probability": round(avg_probability, 3)
            },
            "hourly_stats": hourly_stats,
            "recent_coughs": recent_coughs,
            "patterns": {
                "peak_hours": peak_hours,
                "cough_frequency": cough_frequency,
                "intensity": "Высокая" if avg_probability > 0.7 else "Средняя" if avg_probability > 0.3 else "Низкая"
            }
        }
        
        return result
        
    except Exception as e:
        logger.exception(f"Ошибка статистики: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/debug/stats/{device_id}")
async def debug_stats_redirect(device_id: str):
    """Редирект для обратной совместимости"""
    return await get_stats(device_id)

@app.get("/stats/{device_id}/range")
async def get_range_stats(device_id: str, start_date: str, end_date: str):
    """Статистика за период из агрегированных данных"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Агрегированные данные за период
        cursor.execute('''
            SELECT 
                date,
                SUM(total_recordings) as total_recordings,
                SUM(total_coughs) as total_coughs,
                SUM(total_cough_episodes) as total_cough_episodes,
                AVG(avg_probability) as avg_probability
            FROM daily_aggregated_stats 
            WHERE device_id=? AND date BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date
        ''', (device_id, start_date, end_date))
        
        daily_stats = []
        for row in cursor.fetchall():
            daily_stats.append({
                "date": row[0],
                "total_recordings": int(row[1] or 0),
                "total_coughs": int(row[2] or 0),
                "total_cough_episodes": int(row[3] or 0),
                "avg_probability": float(row[4] or 0.0)
            })
        
        # Общая статистика за период
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT date) as days_count,
                SUM(total_recordings) as total_recordings,
                SUM(total_coughs) as total_coughs,
                SUM(total_cough_episodes) as total_cough_episodes,
                AVG(avg_probability) as avg_probability
            FROM daily_aggregated_stats 
            WHERE device_id=? AND date BETWEEN ? AND ?
        ''', (device_id, start_date, end_date))
        
        period_row = cursor.fetchone()
        
        if period_row:
            period_stats = {
                "start_date": start_date,
                "end_date": end_date,
                "days_count": int(period_row[0] or 0),
                "total_recordings": int(period_row[1] or 0),
                "total_coughs": int(period_row[2] or 0),
                "total_cough_episodes": int(period_row[3] or 0),
                "avg_probability": float(period_row[4] or 0.0)
            }
        else:
            period_stats = {
                "start_date": start_date,
                "end_date": end_date,
                "days_count": 0,
                "total_recordings": 0,
                "total_coughs": 0,
                "total_cough_episodes": 0,
                "avg_probability": 0.0
            }
        
        # Недельная агрегация
        cursor.execute('''
            SELECT 
                strftime('%Y-%W', date) as week,
                SUM(total_recordings) as total_recordings,
                SUM(total_coughs) as total_coughs,
                SUM(total_cough_episodes) as total_cough_episodes,
                AVG(avg_probability) as avg_probability
            FROM daily_aggregated_stats 
            WHERE device_id=? AND date BETWEEN ? AND ?
            GROUP BY strftime('%Y-%W', date)
            ORDER BY week
        ''', (device_id, start_date, end_date))
        
        weekly_stats = []
        for row in cursor.fetchall():
            weekly_stats.append({
                "week": row[0],
                "total_recordings": int(row[1] or 0),
                "total_coughs": int(row[2] or 0),
                "total_cough_episodes": int(row[3] or 0),
                "avg_probability": float(row[4] or 0.0)
            })
        
        conn.close()
        
        return {
            "period_stats": period_stats,
            "daily_stats": daily_stats,
            "weekly_stats": weekly_stats
        }
        
    except Exception as e:
        logger.exception(f"Ошибка статистики периода: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/stats/{device_id}/daily/{date}")
async def get_daily_stats(device_id: str, date: str):
    """Статистика за конкретный день из агрегированных данных"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Агрегированные данные за день
        cursor.execute('''
            SELECT 
                total_recordings,
                total_coughs,
                total_cough_episodes,
                avg_probability
            FROM daily_aggregated_stats 
            WHERE device_id=? AND date=?
        ''', (device_id, date))
        
        row = cursor.fetchone()
        
        if row:
            stats = {
                "total_recordings": int(row[0] or 0),
                "total_coughs": int(row[1] or 0),
                "total_cough_episodes": int(row[2] or 0),
                "avg_probability": float(row[3] or 0.0)
            }
        else:
            stats = {
                "total_recordings": 0,
                "total_coughs": 0,
                "total_cough_episodes": 0,
                "avg_probability": 0.0
            }
        
        # Почасовые данные (из детальных записей, если они еще есть)
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
        
        return {
            "date": date,
            "stats": stats,
            "hourly_stats": hourly_stats
        }
        
    except Exception as e:
        logger.exception(f"Ошибка дневной статистики: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/stats/{device_id}/available_dates")
async def get_available_dates(device_id: str):
    """Получить список дат, для которых есть данные"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT date 
            FROM daily_aggregated_stats 
            WHERE device_id=? 
            ORDER BY date DESC
        ''', (device_id,))
        
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {
            "device_id": device_id,
            "available_dates": dates,
            "count": len(dates)
        }
        
    except Exception as e:
        logger.exception(f"Ошибка получения дат: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ---- Другие endpoints (оставляем без изменений) ----

@app.get("/records/all")
async def get_all_records(
    device_id: str = None, 
    limit: int = 100,
    offset: int = 0
):
    """Получить детальные записи (будут очищаться)"""
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
            records.append(record)
        
        conn.close()
        
        return {
            "status": "success",
            "total_records": total_count,
            "returned_records": len(records),
            "records": records
        }
        
    except Exception as e:
        logger.exception(f"Ошибка получения записей: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.delete("/records/cleanup")
async def manual_cleanup():
    """Ручная очистка детальных записей"""
    try:
        deleted_count = cleanup_old_records()
        return {
            "status": "success",
            "message": f"Удалено {deleted_count} детальных записей старше {CLEANUP_INTERVAL_HOURS} часов",
            "note": "Агрегированная статистика СОХРАНЕНА",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.exception(f"Ошибка ручной очистки: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    model_loaded = OUR_MODEL is not None and SCALER is not None
    db_exists = os.path.exists(DB_PATH)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Проверяем обе таблицы
        cursor.execute('SELECT COUNT(*) FROM cough_records')
        detailed_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM daily_aggregated_stats')
        aggregated_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT date) FROM daily_aggregated_stats')
        days_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "database": {
                "detailed_records": detailed_count,
                "aggregated_days": aggregated_count,
                "unique_days": days_count,
                "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "model_loaded": model_loaded
        }

@app.get("/")
async def root():
    return {
        "message": "🔥 Сервер Детекции Кашля v2.0 (с агрегированной статистикой)",
        "version": "2.0",
        "features": "Агрегированная статистика, накопление данных, очистка детальных записей",
        "endpoints": {
            "POST /upload": "Загрузить аудио + обновить статистику",
            "GET /stats/{device_id}": "Статистика за сегодня",
            "GET /stats/{device_id}/range": "Статистика за период",
            "GET /stats/{device_id}/daily/{date}": "Статистика за день",
            "GET /stats/{device_id}/available_dates": "Доступные даты",
            "GET /records/all": "Детальные записи (очищаются)",
            "DELETE /records/cleanup": "Ручная очистка детальных записей",
            "GET /health": "Проверка здоровья"
        }
    }

# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    """Запуск при старте сервера"""
    logger.info("🚀 Запуск сервера с агрегированной статистикой...")
    init_db()
    load_models()
    logger.info(f"✅ Сервер готов! Очистка детальных записей через {CLEANUP_INTERVAL_HOURS} часов")
    logger.info("📊 Агрегированная статистика сохраняется НАВСЕГДА")

# ---- Main ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting COUGH SERVER v2.0 on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
