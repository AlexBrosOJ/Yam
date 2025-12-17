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
CLEANUP_INTERVAL_HOURS = 5  # Удалять записи старше 1 часа
THRESHOLD = 0.62  # Порог для кашля (можешь поменять)

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
    
    # Основная таблица
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
    
    # Индексы для быстрого поиска
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_time ON cough_records(device_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cough_detected ON cough_records(cough_detected)')
    
    conn.commit()
    conn.close()
    
    # Автоматически чистим старые записи при старте
    cleanup_old_records()
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

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None
SCALER = None

def load_models():
    """Загрузка новой модели и scaler'а"""
    global OUR_MODEL, YAMNET_MODEL, SCALER
    
    try:
        # 1. Новая улучшенная модель (2079 входов)
        OUR_MODEL = tf.keras.models.load_model(
            'cough_detection_improved_model.keras', 
            compile=False
        )
        logger.info("✅ Новая модель загружена (2079 фич)")
        
        # 2. YAMNet
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ YAMNet загружен")
        
        # 3. Scaler из обучения (ОБЯЗАТЕЛЬНО!)
        SCALER = joblib.load('cough_scaler.pkl')
        logger.info("✅ Scaler загружен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки моделей: {e}")
        raise

# ---- Feature Extraction (НОВЫЙ ФОРМАТ) ----
def extract_features_new(waveform, sr, yamnet_model):
    """Извлекает 2079 фич как в новой модели обучения"""
    try:
        # 1. YAMNet embeddings
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform_tf)
        
        # Два типа пуллинга
        avg_pool = tf.reduce_mean(embeddings, axis=0).numpy()      # 1024
        max_pool = tf.reduce_max(embeddings, axis=0).numpy()       # 1024
        
        # 2. MFCC с mean и std
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=13, hop_length=512
        )
        mfcc_mean = np.mean(mfcc, axis=1)    # 13
        mfcc_std = np.std(mfcc, axis=1)      # 13
        
        # 3. Спектральные фичи
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=sr, hop_length=512
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=sr, hop_length=512
        )[0]
        
        spectral_features = np.array([
            np.mean(spectral_centroid),      # 1
            np.std(spectral_centroid),       # 1  
            np.mean(spectral_bandwidth)      # 1
        ])  # 3 фичи
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            waveform, hop_length=512
        )[0]
        zcr_mean = np.mean(zcr)  # 1
        
        # 5. RMS энергии
        rms = librosa.feature.rms(y=waveform, hop_length=512)[0]
        rms_mean = np.mean(rms)  # 1
        
        # Собираем ВСЕ фичи (2079)
        combined = np.concatenate([
            avg_pool,           # 1024
            max_pool,           # 1024  
            mfcc_mean,          # 13
            mfcc_std,           # 13
            spectral_features,  # 3
            [zcr_mean, rms_mean]  # 2
        ])
        
        return combined
        
    except Exception as e:
        logger.error(f"Ошибка извлечения фич: {e}")
        return None

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Анализ аудио с НОВОЙ моделью"""
    if not OUR_MODEL or not SCALER or not YAMNET_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Модели не загружены"}
    
    try:
        # Временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Загрузка аудио
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        # Проверка на тишину
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.03:
            return {"probability": 0.0, "cough_detected": False, "message": "Тишина"}
        
        # Нормализация (как в новой модели обучения)
        max_val = np.max(np.abs(waveform))
        if max_val < 0.01:
            return {"probability": 0.0, "cough_detected": False, "message": "Слишком тихо"}
        
        waveform = waveform / (max_val + 1e-8)
        
        # Дополнение до 1 секунды
        target_length = 16000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        # Извлечение НОВЫХ фич (2079)
        features = extract_features_new(waveform, sr, YAMNET_MODEL)
        if features is None:
            return {"probability": 0.0, "cough_detected": False, "message": "Ошибка извлечения фич"}
        
        # Нормализация через scaler (ВАЖНО!)
        features_scaled = SCALER.transform(features.reshape(1, -1))
        
        # Предсказание
        prediction = OUR_MODEL.predict(features_scaled, verbose=0)
        prob = float(prediction[0][0])
        
        # Классификация
        is_cough = prob > THRESHOLD
        
        logger.info(f"🎯 НОВАЯ МОДЕЛЬ: {filename} | prob={prob:.3f} | cough={is_cough}")
        
        return {
            "probability": prob,
            "cough_detected": bool(is_cough),
            "confidence": prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "cough_count": 1 if is_cough else 0
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
        # Автоочистка старых записей при каждой загрузке
        cleanup_old_records()
        
        # Чтение файла
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Пустой файл")
        
        current_datetime = get_current_datetime()
        
        # Анализ
        result = analyze_audio(raw, audio.filename)
        
        # Сохранение в БД
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, 
            audio.filename,
            "",  # file_path оставляем пустым
            float(result["probability"]),
            int(result["cough_detected"]),
            result["message"],
            "[]",  # top_classes
            "{}",  # cough_stats
            current_datetime
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Результат: {result}")
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """Статистика за сегодня (основной endpoint)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = get_current_date()
        
        # Основная статистика
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
        
        # Почасовые данные
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
        
        # Последние кашли
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
    """Редирект со старого endpoint'а на новый"""
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
        conn.row_factory = sqlite3.Row  # Для доступа по имени колонок
        cursor = conn.cursor()
        
        # Определяем запрос в зависимости от параметров
        if device_id:
            query = '''
                SELECT * FROM cough_records 
                WHERE device_id=? 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            '''
            params = (device_id, limit, offset)
            # Также получаем общее количество для этого device_id
            cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=?', (device_id,))
        else:
            query = '''
                SELECT * FROM cough_records 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            '''
            params = (limit, offset)
            # Общее количество всех записей
            cursor.execute('SELECT COUNT(*) FROM cough_records')
        
        total_count = cursor.fetchone()[0]
        
        # Выполняем основной запрос
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Конвертируем в словари
        records = []
        for row in rows:
            record = dict(row)
            # Конвертируем типы
            record['cough_detected'] = bool(record['cough_detected'])
            record['probability'] = float(record['probability'])
            
            # Если не нужны аудио данные, убираем file_path
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
    """Получить количество записей для устройства"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Общее количество
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=?', (device_id,))
        total = cursor.fetchone()[0]
        
        # Кашли сегодня
        today = get_current_date()
        cursor.execute('''
            SELECT COUNT(*) FROM cough_records 
            WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=?
        ''', (device_id, today))
        today_coughs = cursor.fetchone()[0]
        
        # Все кашли
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
    """Ручная очистка старых записей"""
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
    """Экспорт записей в разных форматах"""
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
        
        # Конвертация в словари
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
        
        else:  # json по умолчанию
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
    """Проверка здоровья сервера"""
    model_loaded = OUR_MODEL is not None and SCALER is not None
    db_exists = os.path.exists(DB_PATH)
    
    # Проверка БД
    db_status = "healthy"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM cough_records')
        db_status = "healthy"
        conn.close()
    except:
        db_status = "unhealthy"
    
    return JSONResponse({
        "status": "healthy" if model_loaded and db_status == "healthy" else "degraded",
        "model_loaded": model_loaded,
        "scaler_loaded": SCALER is not None,
        "database": db_status,
        "database_path": DB_PATH,
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat(),
        "features_dimension": 2079 if model_loaded else "unknown"
    })

@app.get("/")
async def root():
    return {
        "message": "🔥 УЛУЧШЕННЫЙ Сервер Детекции Кашля",
        "version": "2.0",
        "features": "Новая модель (2079 фич), автоочистка, экспорт записей",
        "endpoints": {
            "POST /upload": "Загрузить аудио для анализа",
            "GET /stats/{device_id}": "Статистика за сегодня",
            "GET /records/all": "Все записи (с пагинацией)",
            "GET /records/{device_id}/count": "Количество записей",
            "GET /records/export/{device_id}": "Экспорт записей",
            "DELETE /records/cleanup": "Ручная очистка",
            "GET /health": "Проверка здоровья"
        }
    }

# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    """Запуск при старте сервера"""
    logger.info("🚀 Запуск улучшенного сервера детекции кашля...")
    init_db()
    load_models()
    logger.info(f"✅ Сервер готов! Порог кашля: {THRESHOLD}")
    logger.info(f"🧹 Автоочистка каждые {CLEANUP_INTERVAL_HOURS} часов")

# ---- Main ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting IMPROVED COUGH SERVER on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")




