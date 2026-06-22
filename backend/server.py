from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import datetime
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER,
            label TEXT,
            confidence REAL,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            source TEXT,  -- 'online' or 'gt'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        cursor.execute("ALTER TABLE detections ADD COLUMN latitude REAL")
        cursor.execute("ALTER TABLE detections ADD COLUMN longitude REAL")
        cursor.execute("ALTER TABLE detections ADD COLUMN altitude REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

class Detection(BaseModel):
    timestamp_ms: int
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    source: str = "online"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None

@app.post("/detections")
async def create_detections(detections: List[Detection]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        for d in detections:
            cursor.execute('''
                INSERT INTO detections (timestamp_ms, label, confidence, x_min, y_min, x_max, y_max, source, latitude, longitude, altitude)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (d.timestamp_ms, d.label, d.confidence, d.x_min, d.y_min, d.x_max, d.y_max, d.source, d.latitude, d.longitude, d.altitude))
        conn.commit()
        return {"status": "success", "count": len(detections)}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/detections")
async def get_detections(source: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if source:
        cursor.execute('SELECT * FROM detections WHERE source = ? ORDER BY timestamp_ms ASC', (source,))
    else:
        cursor.execute('SELECT * FROM detections ORDER BY timestamp_ms ASC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
