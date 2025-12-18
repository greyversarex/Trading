import aiosqlite
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict
import os

from .structure_extractor import StructureFeatures, StructureType
from .similarity_matcher import MatchResult


class Database:
    """SQLite database for storing structures, matches, and feedback."""
    
    def __init__(self, db_path: str = "data/structures.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async def initialize(self):
        """Create database tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_structures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    structure_type TEXT,
                    feature_vector TEXT,
                    normalized_line TEXT,
                    pivot_sequence TEXT,
                    relative_distances TEXT,
                    trend_direction REAL,
                    volatility REAL,
                    compression_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    structure_id INTEGER,
                    symbol TEXT,
                    timeframe TEXT,
                    similarity_score REAL,
                    structure_type TEXT,
                    is_mirrored INTEGER,
                    normalized_line TEXT,
                    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (structure_id) REFERENCES uploaded_structures(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    is_relevant INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS scan_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    structure_id INTEGER,
                    started_at TIMESTAMP,
                    stopped_at TIMESTAMP,
                    threshold REAL,
                    total_matches INTEGER DEFAULT 0,
                    FOREIGN KEY (structure_id) REFERENCES uploaded_structures(id)
                )
            """)
            
            await db.commit()
    
    async def save_structure(self, name: str, features: StructureFeatures) -> int:
        """Save an uploaded structure and return its ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO uploaded_structures 
                (name, structure_type, feature_vector, normalized_line, 
                 pivot_sequence, relative_distances, trend_direction, 
                 volatility, compression_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                features.structure_type.value,
                json.dumps(features.feature_vector.tolist()),
                json.dumps(features.normalized_line.tolist()),
                json.dumps(features.pivot_sequence),
                json.dumps(features.relative_distances),
                features.trend_direction,
                features.volatility,
                features.compression_ratio
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_structure(self, structure_id: int) -> Optional[Dict[str, Any]]:
        """Get a saved structure by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM uploaded_structures WHERE id = ?",
                (structure_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
        return None
    
    async def get_all_structures(self) -> List[Dict[str, Any]]:
        """Get all saved structures."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM uploaded_structures ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def save_match(self, structure_id: int, match: MatchResult) -> int:
        """Save a match result."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO matches 
                (structure_id, symbol, timeframe, similarity_score, 
                 structure_type, is_mirrored, normalized_line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                structure_id,
                match.symbol,
                match.timeframe,
                match.similarity_score,
                match.structure_type.value,
                1 if match.is_mirrored else 0,
                json.dumps(match.normalized_line)
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_matches(self, structure_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get matches for a structure."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM matches 
                WHERE structure_id = ?
                ORDER BY matched_at DESC
                LIMIT ?
            """, (structure_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def save_feedback(self, match_id: int, is_relevant: bool):
        """Save user feedback for a match."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO feedback (match_id, is_relevant)
                VALUES (?, ?)
            """, (match_id, 1 if is_relevant else 0))
            await db.commit()
    
    async def get_feedback_stats(self, structure_id: int) -> Dict[str, int]:
        """Get feedback statistics for a structure's matches."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT 
                    SUM(CASE WHEN f.is_relevant = 1 THEN 1 ELSE 0 END) as relevant,
                    SUM(CASE WHEN f.is_relevant = 0 THEN 1 ELSE 0 END) as irrelevant
                FROM feedback f
                JOIN matches m ON f.match_id = m.id
                WHERE m.structure_id = ?
            """, (structure_id,)) as cursor:
                row = await cursor.fetchone()
                return {
                    "relevant": row[0] or 0,
                    "irrelevant": row[1] or 0
                }
    
    async def delete_structure(self, structure_id: int):
        """Delete a structure and its associated data."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM feedback WHERE match_id IN 
                (SELECT id FROM matches WHERE structure_id = ?)
            """, (structure_id,))
            await db.execute("DELETE FROM matches WHERE structure_id = ?", (structure_id,))
            await db.execute("DELETE FROM scan_sessions WHERE structure_id = ?", (structure_id,))
            await db.execute("DELETE FROM uploaded_structures WHERE id = ?", (structure_id,))
            await db.commit()
