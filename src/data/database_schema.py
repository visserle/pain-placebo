import logging
import sqlite3

import polars as pl

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DatabaseSchema:
    @staticmethod
    def create_tables(cursor):
        cursor.execute("""CREATE TABLE IF NOT EXISTS Participants (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            participant_key INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            gender TEXT,
            age INTEGER,
            comment TEXT
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Stimuli (
            stimulus_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            name TEXT,
            config TEXT,
            time_points TEXT,
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            trial_number INTEGER,
            stimulus_name TEXT,
            stimulus_id INTEGER,
            timestamp TEXT,
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key),
            FOREIGN KEY (stimulus_id) REFERENCES Stimuli(stimulus_id)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS DataPoints (
            data_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            time REAL,
            temperature REAL,
            rating REAL,
            FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS CalibrationResults (
            calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            vas_0 REAL,
            vas_70 REAL,    
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
        );""")
