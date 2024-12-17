import sqlite3


class DatabaseSchema:
    @staticmethod
    def initialize_tables(cursor: sqlite3.Cursor) -> None:
        cursor.execute("""CREATE TABLE IF NOT EXISTS Participants (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            participant_key INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            gender TEXT,
            age INTEGER,
            comment TEXT
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            trial_number INTEGER,
            stimulus_name TEXT,
            stimulus_id INTEGER,
            stimulus_seed INTEGER,
            -- stimulus_config TEXT, # TODO: maybe add this column
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Markers (
            marker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            time REAL,
            marker TEXT,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Data_Points (
            data_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            time REAL,
            temperature REAL,
            rating REAL,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Calibration_Results (
            calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            vas_0 REAL,
            vas_70 REAL,
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Placebo_Results (
            placebo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            -- TODO: Add columns for placebo results here 
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
        );""")  # TODO: Add columns for placebo results here
