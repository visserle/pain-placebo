import sqlite3


class DatabaseSchema:
    @staticmethod
    def initialize_tables(cursor: sqlite3.Cursor) -> None:
        for table in TABLES + QUESTIONNAIRE_TABLES:
            cursor.execute(table)


TABLES = [
    """CREATE TABLE IF NOT EXISTS Participants (
        participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_number INTEGER NOT NULL,
        age INTEGER,
        gender TEXT,
        comment TEXT,
        is_excluded BOOLEAN DEFAULT FALSE,
        excluded_reason TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000)
        );""",
    """CREATE TABLE IF NOT EXISTS Trials (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        trial_number INTEGER,
        stimulus_name TEXT,
        stimulus_seed INTEGER,
        is_excluded BOOLEAN DEFAULT FALSE,
        excluded_reason TEXT,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Markers (
        trial_id INTEGER,
        marker TEXT,
        time REAL,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Measurements (
        trial_id INTEGER,
        temperature REAL,
        rating REAL,
        time REAL,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Buttons (
        trial_id INTEGER,
        button TEXT,
        time INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (trial_id) REFERENCES Trials(trial_id)
        );""",
    """CREATE TABLE IF NOT EXISTS Calibration_Results (
        calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        vas_0 REAL,
        vas_70 REAL,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Placebo_Results (
        placebo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        -- TODO: Add columns for placebo results here
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",  # TODO: Add columns for placebo results here
]

QUESTIONNAIRE_TABLES = [
    """CREATE TABLE IF NOT EXISTS Questionnaire_General (
        general_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        gender TEXT,
        height REAL,
        weight REAL,
        handedness TEXT,
        education TEXT,
        employment_status TEXT,
        physical_activity TEXT,
        meditation TEXT,
        contact_lenses TEXT,
        ear_wiggling TEXT,
        regular_medication TEXT,
        pain_medication_last_24h TEXT,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_BDI_II (
        bdi_ii_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        q14 INTEGER,
        q15 INTEGER,
        q16 TEXT,  
        q17 INTEGER,
        q18 TEXT,
        q19 INTEGER,
        q20 INTEGER,
        q21 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_MAAS (
        maas_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        q14 INTEGER,
        q15 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_PANAS (
        panas_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        positive_affect INTEGER,
        negative_affect INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        q14 INTEGER,
        q15 INTEGER,
        q16 INTEGER,
        q17 INTEGER,
        q18 INTEGER,
        q19 INTEGER,
        q20 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_PCS (
        pcs_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        rumination_score INTEGER,
        magnification_score INTEGER,
        helplessness_score INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_PHQ_15 (
        phq_15_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        q14 INTEGER,
        q15 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_PVAQ (
        pvaq_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        attention_to_pain_score INTEGER,
        attention_to_changes_score INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        q11 INTEGER,
        q12 INTEGER,
        q13 INTEGER,
        q14 INTEGER,
        q15 INTEGER,
        q16 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
    """CREATE TABLE IF NOT EXISTS Questionnaire_STAI_T_10 (
        stai_t_10_id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        total INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 INTEGER,
        unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
        FOREIGN KEY (participant_id) REFERENCES Participants(participant_id)
            ON DELETE CASCADE
        );""",
]
