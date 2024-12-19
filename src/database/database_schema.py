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
            comment TEXT,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000)
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Trials (
            trial_key INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            trial_number INTEGER,
            stimulus_name TEXT,
            stimulus_id INTEGER,
            stimulus_seed INTEGER,
            -- stimulus_config TEXT, # TODO: maybe add this column back
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Markers (
            marker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_key INTEGER,
            time REAL,
            marker TEXT,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (trial_key) REFERENCES Trials(trial_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Data_Points (
            data_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_key INTEGER,
            time REAL,
            temperature REAL,
            rating REAL,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (trial_key) REFERENCES Trials(trial_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Calibration_Results (
            calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            vas_0 REAL,
            vas_70 REAL,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Placebo_Results (
            placebo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            -- TODO: Add columns for placebo results here 
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")  # TODO: Add columns for placebo results here
        DatabaseSchema._init_questionnaire_tables(cursor)

    @staticmethod  # TODO add result from scoring schema
    def _init_questionnaire_tables(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_General (
            general_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
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
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_BDI_II (
            bdi_ii_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            q1_sadness INTEGER,
            q2_pessimism INTEGER,
            q3_failure INTEGER,
            q4_loss_of_pleasure INTEGER,
            q5_guilty_feelings INTEGER,
            q6_punishment INTEGER,
            q7_self_dislike INTEGER,
            q8_self_criticalness INTEGER,
            q9_suicidal INTEGER,
            q10_crying INTEGER,
            q11_agitation INTEGER,
            q12_loss_of_interest INTEGER,
            q13_indecisiveness INTEGER,
            q14_worthlessness INTEGER,
            q15_loss_of_energy INTEGER,
            q16_sleep_changes TEXT,
            q17_irritability INTEGER,
            q18_appetite_changes TEXT,
            q19_concentration INTEGER,
            q20_fatigue INTEGER,
            q21_sex_interest INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_MAAS (
            maas_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            q1_delayed_emotion INTEGER,
            q2_breaking_things INTEGER,
            q3_present_difficulty INTEGER,
            q4_rushing INTEGER,
            q5_physical_tension INTEGER,
            q6_forgetting_names INTEGER,
            q7_automatic_functioning INTEGER,
            q8_rushing_activities INTEGER,
            q9_goal_focus INTEGER,
            q10_automatic_tasks INTEGER,
            q11_half_listening INTEGER,
            q12_autopilot INTEGER,
            q13_mind_wandering INTEGER,
            q14_mindless_actions INTEGER,
            q15_mindless_eating INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_PANAS (
            panas_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            positive_affect INTEGER,
            negative_affect INTEGER,
            active INTEGER,
            distressed INTEGER,
            interested INTEGER,
            excited INTEGER,
            upset INTEGER,
            strong INTEGER,
            guilty INTEGER,
            scared INTEGER,
            hostile INTEGER,
            enthusiastic INTEGER,
            proud INTEGER,
            irritable INTEGER,
            alert INTEGER,
            ashamed INTEGER,
            inspired INTEGER,
            nervous INTEGER,
            determined INTEGER,
            attentive INTEGER,
            jittery INTEGER,
            afraid INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_PCS (
            pcs_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            rumination_score INTEGER,
            magnification_score INTEGER,
            helplessness_score INTEGER,
            q1_worry_endless INTEGER,
            q2_cant_go_on INTEGER,
            q3_terrible_never_better INTEGER,
            q4_awful_overwhelming INTEGER,
            q5_cant_stand_it INTEGER,
            q6_fear_worse_pain INTEGER,
            q7_other_painful_situations INTEGER,
            q8_desperately_want_pain_gone INTEGER,
            q9_cant_stop_thinking INTEGER,
            q10_keep_thinking_how_much_hurts INTEGER,
            q11_keep_thinking_want_stop INTEGER,
            q12_nothing_to_reduce INTEGER,
            q13_something_serious INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_PHQ_15 (
            phq15_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            q1_stomach INTEGER,
            q2_back INTEGER,
            q3_limb_joint INTEGER,
            q4_menstrual INTEGER,
            q5_sexual_intercourse INTEGER,
            q6_headache INTEGER,
            q7_chest INTEGER,
            q8_dizziness INTEGER,
            q9_fainting INTEGER,
            q10_heart_racing INTEGER,
            q11_shortness_breath INTEGER,
            q12_bowel_problems INTEGER,
            q13_nausea_digestion INTEGER,
            q14_sleep_problems INTEGER,
            q15_fatigue INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_PVAQ (
            pvaq_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            attention_to_pain_score INTEGER,    
            attention_to_changes_score INTEGER, 
            q1_pain_sensitivity INTEGER,
            q2_temp_changes INTEGER,
            q3_intensity_changes INTEGER,
            q4_medication_effects INTEGER,
            q5_location_changes INTEGER,
            q6_focus_on_pain INTEGER,
            q7_notice_during_activity INTEGER,
            q8_ignore_pain INTEGER,
            q9_onset_awareness INTEGER,
            q10_increase_checking INTEGER,
            q11_decrease_awareness INTEGER,
            q12_pain_consciousness INTEGER,
            q13_pain_attention INTEGER,
            q14_pain_monitoring INTEGER,
            q15_preoccupation INTEGER,
            q16_rumination INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS Questionnaire_STAI_T_10 (
            stai_t_10_id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_key INTEGER,
            total_score INTEGER,
            q1_fatigue INTEGER,
            q2_indecisiveness INTEGER,
            q3_calmness INTEGER,
            q4_happiness INTEGER,
            q5_heaviness INTEGER,
            q6_confidence_lack INTEGER,
            q7_security INTEGER,
            q8_depression INTEGER,
            q9_worrying_thoughts INTEGER,
            q10_nervousness INTEGER,
            unix_time REAL DEFAULT (UNIXEPOCH('subsecond')*1000),
            FOREIGN KEY (participant_key) REFERENCES Participants(participant_key)
                ON DELETE CASCADE
        );""")
