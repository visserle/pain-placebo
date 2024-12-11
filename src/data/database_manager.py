import logging
import sqlite3
from pathlib import Path

from src.experiments.measurement.stimulus_generator import StimulusGenerator

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

DB_PATH = Path("experiment.db")


class DatabaseSchema:
    @staticmethod
    def create_tables(cursor):
        cursor.execute("""CREATE TABLE IF NOT EXISTS Participants (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            participant_key INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            gender TEXT,
            age INTEGER,
            comments TEXT
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


class DatabaseManager:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
    ) -> None:
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self) -> None:
        """Establishes a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        self.cursor = self.conn.cursor()
        DatabaseSchema.create_tables(self.cursor)

    def disconnect(self) -> None:
        """Closes the connection to the SQLite database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    @property
    def last_participant_key(self) -> int:
        self.cursor.execute(
            "SELECT participant_key FROM Participants ORDER BY participant_key DESC LIMIT 1;"
        )
        return self.cursor.fetchone()[0]

    def add_participant(self) -> tuple[int, int]:
        """
        Add a participant to the database.

        Returns both the participant key and the participant ID.

        TODO: Add missing fields from the participant table.
        """
        participant_id = input("Enter participant id: ")
        participant_id = int(participant_id) if participant_id else 0
        comments = input("Enter comment (optional): ")

        self.cursor.execute(
            """
            INSERT INTO Participants (participant_id, comments)
            VALUES (?, ?);
        """,
            (participant_id, comments),
        )
        dummy = " (dummy)" if participant_id == 0 else ""
        logger.info(f"Participant {participant_id}{dummy} added to the database.")
        return self.cursor.lastrowid, participant_id

    def add_stimuli(
        self,
        participant_key: int,
        stimulus_config: dict,
    ) -> None:
        for name, config in stimulus_config.items():
            stimulus = StimulusGenerator(config)
            self.cursor.execute(
                """
                INSERT INTO Stimuli (participant_key, name, config, time_points)
                VALUES (?, ?, ?, ?);
                """,
                (
                    participant_key,
                    name,
                    stimulus.serialize_config(),
                    stimulus.serialize_time_points(),
                ),
            )
        logger.debug("Stimuli from the config added to the database.")

    def add_trial(
        self,
        participant_key: int,
        trial_number: int,
        stimulus_name: str,
    ) -> int:
        """
        Add a trial to the database.

        Returns the trial ID.
        """
        self.cursor.execute(
            """
            INSERT INTO Trials (participant_key, trial_number, stimulus_name, stimulus_id, timestamp)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
            """,
            (
                participant_key,
                trial_number,
                stimulus_name,
                self._get_stimuli_id_for_trial(participant_key, stimulus_name),
            ),
        )
        logger.debug(f"Trial {trial_number} added to the database.")
        return self.cursor.lastrowid

    def _get_stimuli_id_for_trial(
        self,
        participant_key: int,
        stimulus_name: str,
    ) -> int:
        """Auxiliary function to get the stimulus ID for a given participant and
        stimulus name. Neccessary, because stimuli are added in batches and I am
        a noob at SQL."""
        self.cursor.execute(
            """
            SELECT stimulus_id FROM Stimuli
            WHERE participant_key = ? AND name = ?
            ORDER BY stimulus_id DESC
            LIMIT 1;
            """,
            (participant_key, stimulus_name),
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    def add_data_point(
        self,
        trial_id: int,
        time: float,
        temperature: float,
        rating: float,
        debug: bool = False,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO DataPoints (trial_id, time, temperature, rating)
            VALUES (?, ?, ?, ?);
            """,
            (trial_id, time, temperature, rating),
        )
        if debug:
            logger.debug(
                f"Data point added to the database: {time=}, {temperature=}, {rating=}"
            )

    def delete_database(self) -> None:
        input_ = input(f"Delete database {self.db_path}? (y/n) ")
        if input_.lower() != "y":
            logger.info("Database deletion cancelled.")
            return

        if self.db_path.exists():
            self.disconnect()
            self.db_path.unlink()
            logger.info(f"Database {self.db_path} deleted.")
        else:
            logger.error(f"Database {self.db_path} does not exist.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Example usage of DatabaseManager
    db_manager = DatabaseManager()
    db_manager.add_participant()
    db_manager.delete_database()
