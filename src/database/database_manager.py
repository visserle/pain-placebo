import logging
import sqlite3
from pathlib import Path

from src.database.data_config import DataConfig
from src.database.database_schema import DatabaseSchema
from src.experiments.placebo.stimulus_generator import StimulusGenerator

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

DB_FILE = DataConfig.DB_FILE


class DatabaseManager:
    def __init__(self) -> None:
        self.conn = None
        self.cursor = None
        self.connect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        if exc_type:
            logger.error(f"Exception occurred: {exc_type}: {exc_value}")
            return False  # re-raise the exception
        return True

    def connect(self) -> None:
        """Establishes a connection to the SQLite database."""
        self.conn = sqlite3.connect(
            DB_FILE,
            isolation_level=None,  # autocommit
        )
        self.cursor = self.conn.cursor()
        DatabaseSchema.initialize_tables(self.cursor)

    def disconnect(self) -> None:
        """Closes the connection to the SQLite database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    @property
    def last_participant_key(self) -> int:
        self.cursor.execute(
            """
            SELECT participant_key FROM Participants
            ORDER BY ROWID DESC LIMIT 1;  -- ROWID is a built-in column
            """
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    @property
    def last_participant_id(self) -> int:
        self.cursor.execute(
            """
            SELECT participant_id FROM Participants
            ORDER BY ROWID DESC LIMIT 1;
            """
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

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
        input_ = input(f"Delete database {DB_FILE}? (y/n) ")
        if input_.lower() != "y":
            logger.info("Database deletion cancelled.")
            return

        if DB_FILE.exists():
            self.disconnect()
            DB_FILE.unlink()
            logger.info(f"Database {DB_FILE} deleted.")
        else:
            logger.error(f"Database {DB_FILE} does not exist.")


if __name__ == "__main__":
    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    # Example usage of DatabaseManager
    db_manager = DatabaseManager()
    db_manager.add_participant()
    db_manager.delete_database()
