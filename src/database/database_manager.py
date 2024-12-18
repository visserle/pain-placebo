import logging
import sqlite3

import polars as pl

from src.database.data_config import DataConfig
from src.database.database_schema import DatabaseSchema

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
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        DatabaseSchema.initialize_tables(self.cursor)

    def disconnect(self) -> None:
        """Closes the connection to the SQLite database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def get_table(
        self,
        table_name: str,
        exclude_invalid_data: bool = True,  # TODO: implement
    ) -> pl.DataFrame:
        """Returns the table as a polars DataFrame."""
        # note that sqlite fetchall does not return column names
        # pl function is more convenient
        return pl.read_database(f"SELECT * FROM {table_name};", self.cursor)

    @property
    def last_participant_key(self) -> int:  # keys are autoincremented and unique
        self.cursor.execute(
            """
            SELECT participant_key FROM Participants
            ORDER BY ROWID DESC LIMIT 1;  -- ROWID is a built-in column
            """
        )
        result = self.cursor.fetchone()
        if result is None:
            self._insert_dummy_participant_into_empty_db()
            return 0
        return result[0]

    @property
    def last_participant_id(self) -> int:  # ids are given by the user
        """Only used in add_participant.py."""
        self.cursor.execute(
            """
            SELECT participant_id FROM Participants
            ORDER BY ROWID DESC LIMIT 1;
            """
        )
        result = self.cursor.fetchone()
        if result is None:
            self._insert_dummy_participant_into_empty_db()
            return 0
        return result[0]

    def _insert_dummy_participant_into_empty_db(self) -> None:
        self.cursor.execute(
            """
            INSERT INTO Participants (participant_id, comment
            ) VALUES (0, 'Empty database: Dummy participant added.');
            """
        )
        logger.debug("Empty database: Dummy participant added.")

    def insert_participant(
        self,
        participant_id: int,
        gender: str,
        age: int,
        comment: str | None = None,
    ) -> int:
        # Check if participant already exists
        if not participant_id == 0:  # Dummy participant
            result = self.conn.execute(f"""
                SELECT COUNT(participant_id) FROM Participants
                WHERE participant_id = {participant_id};
            """).fetchone()[0]
            if result:
                logger.warning(f"Participant with ID {participant_id} already exists.")

        # Insert participant
        self.cursor.execute(
            """
            INSERT INTO Participants (participant_id, gender, age, comment)
            VALUES (?, ?, ?, ?);
            """,
            (
                participant_id,
                gender,
                age,
                comment,
            ),
        )
        return self.cursor.lastrowid

    def insert_calibration_results(
        self,
        vas_0: float,
        vas_70: float,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Calibration_Results (participant_key, vas_0, vas_70, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP);
            """,
            (
                self.last_participant_key,
                vas_0,
                vas_70,
            ),
        )

    def insert_trial(
        self,
        trial_number: int,
        stimulus_name: str,
        stimulus_seed: int,
    ) -> int:
        """
        Add a trial to the database.

        Returns the trial key.
        """
        self.cursor.execute(
            """
            INSERT INTO Trials (trial_number, participant_key, stimulus_name, stimulus_seed)
            VALUES (?, ?, ?, ?);
            """,
            (
                trial_number,
                self.last_participant_key,
                stimulus_name,
                stimulus_seed,
            ),
        )
        logger.debug(f"Trial {trial_number} added to the database.")
        return self.cursor.lastrowid  # return the trial key

    def insert_data_point(
        self,
        trial_key: int,
        time: float,
        temperature: float,
        rating: float,
        debug: bool = False,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Data_Points (trial_key, time, temperature, rating)
            VALUES (?, ?, ?, ?);
            """,
            (
                trial_key,
                time,
                temperature,
                rating,
            ),
        )
        if debug:
            logger.debug(
                f"Data point added to the database: {time=}, {temperature=}, {rating=}"
            )

    def remove_participant(
        self,
        id: int,
    ) -> None:
        response = input(f"Remove participant {id}? (y/n) ")
        if response.lower() != "y":
            logger.info("Participant removal cancelled.")
            return

        try:
            # remove all data from dummy participants and cascade delete
            self.cursor.execute(
                """
                DELETE FROM Participants
                WHERE participant_id = ?;
                """,
                (id,),
            )
            if id == 0:
                logger.info("Dummy participant removed from the database.")
            else:
                logger.info(f"Participant {id} removed from the database.")
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to remove participant {id}: {e}")

    def remove_dummy_participant(self) -> None:
        self.remove_participant(0)

    def anonymize_database(self) -> None:
        input_ = input(f"Anonymize database {DB_FILE}? (y/n) ")
        if input_.lower() != "y":
            logger.info("Database anonymization cancelled.")
            return

        # TODO: remove timestamps / unix_time
        # TODO: scramble participant ids

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
    # db_manager.delete_database()
    db_manager.remove_dummy_participant()
