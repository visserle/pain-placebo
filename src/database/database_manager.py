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
        return result[0] if result else 0

    @property
    def last_participant_id(self) -> int:  # ids are given by the user
        self.cursor.execute(
            """
            SELECT participant_id FROM Participants
            ORDER BY ROWID DESC LIMIT 1;
            """
        )
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def insert_participant(
        self,
        participant_data: dict,
    ):
        # Check if participant already exists
        if not participant_data["id"] == 0:  # Dummy participant
            result = self.conn.execute(f"""
                SELECT COUNT(participant_id) FROM Participants
                WHERE participant_id = {participant_data["id"]};
            """).fetchone()[0]
            if result:
                logger.warning(
                    f"Participant with ID {participant_data['id']} already exists."
                )

        # Insert participant
        self.cursor.execute(
            """
            INSERT INTO Participants (participant_id, comment, age, gender)
            VALUES (?, ?, ?, ?);
            """,
            (
                participant_data["id"],
                participant_data["comment"],
                participant_data["age"],
                participant_data["gender"],
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
                self.last_participant_key,
                trial_number,
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

    def remove_dummy_data(self) -> None:
        # remove all data from dummy participants and cascade delete
        # TODO check if this is correct
        self.cursor.execute(
            """
            DELETE FROM Participants
            WHERE participant_id = 0;
            """
        )
        logger.info("Dummy data removed from the database.")

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
