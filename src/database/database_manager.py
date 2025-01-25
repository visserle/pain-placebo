import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import polars as pl

from src.database.database_schema import DatabaseSchema

DB_FILE = Path("data/pain-placebo.db")
BACKUP_DIR = DB_FILE.parent / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DatabaseManager:
    def __init__(self) -> None:
        self.conn = None
        self.cursor = None

        # Initialize database and tables
        try:
            self.connect()
            DatabaseSchema.initialize_tables(self.cursor)
        finally:
            self.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def connect(self) -> None:
        """Establishes a connection to the SQLite database."""
        if not self.conn:
            self.conn = sqlite3.connect(
                DB_FILE,
                isolation_level=None,  # autocommit
            )
            self.cursor = self.conn.cursor()  # bit of an anti-pattern
            self.cursor.execute("PRAGMA foreign_keys = ON;")

    def disconnect(self) -> None:
        """Closes the connection to the SQLite database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def execute(
        self,
        sql: str,
    ) -> "DatabaseManager":
        """Shortcut for executing SQL queries. Does not fetch results."""
        if self.conn is None:
            raise ConnectionError(
                "Database not connected. Use 'with' statement or call connect() first."
            )
        return self.cursor.execute(sql)

    def get_table(
        self,
        table_name: str,
        exclude_invalid_data: bool = True,  # TODO: implement
    ) -> pl.DataFrame:
        """Returns the table as a polars DataFrame."""
        # alternative: pl.read_database(f"SELECT * FROM {table_name};", self.cursor)
        result = self.cursor.execute(f"SELECT * FROM {table_name};").fetchall()
        return pl.DataFrame(
            result,
            schema=next(zip(*self.cursor.description)),
            orient="row",
        )

    @property
    def last_participant_id(self) -> int:  # ids are autoincremented and unique
        self.cursor.execute(
            """
            SELECT participant_id FROM Participants
            ORDER BY ROWID DESC LIMIT 1; -- ROWID for the last inserted row
            """
        )
        result = self.cursor.fetchone()
        if result is None:
            self._insert_dummy_participant_into_empty_db()
            return 1  # autoincrement id starts at 1 (while dummy number is 0)
        return result[0]

    @property
    def last_participant_number(self) -> int:  # numbers are given by the user
        """Only used in add_participant.py."""
        self.cursor.execute(
            """
            SELECT participant_number FROM Participants
            ORDER BY ROWID DESC LIMIT 1;  
            """
        )
        result = self.cursor.fetchone()
        if result is None:
            self._insert_dummy_participant_into_empty_db()
            return 0  # dummy participant number
        return result[0]

    @property
    def last_trial_id(self) -> int:
        self.cursor.execute(
            """
            SELECT trial_id FROM Trials
            ORDER BY ROWID DESC LIMIT 1;
            """
        )
        result = self.cursor.fetchone()
        if result is None:
            return 1
        return result[0]

    def _insert_dummy_participant_into_empty_db(self) -> None:
        self.cursor.execute(
            """
            INSERT INTO Participants (participant_number, comment
            ) VALUES (0, 'Empty database: Dummy participant added.');
            """
        )
        logger.debug("Empty database: Dummy participant added.")

    def insert_participant(
        self,
        participant_number: int,
        age: int,
        gender: str,
        comment: str | None = None,
    ) -> int:
        # Check if participant already exists
        if not participant_number == 0:  # Dummy participant
            result = self.conn.execute(f"""
                SELECT COUNT(participant_number) FROM Participants
                WHERE participant_number = {participant_number};
            """).fetchone()[0]
            if result:
                logger.warning(
                    f"Participant with ID {participant_number} already exists."
                )

        # Insert participant
        self.cursor.execute(
            """
            INSERT INTO Participants (participant_number, age, gender, comment)
            VALUES (?, ?, ?, ?);
            """,
            (
                participant_number,
                age,
                gender,
                comment,
            ),
        )
        # Create backup after successful insertion
        if not participant_number == 0:  # Dummy participant
            self.backup_database()

    def insert_calibration_results(
        self,
        vas_0: float,
        vas_70: float,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Calibration_Results (participant_id, vas_0, vas_70)
            VALUES (?, ?, ?);
            """,
            (
                self.last_participant_id,
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
            INSERT INTO Trials (trial_number, participant_id, stimulus_name, stimulus_seed)
            VALUES (?, ?, ?, ?);
            """,
            (
                trial_number,
                self.last_participant_id,
                stimulus_name,
                stimulus_seed,
            ),
        )
        logger.debug(f"Trial {trial_number} added to the database.")

    def insert_marker(
        self,
        trial_id: int,
        marker: str,
        time: float,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Markers (trial_id, marker, time)
            VALUES (?, ?, ?);
            """,
            (
                trial_id,
                marker,
                time,
            ),
        )
        logger.debug(f"Marker added to the database: {marker}")

    def insert_measurement(
        self,
        trial_id: int,
        temperature: float,
        rating: float,
        time: float,
        debug: bool = False,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Measurements (trial_id, temperature, rating, time)
            VALUES (?, ?, ?, ?);
            """,
            (
                trial_id,  # no self.current_trial_id to avoid unnecessary queries
                temperature,
                rating,
                time,
            ),
        )
        if debug:
            logger.debug(
                f"Measurement added to the database: {temperature = }, {rating = }, {time = }"
            )

    def insert_button(
        self,
        trial_id: int,
        button: str,
        time: float,
        debug: bool = False,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO Buttons (trial_id, button, time)
            VALUES (?, ?, ?);
            """,
            (
                trial_id,
                button,
                time,
            ),
        )
        if debug:
            logger.debug(f"Button '{button}' added to the database.")

    def insert_questionnaire(
        self,
        scale: str,
        results: dict,
    ) -> None:
        scale = scale.replace("-", "_")
        self.cursor.execute(
            f"""
            INSERT INTO Questionnaire_{scale.upper()} (participant_id, {", ".join(results.keys())})
            VALUES (?, {", ".join("?" for _ in results)});
            """,
            (
                self.last_participant_id,
                *results.values(),
            ),
        )
        logger.debug(f"Questionnaire {scale.upper()} added to the database.")

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
                WHERE participant_number = ?;
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

    def backup_database(self) -> None:
        """Create a backup of the database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"pain-placebo_{timestamp}.db"

        try:
            # Ensure the database is not being written to during backup
            self.cursor.execute("BEGIN IMMEDIATE")
            shutil.copy2(DB_FILE, backup_path)
            logger.debug(f"Database backed up to {backup_path}")

            # Keep only last 100 backups
            backups = sorted(BACKUP_DIR.glob("pain-placebo_*.db"))
            if len(backups) > 100:
                for backup in backups[:-100]:
                    backup.unlink()
                    logger.debug(f"Removed old backup: {backup}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def anonymize_database(self) -> None:
        input_ = input(f"Anonymize database {DB_FILE}? (y/n) ")
        if input_.lower() != "y":
            logger.info("Database anonymization cancelled.")
            return
        # TODO: remove comments, timestamps / unix_time and scramble participant ids
        logging.info("Anonymizing database not implemented yet.")


if __name__ == "__main__":
    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    # Example usage of DatabaseManager
    db_manager = DatabaseManager()
    # db_manager.delete_database()
    db_manager.remove_dummy_participant()
