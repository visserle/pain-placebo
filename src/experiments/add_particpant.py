import logging

from src.database.database_manager import DatabaseManager
from src.log_config import configure_logging

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def add_participant(cursor) -> tuple[int, int, dict]:
    """
    Handle the participant registration process.

    Args:
        cursor: SQLite database cursor

    Returns:
        Tuple containing:
        - participant_key: int
        - participant_id: int
        - participant_data: dict with age, gender, and comment
    """
    logging.info(f"Last participant id: {_get_last_participant_id(cursor)}")
    participant_id = input("Enter new participant id: ")
    participant_id = int(participant_id) if participant_id else 0
    dummy = " (dummy)" if participant_id == 0 else ""

    participant_data = {}

    # Ask for age and gender if the participant is not a dummy
    if not dummy:
        participant_data["age"] = _get_valid_age()
        participant_data["gender"] = _get_valid_gender()
    else:
        participant_data["age"] = 20
        participant_data["gender"] = "f"

    participant_data["comment"] = input("Enter comment (optional): ")

    cursor.execute(
        """
        INSERT INTO Participants (participant_id, comment, age, gender)
        VALUES (?, ?, ?, ?);
        """,
        (
            participant_id,
            participant_data["comment"],
            participant_data["age"],
            participant_data["gender"],
        ),
    )

    logger.info(f"Participant {participant_id}{dummy} added to the database.")
    logger.debug(f"Age: {participant_data['age']}")
    logger.debug(f"Gender: {participant_data['gender']}")
    if participant_data["comment"]:
        logger.info(f"Comment: {participant_data['comment']}")

    return cursor.lastrowid, participant_id


def _get_valid_age() -> int:
    """Get and validate participant age."""
    age = input("Enter age: ")
    while not age.isdigit():
        logger.error("Age must be an integer.")
        age = input("Enter age: ")
    return int(age)


def _get_valid_gender() -> str:
    """Get and validate participant gender."""
    gender = input("Enter gender (f/m): ")
    while gender not in ["f", "m"]:
        logger.error("Gender must be 'f' or 'm'.")
        gender = input("Enter gender (f/m): ")
    return gender


def _get_last_participant_id(cursor) -> int:
    """Get the last participant ID from the database."""
    cursor.execute(
        """
        SELECT participant_id FROM Participants
        ORDER BY participant_id DESC LIMIT 1;
        """
    )
    result = cursor.fetchone()
    return result[0] if result else None


if __name__ == "__main__":
    configure_logging(stream_level=logging.DEBUG)

    with DatabaseManager() as db:
        output = add_participant(db.cursor)

    print(output)
