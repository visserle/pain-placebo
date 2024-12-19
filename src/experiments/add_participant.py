import logging

from src.database.database_manager import DatabaseManager
from src.log_config import configure_logging

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def add_participant() -> int:
    with DatabaseManager() as db:
        logging.info(f"Last participant id: {db.last_participant_number}")
        participant_number = input("Enter new participant id: ")
        participant_number = int(participant_number) if participant_number else 0
        dummy = " (dummy)" if participant_number == 0 else ""

        # Ask for age and gender if the participant is not a dummy
        if not dummy:
            age = _get_valid_age()
            gender = _get_valid_gender()
        else:
            age = 20
            gender = "f"
        comment = input("Enter comment (optional): ")

        db.insert_participant(
            participant_number=participant_number,
            age=age,
            gender=gender,
            comment=comment,
        )

    logger.info(f"Participant {participant_number}{dummy} added to the database.")
    logger.debug(f"Age: {age}")
    logger.debug(f"Gender: {gender}")
    if comment:
        logger.info(f"Comment: {comment}")


def _get_valid_age() -> int:
    """Get and validate participant age."""
    age = input("Enter age: ")

    while not age.isdigit():
        logger.error("Age must be an integer.")
        age = input("Enter age: ")
    return int(age)


def _get_valid_gender() -> str:
    """Get and validate participant gender."""
    gender = input("Enter gender (m/f/d): ")
    while gender not in ["m", "f", "d"]:
        logger.error("Gender must be 'm', 'f' or 'd'.")
        gender = input("Enter gender (m/f/d): ")
    return gender


if __name__ == "__main__":
    configure_logging(stream_level=logging.DEBUG)
    participant_id = add_participant()
    print(f"Participant key: {participant_id}")
