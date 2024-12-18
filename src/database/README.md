# Data Pipeline

The data pipeline consists of #TODO

## Files

- `database_schema.py` defines the schema for the database.
- `database_manager.py` is for insertig data into the database and extracting data from the database.
- `data_processing.py` coordinates the data processing).
- `data_config.py` contains very basic configuration parameters for the data pipeline and paths to the different data sources.

## Further Details

- The functions for the data transformations are saved in a separate module named feature to keep the data pipeline modular and easy to maintain.
- Invalid participants and trials can be removed at any stage of the pipeline using the `exclude_invalid_data` keyword from the `get_table` method.
