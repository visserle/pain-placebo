import argparse
import copy
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.database.database_manager import DatabaseManager
from src.experiments.placebo.rate_limiter import RateLimiter
from src.experiments.placebo.stimulus_generator import StimulusGenerator
from src.experiments.placebo.visual_analogue_scale import VisualAnalogueScale
from src.experiments.thermoino import ThermoinoComplexTimeCourses
from src.experiments.utils import (
    load_configuration,
    load_script,
    prepare_audio,
    prepare_script,
    scale_1d_value,
    scale_2d_tuple,
)
from src.log_config import configure_logging

EXP_NAME = "pain-placebo"
EXP_DIR = Path("src/experiments/placebo")
LOG_FILE = Path("logs/experiments/placebo") / datetime.now().strftime(
    "%Y_%m_%d__%H_%M_%S.log"
)

# Parse arguments
parser = argparse.ArgumentParser(description="Run the pain-placebo experiment.")
parser.add_argument(
    "-a",
    "--all",
    action="store_false",  # TODO: change back to store_true after testing
    help="Use all flags for a dry run.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode.",
)
parser.add_argument(
    "-w",
    "--windowed",
    action="store_true",
    help="Run in windowed mode.",
)
parser.add_argument(
    "-m",
    "--muted",
    action="store_true",
    help="Mute audio output.",
)
parser.add_argument(
    "-ds",
    "--dummy_stimulus",
    action="store_true",
    help="Use dummy stimulus.",
)
parser.add_argument(
    "-dt",
    "--dummy_thermoino",
    action="store_true",
    help="Use dummy Thermoino device.",
)
args = parser.parse_args()

# Configure logging
configure_logging(
    stream_level=logging.INFO if not (args.debug or args.all) else logging.DEBUG,
    file_path=LOG_FILE,
)

# Load scripts and configurations
script = load_script(EXP_DIR / "placebo_script.yaml")
config = load_configuration(EXP_DIR / "placebo_config.toml")
thermoino_config = load_configuration(EXP_DIR.parent / "thermoino_config.toml")
stimulus_config = load_configuration(EXP_DIR / "stimulus_config.toml")
# Experiment settings
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = config["experiment"]["element_color"]
stimuli.defaults.textbox_text_colour = config["experiment"]["element_color"]
stimuli.defaults.rectangle_colour = config["experiment"]["element_color"]
io.defaults.mouse_show_cursor = False
control.defaults.initialize_delay = 3
# Adjust settings
if args.all:
    logging.debug("Using all flags for a dry run.")
    for flag in vars(args).keys():
        setattr(args, flag, True)
if args.debug or args.windowed:
    control.set_develop_mode(True)
    control.defaults.window_mode = False
if args.debug:
    logging.debug("Running in debug mode.")
if args.windowed:
    logging.debug("Running in windowed mode.")
    control.defaults.window_mode = True
    control.defaults.window_size = (860, 600)
if args.muted:
    logging.debug("Muting the audio output.")
if args.dummy_stimulus:
    logging.debug("Using dummy stimulus.")
    stimulus_config |= config["dummy_stimulus"]
else:
    stimulus_config.pop("dummy", None)

# Prepare to database
db_manager = DatabaseManager()
with db_manager:
    participant_number = db_manager.last_participant_number

db_rate_limiter = RateLimiter(rate=config["database"]["rate_limit"], use_intervals=True)

# # Load participant info and update stimulus config with calibration data
# participant_info = read_last_participant(CALIBRATION_RESULTS)
# participant_info["vas0"] = float(participant_info["vas0"])
# participant_info["vas70"] = float(participant_info["vas70"])
# participant_info["temperature_range"] = float(participant_info["temperature_range"])
# participant_info["temperature_baseline"] = round(
#     (participant_info["vas0"] + participant_info["vas70"]) / 2, 2
# )
# # check if VAS 70 is too high (risk of burn)
# readjustment = False
# if participant_info["vas70"] > 48.0:
#     readjustment = True
#     participant_info["vas70"] = 48.0
#     logging.warning("VAS 70 is too high. Adjusting maximum temperature to 48.0 °C.")
# # check if delta is too low (under 1.5 °C); round results to avoid float weirdness
# temp_range = participant_info["vas70"] - participant_info["vas0"]
# if temp_range < 1.5:
#     readjustment = True
#     missing_amount = 1.5 - temp_range
#     participant_info["vas70"] = round(participant_info["vas70"] + missing_amount, 1)

#     if participant_info["vas70"] > 48.0:
#         overflow = participant_info["vas70"] - 48.0
#         participant_info["vas0"] = round(participant_info["vas0"] - overflow, 1)
#         participant_info["vas70"] = 48.0
#     logging.warning("Temperature range is too low. Adjusting delta to 1.5 °C.")
# # calculate new baseline and range
# if readjustment:
#     participant_info["temperature_baseline"] = round(
#         (participant_info["vas0"] + participant_info["vas70"]) / 2, 1
#     )
#     participant_info["temperature_range"] = round(
#         (participant_info["vas70"] - participant_info["vas0"]), 1
#     )
#     logging.info(
#         f"New values: VAS 70 = {participant_info['vas70']} °C, "
#         f"VAS 0 = {participant_info['vas0']} °C, "
#         f"baseline = {participant_info['temperature_baseline']} °C, "
#         f"range = {participant_info['temperature_range']} °C."
#     )

# determine order of skin patches based on participant ID
id_is_odd = int(participant_number) % 2
skin_patches = range(1, 7) if id_is_odd else range(6, 0, -1)
logging.info(f"Start with skin patch {skin_patches[0]}.")
# update config with calibration data
# config["stimulus"] |= participant_info
# shuffle seeds for randomization
random.shuffle(config["stimulus"]["seeds"])

# Experiment setup
exp = design.Experiment(name="pain-placebo")
exp.set_log_level(0)
control.initialize(exp)
screen_size = exp.screen.size
audio = copy.deepcopy(script)  # audio shares the same keys
prepare_audio(audio, audio_dir=EXP_DIR / "audio")
prepare_script(
    script,
    text_box_size=scale_2d_tuple(config["experiment"]["text_box_size"], screen_size),
    text_size=scale_1d_value(config["experiment"]["text_size"], screen_size),
)
vas_slider = VisualAnalogueScale(experiment=exp, config=config["visual_analogue_scale"])


# Initialize Thermoino
thermoino = ThermoinoComplexTimeCourses(
    mms_baseline=thermoino_config["mms_baseline"],
    mms_rate_of_rise=thermoino_config["mms_rate_of_rise"],
    dummy=args.dummy_thermoino,
)
thermoino.connect()


def callback_function(
    trial_id: int,
    stimulus: StimulusGenerator,
) -> None:
    """
    Callback function for the trial loop.
    """
    vas_slider.rate()  # slider has its own rate limiters (see VisualAnalogueScale)
    time = exp.clock.stopwatch_time

    # Insert mouse clicks
    mouse_event = exp.mouse.get_last_button_down_event()
    if mouse_event is not None:
        try:
            db_manager.insert_button(
                trial_id=trial_id,
                button=mouse_event,
                time=time,
                debug=args.debug or args.all,
            )
        except Exception as e:
            logging.error(f"Error while logging mouse click: {e}")

    # Insert temperature and rating
    if db_rate_limiter.is_allowed(time):
        index = int((time / 1000) * stimulus.sample_rate)
        index = min(index, len(stimulus.y) - 1)  # prevent index out of bounds
        db_manager.insert_measurement(
            trial_id=trial_id,
            time=time,
            temperature=stimulus.y[index],
            rating=vas_slider.rating,
            debug=args.debug or args.all,
        )


def main():
    # Start experiment
    control.start(skip_ready_screen=True, subject_id=participant_number)
    logging.info(
        f"Started measurement with seed order {stimulus_config['stimulus']['seeds']}."
    )

    # Introduction
    for text, sound in zip(script[s := "welcome"].values(), audio[s].values()):
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    # Instruction with VAS slider
    for text, sound in zip(script[s := "instruction"].values(), audio[s].values()):
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(
            K_SPACE,
            callback_function=lambda text=text: vas_slider.rate(text),
        )

    # Ready
    for text, sound in zip(script[s := "ready"].values(), audio[s].values()):
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    # Trial loop
    total_trials = len(stimulus_config.keys())
    correlations = []  # between temperature and rating
    reward = 0.0
    for trial, seed in enumerate(stimulus_config["stimulus"]["seeds"]):
        db_manager.connect()

        name = "TODO"  # TODO: add name for placebo vs no placebo, etc.
        logging.info(
            f"Started trial ({trial + 1}/{total_trials}) with stimulus {name}."
        )
        db_manager.insert_trial(trial + 1, name, seed)
        trial_id = db_manager.last_trial_id

        # Start with a waiting screen for the initalization of the complex time course
        script["wait"].present()
        stimulus = StimulusGenerator(stimulus_config)
        thermoino.flush_ctc()
        thermoino.init_ctc(bin_size_ms=thermoino_config["bin_size_ms"])
        thermoino.create_ctc(temp_course=stimulus.y, sample_rate=stimulus.sample_rate)
        thermoino.load_ctc()
        thermoino.trigger()

        # Present the VAS slider and wait for the temperature to ramp up
        time_to_ramp_up = thermoino.prep_ctc()
        db_manager.insert_marker(trial_id, "thermode_ramp_up", exp.clock.stopwatch_time)
        exp.clock.wait_seconds(
            time_to_ramp_up + 1.5,  # give participant time to prepare
            callback_function=lambda: vas_slider.rate(),  # lamdba needed for callback
        )

        # Measure temperature and rating
        thermoino.exec_ctc()
        logging.info("Stimulus started.")
        db_manager.insert_marker(trial_id, "stimulus_start", exp.clock.stopwatch_time)
        db_rate_limiter.reset()
        exp.clock.reset_stopwatch()  # needed for the callback
        exp.clock.wait_seconds(
            stimulus.duration,
            callback_function=lambda: callback_function(trial_id, stimulus),
        )
        db_manager.insert_marker(trial_id, "stimulus_end", exp.clock.stopwatch_time)
        logging.info("Stimulus ended.")

        # Add delay at the end of the complex time course (see thermoino.py)
        exp.clock.wait_seconds(1, callback_function=lambda: vas_slider.rate())

        # Ramp down temperature
        time_to_ramp_down, _ = thermoino.set_temp(thermoino_config["mms_baseline"])
        exp.clock.wait_seconds(
            time_to_ramp_down, callback_function=lambda: vas_slider.rate()
        )
        db_manager.insert_marker(
            trial_id, "thermode_ramp_down", exp.clock.stopwatch_time
        )
        logging.info(
            f"Finished trial ({trial + 1}/{total_trials}) with stimulus {name}."
        )
        db_manager.insert_marker(
            trial_id, "thermode_baseline", exp.clock.stopwatch_time
        )

        # Log data
        query = f"SELECT * FROM Measurements WHERE trial_id = {trial_id};"
        df = pl.read_database(query, db_manager.conn)
        logging.info(
            f"Rating of the stimulus: "
            f"min = {int(df['rating'].min())}, "
            f"max = {int(df['rating'].max())}, "
            f"mean = {int(df['rating'].mean())}, "
            f"std = {int(df['rating'].std())}."
        )
        db_manager.disconnect()

        # Next trial
        if trial == total_trials - 1:
            break
        logging.info(
            f"Next, use skin patch {skin_patches[(trial + 1) % len(skin_patches)]}."
        )
        script["next_trial"].present()
        audio["next_trial"].play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)
        script["approve"].present()
        exp.keyboard.wait(K_SPACE)

    # Save participant data
    # participant_info_ = read_last_participant()  # reload to remove calibration data
    # participant_info_["readjustment"] = readjustment
    # participant_info_["seed_order"] = config["stimulus"]["seeds"]
    # participant_info_["correlations"] = correlations
    # participant_info_["reward"] = reward
    # add_participant_info(participant_info_, MEASUREMENT_RESULTS)

    # End of Experiment
    script["bye"].present()
    audio["bye"].play(maxtime=args.muted)
    exp.clock.wait_seconds(3)

    control.end()
    # add marker for experiment end here
    for instance in [thermoino]:
        instance.close()
    logging.info("Measurement successfully finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
