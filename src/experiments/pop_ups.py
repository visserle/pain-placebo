"""
Note that all pop_ups have to be called before expyriment window creation or bad things
will happen.
"""

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from screeninfo import get_monitors

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def ask_for_calibration_start() -> bool:
    """Custom confirmation dialog window with checkboxes for each item."""
    items = [
        "MMS Programm pain-calibration?",
        "MMS Trigger-bereit?",
        "Jalousien unten?",
        "Alle Sensoren in iMotions connected?"
        "Sensor Preview mit PPG, EDA und Pupillometrie?",
        "Terminal, iMotions, NIC mit EEG Qualitätssignalen?",
        "Thermodenkopf angebracht?",
    ]
    return _start_window(items, "calibration")


def ask_for_measurement_start() -> bool:
    """Custom confirmation dialog window with checkboxes for each item."""
    items = [
        "MMS Programm auf pain-measurement umgestellt?",
        "MMS Trigger-bereit?",
        "iMotions' Kalibrierung bestätigt?",
        "Sensor Preview geöffnet?",
        "Signale überprüft (PPG, EDA, Pupillometrie)?",
        "Hintergrundbeleuchtung blendet nicht?",
        "Eye-Tracker Postionsfeedback sichtbar?",
        "Terminal, iMotions, NIC mit EEG Qualitätssignalen?",
        "Hautareal gewechselt?",
    ]
    return _start_window(items, "measurement")


def _start_window(items: list[str], title: str) -> bool:
    """Custom confirmation dialog window with checkboxes for each item."""
    root = tk.Tk()
    root.withdraw()
    dialog = ChecklistDialog(root, items)
    _center_tk_window(root)
    response = dialog.show()
    logger.debug(
        f"Confirmation for {title} start {'recieved' if response else 'denied'}."
    )
    return response


class ChecklistDialog:
    def __init__(self, root, items):
        self.root = root
        self.root.title("Ready to Start?")
        self.items = items
        self.response = False
        self.setup_ui()

    def setup_ui(self):
        """Sets up the UI components of the dialog."""
        self._create_checkboxes()
        self._create_proceed_button()
        self.root.deiconify()

    def _create_checkboxes(self):
        """Creates a checkbox for each item."""
        self.check_vars = []
        for item in self.items:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.root, text=item, variable=var)
            chk.pack(anchor="w", padx=20, pady=5)
            self.check_vars.append(var)

    def _create_proceed_button(self):
        """Creates the proceed button."""
        self.proceed_button = ttk.Button(
            self.root, text="Proceed", command=self.on_proceed
        )
        self.proceed_button.pack(pady=20)

    def on_proceed(self):
        """Handles the proceed button click event."""
        if all(var.get() for var in self.check_vars):
            self.response = True
            self.root.destroy()
        else:
            messagebox.showwarning(
                "Warning", "Please confirm all items before proceeding."
            )

    def show(self):
        """Displays the dialog and returns the user's response."""
        self.root.mainloop()
        return self.response


def _center_tk_window(
    window: tk.Tk,
    primary_screen: bool = False,
) -> None:
    """
    Center a window, by default on the first available non-primary screen if available,
    otherwise on the primary screen.
    """
    # Get sorted list of monitors
    monitors = sorted(get_monitors(), key=lambda m: m.is_primary)
    # non-primary monitor comes first (False < True)
    monitor = monitors[0] if not primary_screen else monitors[-1]

    # Get window size
    window.update_idletasks()
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # Calculate center coordinates
    center_x = int(monitor.x + (monitor.width / 2 - window_width / 2))
    center_y = int(monitor.y + (monitor.height / 2 - window_height / 2))

    # Move window to the center of the chosen monitor
    window.geometry(f"+{center_x}+{center_y}")


if __name__ == "__main__":
    # Set basic configuration for the logger
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    ask_for_calibration_start()
