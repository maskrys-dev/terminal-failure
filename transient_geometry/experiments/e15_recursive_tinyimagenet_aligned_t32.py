"""Run the frozen E15 aligned-horizon variant through the validated E14 driver."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["TERMINAL_FAILURE_RECURSIVE_VARIANT"] = "e15"
driver = Path(__file__).with_name("e14_recursive_tinyimagenet_rebuttal.py")
runpy.run_path(str(driver), run_name="__main__")
