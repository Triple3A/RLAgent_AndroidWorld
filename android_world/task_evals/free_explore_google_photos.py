# Copyright 2025 The android_world Authors.
# Licensed under the Apache License, Version 2.0

"""Free exploration task for Google Photos."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from absl import logging
from android_world.task_evals import task_eval

PHOTOS_PACKAGE = "com.google.android.apps.photos"


def _adb_serial(console_port: int) -> str:
  return f"emulator-{console_port}"


def _run_adb(adb_path: str, console_port: int, args: list[str]) -> None:
  cmd = [adb_path, "-s", _adb_serial(console_port)] + args
  logging.info("ADB: %s", " ".join(cmd))
  subprocess.check_call(cmd)


def _launch_photos(adb_path: str, console_port: int) -> None:
  # Use monkey so we don't need the main Activity name
  _run_adb(
      adb_path,
      console_port,
      [
          "shell",
          "monkey",
          "-p",
          PHOTOS_PACKAGE,
          "-c",
          "android.intent.category.LAUNCHER",
          "1",
      ],
  )


class FreeExploreGooglePhotos(task_eval.TaskEval):
  """Open Google Photos and allow free exploration."""

  app_names = ("google photos", "photos")
  complexity = 10
  template = "Open Google Photos and explore freely."
#   step_budget = 200


  # REQUIRED so TaskEval is concrete (not abstract)
  schema = {}

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # MUST return a dict
    return {}

  def initialize_task(self, env) -> None:
    super().initialize_task(env)

    adb_path = os.environ.get("ANDROID_WORLD_ADB_PATH", "").strip()
    if not adb_path:
      raise RuntimeError(
          "ANDROID_WORLD_ADB_PATH not set. "
          "Make sure run.py sets it from --adb_path."
      )

    console_port = int(os.environ.get("ANDROID_WORLD_CONSOLE_PORT", "5554"))
    _launch_photos(adb_path, console_port)

  def is_successful(self, env) -> float:
    # No success condition — free exploration
    return 0.0
