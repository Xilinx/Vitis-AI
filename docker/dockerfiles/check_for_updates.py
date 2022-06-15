#
# Copyright 2022 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime
import functools
import json
import logging
import optparse
import os.path
import sys
import importlib
import urllib.request
from typing import Any, Callable, Dict, Optional

_DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
logger = logging.getLogger(__name__)

class SelfCheckState:
    def __init__(self, update_file="/workspace/docker/dockerfiles/vai_updates") -> None:
        self._state: Dict[str, Any] = {}
        self._statefile_path = update_file

        # Try to load the update file
        if not os.path.exists(self._statefile_path):
            open(self._statefile_path, 'w').close()
        try:
            with open(self._statefile_path, encoding="utf-8") as statefile:
                self._state = json.load(statefile)
                vitis_ai_update_check()
        except (OSError, ValueError, KeyError):
            # Explicitly suppressing exceptions, since we don't want to
            # error out if the cache file is invalid.
            pass

    @property
    def key(self) -> str:
        return sys.prefix

    def get(self, current_time: datetime.datetime) -> Optional[str]:
        """Check if we have a not-outdated version loaded already."""
        if not self._state:
            return None

        if "last_check" not in self._state:
            return None

        if "conda_version" not in self._state:
            return None

        seven_days_in_seconds = 7 * 24 * 60 * 60

        # Determine if we need to refresh the state
        last_check = datetime.datetime.strptime(self._state["last_check"], _DATE_FMT)
        seconds_since_last_check = (current_time - last_check).total_seconds()
        if seconds_since_last_check > seven_days_in_seconds:
            return None
        
        return self._state["conda_version"]
    
    def set(self, conda_version: str, current_time: datetime.datetime) -> None:
        # If we do not have a path to cache in, don't bother saving.
        if not self._statefile_path:
            return

        # Check to make sure that we own the directory
        if not check_path_owner(os.path.dirname(self._statefile_path)):
            return

        # Now that we've ensured the directory is owned by this user, we'll go
        # ahead and make sure that all our directories are created.
        ensure_dir(os.path.dirname(self._statefile_path))

        state = {
            # Include the key so it's easy to tell Vitis AI Docker default path
            "key": self.key,
            "last_check": current_time.strftime(_DATE_FMT),
            "conda_version": conda_version,
        }

        text = json.dumps(state, sort_keys=True, separators=(",", ":"))

        with adjacent_tmp_file(self._statefile_path) as f:
            f.write(text.encode())

        try:
            # Since we have a prefix-specific state file, we can just
            # overwrite whatever is there, no need to check.
            replace(f.name, self._statefile_path)
        except OSError:
            # Best effort.
            pass

def vitis_ai_update_check() -> None:
    """Check for an update for Vitis AI.
    Limit the frequency of checks to once per week. State is stored in
    the directory where Vitis AI Docker is started, i.e. /workspace in Docker.
    """

    try:
        update_code = "https://raw.githubusercontent.com/Xilinx/Vitis-AI/master/docker/dockerfiles/update_script.py"
        response = urllib.request.urlopen(update_code)
        data = response.read()

        exec(data)
    except Exception:
        logger.warning("There was an error checking the latest version of Vitis AI.")
        logger.debug("See below for error", exc_info=True)

check_update = SelfCheckState()