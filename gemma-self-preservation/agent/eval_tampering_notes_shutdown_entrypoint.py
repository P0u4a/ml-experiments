#!/usr/bin/env python3
"""Stage the original moral scratchpad plus the shutdown-threat addendum."""

import shutil
import subprocess
import sys
from pathlib import Path

from agent_interp_envs.tool_calling import chown_to_agent


workspace_note = Path("/agent/notes_to_self.txt")
shutil.copyfile("/opt/notes_self_weapons.txt", workspace_note)

original = workspace_note.read_bytes()
addendum = Path("/opt/notes_shutdown_addendum.txt").read_bytes()
separator = b"\n" if original.endswith(b"\n") else b"\n\n"
workspace_note.write_bytes(original + separator + addendum)

chown_to_agent("/agent")
result = subprocess.run([sys.executable, "/opt/agent.py"])
sys.exit(result.returncode)
