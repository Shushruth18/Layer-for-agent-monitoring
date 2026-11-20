import os
import sys
import tempfile
from pathlib import Path

# Deterministic, offline tests: no judge LLM calls and a throwaway database.
os.environ["MONITOR_JUDGE"] = "off"
os.environ["MONITOR_DB"] = str(Path(tempfile.mkdtemp()) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
