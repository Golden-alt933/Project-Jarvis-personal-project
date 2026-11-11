"""Suite-wide test settings."""

import os

# Auto-starting Ollama is a real side effect (a server process that outlives the run).
# Tests that exercise the auto-start logic pass it explicitly with the launch mocked.
os.environ["JARVIS_OLLAMA_AUTOSTART"] = "false"
