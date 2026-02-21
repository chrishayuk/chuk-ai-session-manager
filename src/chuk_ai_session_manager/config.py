from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Central model config: can be overridden by environment variable
DEFAULT_TOKEN_MODEL = os.getenv("CHUK_DEFAULT_MODEL", "gpt-4o-mini")
