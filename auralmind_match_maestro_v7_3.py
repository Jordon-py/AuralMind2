from __future__ import annotations

"""
Legacy compatibility wrapper for the canonical maestro engine.

This file is retained so older scripts or local notes that reference the old
top-level path continue to work. The maintained implementation lives in
`tools/auralmind_maestro.py`.
"""

from tools.auralmind_maestro import main


if __name__ == "__main__":
    main()
