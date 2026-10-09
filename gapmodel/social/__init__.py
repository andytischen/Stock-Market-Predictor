"""Social-arbitrage sub-package for gapmodel.

Exposes the public API in a single import:

    from gapmodel.social import SocialSignal, scan, render_text, render_html
"""

from __future__ import annotations

from .report import render_html, render_text
from .signals import SocialSignal, scan

__all__ = ["SocialSignal", "render_html", "render_text", "scan"]
