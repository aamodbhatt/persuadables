"""Capture README screenshots of the real dashboard with Playwright and an installed Edge/Chrome.

    uv run python scripts/capture_screenshots.py [--dashboard reports/dashboard.html]

No browser is downloaded: ``--channel`` picks an installed Chromium-based browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def _settle(page: Page) -> None:
    # Plotly draws asynchronously; wait until every chart has an SVG and fonts are ready.
    page.wait_for_function(
        "() => [...document.querySelectorAll('.chart')].every(c => c.querySelector('svg'))"
    )
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(400)


def capture(dashboard: Path, out_dir: Path, channel: str) -> list[Path]:
    if not dashboard.is_file():
        msg = f"{dashboard} not found; run `persuadables report` first"
        raise FileNotFoundError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1280, "height": 900},
                device_scale_factor=1.5,
                color_scheme="light",
            )
            page.goto(dashboard.resolve().as_uri())
            _settle(page)
            hero = out_dir / "dashboard.png"
            page.screenshot(path=str(hero), clip={"x": 0, "y": 0, "width": 1280, "height": 900})
            shots.append(hero)
            for section, name in (
                ("#uplift", "dashboard_uplift.png"),
                ("#policy", "dashboard_policy.png"),
            ):
                path = out_dir / name
                page.locator(section).screenshot(path=str(path))
                shots.append(path)
            # The simulator at a smaller budget, to show it responds.
            page.locator("#budget").fill("10")
            page.locator("#budget").dispatch_event("input")
            page.wait_for_timeout(300)
            path = out_dir / "dashboard_policy_10.png"
            page.locator("#policy").screenshot(path=str(path))
            shots.append(path)
        finally:
            browser.close()
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=ROOT / "reports" / "dashboard.html")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--channel", default="msedge", help="installed browser: msedge or chrome")
    args = parser.parse_args()
    for path in capture(args.dashboard, args.out, args.channel):
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
