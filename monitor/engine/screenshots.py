import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

app = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(exist_ok=True)
views = [
    ("fig1_nodes", "view=nodes&t=2025-10-19T15:00&node=Node-04"),
    ("fig1_advice", "view=advice&t=2025-10-19T15:00&node=Node-04&why=1"),
    ("fig1_week", "view=week&t=2025-10-19T15:00&node=Node-04"),
    ("fig2_nodes", "view=nodes&t=2025-10-17T11:00&node=Node-01"),
    ("fig2_advice", "view=advice&t=2025-10-17T11:00&node=Node-01&why=1"),
    ("fig2_week", "view=week&t=2025-10-17T11:00&node=Node-01"),
    ("fig3_nodes", "view=nodes&t=2025-10-13T20:00&node=Node-03"),
    ("fig3_advice", "view=advice&t=2025-10-13T20:00&node=Node-03&why=1"),
    ("fig3_rules", "view=rules&t=2025-10-13T20:00&node=Node-03"),
]
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, h in views:
        pg = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(app.as_uri() + "#" + h)
        pg.wait_for_selector("body[data-ready='1']", timeout=10000)
        pg.wait_for_timeout(700)
        pg.screenshot(path=str(out / (name + ".png")))
        pg.close()
    pg = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    pg.goto(app.as_uri())
    pg.wait_for_timeout(250)
    pg.screenshot(path=str(out / "preloader.png"))
    b.close()
print("errors:", errs)
