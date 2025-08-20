#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List

ANCHORAGE = ZoneInfo("America/Anchorage")
START_ENV = os.environ.get("QELM_START_DATE")
STATUS_FILE = os.environ.get("STATUS_FILE", "STATUS.md")
MILESTONES_FILE = os.environ.get("MILESTONES_FILE", "STATUS_MILESTONES.json")
MAX_COMMITS_PER_DAY = int(os.environ.get("MAX_COMMITS_PER_DAY", "3"))
BADGE_JSON_FILE = os.environ.get("BADGE_JSON_FILE", "badges/days_active.json")  # <-- NEW

if not START_ENV:
    raise SystemExit("QELM_START_DATE env var is required (YYYY-MM-DD)")

try:
    START_DATE = date.fromisoformat(START_ENV)
except ValueError as e:
    raise SystemExit(f"Invalid QELM_START_DATE '{START_ENV}': {e}")

now_local = datetime.now(ANCHORAGE)
today = now_local.date()
if today < START_DATE:
    raise SystemExit("Start date is in the future relative to Alaska time.")

def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_milestones(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

data = load_milestones(MILESTONES_FILE)
manual = data.get("manual", []) if isinstance(data.get("manual"), list) else (data if isinstance(data, list) else [])
auto_commits: List[Dict[str, Any]] = data.get("auto_commits", []) if isinstance(data.get("auto_commits"), list) else []

notes_by_date: Dict[date, List[str]] = {}
def add_note(d: date, text: str):
    if text:
        notes_by_date.setdefault(d, []).append(text)

if isinstance(manual, list):
    for item in manual:
        try:
            d = date.fromisoformat(str(item.get("date", "")))
            txt = str(item.get("text", "")).strip()
            if d and txt:
                add_note(d, txt)
        except Exception:
            continue

for c in auto_commits:
    try:
        d = date.fromisoformat(str(c.get("date", "")))
        subj = str(c.get("subject", "")).strip()
        sha = str(c.get("sha", ""))[:7]
        if d and subj:
            add_note(d, f"{subj} ({sha})")
    except Exception:
        continue

content_old = load_text(STATUS_FILE)
HISTORY_HDR = "## History\n"
line_re = re.compile(r"^- (\d{4}-\d{2}-\d{2}) — Day (\d+)(?: — (.*))?$")
if HISTORY_HDR in content_old:
    _, _, tail = content_old.partition(HISTORY_HDR)
    for raw in tail.splitlines():
        m = line_re.match(raw.strip())
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        extra = (m.group(3) or "").strip()
        if extra:
            add_note(d, extra)

lines: List[str] = []
current = today
while current >= START_DATE:
    day_num = (current - START_DATE).days + 1
    base = f"- {current.isoformat()} — Day {day_num}"
    notes = notes_by_date.get(current, [])
    if notes:
        seen = set()
        uniq = [n for n in notes if not (n in seen or seen.add(n))]
        if len(uniq) > MAX_COMMITS_PER_DAY:
            shown = " · ".join(uniq[:MAX_COMMITS_PER_DAY])
            base += f" — {shown} (+{len(uniq) - MAX_COMMITS_PER_DAY})"
        else:
            base += " — " + " · ".join(uniq)
    lines.append(base)
    current -= timedelta(days=1)

day_number = (today - START_DATE).days + 1
header_line = f"QELM — Day {day_number}"
subheader_line = now_local.strftime("%Y-%m-%d %H:%M %Z (%z) — Alaska time")

top = (
    f"# {header_line}\n\n"
    f"**Date:** {subheader_line}\n\n"
    f"QELM has been active for **{day_number} day"
    f"{'s' if day_number != 1 else ''}** since **{START_DATE.isoformat()}**.\n\n"
    "This file is updated once per day. No history rewrites.\n\n"
)

new_content = top + HISTORY_HDR + "\n".join(lines) + "\n"

if new_content != content_old:
    save_text(STATUS_FILE, new_content)

badge_payload = {
    "schemaVersion": 1,
    "label": "days active",
    "message": str(day_number),
    "color": "blue",
    "cacheSeconds": 300  # 5 minutes
}
save_text(BADGE_JSON_FILE, json.dumps(badge_payload, ensure_ascii=False, indent=2) + "\n")
