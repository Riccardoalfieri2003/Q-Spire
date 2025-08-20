#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, requests, datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional

MILESTONES_FILE = os.environ.get("MILESTONES_FILE", "STATUS_MILESTONES.json")
TZ = ZoneInfo(os.environ.get("TIMEZONE", "America/Anchorage"))

repo = os.environ.get("GITHUB_REPOSITORY", "")
if not repo or "/" not in repo:
    sys.exit("GITHUB_REPOSITORY not set (owner/repo).")

owner, name = repo.split("/", 1)
token = os.environ.get("GITHUB_TOKEN")
if not token:
    sys.exit("GITHUB_TOKEN not set.")

headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

data = load_json(MILESTONES_FILE)
manual: List[Dict[str, Any]] = data.get("manual", []) if isinstance(data.get("manual"), list) else []
auto_commits: List[Dict[str, Any]] = data.get("auto_commits", []) if isinstance(data.get("auto_commits"), list) else []
last_sha: str = data.get("last_commit_sha", "") or ""

seen_shas = {c.get("sha") for c in auto_commits if isinstance(c, dict) and "sha" in c}

since_iso: Optional[str] = None
if last_sha:
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{name}/commits/{last_sha}",
        headers=headers, timeout=30
    )
    if r.status_code == 200:
        commit = r.json()
        since_iso = commit["commit"]["author"]["date"]

params = {"per_page": 100}
if since_iso:
    params["since"] = since_iso

new_entries: List[Dict[str, Any]] = []
page = 1
latest_commit_utc: Optional[dt.datetime] = None
latest_commit_sha: Optional[str] = None

while True:
    params["page"] = page
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{name}/commits",
        headers=headers, params=params, timeout=30
    )
    if r.status_code != 200:
        sys.exit(f"GitHub API error {r.status_code}: {r.text}")
    items = r.json()
    if not items:
        break

    for it in items:
        full_sha = it.get("sha", "")
        if not full_sha:
            continue
        sha7 = full_sha[:7]
        if sha7 in seen_shas:
            continue

        msg = (it["commit"]["message"] or "").strip()
        subject = msg.splitlines()[0][:200]

        adate = it["commit"]["author"]["date"]  
        d_utc = dt.datetime.fromisoformat(adate.replace("Z", "+00:00"))
        d_local = d_utc.astimezone(TZ)

        entry = {
            "date": d_local.date().isoformat(),
            "sha": sha7,
            "subject": subject
        }
        new_entries.append(entry)
        seen_shas.add(sha7)

        if (latest_commit_utc is None) or (d_utc > latest_commit_utc):
            latest_commit_utc = d_utc
            latest_commit_sha = full_sha

    page += 1

new_entries.sort(key=lambda e: (e["date"], e["sha"]))
if new_entries:
    auto_commits.extend(new_entries)

out = {
    "manual": manual,
    "auto_commits": auto_commits,
    "last_commit_sha": latest_commit_sha or last_sha,
    "last_checked": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
}
save_json(MILESTONES_FILE, out)
