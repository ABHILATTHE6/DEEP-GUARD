"""Upload or download public DEEP-GUARD checkpoint assets using GitHub Releases.

Upload requires GH_TOKEN with Contents: read/write permission for the repository.
Release assets are public when attached to a public repository. Never put the token
in code or commit it to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"


def _request(
    method: str,
    url: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    content_type: str = "application/json",
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "DEEP-GUARD-checkpoint-sync",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach GitHub: {exc}") from exc

    if not raw:
        return None
    if "json" in content_type or "api.github.com" in url:
        return json.loads(raw.decode("utf-8"))
    return raw


def _api_url(repo: str, suffix: str) -> str:
    if "/" not in repo or len(repo.split("/")) != 2:
        raise ValueError("--repo must be in owner/name format")
    return f"{API_ROOT}/repos/{repo}/{suffix.lstrip('/')}"


def _get_release(repo: str, tag: str, token: str) -> dict[str, Any]:
    encoded_tag = urllib.parse.quote(tag, safe="")
    return _request("GET", _api_url(repo, f"releases/tags/{encoded_tag}"), token=token)


def _ensure_release(repo: str, tag: str, token: str) -> dict[str, Any]:
    try:
        return _get_release(repo, tag, token)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
    created = _request(
        "POST",
        _api_url(repo, "releases"),
        token=token,
        payload={
            "tag_name": tag,
            "target_commitish": "main",
            "name": f"DEEP-GUARD checkpoint storage ({tag})",
            "body": (
                "Public model-training checkpoint storage for DEEP-GUARD. "
                "Assets may be replaced as training progresses; see the repository's "
                "training documentation for reproducibility details."
            ),
            "draft": False,
            "prerelease": True,
        },
    )
    print(f"Created release: {created.get('html_url', '')}")
    return created


def _asset_by_name(release: dict[str, Any], asset_name: str) -> dict[str, Any] | None:
    return next((a for a in release.get("assets", []) if a.get("name") == asset_name), None)


def upload(args: argparse.Namespace) -> None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Upload requires GH_TOKEN (or GITHUB_TOKEN) in the environment. "
            "Use a secret store; never hardcode or commit the token."
        )
    path = args.file.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    repo = args.repo
    release = _ensure_release(repo, args.tag, token)
    existing = _asset_by_name(release, args.asset_name)
    if existing is not None:
        _request(
            "DELETE",
            _api_url(repo, f"releases/assets/{existing['id']}"),
            token=token,
        )
        print(f"Replaced existing asset: {args.asset_name}")

    upload_url = f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets"
    upload_url += "?" + urllib.parse.urlencode({"name": args.asset_name})
    file_size = path.stat().st_size
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "DEEP-GUARD-checkpoint-sync",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(file_size),
    }
    request = urllib.request.Request(
        upload_url,
        data=path.read_bytes(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            asset = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub asset upload failed (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Checkpoint upload failed: {exc}") from exc

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Uploaded: {asset.get('browser_download_url', '')}")
    print(f"size_bytes={file_size}")
    print(f"sha256={digest}")


def download(args: argparse.Namespace) -> None:
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    release = _get_release(args.repo, args.tag, token or "")
    asset = _asset_by_name(release, args.asset_name)
    if asset is None:
        raise FileNotFoundError(
            f"Release '{args.tag}' has no asset named '{args.asset_name}'. "
            "Upload a checkpoint first or check the repository/tag/asset name."
        )

    url = asset.get("browser_download_url")
    if not url:
        raise RuntimeError("GitHub did not return a download URL for the release asset.")
    headers = {"User-Agent": "DEEP-GUARD-checkpoint-sync"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    temporary = target.with_suffix(target.suffix + ".download")
    try:
        with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"Downloaded: {url}")
    print(f"checkpoint={target}")
    print(f"size_bytes={target.stat().st_size}")
    print(f"sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("upload", "download"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo", default="ABHILATTHE6/DEEP-GUARD")
        child.add_argument("--tag", default="training-checkpoints")
        child.add_argument("--asset-name", default="image-training-latest.pt")
        if command == "upload":
            child.add_argument("--file", type=Path, required=True)
        else:
            child.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "upload":
            upload(args)
        else:
            download(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
