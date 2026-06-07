#!/usr/bin/env python3
"""
RAKSHAK-ICS — Pre-commit Data Privacy Checker
==============================================

Scans staged git files to prevent accidental commits of raw SWaT data
or other sensitive ICS dataset artefacts.

Usage:
    # Standalone check
    python scripts/check_data_privacy.py

    # As a pre-commit hook (symlink or copy into .git/hooks/pre-commit)
    python scripts/check_data_privacy.py --strict

Exit codes:
    0 — No violations found (safe to commit)
    1 — One or more violations detected (commit blocked)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Colour helpers (graceful fallback on Windows without ANSI support)
# ---------------------------------------------------------------------------

def _supports_colour() -> bool:
    """Return True if the terminal likely supports ANSI colour codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if sys.platform == "win32":
        # Windows 10+ supports ANSI if virtual terminal processing is on.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Enable ANSI escape sequences on stdout
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOUR = _supports_colour()

_RED = "\033[91m" if _COLOUR else ""
_YELLOW = "\033[93m" if _COLOUR else ""
_GREEN = "\033[92m" if _COLOUR else ""
_CYAN = "\033[96m" if _COLOUR else ""
_BOLD = "\033[1m" if _COLOUR else ""
_RESET = "\033[0m" if _COLOUR else ""


def _error(msg: str) -> str:
    return f"{_RED}{_BOLD}[ERROR]{_RESET} {_RED}{msg}{_RESET}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}{_BOLD}[WARN]{_RESET}  {_YELLOW}{msg}{_RESET}"


def _ok(msg: str) -> str:
    return f"{_GREEN}{_BOLD}[OK]{_RESET}    {_GREEN}{msg}{_RESET}"


def _info(msg: str) -> str:
    return f"{_CYAN}{_BOLD}[INFO]{_RESET}  {msg}"


# ---------------------------------------------------------------------------
# Violation data model
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A single privacy violation."""
    file: str
    rule: str
    detail: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ScanResult:
    """Aggregated scan results."""
    violations: List[Violation] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == "warning" for v in self.violations)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(*args: str) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_staged_files() -> List[str]:
    """Return list of staged file paths (relative to repo root)."""
    output = _run_git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    if output is None:
        return []
    return [f for f in output.splitlines() if f.strip()]


def get_repo_root() -> Optional[Path]:
    """Return the repository root directory."""
    output = _run_git("rev-parse", "--show-toplevel")
    if output is None:
        return None
    return Path(output)


# ---------------------------------------------------------------------------
# Patterns to detect
# ---------------------------------------------------------------------------

# Raw CSV / data file paths that look like SWaT dataset references
RAW_PATH_PATTERNS = [
    re.compile(r"data[/\\]swat[/\\].*\.csv", re.IGNORECASE),
    re.compile(r"dataset[123]\.csv", re.IGNORECASE),
    re.compile(r"SWaT.*\.csv", re.IGNORECASE),
    re.compile(r"SWaT.*\.xlsx", re.IGNORECASE),
]

# SWaT timestamp format: "28-Dec-2015 10:00:00" → dd-Mon-YYYY HH:MM:SS
SWAT_TIMESTAMP_RE = re.compile(
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\s+\d{2}:\d{2}:\d{2}",
    re.IGNORECASE,
)

# Lines that look like raw sensor readings (non-normalised floats > 1.0)
# Heuristic: a line with many comma-separated numbers, at least one > 1.0
RAW_SENSOR_LINE_RE = re.compile(
    r"(?:^|,)\s*(\d+\.\d+)\s*(?:,|$)"
)

# Binary / large data extensions that should never be committed
FORBIDDEN_EXTENSIONS = {".csv", ".xlsx", ".xls", ".npy", ".npz", ".pkl", ".pickle", ".h5", ".hdf5"}

# Maximum file size for staged files (1 MB)
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024

# SWaT sensor column name patterns (to detect in notebook outputs)
SENSOR_COL_PATTERNS = [
    re.compile(r"\bLIT\d{3}\b"),
    re.compile(r"\bFIT\d{3}\b"),
    re.compile(r"\bAIT\d{3}\b"),
    re.compile(r"\bDPIT\d{3}\b"),
    re.compile(r"\bMV\d{3}\b"),
    re.compile(r"\bP\d+_STATE\b"),
    re.compile(r"\b\w+\.Pv\b"),
    re.compile(r"\b\w+\.Status\b"),
    re.compile(r"\b\w+\.Alarm\b"),
    re.compile(r"\b\w+\.Speed\b"),
]


# ---------------------------------------------------------------------------
# Scanning rules
# ---------------------------------------------------------------------------

def check_forbidden_extensions(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Flag staged files with forbidden data extensions."""
    for fpath in staged:
        ext = Path(fpath).suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            result.violations.append(Violation(
                file=fpath,
                rule="forbidden_extension",
                detail=f"Data file with extension '{ext}' should not be committed. "
                       f"Add to .gitignore instead.",
                severity="error",
            ))


def check_file_sizes(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Flag staged files larger than MAX_FILE_SIZE_BYTES."""
    for fpath in staged:
        full_path = repo_root / fpath
        if full_path.exists() and full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            result.violations.append(Violation(
                file=fpath,
                rule="large_file",
                detail=f"File is {size_mb:.2f} MB (limit: {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB). "
                       f"Raw data files must not be committed.",
                severity="error",
            ))


def check_raw_paths_in_source(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Scan source files for hard-coded raw data paths."""
    source_exts = {".py", ".yaml", ".yml", ".toml", ".json", ".md", ".rst", ".txt"}
    for fpath in staged:
        ext = Path(fpath).suffix.lower()
        if ext not in source_exts:
            continue
        full_path = repo_root / fpath
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pattern in RAW_PATH_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                # Allow references in config files and .gitignore
                basename = Path(fpath).name.lower()
                if basename in ("default.yaml", ".gitignore", "config.yaml",
                                "check_data_privacy.py", "readme.md"):
                    continue
                result.violations.append(Violation(
                    file=fpath,
                    rule="raw_data_path",
                    detail=f"Contains raw data path pattern: '{matches[0]}'. "
                           f"Use config references instead of hard-coded paths.",
                    severity="warning",
                ))
                break  # one warning per file


def check_raw_sensor_values(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Detect lines with un-normalised sensor readings (values > 1.0)."""
    source_exts = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
    for fpath in staged:
        ext = Path(fpath).suffix.lower()
        if ext not in source_exts:
            continue
        full_path = repo_root / fpath
        if not full_path.exists():
            continue
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Look for CSV-like lines with many numeric values
            nums = RAW_SENSOR_LINE_RE.findall(line)
            if len(nums) >= 5:
                # Check if any value exceeds 1.0 (not normalised)
                large_vals = [float(n) for n in nums if float(n) > 1.0]
                if len(large_vals) >= 3:
                    result.violations.append(Violation(
                        file=fpath,
                        rule="raw_sensor_values",
                        detail=f"Line {line_no}: contains {len(large_vals)} values > 1.0 "
                               f"(possibly un-normalised sensor data). "
                               f"Max value: {max(large_vals):.4f}",
                        severity="error",
                    ))
                    break  # one per file


def check_swat_timestamps(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Detect SWaT-format timestamps in staged files."""
    text_exts = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".log"}
    for fpath in staged:
        ext = Path(fpath).suffix.lower()
        if ext not in text_exts:
            continue
        # Allow the config and this script to reference the format string
        basename = Path(fpath).name.lower()
        if basename in ("default.yaml", "config.yaml", "check_data_privacy.py"):
            continue
        full_path = repo_root / fpath
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        matches = SWAT_TIMESTAMP_RE.findall(content)
        if len(matches) >= 3:
            result.violations.append(Violation(
                file=fpath,
                rule="raw_timestamps",
                detail=f"Contains {len(matches)} SWaT-format timestamps "
                       f"(e.g. '{matches[0]}'). This may be raw dataset content.",
                severity="error",
            ))


def check_notebook_outputs(staged: List[str], repo_root: Path, result: ScanResult) -> None:
    """Scan Jupyter notebook output cells for raw sensor data."""
    for fpath in staged:
        if not fpath.endswith(".ipynb"):
            continue
        full_path = repo_root / fpath
        if not full_path.exists():
            continue
        try:
            nb = json.loads(full_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, Exception):
            continue
        cells = nb.get("cells", [])
        for cell_idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            outputs = cell.get("outputs", [])
            for output in outputs:
                text_data = ""
                if "text" in output:
                    text_data = "".join(output["text"])
                elif "data" in output:
                    text_data = "".join(output["data"].get("text/plain", []))
                if not text_data:
                    continue
                # Check for sensor column names in output
                sensor_hits = []
                for pat in SENSOR_COL_PATTERNS:
                    if pat.search(text_data):
                        sensor_hits.append(pat.pattern)
                if len(sensor_hits) >= 2:
                    result.violations.append(Violation(
                        file=fpath,
                        rule="notebook_raw_output",
                        detail=f"Cell {cell_idx}: output contains raw sensor column names "
                               f"({', '.join(sensor_hits[:3])}...). "
                               f"Clear outputs before committing.",
                        severity="warning",
                    ))
                # Check for raw timestamp patterns in output
                ts_matches = SWAT_TIMESTAMP_RE.findall(text_data)
                if ts_matches:
                    result.violations.append(Violation(
                        file=fpath,
                        rule="notebook_raw_timestamps",
                        detail=f"Cell {cell_idx}: output contains {len(ts_matches)} "
                               f"SWaT timestamps. Clear outputs before committing.",
                        severity="error",
                    ))
                # Check for raw numeric rows
                lines = text_data.splitlines()
                for line in lines:
                    nums = RAW_SENSOR_LINE_RE.findall(line)
                    if len(nums) >= 5:
                        large_vals = [float(n) for n in nums if float(n) > 1.0]
                        if len(large_vals) >= 3:
                            result.violations.append(Violation(
                                file=fpath,
                                rule="notebook_raw_values",
                                detail=f"Cell {cell_idx}: output contains un-normalised "
                                       f"sensor values (max: {max(large_vals):.4f}). "
                                       f"Clear outputs before committing.",
                                severity="error",
                            ))
                            break


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan(staged_files: Optional[List[str]] = None,
         repo_root: Optional[Path] = None,
         strict: bool = False) -> ScanResult:
    """
    Run all privacy checks on staged files.

    Parameters
    ----------
    staged_files : list of str, optional
        Override list of files to scan (for testing).
    repo_root : Path, optional
        Override repo root (for testing).
    strict : bool
        If True, treat warnings as errors.

    Returns
    -------
    ScanResult
    """
    if repo_root is None:
        repo_root = get_repo_root()
        if repo_root is None:
            # Not in a git repo — try current directory
            repo_root = Path.cwd()

    if staged_files is None:
        staged_files = get_staged_files()

    result = ScanResult()

    # Run all checks
    check_forbidden_extensions(staged_files, repo_root, result)
    check_file_sizes(staged_files, repo_root, result)
    check_raw_paths_in_source(staged_files, repo_root, result)
    check_raw_sensor_values(staged_files, repo_root, result)
    check_swat_timestamps(staged_files, repo_root, result)
    check_notebook_outputs(staged_files, repo_root, result)

    # In strict mode, promote warnings to errors
    if strict:
        for v in result.violations:
            if v.severity == "warning":
                v.severity = "error"

    return result


def print_report(result: ScanResult) -> None:
    """Print a formatted report of scan results."""
    print()
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}  RAKSHAK-ICS  —  Data Privacy Check{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print()

    if not result.violations:
        print(_ok("No privacy violations detected. Safe to commit."))
        print()
        return

    errors = [v for v in result.violations if v.severity == "error"]
    warnings = [v for v in result.violations if v.severity == "warning"]

    if errors:
        print(f"  {_RED}{_BOLD}ERRORS ({len(errors)}):{_RESET}")
        print(f"  {'-' * 50}")
        for v in errors:
            print(f"  {_error(v.file)}")
            print(f"         Rule: {v.rule}")
            print(f"         {v.detail}")
            print()

    if warnings:
        print(f"  {_YELLOW}{_BOLD}WARNINGS ({len(warnings)}):{_RESET}")
        print(f"  {'-' * 50}")
        for v in warnings:
            print(f"  {_warn(v.file)}")
            print(f"         Rule: {v.rule}")
            print(f"         {v.detail}")
            print()

    # Summary
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    if result.has_errors:
        print(_error(
            f"COMMIT BLOCKED — {len(errors)} error(s), {len(warnings)} warning(s)."
        ))
        print(_info("Fix the errors above, then try again."))
        print(_info("If a data file is needed, add it to .gitignore instead."))
    else:
        print(_warn(
            f"{len(warnings)} warning(s) found, but no blocking errors."
        ))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_data_privacy",
        description="RAKSHAK-ICS pre-commit hook: prevents raw ICS data leaks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/check_data_privacy.py              # check staged files
  python scripts/check_data_privacy.py --strict      # warnings become errors
  python scripts/check_data_privacy.py --files a.py  # check specific files
        """,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (recommended for CI).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Manually specify files to check (overrides git staging).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repository root directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output on success.",
    )
    return parser


def main() -> int:
    """Entry point. Returns 0 on success, 1 on violation."""
    parser = build_parser()
    args = parser.parse_args()

    result = scan(
        staged_files=args.files,
        repo_root=args.repo_root,
        strict=args.strict,
    )

    if not args.quiet or result.violations:
        print_report(result)

    return 1 if result.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
