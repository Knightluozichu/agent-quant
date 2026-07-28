"""Tests to ensure no secrets are committed to the repository.

These tests scan tracked files for common secret patterns and ensure
.env files are properly gitignored.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# Patterns that might indicate hardcoded secrets
SECRET_PATTERNS = [
    # API keys and tokens (at least 8 chars after =)
    re.compile(r"""(api[_-]?key|apikey)\s*[=:]\s*["'][^"']{8,}["']""", re.IGNORECASE),
    re.compile(r"""(token|secret|password|passwd)\s*[=:]\s*["'][^"']{8,}["']""", re.IGNORECASE),
    # Tushare token format
    re.compile(r"""tushare[_-]?token\s*[=:]\s*["'][a-f0-9]{32,}["']""", re.IGNORECASE),
    # Private keys
    re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
    # AWS keys
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

# Files/directories to skip
SKIP_PATTERNS = [
    ".env.example",  # Template file with empty values
    "test_secrets.py",  # This test file itself
    ".git/",
    "__pycache__/",
    ".venv/",
    "uv.lock",
]


def get_tracked_files() -> list[Path]:
    """Get list of git-tracked files."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return [Path(f) for f in result.stdout.strip().split("\n") if f]
    except subprocess.CalledProcessError:
        # Not a git repo or git not available
        return []


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped from secret scanning."""
    path_str = str(file_path)
    return any(skip in path_str for skip in SKIP_PATTERNS)


@pytest.mark.unit
class TestSecretsNotCommitted:
    """Ensure no secrets are hardcoded in the codebase."""

    def test_env_file_not_tracked(self) -> None:
        """Verify .env is not tracked by git."""
        tracked = get_tracked_files()
        env_files = [f for f in tracked if f.name == ".env" or f.suffix == ".env"]
        assert not env_files, f".env files should not be tracked: {env_files}"

    def test_env_in_gitignore(self) -> None:
        """Verify .env is in .gitignore."""
        gitignore_path = Path(__file__).parent.parent.parent / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert ".env" in content, ".env should be in .gitignore"

    def test_no_hardcoded_secrets(self) -> None:
        """Scan tracked files for potential hardcoded secrets."""
        repo_root = Path(__file__).parent.parent.parent
        tracked = get_tracked_files()

        violations: list[str] = []

        for file_path in tracked:
            if should_skip_file(file_path):
                continue

            full_path = repo_root / file_path

            # Skip binary files
            if full_path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".whl"}:
                continue

            if not full_path.exists():
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for pattern in SECRET_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    violations.append(f"{file_path}: potential secret pattern found")

        assert not violations, "Potential secrets found:\n" + "\n".join(violations)

    def test_env_example_has_no_real_secrets(self) -> None:
        """Verify .env.example contains only empty or placeholder values."""
        repo_root = Path(__file__).parent.parent.parent
        env_example = repo_root / ".env.example"

        if not env_example.exists():
            pytest.skip(".env.example not found")

        content = env_example.read_text()

        # Check that secret fields are empty
        secret_fields = [
            "JOINQUANT_PASSWORD",
            "TUSHARE_TOKEN",
            "LIVE_TRADING_ACKNOWLEDGEMENT",
        ]

        for field in secret_fields:
            # Find the line with this field
            for line in content.split("\n"):
                if line.startswith(field):
                    # Value should be empty
                    value = line.split("=", 1)[1].strip() if "=" in line else ""
                    assert value == "", f"{field} should be empty in .env.example, got: {value}"
