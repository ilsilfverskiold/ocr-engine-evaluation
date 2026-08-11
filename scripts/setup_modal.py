"""
One-time Modal setup script.
Run this before scripts/deploy_modal.py.

Does:
  1. Verifies Modal CLI is installed and authenticated
  2. Checks for the 'huggingface-token' secret; creates it if missing
  3. Prints next steps
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def check_auth() -> bool:
    result = subprocess.run(
        ["modal", "profile", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Modal is not authenticated.")
        print("\nFix: run  modal setup  (opens browser OAuth, ~30 seconds)")
        return False
    print("[OK] Modal authenticated")
    return True


def check_hf_secret() -> bool:
    result = subprocess.run(
        ["modal", "secret", "list"],
        capture_output=True,
        text=True,
    )
    if "huggingface-token" in result.stdout:
        print("[OK] Modal secret 'huggingface-token' exists")
        return True

    print("\n[MISSING] Modal secret 'huggingface-token' not found.")
    print("This is needed so the Modal containers can download models from HuggingFace.")
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        print(f"Using HF_TOKEN from .env")
    else:
        print("Get your token at: huggingface.co → Settings → Access Tokens (read is enough)")
        token = input("\nPaste your HuggingFace token (hf_...): ").strip()
    if not token:
        print("No token provided. Skipping — re-run this script before deploying.")
        return False

    create = subprocess.run(
        ["modal", "secret", "create", "huggingface-token", f"HF_TOKEN={token}"],
        capture_output=True,
        text=True,
    )
    if create.returncode == 0:
        print("[OK] Secret 'huggingface-token' created")
        return True
    else:
        print(f"Failed to create secret:\n{create.stderr}")
        return False


def main() -> None:
    _load_env()
    print("=== Modal Setup Check ===\n")

    ok = True
    ok = check_auth() and ok
    if not ok:
        sys.exit(1)

    ok = check_hf_secret() and ok

    print()
    if ok:
        print("All checks passed. Next step:")
        print("  python scripts/deploy_modal.py")
        print("\nNote: first deploy takes ~5 min per model (~30 min total).")
        print("Re-deploys are fast once images are cached.")
    else:
        print("Fix the issues above, then re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
