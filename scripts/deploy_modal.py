"""Deploy selected benchmark Modal apps and optionally run smoke tests."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engines import ENGINES  # noqa: E402

# (engine key, app file path, modal.Cls name) — derived from the single registry
APPS = [
    (name, ROOT / spec.modal_app_file, spec.modal_class)
    for name, spec in ENGINES.items()
    if spec.kind == "modal"
]

# Deployed Modal app name per engine key (e.g. "ocr-bench-glm-ocr")
APP_NAMES = {
    name: spec.modal_app for name, spec in ENGINES.items() if spec.kind == "modal"
}

BATCH_APPS = {name for name, spec in ENGINES.items() if spec.kind == "modal" and spec.batch_size > 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy and smoke-test Modal OCR apps")
    parser.add_argument(
        "--apps",
        default=",".join(name for name, _, _ in APPS),
        help="Comma-separated Modal app keys to deploy/check",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Deploy selected apps but do not run smoke tests",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Skip deploy and only run smoke tests for selected apps",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel app deploy/smoke workers (default: all selected apps)",
    )
    return parser.parse_args()


def selected_apps(app_names: str) -> list[tuple[str, Path, str]]:
    selected = [name.strip() for name in app_names.split(",") if name.strip()]
    app_map = {name: (name, app_path, class_name) for name, app_path, class_name in APPS}
    unknown = [name for name in selected if name not in app_map]
    if unknown:
        known = ", ".join(app_map)
        raise SystemExit(f"Unknown Modal app(s): {', '.join(unknown)}. Known: {known}")
    return [app_map[name] for name in selected]


def deploy_app(name: str, app_path: Path) -> bool:
    print(f"\n  [{name}] Deploying {app_path.name} ...")
    print(f"  (First deploy: ~5 min. Re-deploy: ~30s)", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", str(app_path)],
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED after {elapsed:.0f}s")
        return False
    print(f"  OK  ({elapsed:.0f}s)")
    return True


def smoke_test(name: str, app_name: str, class_name: str) -> bool:
    """Send a small PNG to verify the deployed app is live."""
    try:
        import modal
        from PIL import Image

        from PIL import ImageDraw

        white = Image.new("RGB", (160, 64), (255, 255, 255))
        draw = ImageDraw.Draw(white)
        draw.text((12, 20), "TEST 123", fill=(0, 0, 0))
        buf = io.BytesIO()
        white.save(buf, "PNG")
        ping_bytes = buf.getvalue()

        cls = modal.Cls.from_name(app_name, class_name)
        if name in BATCH_APPS:
            response = cls().run_ocr_batch.remote([
                {
                    "document_path": "smoke.png",
                    "bytes": ping_bytes,
                    "mime_type": "image/png",
                }
            ])
            if not isinstance(response, dict):
                raise RuntimeError(f"Unexpected batch smoke response: {response!r}")
            results = response.get("results")
            if not isinstance(results, list) or not results:
                raise RuntimeError(f"Unexpected batch smoke results: {response!r}")
            if not response.get("batch_gpu_seconds", 0) > 0:
                raise RuntimeError(f"Missing batch GPU timing: {response!r}")
            _ = results[0].get("raw_text", "")
        else:
            response = cls().run_ocr.remote(ping_bytes, "image/png")
            _ = response.get("raw_text", "")
        print(f"  [{name}] smoke test OK")
        return True
    except Exception as exc:
        print(f"  [{name}] smoke test FAILED: {exc}")
        return False


def main() -> None:
    args = parse_args()
    apps = selected_apps(args.apps)
    workers = args.workers if args.workers > 0 else len(apps)

    print("=== Modal OCR App Deployment ===")
    print(f"Selected apps: {', '.join(name for name, _, _ in apps)}\n")

    failed_deploy = []
    if not args.smoke_only:
        print(f"Deploying {len(apps)} app(s) with {workers} parallel workers...\n")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(deploy_app, name, app_path): name
                for name, app_path, _ in apps
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    if not future.result():
                        failed_deploy.append(name)
                except Exception as exc:
                    print(f"  [{name}] deploy worker FAILED: {exc}")
                    failed_deploy.append(name)

        if failed_deploy:
            print(f"\nDeploy failures: {failed_deploy}")
            print("Fix the errors above and re-run.")
            sys.exit(1)

        if args.skip_smoke:
            print(f"\nSelected app(s) deployed. Smoke tests skipped.")
            return

    print(
        f"\nRunning smoke tests for {len(apps)} app(s) "
        f"with {workers} parallel workers...\n"
    )

    # Smoke test phase — note: cold start may take 30–90s per model
    print("(Cold-start note: first call per model may take 30–90s to spin up GPU container)")
    failed_smoke = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for name, _, class_name in apps:
            app_name = APP_NAMES[name]
            futures[
                pool.submit(smoke_test, name, app_name, class_name)
            ] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                if not future.result():
                    failed_smoke.append(name)
            except Exception as exc:
                print(f"  [{name}] smoke worker FAILED: {exc}")
                failed_smoke.append(name)

    print(f"\n{'='*40}")
    if failed_smoke:
        print(f"Smoke test failures: {failed_smoke}")
        print("The apps are deployed but may need a minute to warm up. Try --check again.")
        sys.exit(1)
    else:
        print("All apps deployed and responding.")
        print("\nNext step:")
        print("  python run_benchmark.py --estimate")
        print("  python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge")


if __name__ == "__main__":
    main()
