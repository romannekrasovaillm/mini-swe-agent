#!/usr/bin/env python3
"""
Run SWE-bench evaluation WITHOUT containers (LocalEnvironment).

WARNING: This runs code directly on your machine without isolation!
Use only for testing purposes.

This script:
1. Downloads SWE-bench instances
2. Clones repositories locally
3. Applies base commits
4. Runs the agent in LocalEnvironment
5. Collects patches
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml
from datasets import load_dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def setup_repo(instance: dict, work_dir: Path) -> Path:
    """Clone repo and checkout base commit."""
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    repo_dir = work_dir / repo.replace("/", "_")

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print(f"  Cloning {repo} (full clone for old commits)...")
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600  # 10 min timeout for large repos
    )

    print(f"  Checking out {base_commit[:8]}...")
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True
    )

    # Try to install dependencies (best effort)
    print(f"  Installing dependencies...")
    try:
        if (repo_dir / "requirements.txt").exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=repo_dir,
                timeout=120,
                capture_output=True
            )
        if (repo_dir / "setup.py").exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=repo_dir,
                timeout=120,
                capture_output=True
            )
        elif (repo_dir / "pyproject.toml").exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=repo_dir,
                timeout=120,
                capture_output=True
            )
    except Exception as e:
        print(f"  Warning: Failed to install deps: {e}")

    return repo_dir


def get_patch(repo_dir: Path) -> str:
    """Get git diff of changes made."""
    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    return result.stdout


def process_instance(
    instance: dict,
    config: dict,
    model_name: str | None,
    work_dir: Path,
    output_dir: Path,
) -> dict:
    """Process a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    print(f"\n{'='*60}")
    print(f"Processing: {instance_id}")
    print(f"{'='*60}")

    result = {
        "instance_id": instance_id,
        "model_name_or_path": model_name or config.get("model", {}).get("model_name", "unknown"),
        "model_patch": "",
        "exit_status": "error",
        "steps": 0,
        "error": None,
    }

    repo_dir = None
    try:
        # Setup repository
        repo_dir = setup_repo(instance, work_dir)

        # Create model and environment
        model = get_model(model_name, config=config.get("model", {}))

        env_config = config.get("environment", {}).copy()
        env_config["cwd"] = str(repo_dir)
        env_config.pop("executable", None)  # Remove docker/podman setting
        env_config.pop("image", None)

        env = LocalEnvironment(**env_config)

        # Create agent
        agent_config = config.get("agent", {})
        agent = DefaultAgent(model, env, **agent_config)

        # Run agent
        task = instance["problem_statement"]
        print(f"\nRunning agent on task...")
        exit_status, agent_result = agent.run(task)

        # Get patch
        patch = get_patch(repo_dir)

        result["model_patch"] = patch
        result["exit_status"] = exit_status
        result["steps"] = model.n_calls

        print(f"\nCompleted: {instance_id}")
        print(f"  Exit status: {exit_status}")
        print(f"  Steps: {model.n_calls}")
        print(f"  Patch size: {len(patch)} chars")

    except Exception as e:
        print(f"\nError processing {instance_id}: {e}")
        traceback.print_exc()
        result["error"] = str(e)
        result["exit_status"] = type(e).__name__

    finally:
        # Save trajectory
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        traj_file = instance_dir / f"{instance_id}.traj.json"
        traj_file.write_text(json.dumps({
            "instance_id": instance_id,
            "exit_status": result["exit_status"],
            "steps": result["steps"],
            "error": result.get("error"),
        }, indent=2))

        # Cleanup repo
        if repo_dir and repo_dir.exists():
            try:
                shutil.rmtree(repo_dir)
            except Exception:
                pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench locally WITHOUT containers (dangerous!)"
    )
    parser.add_argument(
        "--subset", default="lite",
        help="SWE-bench subset (lite, verified, full)"
    )
    parser.add_argument(
        "--split", default="dev",
        help="Dataset split"
    )
    parser.add_argument(
        "--slice", default="",
        help="Instance slice (e.g., 0:5)"
    )
    parser.add_argument(
        "--filter", default="",
        help="Filter instance IDs by regex"
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Model name override"
    )
    parser.add_argument(
        "-c", "--config",
        default=Path(__file__).parent / "gigachat_swebench_local.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "-o", "--output",
        default=Path(__file__).parent / "gigachat_local_results",
        help="Output directory"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for repos (default: temp dir)"
    )

    args = parser.parse_args()

    print("="*60)
    print("  SWE-bench Local Runner (NO CONTAINERS)")
    print("  WARNING: Code runs directly on your machine!")
    print("="*60)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Creating default local config...")
        # Will be created separately
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text())

    # Load dataset
    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    print(f"\nLoading dataset: {dataset_path}, split: {args.split}")
    instances = list(load_dataset(dataset_path, split=args.split))

    # Filter instances
    if args.filter:
        import re
        instances = [i for i in instances if re.match(args.filter, i["instance_id"])]

    if args.slice:
        parts = [int(x) if x else None for x in args.slice.split(":")]
        instances = instances[slice(*parts)]

    print(f"Running on {len(instances)} instances")

    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="swebench_local_"))

    print(f"Output: {output_dir}")
    print(f"Work dir: {work_dir}")

    # Process instances
    results = []
    preds = {}

    for i, instance in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}]", end="")

        result = process_instance(
            instance,
            config,
            args.model,
            work_dir,
            output_dir
        )

        results.append(result)
        preds[result["instance_id"]] = {
            "model_name_or_path": result["model_name_or_path"],
            "instance_id": result["instance_id"],
            "model_patch": result["model_patch"],
        }

        # Save predictions incrementally
        (output_dir / "preds.json").write_text(json.dumps(preds, indent=2))

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    completed = sum(1 for r in results if r["exit_status"] == "done")
    with_patch = sum(1 for r in results if r["model_patch"])

    print(f"Total: {len(results)}")
    print(f"Completed: {completed} ({completed/len(results)*100:.1f}%)")
    print(f"With patch: {with_patch} ({with_patch/len(results)*100:.1f}%)")
    print(f"\nResults saved to: {output_dir}")

    # Cleanup work dir if temp
    if not args.work_dir:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
