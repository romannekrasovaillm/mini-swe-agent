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
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
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

# Setup logging
def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    log_file = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger('swebench')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def truncate_text(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text) - max_len} chars truncated]"


class LoggingAgent(DefaultAgent):
    """Agent with logging of model responses."""

    def __init__(self, *args, logger=None, instance_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger or logging.getLogger('swebench')
        self.instance_id = instance_id
        self.step_count = 0

    def step(self) -> dict:
        """Override step to log model responses."""
        self.step_count += 1
        self.logger.info(f"[{self.instance_id}] Step {self.step_count} starting...")
        return super().step()

    def query(self) -> dict:
        """Override query to log model response."""
        response = super().query()
        content = response.get("content", "")

        # Log to console
        print(f"\n{'='*60}")
        print(f"[{self.instance_id}] STEP {self.step_count} - MODEL RESPONSE:")
        print("-" * 60)
        print(content[:2000] if len(content) > 2000 else content)
        if len(content) > 2000:
            print(f"... [{len(content) - 2000} chars truncated]")
        print("=" * 60)

        # Log to file
        self.logger.info(f"[{self.instance_id}] Step {self.step_count} MODEL RESPONSE ({len(content)} chars)")
        self.logger.debug(f"[{self.instance_id}] Full response:\n{content}")

        return response

    def parse_action(self, response: dict) -> dict:
        """Override parse_action to log extracted command."""
        import re
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)

        if len(actions) == 1:
            cmd = actions[0].strip()
            # Log command
            print(f"\n>>> EXTRACTED COMMAND:")
            print("-" * 40)
            print(cmd[:500] if len(cmd) > 500 else cmd)
            if len(cmd) > 500:
                print(f"... [{len(cmd) - 500} chars truncated]")
            print("-" * 40)
            self.logger.info(f"[{self.instance_id}] Extracted command: {cmd[:200]}...")
            return {"action": cmd, **response}

        # No valid action found
        print(f"\n[!] FORMAT ERROR: Found {len(actions)} bash blocks (expected 1)")
        if actions:
            for i, a in enumerate(actions):
                print(f"  Block {i+1}: {a[:100]}...")
        self.logger.warning(f"[{self.instance_id}] Format error: {len(actions)} bash blocks")

        from minisweagent.agents.default import FormatError
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        """Override execute_action to log output."""
        import subprocess
        try:
            output = self.env.execute(action["action"])

            # Log output
            out_text = output.get("output", "")
            print(f"\n<<< COMMAND OUTPUT ({len(out_text)} chars):")
            print("-" * 40)
            print(out_text[:1000] if len(out_text) > 1000 else out_text)
            if len(out_text) > 1000:
                print(f"... [{len(out_text) - 1000} chars truncated]")
            print("-" * 40)
            self.logger.info(f"[{self.instance_id}] Command output: {len(out_text)} chars")
            self.logger.debug(f"[{self.instance_id}] Full output:\n{out_text}")

        except subprocess.TimeoutExpired as e:
            out = e.output.decode("utf-8", errors="replace") if e.output else ""
            print(f"\n[!] COMMAND TIMEOUT")
            self.logger.warning(f"[{self.instance_id}] Command timeout")
            from minisweagent.agents.default import ExecutionTimeoutError
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=out)
            )
        except TimeoutError:
            print(f"\n[!] COMMAND TIMEOUT")
            self.logger.warning(f"[{self.instance_id}] Command timeout")
            from minisweagent.agents.default import ExecutionTimeoutError
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output="")
            )

        self.has_finished(output)
        return output


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
    logger=None,
) -> dict:
    """Process a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    logger = logger or logging.getLogger('swebench')

    print(f"\n{'='*60}")
    print(f"Processing: {instance_id}")
    print(f"{'='*60}")
    logger.info(f"Starting instance: {instance_id}")

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

        # Get and log task
        task = instance["problem_statement"]

        # Log task preview
        print(f"\n{'='*60}")
        print("TASK (problem_statement):")
        print("-" * 60)
        print(truncate_text(task, 800))
        print("=" * 60)

        logger.info(f"[{instance_id}] TASK preview: {truncate_text(task, 500)}")
        logger.debug(f"[{instance_id}] Full task:\n{task}")

        # Create agent with logging
        agent_config = config.get("agent", {})
        agent = LoggingAgent(
            model, env,
            logger=logger,
            instance_id=instance_id,
            **agent_config
        )

        # Run agent
        print(f"\nRunning agent on task...")
        logger.info(f"[{instance_id}] Agent starting...")
        exit_status, agent_result = agent.run(task)

        # Get patch
        patch = get_patch(repo_dir)

        result["model_patch"] = patch
        result["exit_status"] = exit_status
        result["steps"] = model.n_calls

        # Log completion
        print(f"\n{'='*60}")
        print(f"Completed: {instance_id}")
        print(f"  Exit status: {exit_status}")
        print(f"  Steps: {model.n_calls}")
        print(f"  Patch size: {len(patch)} chars")

        if patch:
            print(f"\nGENERATED PATCH:")
            print("-" * 60)
            print(truncate_text(patch, 1000))
            print("=" * 60)
            logger.info(f"[{instance_id}] Generated patch ({len(patch)} chars):")
            logger.debug(f"[{instance_id}] Full patch:\n{patch}")
        else:
            print(f"\n[WARNING] No patch generated!")
            logger.warning(f"[{instance_id}] No patch generated")

        logger.info(f"[{instance_id}] Completed: status={exit_status}, steps={model.n_calls}, patch_size={len(patch)}")

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

    # Setup logging
    logger, log_file = setup_logging(output_dir)
    logger.info(f"Starting SWE-bench run: {len(instances)} instances")
    logger.info(f"Subset: {args.subset}, Split: {args.split}")
    print(f"Log file: {log_file}")

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
        logger.info(f"Processing instance {i+1}/{len(instances)}: {instance['instance_id']}")

        result = process_instance(
            instance,
            config,
            args.model,
            work_dir,
            output_dir,
            logger=logger
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
    print(f"Log file: {log_file}")

    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Completed: {completed} ({completed/len(results)*100:.1f}%)")
    logger.info(f"With patch: {with_patch} ({with_patch/len(results)*100:.1f}%)")
    logger.info("="*60)

    # Cleanup work dir if temp
    if not args.work_dir:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
