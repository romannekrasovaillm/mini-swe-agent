#!/usr/bin/env python3
"""
Comprehensive SWE-bench results analyzer for GigaChat baseline evaluation.

Generates detailed analytics including:
- Overall success/failure rates
- Per-repository breakdown
- Difficulty analysis
- Step distribution
- Error categorization
- Comparison with baselines
- Trajectory analysis
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class InstanceResult:
    """Result for a single SWE-bench instance."""
    instance_id: str
    exit_status: str
    model_patch: str
    steps: int = 0
    cost: float = 0.0
    has_patch: bool = False
    repo: str = ""
    error_type: str = ""
    trajectory_file: Path | None = None


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    total_instances: int = 0
    completed: int = 0
    failed: int = 0
    with_patch: int = 0
    empty_patch: int = 0

    # Detailed breakdowns
    exit_status_counts: dict = field(default_factory=dict)
    repo_stats: dict = field(default_factory=dict)
    step_distribution: list = field(default_factory=list)
    error_categories: dict = field(default_factory=dict)

    # Performance metrics
    avg_steps: float = 0.0
    median_steps: float = 0.0
    total_cost: float = 0.0
    avg_cost: float = 0.0

    # Instance lists
    successful_instances: list = field(default_factory=list)
    failed_instances: list = field(default_factory=list)


def parse_instance_id(instance_id: str) -> dict:
    """Parse instance ID to extract repository and issue info."""
    # Format: repo__org__issue-NUMBER
    parts = instance_id.split("__")
    if len(parts) >= 2:
        return {
            "repo": f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0],
            "issue": parts[-1] if len(parts) > 2 else "",
        }
    return {"repo": instance_id, "issue": ""}


def load_predictions(preds_file: Path) -> dict:
    """Load predictions from preds.json."""
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_file}")
    return json.loads(preds_file.read_text())


def load_exit_statuses(output_dir: Path) -> dict:
    """Load all exit status files from output directory."""
    statuses = {}
    for status_file in output_dir.glob("exit_statuses_*.yaml"):
        try:
            data = yaml.safe_load(status_file.read_text())
            if data:
                statuses.update(data)
        except Exception as e:
            print(f"Warning: Failed to load {status_file}: {e}")
    return statuses


def load_trajectory(traj_file: Path) -> dict | None:
    """Load trajectory file for an instance."""
    if not traj_file.exists():
        return None
    try:
        return json.loads(traj_file.read_text())
    except Exception:
        return None


def analyze_trajectory(trajectory: dict) -> dict:
    """Analyze a trajectory for detailed metrics."""
    analysis = {
        "steps": 0,
        "cost": 0.0,
        "commands": [],
        "errors": [],
        "files_modified": set(),
        "tools_used": Counter(),
    }

    if not trajectory:
        return analysis

    history = trajectory.get("history", [])
    analysis["steps"] = len(history)

    # Analyze each step
    for step in history:
        if isinstance(step, dict):
            # Extract command if present
            action = step.get("action", "")
            if action:
                analysis["commands"].append(action)

                # Count tool usage
                if "git " in action:
                    analysis["tools_used"]["git"] += 1
                if "sed " in action:
                    analysis["tools_used"]["sed"] += 1
                if "grep " in action:
                    analysis["tools_used"]["grep"] += 1
                if "find " in action:
                    analysis["tools_used"]["find"] += 1
                if "cat " in action:
                    analysis["tools_used"]["cat"] += 1
                if "python " in action:
                    analysis["tools_used"]["python"] += 1

            # Check for errors in output
            output = step.get("observation", "")
            if "error" in output.lower() or "traceback" in output.lower():
                analysis["errors"].append(output[:200])

    # Get cost if available
    analysis["cost"] = trajectory.get("cost", 0.0)

    return analysis


def categorize_error(exit_status: str, trajectory: dict | None) -> str:
    """Categorize the type of error/failure."""
    if exit_status == "done":
        return "success"

    exit_lower = exit_status.lower()

    if "timeout" in exit_lower:
        return "timeout"
    if "context" in exit_lower or "token" in exit_lower:
        return "context_exceeded"
    if "format" in exit_lower or "parse" in exit_lower:
        return "format_error"
    if "api" in exit_lower or "rate" in exit_lower:
        return "api_error"
    if "docker" in exit_lower or "container" in exit_lower:
        return "environment_error"

    # Check trajectory for more context
    if trajectory:
        history = trajectory.get("history", [])
        if len(history) == 0:
            return "no_action"
        if len(history) >= 250:
            return "step_limit"

    return "other"


def collect_results(output_dir: Path) -> list[InstanceResult]:
    """Collect all results from output directory."""
    results = []

    preds_file = output_dir / "preds.json"
    predictions = load_predictions(preds_file)
    exit_statuses = load_exit_statuses(output_dir)

    for instance_id, pred_data in predictions.items():
        model_patch = pred_data.get("model_patch", "")
        exit_status = exit_statuses.get(instance_id, "unknown")

        # Parse instance ID
        parsed = parse_instance_id(instance_id)

        # Load trajectory
        traj_file = output_dir / instance_id / f"{instance_id}.traj.json"
        trajectory = load_trajectory(traj_file)
        traj_analysis = analyze_trajectory(trajectory)

        result = InstanceResult(
            instance_id=instance_id,
            exit_status=str(exit_status),
            model_patch=model_patch,
            steps=traj_analysis["steps"],
            cost=traj_analysis["cost"],
            has_patch=bool(model_patch.strip()),
            repo=parsed["repo"],
            error_type=categorize_error(str(exit_status), trajectory),
            trajectory_file=traj_file if traj_file.exists() else None,
        )
        results.append(result)

    return results


def generate_report(results: list[InstanceResult]) -> AnalysisReport:
    """Generate comprehensive analysis report."""
    report = AnalysisReport()
    report.total_instances = len(results)

    if not results:
        return report

    # Basic counts
    report.completed = sum(1 for r in results if r.exit_status == "done")
    report.failed = report.total_instances - report.completed
    report.with_patch = sum(1 for r in results if r.has_patch)
    report.empty_patch = report.total_instances - report.with_patch

    # Exit status breakdown
    report.exit_status_counts = Counter(r.exit_status for r in results)

    # Error categories
    report.error_categories = Counter(r.error_type for r in results)

    # Repository stats
    repo_results = defaultdict(list)
    for r in results:
        repo_results[r.repo].append(r)

    for repo, repo_res in repo_results.items():
        completed = sum(1 for r in repo_res if r.exit_status == "done")
        with_patch = sum(1 for r in repo_res if r.has_patch)
        report.repo_stats[repo] = {
            "total": len(repo_res),
            "completed": completed,
            "with_patch": with_patch,
            "completion_rate": completed / len(repo_res) * 100 if repo_res else 0,
            "patch_rate": with_patch / len(repo_res) * 100 if repo_res else 0,
        }

    # Step distribution
    report.step_distribution = [r.steps for r in results]
    if report.step_distribution:
        report.avg_steps = sum(report.step_distribution) / len(report.step_distribution)
        sorted_steps = sorted(report.step_distribution)
        mid = len(sorted_steps) // 2
        report.median_steps = sorted_steps[mid] if len(sorted_steps) % 2 else \
            (sorted_steps[mid-1] + sorted_steps[mid]) / 2

    # Cost metrics
    costs = [r.cost for r in results]
    report.total_cost = sum(costs)
    report.avg_cost = report.total_cost / len(costs) if costs else 0

    # Instance lists
    report.successful_instances = [r.instance_id for r in results if r.exit_status == "done"]
    report.failed_instances = [r.instance_id for r in results if r.exit_status != "done"]

    return report


def print_report(report: AnalysisReport, output_dir: Path):
    """Print formatted analysis report."""
    print("\n" + "=" * 80)
    print("  GigaChat3-10B-A1.8B SWE-bench Baseline Analysis Report")
    print("=" * 80)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results directory: {output_dir}")
    print("=" * 80)

    # Overall Summary
    print("\n" + "-" * 40)
    print("  OVERALL SUMMARY")
    print("-" * 40)
    print(f"  Total instances evaluated:  {report.total_instances}")
    print(f"  Completed (exit=done):      {report.completed} ({report.completed/report.total_instances*100:.1f}%)" if report.total_instances else "  Completed: 0")
    print(f"  With non-empty patch:       {report.with_patch} ({report.with_patch/report.total_instances*100:.1f}%)" if report.total_instances else "  With patch: 0")
    print(f"  Failed/Error:               {report.failed} ({report.failed/report.total_instances*100:.1f}%)" if report.total_instances else "  Failed: 0")

    # Step Statistics
    print("\n" + "-" * 40)
    print("  STEP STATISTICS")
    print("-" * 40)
    print(f"  Average steps per instance: {report.avg_steps:.1f}")
    print(f"  Median steps:               {report.median_steps:.1f}")
    if report.step_distribution:
        print(f"  Min steps:                  {min(report.step_distribution)}")
        print(f"  Max steps:                  {max(report.step_distribution)}")

    # Cost Statistics (for local model, should be 0)
    print("\n" + "-" * 40)
    print("  COST STATISTICS")
    print("-" * 40)
    print(f"  Total cost:                 ${report.total_cost:.4f}")
    print(f"  Average cost per instance:  ${report.avg_cost:.4f}")

    # Exit Status Breakdown
    print("\n" + "-" * 40)
    print("  EXIT STATUS BREAKDOWN")
    print("-" * 40)
    for status, count in sorted(report.exit_status_counts.items(), key=lambda x: -x[1]):
        pct = count / report.total_instances * 100 if report.total_instances else 0
        print(f"  {status:30s} {count:5d} ({pct:5.1f}%)")

    # Error Category Breakdown
    print("\n" + "-" * 40)
    print("  ERROR CATEGORY BREAKDOWN")
    print("-" * 40)
    for category, count in sorted(report.error_categories.items(), key=lambda x: -x[1]):
        pct = count / report.total_instances * 100 if report.total_instances else 0
        print(f"  {category:30s} {count:5d} ({pct:5.1f}%)")

    # Repository Breakdown
    print("\n" + "-" * 40)
    print("  REPOSITORY BREAKDOWN")
    print("-" * 40)
    print(f"  {'Repository':<35s} {'Total':>6s} {'Done':>6s} {'Patch':>6s} {'Done%':>7s}")
    print("  " + "-" * 60)

    for repo, stats in sorted(report.repo_stats.items(), key=lambda x: -x[1]["total"]):
        print(f"  {repo:<35s} {stats['total']:>6d} {stats['completed']:>6d} "
              f"{stats['with_patch']:>6d} {stats['completion_rate']:>6.1f}%")

    # Baseline Comparison Reference
    print("\n" + "-" * 40)
    print("  BASELINE COMPARISON REFERENCE")
    print("-" * 40)
    print("  Reference scores for SWE-bench Lite (300 instances):")
    print("  --------------------------------------------------------")
    print("  Model                          Resolved%")
    print("  --------------------------------------------------------")
    print("  Claude 3.5 Sonnet (Anthropic)     49.0%")
    print("  GPT-4o (OpenAI)                   38.0%")
    print("  Claude 3 Opus                     22.0%")
    print("  Llama 3.1 405B                    14.0%")
    print("  DeepSeek-V2                       12.0%")
    print("  Mixtral 8x22B                      4.3%")
    print("  --------------------------------------------------------")
    print(f"  GigaChat3-10B (this run):        ~{report.with_patch/report.total_instances*100:.1f}% (patch generated)" if report.total_instances else "  GigaChat3-10B: N/A")
    print("  Note: Actual resolve rate requires running SWE-bench evaluation")

    # Successful instances sample
    if report.successful_instances:
        print("\n" + "-" * 40)
        print("  SAMPLE SUCCESSFUL INSTANCES (up to 10)")
        print("-" * 40)
        for instance_id in report.successful_instances[:10]:
            print(f"  - {instance_id}")

    # Failed instances sample
    if report.failed_instances:
        print("\n" + "-" * 40)
        print("  SAMPLE FAILED INSTANCES (up to 10)")
        print("-" * 40)
        for instance_id in report.failed_instances[:10]:
            print(f"  - {instance_id}")

    print("\n" + "=" * 80)
    print("  END OF REPORT")
    print("=" * 80 + "\n")


def generate_plots(report: AnalysisReport, output_dir: Path):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Exit status pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = list(report.exit_status_counts.keys())
    sizes = list(report.exit_status_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Exit Status Distribution')
    plt.savefig(plots_dir / 'exit_status_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Step distribution histogram
    if report.step_distribution:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(report.step_distribution, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(report.avg_steps, color='red', linestyle='--', label=f'Mean: {report.avg_steps:.1f}')
        ax.axvline(report.median_steps, color='green', linestyle='--', label=f'Median: {report.median_steps:.1f}')
        ax.set_xlabel('Number of Steps')
        ax.set_ylabel('Frequency')
        ax.set_title('Step Distribution Across Instances')
        ax.legend()
        plt.savefig(plots_dir / 'step_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Repository performance bar chart
    if report.repo_stats:
        repos = list(report.repo_stats.keys())[:15]  # Top 15 repos
        completion_rates = [report.repo_stats[r]['completion_rate'] for r in repos]

        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(repos, completion_rates, color='steelblue', edgecolor='black')
        ax.set_xlabel('Completion Rate (%)')
        ax.set_title('Completion Rate by Repository (Top 15)')
        ax.set_xlim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, completion_rates):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(plots_dir / 'repo_performance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Error category breakdown
    if report.error_categories:
        fig, ax = plt.subplots(figsize=(10, 8))
        categories = list(report.error_categories.keys())
        counts = list(report.error_categories.values())

        colors = plt.cm.Set3(range(len(categories)))
        ax.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Error Category Distribution')
        plt.savefig(plots_dir / 'error_categories.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to: {plots_dir}")


def export_to_csv(results: list[InstanceResult], output_dir: Path):
    """Export results to CSV for further analysis."""
    if not HAS_PANDAS:
        print("Warning: pandas not installed, skipping CSV export")
        return

    data = []
    for r in results:
        data.append({
            'instance_id': r.instance_id,
            'repository': r.repo,
            'exit_status': r.exit_status,
            'error_type': r.error_type,
            'steps': r.steps,
            'cost': r.cost,
            'has_patch': r.has_patch,
            'patch_length': len(r.model_patch) if r.model_patch else 0,
        })

    df = pd.DataFrame(data)
    csv_path = output_dir / 'results_detailed.csv'
    df.to_csv(csv_path, index=False)
    print(f"Detailed results exported to: {csv_path}")

    # Summary statistics
    summary_path = output_dir / 'results_summary.csv'
    summary = df.groupby('repository').agg({
        'instance_id': 'count',
        'has_patch': 'sum',
        'steps': ['mean', 'median', 'std'],
    }).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(summary_path)
    print(f"Summary statistics exported to: {summary_path}")


def save_report_json(report: AnalysisReport, output_dir: Path):
    """Save report as JSON for programmatic access."""
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'total_instances': report.total_instances,
        'completed': report.completed,
        'failed': report.failed,
        'with_patch': report.with_patch,
        'completion_rate': report.completed / report.total_instances * 100 if report.total_instances else 0,
        'patch_rate': report.with_patch / report.total_instances * 100 if report.total_instances else 0,
        'avg_steps': report.avg_steps,
        'median_steps': report.median_steps,
        'total_cost': report.total_cost,
        'exit_status_counts': dict(report.exit_status_counts),
        'error_categories': dict(report.error_categories),
        'repo_stats': report.repo_stats,
        'successful_instances': report.successful_instances,
        'failed_instances': report.failed_instances,
    }

    json_path = output_dir / 'analysis_report.json'
    json_path.write_text(json.dumps(report_data, indent=2))
    print(f"Report JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GigaChat SWE-bench evaluation results"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save report as JSON"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all outputs (plots, CSV, JSON)"
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        sys.exit(1)

    print(f"Analyzing results in: {args.output_dir}")

    # Collect results
    try:
        results = collect_results(args.output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not results:
        print("No results found to analyze")
        sys.exit(1)

    print(f"Found {len(results)} instances to analyze")

    # Generate report
    report = generate_report(results)

    # Print text report
    print_report(report, args.output_dir)

    # Generate additional outputs
    if args.all or args.plots:
        generate_plots(report, args.output_dir)

    if args.all or args.csv:
        export_to_csv(results, args.output_dir)

    if args.all or args.json:
        save_report_json(report, args.output_dir)


if __name__ == "__main__":
    main()
