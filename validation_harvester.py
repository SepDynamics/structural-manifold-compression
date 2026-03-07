import subprocess
import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt

# Add SEP-mcp to path
sys.path.insert(0, str(Path("SEP-mcp").resolve()))
from mcp_server import ingest_repo, batch_chaos_scan

REPO_DIR = "langchain_test"
OLD_TAG = "v0.0.300"


def get_git_churn(repo_path, old_tag):
    """
    Simulates checking the actual github history for what files got rewritten.
    We measure churn (lines changed) from old_tag to HEAD.
    """
    print(
        f"Fetching full git history for {repo_path} to calculate ground-truth churn..."
    )
    # First, fetch everything so we have origin/master
    # But for this test, we cloned depth=1, so we need to fetch the rest
    subprocess.run(["git", "fetch", "--unshallow"], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "fetch", "origin", "master"], cwd=repo_path, capture_output=True
    )

    cmd = ["git", "log", f"{old_tag}..FETCH_HEAD", "--numstat", "--pretty=format:"]
    res = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

    churn_dict = {}
    for line in res.stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
                file_path = parts[2]
                if file_path.endswith(".py"):
                    churn_dict[file_path] = (
                        churn_dict.get(file_path, 0) + added + deleted
                    )
            except ValueError:
                pass

    return churn_dict


def run_radon_complexity(repo_path):
    """Runs standard cyclomatic complexity analyzer for comparison."""
    print("Running Radon Cyclomatic Complexity...")
    radon_path = str((Path(__file__).parent / ".venv" / "bin" / "radon").resolve())
    cmd = [radon_path, "cc", "-j", "libs/langchain/langchain"]  # Targeting the core lib
    res = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    try:
        radon_data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return {}

    complexity_dict = {}
    for filepath, blocks in radon_data.items():
        if not blocks:
            continue
        # Average complexity of all blocks in the file
        avg_cc = sum(b.get("complexity", 1) for b in blocks) / len(blocks)
        complexity_dict[filepath] = avg_cc

    return complexity_dict


def main():
    print("--- 🔬 Starting True Validation Empirical Study ---")
    repo = Path(REPO_DIR)

    # 1. Manifold Ingest & Scan
    print(f"\n1. Ingesting {REPO_DIR} into Manifold Engine...")
    ingest_repo(
        root_dir=f"{REPO_DIR}/libs/langchain/langchain",
        compute_chaos=True,
        clear_first=True,
        max_bytes_per_file=512000,
    )

    print("Extracting Manifold Chaos Scores...")
    # We want scores for ALL files to do a distribution threshold test
    scan_results = batch_chaos_scan(pattern="*.py", max_files=1000)

    # Parse the scan report block to extract the scores
    manifold_scores = {}
    for line in scan_results.split("\n"):
        if "|" in line and "HIGH" in line or "LOW" in line or "OSCILLATION" in line:
            parts = line.split("|")
            try:
                score = float(parts[0].split("]")[1].strip())
                filepath = parts[1].strip().replace(f"{REPO_DIR}/", "")
                manifold_scores[filepath] = score
            except:
                pass

    # 2. Get Radon Complexity Baseline
    radon_scores = run_radon_complexity(REPO_DIR)

    # 3. Get Ground Truth Churn
    churn_data = get_git_churn(REPO_DIR, OLD_TAG)

    # 4. Integrate Data
    records = []

    for fp, m_score in manifold_scores.items():
        r_score = radon_scores.get(fp, 0)
        churn = churn_data.get(fp, 0)

        # Ground truth: Was this file architecturally problematic?
        # Define "ejected/rewritten" as excessive churn (> 500 lines modified since the tag)
        is_ejected = 1 if churn > 500 else 0

        records.append(
            {
                "file": fp,
                "chaos_score": m_score,
                "cyclomatic_complexity": r_score,
                "total_churn": churn,
                "is_ejected": is_ejected,
            }
        )

    df = pd.DataFrame(records)
    print(f"\nConstructed unified DataFrame with {len(df)} files.")

    # 5. Threshold Optimization & Output
    if df["is_ejected"].sum() == 0:
        print(
            "ERROR: No files met the ejection/churn threshold. Ground truth is empty."
        )
        return

    # Analyze Manifold
    fpr, tpr, thresholds = roc_curve(df["is_ejected"], df["chaos_score"])
    manifold_auc = auc(fpr, tpr)

    # Find best f1 threshold
    best_thresh = 0
    best_f1 = 0
    for t in thresholds:
        preds = (df["chaos_score"] >= t).astype(int)
        f1 = f1_score(df["is_ejected"], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\n📊 Manifold Engine Predictive Performance:")
    print(f"  - ROC AUC: {manifold_auc:.3f}")
    print(f"  - Optimal Chaos Threshold: {best_thresh:.3f} (F1: {best_f1:.3f})")

    # Analyze Radon
    fpr_r, tpr_r, thresh_r = roc_curve(df["is_ejected"], df["cyclomatic_complexity"])
    radon_auc = auc(fpr_r, tpr_r)
    print(f"\n📊 Baseline (Radon Cyclomatic Complexity) Performance:")
    print(f"  - ROC AUC: {radon_auc:.3f}")

    # Generate CSV artifact
    df.to_csv("SEP-mcp/reports/langchain_validation_data.csv", index=False)
    print("\nData exported to SEP-mcp/reports/langchain_validation_data.csv")


if __name__ == "__main__":
    main()
