"""Compare results.csv (missing rows only) against ground_truth.csv and report
TP/FP/FN, precision, and recall per video and overall."""
import argparse
import csv
from collections import defaultdict


def read_ground_truth(path: str) -> dict:
    gt = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["video"]].add(row["missing_class"].strip())
    return gt


def read_predicted_missing(path: str, use_aliases: bool) -> dict:
    """Per video, one set of labels per reported object: just its class, or
    its class plus every alias label the detector also gave it."""
    pred = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "missing":
                continue
            labels = {row["class"].strip()}
            if use_aliases and row.get("aliases"):
                labels |= {a.strip() for a in row["aliases"].split(";") if a.strip()}
            pred[row["video"]].append(labels)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate missing-object results against ground truth.")
    parser.add_argument("--ground-truth", type=str, default="ground_truth.csv")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--use-aliases", action="store_true",
                        help="Count a reported object as correct if its class OR any of its alias labels "
                             "matches the ground truth (open-vocabulary models often mislabel the right object)")
    args = parser.parse_args()

    gt = read_ground_truth(args.ground_truth)
    pred = read_predicted_missing(args.results, args.use_aliases)

    videos = sorted(set(gt) | set(pred))
    total_tp = total_fp = total_fn = 0

    header = f"{'video':<15}{'TP':>4}{'FP':>4}{'FN':>4}{'precision':>11}{'recall':>9}"
    print(header)
    print("-" * len(header))
    for video in videos:
        gt_classes = gt.get(video, set())
        objects = pred.get(video, [])

        # Class granularity: several reported objects with the same label (or
        # sharing a label via aliases) count once, like a single prediction.
        hit, wrong = set(), set()
        for labels in objects:
            if labels & gt_classes:
                hit |= labels & gt_classes
            else:
                wrong.add("/".join(sorted(labels)))

        tp, fp, fn = len(hit), len(wrong), len(gt_classes - hit)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{video:<15}{tp:>4}{fp:>4}{fn:>4}{precision:>11.2f}{recall:>9.2f}")

        for cls in sorted(wrong):
            print(f"    FP: reported '{cls}' missing but it wasn't")
        for cls in sorted(gt_classes - hit):
            print(f"    FN: '{cls}' was missing but not reported")

    print("-" * len(header))
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else float("nan")
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else float("nan")
    print(f"{'overall':<15}{total_tp:>4}{total_fp:>4}{total_fn:>4}{overall_precision:>11.2f}{overall_recall:>9.2f}")


if __name__ == "__main__":
    main()
