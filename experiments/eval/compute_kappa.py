"""
Compute Cohen's kappa between the auto 'Exact' label and a human reviewer.

Reads the 'Label Validation' tab of Result_robot.xlsx. The reviewer fills
column H ('Human correct?') with Y/N. This script then compares Human vs Auto.

Until a human fills column H, pass --provisional to use a transparent
second-pass heuristic reviewer (NOT a human) just to exercise the machinery;
those rows are clearly marked provisional and must not be reported as human kappa.
"""
import argparse
from pathlib import Path
import openpyxl
from eval.stats import cohens_kappa

WB = Path(__file__).parent / "Result_robot.xlsx"


def read_rows():
    wb = openpyxl.load_workbook(WB)
    ws = wb["Label Validation"]
    rows = []
    for r in ws.iter_rows(min_row=5, values_only=True):
        if not r or not r[0]:
            continue
        rows.append({"command": r[0], "auto_exact": r[3], "human": r[7]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provisional", action="store_true",
                    help="fill missing Human labels with a heuristic (NOT human)")
    args = ap.parse_args()
    rows = read_rows()
    auto, human = [], []
    filled = 0
    for r in rows:
        a = "Y" if str(r["auto_exact"]).strip().upper() == "Y" else "N"
        h = str(r["human"]).strip().upper()
        if h in ("Y", "N"):
            filled += 1
        elif args.provisional:
            h = a            # heuristic: assume reviewer agrees (sanity check only)
        else:
            continue
        auto.append(a); human.append(h)
    if not auto:
        print("No human labels yet. Fill column H in the 'Label Validation' tab, "
              "or run with --provisional to test the pipeline.")
        return
    k = cohens_kappa(human, auto)
    tag = "PROVISIONAL (heuristic, not human)" if (args.provisional and filled < len(rows)) else "human"
    print(f"Cohen's kappa ({tag}): {k['kappa']:.3f}  "
          f"(agreement={k['po']:.3f}, n={k['n']}, human-filled={filled}/{len(rows)})")
    if k["kappa"] != k["kappa"]:  # nan
        print("  (kappa undefined when all labels identical — add disagreement cases)")


if __name__ == "__main__":
    main()
