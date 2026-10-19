"""
Join FID results and privacy accounting results into one CSV.

Two modes:

  1D sweep (prop only) — original empirical_first_10k experiment:
    python combine_fid_privacy.py \\
        --mode prop \\
        --fid_root   /scratch/$USER/results_empirical_first_10k_v2 \\
        --privacy_root /scratch/$USER/results_privacy_accounting_10k \\
        --out combined_results_prop.csv

  2D sweep (prop x subsample_batch_size) — empirical_first_10k_subsample experiment:
    python combine_fid_privacy.py \\
        --mode subsample \\
        --root /scratch/$USER/results_empirical_first_10k_subsample \\
        --out combined_results_subsample.csv
"""
import argparse
import csv
import glob
import json
import os


def load_json(path):
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_json(directory, suffix):
    """Find the first JSON file under directory (recursively) ending with suffix."""
    matches = glob.glob(os.path.join(directory, "**", f"*{suffix}"), recursive=True)
    return matches[0] if matches else None


# ── Mode 1: 1D prop sweep ────────────────────────────────────────────────────

def combine_prop_sweep(fid_root, privacy_root, props, out):
    rows = []
    for pct in props:
        fid_dir    = glob.glob(os.path.join(fid_root,     f"*prop{pct}", "fid"))
        priv_dir   = glob.glob(os.path.join(privacy_root, f"*prop{pct}", "privacy"))

        fid_json  = find_json(fid_dir[0],  "_fid_score.json")  if fid_dir  else None
        priv_json = find_json(priv_dir[0], "_privacy.json")     if priv_dir else None

        fid_data  = load_json(fid_json)  or {}
        priv_data = load_json(priv_json) or {}

        rows.append({
            "prop":            pct / 100.0,
            "prop_pct":        pct,
            "fid":             fid_data.get("FID Score"),
            "mu_N":            priv_data.get("multistep_mu_N"),
            "eps_clt":         priv_data.get("multistep_eps_clt"),
            "eps_edgeworth":   priv_data.get("multistep_eps_edgeworth"),
            "eps_rigorous":    priv_data.get("multistep_eps_rigorous"),
            "be_error":        priv_data.get("multistep_be_error"),
            "fid_path":        fid_json,
            "privacy_path":    priv_json,
        })

    _write_csv(out, rows)


# ── Mode 2: 2D prop x subsample_batch_size sweep ────────────────────────────

def combine_subsample_sweep(root, props, subsample_sizes, dataset_size, out):
    rows = []
    for pct in props:
        for sub in subsample_sizes:
            out_dir = os.path.join(root, f"cifar10_empirical_first_10k_sub{sub}_prop{pct}")

            fid_json  = find_json(os.path.join(out_dir, "fid"),     "_fid_score.json")
            priv_json = find_json(out_dir,                           "_privacy.json")

            fid_data  = load_json(fid_json)  or {}
            priv_data = load_json(priv_json) or {}

            rows.append({
                "prop":                 pct / 100.0,
                "prop_pct":             pct,
                "subsample_batch_size": sub,
                "p":                    sub / dataset_size,
                "fid":                  fid_data.get("FID Score"),
                "mu_N":                 priv_data.get("multistep_mu_N"),
                "eps_clt":              priv_data.get("multistep_eps_clt"),
                "eps_edgeworth":        priv_data.get("multistep_eps_edgeworth"),
                "eps_rigorous":         priv_data.get("multistep_eps_rigorous"),
                "be_error":             priv_data.get("multistep_be_error"),
                "fid_path":             fid_json,
                "privacy_path":         priv_json,
            })

    _write_csv(out, rows)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    if not rows:
        print("No rows to write.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")
    for r in rows:
        parts = [f"prop={r['prop']:.2f}"]
        if "subsample_batch_size" in r:
            parts.append(f"sub={r['subsample_batch_size']}")
        parts += [f"FID={r['fid']}", f"eps_clt={r['eps_clt']}"]
        print(" ", "  ".join(str(p) for p in parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prop", "subsample"], default="subsample",
                        help="'prop' for 1D sweep, 'subsample' for 2D sweep")
    # 2D mode
    parser.add_argument("--root", help="Base dir for subsample sweep results")
    # 1D mode
    parser.add_argument("--fid_root",     help="Base dir containing FID results (1D mode)")
    parser.add_argument("--privacy_root", help="Base dir containing privacy results (1D mode)")
    # Shared
    parser.add_argument("--out", default="combined_results.csv")
    parser.add_argument("--props", type=float, nargs="+",
                        help="Proportion percentages to include (e.g. 10 20 ... 95)")
    parser.add_argument("--subsample_sizes", type=int, nargs="+",
                        default=[100, 25, 10, 5])
    parser.add_argument("--dataset_size", type=int, default=50000)
    args = parser.parse_args()

    if args.mode == "subsample":
        props = args.props or [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
        combine_subsample_sweep(
            root=args.root,
            props=[int(p) if p == int(p) else p for p in props],
            subsample_sizes=args.subsample_sizes,
            dataset_size=args.dataset_size,
            out=args.out,
        )
    else:
        props = args.props or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        combine_prop_sweep(
            fid_root=args.fid_root,
            privacy_root=args.privacy_root,
            props=[int(p) for p in props],
            out=args.out,
        )


if __name__ == "__main__":
    main()
