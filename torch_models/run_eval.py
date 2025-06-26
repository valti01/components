import subprocess
import sys


def run_command(cmd):
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")


targetlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'all', 'own']

distance_norms = {
    "L2": [1, 2, 3, 4, 5, 6],
    "Linf": [0.09411764705, 0.18823529411, 0.37647058823, 0.56470588235, 0.75294117647]
}

model_names = ["hat", "ratio"]

for model_name in model_names:
    if model_name == "ratio":
        dist_norm = "Linf"
        distances = distance_norms[dist_norm]
        for dist in distances:
            for label in targetlabels:
                cmd_global = [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--global_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv"
                    "--fname",'ratio_05.pth'
                ]
                run_command(cmd_global)
                cmd_adaptive =  [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--adaptive_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv",
                    "--fname",'ratio_05.pth'
                ]
                run_command(cmd_adaptive)
                cmd_global = [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--global_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv",
                    "--fname",'ratio_025.pth'
                ]
                run_command(cmd_global)
                cmd_adaptive =  [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--adaptive_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv"
                    "--fname",'ratio_025.pth'
                ]
                run_command(cmd_adaptive)
    elif model_name == "hat":
        dist_norm = "L2"
        distances = distance_norms[dist_norm]
        for dist in distances:
            for label in targetlabels:
                cmd_global = [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--global_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv"
                ]
                run_command(cmd_global)
                cmd_adaptive =  [
                    sys.executable,
                    "eval_auc.py",
                    "--model_name", model_name,
                    "--distance_norm", dist_norm,
                    "--adaptive_min_distance", str(dist),
                    "--targetlabel", str(label),
                    "--out_fname", "bigmeasurement_results.csv"
                ]
                run_command(cmd_adaptive)
    elif model_name == "ramp":
        for dist_norm in ["L2", "Linf"]:
            distances = distance_norms[dist_norm]
            for dist in distances:
                for label in targetlabels:
                    if dist_norm == "L2":
                        continue
                    cmd_global = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--global_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "bigmeasurement_results_linf.csv"
                    ]
                    run_command(cmd_global)
                    # Run with adaptive_min_distance
                    cmd_adaptive = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "bigmeasurement_results_linf.csv"
                    ]
                    run_command(cmd_adaptive)
