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


targetlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "all", "own", "random", "second"]
#    "Linf": [0.09411764705, 0.18823529411, 0.37647058823, 0.56470588235, 0.75294117647]
distance_norms = {
    "L2": [1.25, 2, 3, 4, 5, 6],
    "Linf": [0.09411764705, 0.18823529411, 0.37647058823, 0.56470588235, 0.75294117647]
}

model_names = ["ramp","ratio"]

for model_name in model_names:
    if model_name == "ratio":
        for dist_norm in ["Linf"]:
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
                        "--out_fname", "2025_11_results.csv",
                        "--fname", 'ratio_05.pth'
                    ]
                    run_command(cmd_global)
                    cmd_adaptive = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "2025_11_results.csv",
                        "--fname", 'ratio_05.pth'
                    ]
                    run_command(cmd_adaptive)
                    cmd_global = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--global_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "2025_11_results.csv",
                        "--fname", 'ratio_025.pth'
                    ]
                    run_command(cmd_global)
                    cmd_adaptive = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "2025_11_results.csv",
                        "--fname", 'ratio_025.pth'
                    ]
                    run_command(cmd_adaptive)
    elif model_name == "hat":
        for dist_norm in ["Linf"]:
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
                        "--out_fname", "2025_11_results.csv"
                    ]
                    #run_command(cmd_global)

                    cmd_adaptive = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "2025_11_results.csv"
                    ]
                    run_command(cmd_adaptive)
    elif model_name == "ramp":
        for dist_norm in ["Linf"]:
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
                        "--out_fname", "2025_11_results.csv"
                    ]
                    #run_command(cmd_global)

                    cmd_adaptive = [
                        sys.executable,
                        "eval_auc.py",
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label),
                        "--out_fname", "2025_11_results.csv",
                        "--imgdir", "linf_bug_ramp"
                    ]
                    run_command(cmd_adaptive)
    elif model_name == "atd":
        for dist_norm in ["Linf"]:
            distances = distance_norms[dist_norm]
            for dist in distances:
                for label in targetlabels:
                    cmd_adaptive = [
                        sys.executable,
                        "atd/test_ATD.py",
                        "--distance_norm", dist_norm,
                        "--adaptive_min_distance", str(dist),
                        "--targetlabel", str(label)
                    ]
                    run_command(cmd_adaptive)

                    cmd_global = [
                        sys.executable,
                        "atd/test_ATD.py",
                        "--distance_norm", dist_norm,
                        "--global_min_distance", str(dist),
                        "--targetlabel", str(label)
                    ]
                    run_command(cmd_global)
