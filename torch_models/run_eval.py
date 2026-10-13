import subprocess
import sys

def run_command(cmd):
    print("Running:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")


targetlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "all", "own", "random", "second"]

distance_norms = {
    "L2": [1.25, 2, 3, 4, 5, 6],
    "Linf": [0.09411764705, 0.18823529411, 0.37647058823, 0.56470588235, 0.75294117647]
}

model_configs = {
    "ratio": {
        "files": ['ratio_05.pth', 'ratio_025.pth'],
        "has_fname": True
    },
    "ramp": {
        "files": [None],
        "has_fname": False
    },
    "hat": {
        "files": [None],
        "has_fname": False
    },
    "atd": {
        "files": [None],
        "has_fname": False,
        "script": "atd/test_ATD.py"
    }
}


def get_eps_args(distance_norm: str):
    if distance_norm == "L2":
        return ["--eps_in", "0.5", "--eps_out", "0.5"]
    elif distance_norm == "Linf":
        return ["--eps_in", "0.03137254901", "--eps_out", "0.03137254901"]
    return []


for model_name, config in model_configs.items():
    script_name = config.get("script", "eval_auc.py")
    
    for dist_norm in (["Linf", "L2"] if model_name != "atd" else ["Linf"]):
        distances = distance_norms[dist_norm]
        eps_args = get_eps_args(dist_norm)

        for dist in distances:
            for label in targetlabels:
                for fname in config["files"]:

                    base_cmd = [
                        sys.executable,
                        script_name,
                        "--model_name", model_name,
                        "--distance_norm", dist_norm,
                        "--targetlabel", str(label),
                        "--out_fname", "2026_03_rerun_results.xlsx"
                    ]

                    # Add --fname only for ratio models
                    if config["has_fname"] and fname:
                        base_cmd.extend(["--fname", fname])

                    # Global min distance version
                    cmd_global = base_cmd + [
                        "--global_min_distance", str(dist)
                    ] + eps_args
                    run_command(cmd_global)

                    # Adaptive min distance version
                    cmd_adaptive = base_cmd + [
                        "--adaptive_min_distance", str(dist)
                    ] + eps_args
                    run_command(cmd_adaptive)


