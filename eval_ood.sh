function run() {
  for rst in $5; do
    for ps in $4; do
      for stps in $3; do
        for m in $(ls $1); do
          for o in $2; do
            python eval_auc.py --fname $m --batch_size 100 --steps $stps --eps_in $ps --n_restarts $rst --obj $o --out_fname result_auc.csv --gpu $6
          done
        done
      done
    done
  done
}

objectives="msp ul lse ml"
run "ratio_025.pth" "$objectives" "20" "0.25 0.5" "1 10" 0
run "ratio_05.pth" "$objectives" "20" "0.25 0.5" "1 10" 0
