for m in ` ls *.pth`; do
  python eval_acc.py --fname $m --batch_size 50 --steps 7 --n_restarts 1 --out_fname result.csv
  python eval_acc.py --fname $m --batch_size 50 --steps 20 --n_restarts 10 --out_fname result.csv
done