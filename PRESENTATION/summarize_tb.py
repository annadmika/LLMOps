# python
# pip install tensorboard pandas
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys, os

logdir = sys.argv[1] if len(sys.argv)>1 else "pipeline_results_test3/tensorboard_logs_test3/metrics"
ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
ea.Reload()

def summarize(tag):
    if tag not in ea.Tags().get('scalars', []):
        print(f"{tag}: not found")
        return
    vals = ea.Scalars(tag)
    steps = [v.step for v in vals]
    vals_f = [v.value for v in vals]
    print(f"--- {tag} ---")
    print(f"points: {len(vals_f)}, first: {vals_f[0]:.6f} (step {steps[0]}), last: {vals_f[-1]:.6f} (step {steps[-1]})")
    print(f"min: {min(vals_f):.6f}, max: {max(vals_f):.6f}")
    # print last 5 values
    print("last 5:", [f"{v:.6f}" for v in vals_f[-5:]])
    print()

for t in ["train/loss","eval/loss","learning_rate","steps_per_second","train/grad_norm","eval/bleu","eval/rouge","bleu","rouge"]:
    summarize(t)