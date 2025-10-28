from tensorboard.backend.event_processing import event_accumulator
import os

# Load the TensorBoard logs
log_dir = "tensorboard_logs_test3/metrics"
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("=== TEST 3 TensorBoard Analysis ===\n")

# Training Loss
if 'train/loss' in ea.Tags()['scalars']:
    train_loss = ea.Scalars('train/loss')
    print(f"Train Loss: {len(train_loss)} steps")
    print(f"  Start: {train_loss[0].value:.4f} (step {train_loss[0].step})")
    print(f"  End: {train_loss[-1].value:.4f} (step {train_loss[-1].step})")
    reduction = ((train_loss[0].value - train_loss[-1].value) / train_loss[0].value) * 100
    print(f"  Reduction: {reduction:.1f}%\n")

# Eval Loss
if 'eval/loss' in ea.Tags()['scalars']:
    eval_loss = ea.Scalars('eval/loss')
    print(f"Eval Loss: {len(eval_loss)} evaluations")
    for i, e in enumerate(eval_loss):
        print(f"  Eval {i+1}: {e.value:.4f} (step {e.step})")
    print(f"  Final: {eval_loss[-1].value:.4f}\n")

# Learning Rate
if 'train/learning_rate' in ea.Tags()['scalars']:
    lr = ea.Scalars('train/learning_rate')
    print(f"Learning Rate: Max {max([x.value for x in lr]):.6f}\n")

# Calculate training time from metadata
training_time_hours = 13173.561066865921 / 3600
print(f"Training Time: {training_time_hours:.2f} hours ({training_time_hours*60:.1f} minutes)")
