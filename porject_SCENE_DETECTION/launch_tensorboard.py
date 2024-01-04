import os

"""
this script launch tensorboard
"""

# Define the log directory
log_dir = "runs"  # Change this to the actual path of your 'runs' directory

# Launch TensorBoard
os.system(f"tensorboard --logdir={log_dir}")