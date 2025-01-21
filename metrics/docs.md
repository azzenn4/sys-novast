Tools for Checking Metrics

If you're not training a model, you can safely ignore the ./metrics folder.

Checking Metrics During Training

To check metrics on saved checkpoints during training, follow these steps:

1. Specify the File Path

Define the path to the checkpoint and the required metrics file. For example:

import os

folder = 'path/to/saved_checkpoint'  # Replace with your actual path
state = 1000  # Replace with the desired checkpoint state
checkpoint = f'checkpoint-{state}'

trainer_state_path = os.path.join(folder, checkpoint, 'trainer_state.json')

if os.path.exists(trainer_state_path):
    print("Checkpoint Found. Continue")
else:
    print(f"Path does not exist: {trainer_state_path}")

2. Run the Code

Execute the above code to verify the existence of the checkpoint and analyze the metrics.

In-Depth Metrics Analysis

For a more accurate and detailed metrics analysis, this is the recommended method:

Specify the correct path to the checkpoint.

Use a script or tool to load and analyze the metrics file (e.g., trainer_state.json).

Alternatively, you can use Tensorboard for a visual representation of metrics.

Using Tensorboard

Ensure that the following training arguments are set in your trainer.py script:

Training_Args = {
  'report_to': 'tensorboard',
  'other-arguments': '...'  # Add any additional training arguments here
}

Steps to Use Tensorboard:

During training, Tensorboard logs will be created in the specified directory (usually ./runs or a user-defined folder).

Launch Tensorboard:

tensorboard --logdir=<log_directory>

Open the provided URL in your browser to monitor metrics visually.

For quick and simple metrics, Tensorboard is usually sufficient. For advanced metrics evaluation, use the recommended script-based approach as detailed above.


