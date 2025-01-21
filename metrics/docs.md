# Metrics Analysis Guide

## Tools for Checking Metrics

If you're not training a model, you can safely ignore the `./metrics` folder.

---

## Checking Metrics During Training

### 1. Specify the File Path

Define the path to the checkpoint and the required metrics file. For example:

```python
import os

folder = 'path/to/saved_checkpoint'  # Replace with your actual path
state = 1000  # Replace with the desired checkpoint state
checkpoint = f'checkpoint-{state}'

trainer_state_path = os.path.join(folder, checkpoint, 'trainer_state.json')

if os.path.exists(trainer_state_path):
    print("Checkpoint Found. Continue")
else:
    print(f"Path does not exist: {trainer_state_path}")
```

### 2. Run the Code

Execute the above code to verify the existence of the checkpoint and analyze the metrics.

---

## In-Depth Metrics Analysis

For a more accurate and detailed metrics analysis, follow these steps:

### Specify the Path
Ensure you provide the correct path to the checkpoint.

### Analyze Metrics
Use a script or tool to load and analyze the metrics file (e.g., `trainer_state.json`).

---

## Using Tensorboard

Tensorboard provides a visual representation of metrics for quick and simple analysis.

### Setting Up Tensorboard
Ensure that the following training arguments are set in your `trainer.py` script:

```python
Training_Args = {
  'report_to': 'tensorboard',
  'other-arguments': '...'  # Add any additional training arguments here
}
```

### Steps to Use Tensorboard

1. During training, Tensorboard logs will be created in the specified directory (usually `./runs` or a user-defined folder).
2. Launch Tensorboard:

   ```bash
   tensorboard --logdir=<log_directory>
   ```

3. Open the provided URL in your browser to monitor metrics visually.

---

## Summary

- For **basic metric checking**, use the Python script example above.
- For **in-depth analysis**, use a custom script or tool to inspect `trainer_state.json`.
- For **real-time visualization**, Tensorboard is usually sufficient.

Choose the approach that best fits your needs for metrics evaluation!
