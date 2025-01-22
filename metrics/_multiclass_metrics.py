import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import numpy as np

# uncomment below if using wayland compositor
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
checkpoint = 'checkpoint-1500'
trainer_state_path = os.path.join(f'outputs', checkpoint, 'trainer_state.json')
themecolor = '#dae4f5'
gridcolor = 'whitegrid'
# load the trainer_state.json file
with open(trainer_state_path, 'r') as f:
    trainer_state = json.load(f)
# initialize lists to store metrics
steps = []
losses = []
learning_rates = []
eval_losses = []
eval_accuracies = []
eval_precisions = []
eval_anger = []
eval_fear = []
eval_joy = []
eval_love = []
eval_recall = []
eval_runtime = []
eval_samples_per_second = []
eval_steps_per_second = []
eval_sadness = []
eval_surprise = []
# loop through the log history and extract metrics
for entry in trainer_state['log_history']:
    # check for training entries (contains 'loss') for loss metrics
    if 'loss' in entry:
        steps.append(entry['step'])
        losses.append(entry['loss'])
        learning_rates.append(entry['learning_rate'])
    # check for evaluation entries (eval_{str.metrics})
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
        eval_accuracies.append(entry['eval_accuracy'])
        eval_anger.append(entry['eval_anger'])
        eval_fear.append(entry['eval_fear'])
        eval_joy.append(entry['eval_joy'])
        eval_love.append(entry['eval_love'])
        eval_precisions.append(entry['eval_precision'])
        eval_recall.append(entry['eval_recall'])
        eval_runtime.append(entry['eval_runtime'])
        eval_samples_per_second.append(entry['eval_samples_per_second'])
        eval_steps_per_second.append(entry['eval_steps_per_second'])
        eval_sadness.append(entry['eval_sadness'])
        eval_surprise.append(entry['eval_surprise'])
# set seaborn styling
sns.set_style(gridcolor)
plt.rc('font', family='Arial', weight='light', size=11)
fig, axes = plt.subplots(3, 2, figsize=(10, 8),facecolor=themecolor)
# flatten axes for easier iteration
axes = axes.flatten()
# create DataFrames for each metric
train_df = pd.DataFrame({
    'Step': steps,
    'Loss': losses,
    'Learning Rate': learning_rates,
    'Type': ['Training'] * len(losses)
})
# update eval_df to ensure steps align
eval_steps = [entry['step'] for entry in trainer_state['log_history'] if 'eval_loss' in entry]
eval_df = pd.DataFrame({
    'Step': eval_steps,
    'Loss': eval_losses,
    'Type': ['Evaluation'] * len(eval_losses)
})
# combine training and evaluation loss data
loss_df = pd.concat([train_df.set_index('Step'), eval_df.set_index('Step')], axis=0, join='outer').reset_index()

# plot Loss
sns.lineplot(
    data=loss_df, x='Step', y='Loss', hue='Type', 
    ax=axes[0], dashes=False, palette="mako"
)
axes[0].set_title(f'T_loss {losses[-1]:.4f} | V_loss {eval_losses[-1]:.4f}', fontsize=12, pad=15)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=8)  
# slice steps to match the length of eval_accuracies, eval_f1, and eval_precisions..
steps_for_plot = list(range(0, len(eval_accuracies) * 150, 150))
# plot Accuracy
sns.lineplot(data=pd.DataFrame({
    'Step': steps_for_plot,
    'Accuracy': eval_accuracies
}), x='Step', y='Accuracy', ax=axes[1], palette="mako")
axes[1].set_title(f'Eval_acc {eval_accuracies[-1]:.4f}', fontsize=12, pad=15)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Accuracy')
# plot total F1
total_f1 = np.mean([eval_anger, eval_fear, eval_joy, eval_love, eval_sadness, eval_surprise], axis=0)
axes[2].plot(eval_steps, total_f1, label='Total F1 Score', color='purple', marker='x')
axes[2].set_xlabel('Evaluation Steps')
axes[2].set_ylabel('Total F1 Score')
axes[2].set_title(f'F1 {total_f1[-1]:.4f}', fontsize=12, pad=15)
# plot Precision
sns.scatterplot(
    data=pd.DataFrame({
        'Step': steps_for_plot,
        'Precision': eval_precisions
    }),
    x='Step', y='Precision', ax=axes[3], 
    s=11,  # Dot size
    hue='Precision',  # Color by Precision
    palette="mako_r"  # Use a reversed 'viridis' palette
)
axes[3].set_title(f'Eval_prec {eval_precisions[-1]:.4f}', fontsize=12, pad=15)
axes[3].set_xlabel('Step')
axes[3].set_ylabel('Precision')
axes[3].legend(fontsize=8)  
# plot Loss / Learning Rate
sns.scatterplot(
    data=train_df,
    x='Learning Rate',
    y='Loss',
    ax=axes[4],
    hue='Learning Rate',  # Color by Learning Rate
    palette='mako_r',  # You can change the palette to another colormap (e.g., 'coolwarm', 'plasma', etc.)
    legend=None,  # If you don't want a legend
    s=3
)
axes[4].set_title('Loss vs Learning Rate', fontsize=12, pad=15)
axes[4].set_xlabel('Learning Rate')
axes[4].set_ylabel('Loss')
# set the formatter for the x-axis to display in scientific notation
formatter = FuncFormatter(lambda x, _: f'{x:.1e}')  # Format numbers in scientific notation
axes[4].xaxis.set_major_formatter(formatter)
# optionally, you can specify the limits for the x-axis
# set the x-axis limit to a maximum of 2e-5 (refer to _trainer_{multiclass/binary}.py in training args where learning_rate = xe-x)
axes[4].set_xlim(learning_rates[0], learning_rates[-1])  
# scatter plot for Eval Loss vs Accuracy
sns.scatterplot(
    data=pd.DataFrame({
        'Eval Loss': eval_losses,
        'Accuracy': eval_accuracies
    }),
    x='Eval Loss', y='Accuracy', ax=axes[5],
    palette="mako", hue='Eval Loss', s=8
)
axes[5].set_title('Eval Loss vs Accuracy', fontsize=12, pad=15)
axes[5].set_xlabel('Eval Loss')
axes[5].set_ylabel('Accuracy')
axes[5].legend(fontsize=8)   
plt.tight_layout()

fig.suptitle(checkpoint + "| linear_lrscheduler", fontsize=12, y=1)

plt.show()
# plot F1 Score
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# plot each emotion's F1 score as a separate line
ax.plot(eval_steps, eval_anger, zs=0, zdir='y', label='Anger', marker='o')
ax.plot(eval_steps, eval_fear, zs=1, zdir='y', label='Fear', marker='s')
ax.plot(eval_steps, eval_joy, zs=2, zdir='y', label='Joy', marker='^')
ax.plot(eval_steps, eval_love, zs=3, zdir='y', label='Love', marker='v')
ax.plot(eval_steps, eval_sadness, zs=4, zdir='y', label='Sadness', marker='d')
ax.plot(eval_steps, eval_surprise, zs=5, zdir='y', label='Surprise', marker='p')
# Labels and title
ax.set_xlabel('Evaluation Steps')
ax.set_ylabel('Emotion')
ax.set_zlabel('F1 Score')
ax.set_title('F1 Scores for Each Emotion')
# Add legend
ax.legend()
# Show plot
plt.show()
# create a figure and a 3D axis
sns.set_theme(style=gridcolor, palette="muted") 
fig = plt.figure(figsize=(10, 8), facecolor=themecolor)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(themecolor)
# scatter plot with Loss, Learning Rate, and Steps from trainer_state
sc = ax.scatter(
    steps,  # X axis: Steps
    learning_rates,  # Y axis: Learning Rate
    losses,  # Z axis: Loss
    c=learning_rates,  # Color by Learning Rate
    cmap='mako_r',  # Use a colormap (you can change this)
    s=10
)
# set axis labels with labelpad to move them further from the axes
ax.set_xlabel('Steps', fontsize=10, fontweight='light', labelpad=15)
ax.set_ylabel('Learning Rate', fontsize=10, fontweight='light', labelpad=15)
ax.set_zlabel('Loss', fontsize=10, fontweight='light', labelpad=15)

ax.tick_params(axis='x', labelsize=8)  # X-axis tick label size
ax.tick_params(axis='y', labelsize=8)  # Y-axis tick label size
ax.tick_params(axis='z', labelsize=8)  # Z-axis tick label size
# set title
ax.set_title('Loss / Learning Rate / Steps', fontsize=12, fontweight='light')
# format the Learning Rate axis to use scientific notation
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2e}'))
# format the color bar to use scientific notation
cbar = fig.colorbar(sc, label='Learning Rate', pad=0.2, fraction=0.03)
cbar.set_ticks(cbar.get_ticks())  # Ensure color bar ticks are set
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2e}'))
# set font properties for the color bar label
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_ylabel('Learning Rate', fontsize=10, fontweight='light')
# display the plot
plt.show()




















