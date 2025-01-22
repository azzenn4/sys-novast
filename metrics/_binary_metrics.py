import json
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os

# uncomment below if using wayland compositor
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
checkpoint = 'checkpoint-3000'
trainer_state_path = os.path.join(f'outputs', checkpoint, 'trainer_state.json')
themecolor = '#dae4f5'
gridcolor = 'whitegrid'

with open(trainer_state_path, 'r') as f:
    trainer_state = json.load(f)

steps = []
losses = []
learning_rates = []
eval_losses = []
eval_accuracies = []
eval_precisions = []
eval_recall = []
eval_runtime = []
eval_samples_per_second = []
eval_steps_per_second = []
eval_f1_scores = []

# loop through the log history and extract metrics
for entry in trainer_state['log_history']:
    if 'loss' in entry:
        steps.append(entry['step'])
        losses.append(entry['loss'])
        learning_rates.append(entry['learning_rate'])
    
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
        eval_accuracies.append(entry['eval_accuracy'])
        eval_precisions.append(entry['eval_precision'])
        eval_recall.append(entry['eval_recall'])
        eval_runtime.append(entry['eval_runtime'])
        eval_samples_per_second.append(entry['eval_samples_per_second'])
        eval_steps_per_second.append(entry['eval_steps_per_second'])
        if 'eval_f1' in entry:  
            eval_f1_scores.append(entry['eval_f1'])

sns.set_style(gridcolor)
plt.rc('font', family='Arial', weight='light', size=11)
fig, axes = plt.subplots(3, 2, figsize=(10, 8),facecolor=themecolor)

axes = axes.flatten()

train_df = pd.DataFrame({
    'Step': steps,
    'Loss': losses,
    'Learning Rate': learning_rates,
    'Type': ['Training'] * len(losses)
})

eval_steps = [entry['step'] for entry in trainer_state['log_history'] if 'eval_loss' in entry]
eval_df = pd.DataFrame({
    'Step': eval_steps,
    'Loss': eval_losses,
    'Type': ['Evaluation'] * len(eval_losses)
})

loss_df = pd.concat([train_df.set_index('Step'), eval_df.set_index('Step')], axis=0, join='outer').reset_index()

sns.lineplot(
    data=loss_df, x='Step', y='Loss', hue='Type', 
    ax=axes[0], dashes=False, palette="mako"
)
axes[0].set_title(f'T_loss {losses[-1]:.4f} | V_loss {eval_losses[-1]:.4f}', fontsize=12, pad=15)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=8)  

sns.lineplot(data=pd.DataFrame({
    'Step': eval_steps,
    'Accuracy': eval_accuracies
}), x='Step', y='Accuracy', ax=axes[1], palette="mako")
axes[1].set_title(f'Eval_acc {eval_accuracies[-1]:.4f}', fontsize=12, pad=15)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Accuracy')

axes[2].plot(eval_steps, eval_f1_scores, label='Total F1 Score', color='purple', marker='x')

axes[2].set_ylim(0, 1)  .

axes[2].set_xlabel('Evaluation Steps')
axes[2].set_ylabel('Total F1 Score')
axes[2].set_title(f'F1 {eval_f1_scores[-1]:.4f}', fontsize=12, pad=15)

sns.scatterplot(
    data=pd.DataFrame({
        'Step': eval_steps,
        'Precision': eval_precisions
    }),
    x='Step', y='Precision', ax=axes[3], 
    s=11,   
    hue='Precision',   
    palette="mako_r"   
)

axes[3].set_ylim(0, 1)   

axes[3].set_title(f'Eval_prec {eval_precisions[-1]:.4f}', fontsize=12, pad=15)
axes[3].set_xlabel('Step')
axes[3].set_ylabel('Precision')
axes[3].legend(fontsize=8)   

sns.scatterplot(
    data=train_df,
    x='Learning Rate',
    y='Loss',
    ax=axes[4],
    hue='Learning Rate',   
    palette='mako_r',   
    legend=None,   
    s=3
)
axes[4].set_title('Loss vs Learning Rate', fontsize=12, pad=15)
axes[4].set_xlabel('Learning Rate')
axes[4].set_ylabel('Loss')

formatter = FuncFormatter(lambda x, _: f'{x:.1e}')   
axes[4].xaxis.set_major_formatter(formatter)

axes[4].set_xlim(learning_rates[0], learning_rates[-1])  

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




















