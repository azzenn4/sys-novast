Tools for checking metrics,

If you're not Training a model, you can safely ignore ./metrics folder.

Specify the file path,
Run the code to check metrics on a saved checkpoints during Training,

For an accurate more in-depth metrics analysis this is the recommended method,
otherwise just use Tensorboard by specifying in 'trainer.py' training arguments
for example :

Training_Args {
  'report_to = tensorboard',
  'other-arguments'
 }


