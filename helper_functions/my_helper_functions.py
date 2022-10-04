import tensorflow as tf

def create_checkpoint_callback(checkpoint_path):
  """
  Creates a ModelCheckpoint callback
  Args:
    checkpoint_path: target directory or name to store ModelCheckpoint files
  """
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only = True,
    monitor = 'val_accuracy',
    save_best_only = True
  )
  return checkpoint_callback
