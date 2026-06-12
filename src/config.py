from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', './output/', 'Directory to save outputs')
flags.DEFINE_integer('seed', 0, 'Random seed for initialization')
flags.DEFINE_boolean('wandb', False, 'Use Weights & Biases for experiment tracking')
flags.DEFINE_string('project_name', 'hnf_code', 'WandB project name')
flags.DEFINE_string('device', 'cuda:1', 'Device to use for training (cuda or cpu)')
flags.DEFINE_integer('num_workers', 16, 'Number of workers for data loading')

flags.DEFINE_string('model_path', 'output_image_encoder/2025-10-07_11:02/checkpoints/model_best.pth',
                    'Optional path to pretrained encoder checkpoint (if you want to load it manually).')
flags.DEFINE_integer('num_of_materials', 0, 'Number of materials to use for training the haptic signal generation model')

flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_float('learning_rate', 1e-3, 'Base learning rate for NeRF MLPs (recommended <=1e-3)')
flags.DEFINE_float('image_lr', 1e-4, 'Learning rate for image_encoder (recommended smaller than base LR)')
flags.DEFINE_integer('warmup_epochs', 10, 'Freeze image_encoder for first N epochs (helps stability)')

flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs')
flags.DEFINE_integer('validation_interval', 10, 'Interval (in epochs) to validate the model')
flags.DEFINE_integer('checkpoint_interval', 201, 'Interval (in epochs) to save model checkpoints')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for optimizer')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate for neural network')

flags.DEFINE_integer('dft_cutoff_bins', 100, 'Number of DFT bins to keep from the acceleration signal')
flags.DEFINE_integer('num_of_bins', 5, 'Number of bins for discretizing the trajectory ray')
flags.DEFINE_integer('m_dim', 512, 'Dimension of the latent material vector (image_encoder output dimension)')
flags.DEFINE_integer('n_freq', 10, 'Number of frequencies for positional encoding')
flags.DEFINE_integer('hidden', 512, 'Hidden layer size for MLPs')
