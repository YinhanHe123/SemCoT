import logging
import os
import time
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir='./logs', experiment_name=None):
        """
        Initialize a logger for the experiment

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (defaults to timestamp)
        """
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        # Create log directories
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up file logging
        log_file = os.path.join(self.log_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Set up TensorBoard
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        self.logger.info(f"Logging to {self.log_dir}")

    def log_metrics(self, metrics, step):
        """
        Log metrics to console and TensorBoard

        Args:
            metrics: Dictionary of metric names and values
            step: Current step (epoch or batch)
        """
        # Log to console/file
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step} - {metrics_str}")

        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_hyperparams(self, config):
        """
        Log hyperparameters

        Args:
            config: Configuration object with hyperparameters
        """
        # Convert config to dictionary if needed
        if not isinstance(config, dict):
            config_dict = config.__dict__
        else:
            config_dict = config

        # Log to console/file
        self.logger.info(f"Hyperparameters: {config_dict}")

        # Log to TensorBoard
        self.writer.add_hparams(config_dict, {})

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()