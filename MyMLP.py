import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Dict, Union, Optional, Tuple
import yaml
from dataclasses import dataclass
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import datetime
import sys
import json
import csv
import pandas as pd
import seaborn as sns
from typing import Dict, Optional

@dataclass
class LayerConfig:
    units: int
    activation: str = 'ReLU'
    dropout_rate: float = 0.0
    batch_norm: bool = False

@dataclass
class OptimizerConfig:
    name: str = 'Adam'
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD

@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.1


class LoggerSetup:
    @staticmethod
    def setup_logging(experiment_name: str = None, log_dir: str = "logs", log_level: int = logging.INFO) -> tuple:
        """
        Setup logging configuration with custom experiment folder
        
        Args:
            experiment_name: Name of the experiment/training process (creates subfolder)
            log_dir: Base directory to store logs
            log_level: Logging level (default: logging.INFO)
            
        Returns:
            Tuple of (logger instance, log directory path, timestamp)
        """
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base logs directory
        base_log_path = Path(log_dir)
        base_log_path.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific directory if provided
        if experiment_name:
            # Clean experiment name for folder usage
            clean_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in experiment_name)
            log_path = base_log_path / clean_name
        else:
            log_path = base_log_path
            
        # Create experiment directory
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create file paths
        log_file = log_path / f"training_{timestamp}.log"
        metrics_file = log_path / f"metrics_{timestamp}.csv"
        results_file = log_path / f"results_{timestamp}.json"
        
        # Create logger
        logger = logging.getLogger(f"{__name__}.{experiment_name}" if experiment_name else __name__)
        logger.setLevel(log_level)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging setup complete. Logs will be saved to: {log_file}")
        return logger, metrics_file, results_file, log_path, timestamp

class MetricsLogger:
    def __init__(self, metrics_file: Path, results_file: Path):
        self.metrics_file = metrics_file
        self.results_file = results_file
        
        # Initialize metrics CSV file with headers
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 
                           'val_loss', 'val_acc', 'learning_rate'])
    
    def log_epoch_metrics(self, epoch: int, train_loss: float, train_acc: float, 
                         val_loss: float, val_acc: float, learning_rate: float):
        """Log metrics for each epoch to CSV file"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, learning_rate])
    
    def save_final_results(self, results: dict):
        """Save final training results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=4)


class CustomMLP(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 layer_configs: List[LayerConfig],
                 output_size: int):
        super(CustomMLP, self).__init__()

        self.input_shape = input_shape
        self.flatten = nn.Flatten()

        # Calculate input size from shape
        input_size = 1
        for dim in input_shape:
            input_size *= dim

        layers = []
        current_size = input_size

        # Build layers based on configs
        for layer_config in layer_configs:
            # Add linear layer
            layers.append(nn.Linear(current_size, layer_config.units))
            current_size = layer_config.units

            # Add activation
            if layer_config.activation:
                layers.append(getattr(nn, layer_config.activation)())

            # Add batch normalization if specified
            if layer_config.batch_norm:
                layers.append(nn.BatchNorm1d(layer_config.units))

            # Add dropout if specified
            if layer_config.dropout_rate > 0:
                layers.append(nn.Dropout(layer_config.dropout_rate))

        # Add output layer
        layers.append(nn.Linear(current_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)

class MLPTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer_config: OptimizerConfig, 
                 training_config: TrainingConfig, 
                 experiment_name: str = None,
                 device: Optional[torch.device] = None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.training_config = training_config
        self.experiment_name = experiment_name


        # Initialize optimizer
        optimizer_class = getattr(optim, optimizer_config.name)
        optimizer_params = {
            'lr': optimizer_config.learning_rate,
            'weight_decay': optimizer_config.weight_decay
        }
        if optimizer_config.name == 'SGD':
            optimizer_params['momentum'] = optimizer_config.momentum

        self.optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=training_config.scheduler_patience,
            factor=training_config.scheduler_factor
        )

        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if training_config.mixed_precision else None

        # Setup logging with experiment name
        self.logger, metrics_file, results_file, self.log_dir, self.timestamp = LoggerSetup.setup_logging(
            experiment_name=experiment_name
        )
        self.metrics_logger = MetricsLogger(metrics_file, results_file)
        
        # Log initial configuration
        self.logger.info("=== Training Configuration ===")
        self.logger.info(f"Experiment Name: {experiment_name}")
        self.logger.info(f"Model Architecture:\n{model}")
        self.logger.info(f"Optimizer: {optimizer_config.name}")
        self.logger.info(f"Learning Rate: {optimizer_config.learning_rate}")
        self.logger.info(f"Batch Size: {training_config.batch_size}")
        self.logger.info(f"Epochs: {training_config.epochs}")
        self.logger.info(f"Device: {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Use mixed precision training if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return running_loss / len(train_loader), correct / total

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return running_loss / len(val_loader), correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training pipeline"""
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        start_time = time.time()

        for epoch in range(self.training_config.epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.training_config.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Evaluate
            val_loss, val_acc = self.evaluate(val_loader)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.metrics_logger.log_epoch_metrics(
                epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr
            )
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'logs/{self.experiment_name}/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break

            self.logger.info(
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%\n'
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%'
            )

        total_time = time.time() - start_time

        # Prepare and save final results
        final_results = {
            'training_time': total_time,
            'final_metrics': {
                'train_loss': train_losses[-1],
                'train_accuracy': train_accs[-1],
                'val_loss': val_losses[-1],
                'val_accuracy': val_accs[-1]
            },
            'best_validation_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'early_stopped': patience_counter >= self.training_config.early_stopping_patience
        }
        
        self.metrics_logger.save_final_results(final_results)
        self.logger.info("\n=== Training Complete ===")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        self.logger.info(f"Final validation accuracy: {val_accs[-1]*100:.2f}%")
        self.logger.info(f"Training time: {total_time:.2f} seconds")

        return final_results


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_data_loaders(dataset_name: str, training_config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for specified dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Get dataset class from torchvision.datasets
    dataset_class = getattr(datasets, dataset_name)

    train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )

    return train_loader, test_loader

def plot_metrics(metrics_file: Path):
    """
    Plot training metrics from CSV file
    
    Args:
        metrics_file: Path to the CSV file containing training metrics
    """
    # Read metrics from CSV
    df = pd.read_csv(metrics_file)
    
    # Set style for better visualizations
    sns.set_style('darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Loss Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['train_loss'], label='Train Loss', marker='o', linestyle='-', markersize=4)
    ax1.plot(df['val_loss'], label='Validation Loss', marker='o', linestyle='-', markersize=4)
    ax1.set_title('Loss vs Epoch', fontsize=12, pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['train_acc'] * 100, label='Train Accuracy', 
             marker='o', linestyle='-', markersize=4)
    ax2.plot(df['val_acc'] * 100, label='Validation Accuracy', 
             marker='o', linestyle='-', markersize=4)
    ax2.set_title('Accuracy vs Epoch', fontsize=12, pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['learning_rate'], marker='o', linestyle='-', 
             color='green', markersize=4)
    ax3.set_title('Learning Rate vs Epoch', fontsize=12, pad=10)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss-Accuracy Relationship
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['train_loss'], df['train_acc'] * 100, 
                label='Train', alpha=0.6)
    ax4.scatter(df['val_loss'], df['val_acc'] * 100, 
                label='Validation', alpha=0.6)
    ax4.set_title('Loss-Accuracy Relationship', fontsize=12, pad=10)
    ax4.set_xlabel('Loss')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Progress
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # Create progress plot with dual y-axis
    ax5_acc = ax5.twinx()
    
    # Plot loss
    ln1 = ax5.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', alpha=0.6)
    ln2 = ax5.plot(df['epoch'], df['val_loss'], 'b--', label='Val Loss', alpha=0.6)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    
    # Plot accuracy
    ln3 = ax5_acc.plot(df['epoch'], df['train_acc'] * 100, 'r-', 
                       label='Train Acc', alpha=0.6)
    ln4 = ax5_acc.plot(df['epoch'], df['val_acc'] * 100, 'r--', 
                       label='Val Acc', alpha=0.6)
    ax5_acc.set_ylabel('Accuracy (%)', color='r')
    ax5_acc.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='center right')
    
    ax5.set_title('Training Progress Overview', fontsize=12, pad=10)
    ax5.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots
    save_dir = metrics_file.parent
    timestamp = metrics_file.stem.split('_')[1]  # Extract timestamp from metrics filename
    
    plot_path = save_dir / f'training_plots_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Save additional plot with higher resolution for detailed viewing
    plot_path_high_res = save_dir / f'training_plots_{timestamp}_high_res.pdf'
    plt.savefig(plot_path_high_res, format='pdf', bbox_inches='tight')
    
    plt.show()
    
    # Create and save metrics summary
    metrics_summary = {
        'final_train_loss': df['train_loss'].iloc[-1],
        'final_val_loss': df['val_loss'].iloc[-1],
        'final_train_acc': df['train_acc'].iloc[-1] * 100,
        'final_val_acc': df['val_acc'].iloc[-1] * 100,
        'best_val_loss': df['val_loss'].min(),
        'best_val_acc': df['val_acc'].max() * 100,
        'total_epochs': len(df)
    }
    
    # Save metrics summary
    summary_path = save_dir / f'metrics_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("=== Training Metrics Summary ===\n\n")
        for key, value in metrics_summary.items():
            f.write(f"{key}: {value:.4f}\n")

def generate_training_plots(log_dir: str = "logs", experiment_name: str = None):
    """
    Generate plots for all training runs in the specified directory
    
    Args:
        log_dir: Base directory containing the training logs
        experiment_name: Name of the specific experiment subfolder
    """
    base_path = Path(log_dir)
    
    if experiment_name:
        log_path = base_path / experiment_name
    else:
        log_path = base_path
    
    # Find all metrics CSV files
    metrics_files = list(log_path.glob("metrics_*.csv"))
    
    for metrics_file in metrics_files:
        print(f"Generating plots for {metrics_file.name}")
        plot_metrics(metrics_file)