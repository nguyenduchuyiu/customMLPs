from MyMLP import LayerConfig, CustomMLP, MLPTrainer, OptimizerConfig, TrainingConfig, create_data_loaders, generate_training_plots
import yaml


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

layer_configs = [LayerConfig(**layer) for layer in config['model']['layers']]
optimizer_config = OptimizerConfig(**config['optimizer'])
training_config = TrainingConfig(**config['training'])

# Create model
model = CustomMLP(
    input_shape=config['model']['input_shape'],  # For CIFAR-10
    layer_configs=layer_configs,
    output_size=config['model']['output_size']
)

# Initialize trainer
trainer = MLPTrainer(
    model=model,
    optimizer_config=optimizer_config,
    training_config=training_config,
    experiment_name="train_with_tuned_hyperparameters"
)

# Create data loaders
train_loader, test_loader = create_data_loaders('CIFAR10', training_config)

# Train model
metrics = trainer.train(train_loader, test_loader)

# Plot results
generate_training_plots(experiment_name="train_with_tuned_hyperparameters")