from MyMLP import LayerConfig, CustomMLP, MLPTrainer, OptimizerConfig, TrainingConfig, create_data_loaders, generate_training_plots

layer_configs = [
    LayerConfig(1024, activation='ReLU', dropout_rate=0.3, batch_norm=True),
    LayerConfig(512, activation='ReLU', dropout_rate=0.3, batch_norm=True),
    LayerConfig(256, activation='ReLU', dropout_rate=0.3, batch_norm=True),
]

optimizer_config = OptimizerConfig(
    name='Adam',
    learning_rate=0.001,
    weight_decay=1e-5
)

training_config = TrainingConfig(
    batch_size=256,
    epochs=10,
    num_workers=4,
    mixed_precision=True
)

# Create model
model = CustomMLP(
    input_shape=(3, 32, 32),  # For CIFAR-10
    layer_configs=layer_configs,
    output_size=10
)

# Initialize trainer
trainer = MLPTrainer(
    model=model,
    optimizer_config=optimizer_config,
    training_config=training_config,
    experiment_name="train_with_1_layer"
)

# Create data loaders
train_loader, test_loader = create_data_loaders('CIFAR10', training_config)

# Train model
metrics = trainer.train(train_loader, test_loader)

generate_training_plots(experiment_name="train_with_1_layer")