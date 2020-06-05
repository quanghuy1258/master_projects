import common
import read_dataset

# Default params
batch_size = 128
epochs = 20
training_dir = "training"
checkpoint_format = "weights.{epoch:04d}-{val_loss:.2f}.h5"
period = 5

# Create training dir
common.create_dir_if_not_exists(training_dir)

# Read data
X_train, y_train = read_dataset.read_data('train_new/')
print(f'X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}')

# Create model
# model = common.create_baseline_model() # Baseline model
# model = common.create_lenet5_model() # LeNet5
model = common.create_alexnet_model() # AlexNet

# Summary model
print("=" * 80)
model.summary()
input("Press Enter to continue...")
print("=" * 80)

# Train model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    callbacks=[common.create_checkpoint_callback("{}/{}".format(training_dir, checkpoint_format), period),
                               common.create_CSVLogger_callback(training_dir)],
                    validation_split=0.2, shuffle=True)

# Visualize history
# common.visualize_history(history)