import common
import read_dataset

# Default params
batch_size = 128
epochs = 10
training_dir = "training"
checkpoint_format = "weights.{epoch:04d}-{val_loss:.2f}.h5"
period = 5

# Create training dir
common.create_dir_if_not_exists(training_dir)

# Read data
print('Reading data...')
X_train, y_train, X_test, y_test = read_dataset.read_dataset("train_temp")

# Baseline model -------------------------------------
# Create model
baseline_model = common.create_baseline_model()

# Summary model
print("=" * 80)
baseline_model.summary()
input("Press Enter to continue...")
print("=" * 80)

# Train model
history = baseline_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    callbacks=[common.create_checkpoint_callback("{}/{}".format(training_dir, checkpoint_format), period),
                               common.create_CSVLogger_callback(training_dir)],
                    validation_split=0.3, shuffle=True)

# Visualize history
common.visualize_history(history)