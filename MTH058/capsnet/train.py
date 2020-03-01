#!/usr/bin/python3

import common

# Metadata
batch_size = 128
epochs = 100
training_dir = "training"
checkpoint_format = "weights.{epoch:04d}-{val_loss:.2f}.h5"
period = 5
shift_range = 2

# Create checkpoint directory if not exists
common.create_dir_if_not_exists(training_dir)

# Load data
images = common.read_images("train")
ori_labels = common.read_labels("train")
new_labels = common.category2binary(ori_labels)

# Load test
test_images = common.read_images("t10k")
test_labels = common.category2binary(common.read_labels("t10k"))

# Create model
model = common.create_model()

# Summary model
print("=" * 80)
model.summary()
input("Press Enter to continue...")
print("=" * 80)

# Train model
history = model.fit_generator(common.data_generator(images, new_labels, batch_size=batch_size, shift_range=shift_range),
                              steps_per_epoch=ori_labels.shape[0] / batch_size, epochs=epochs,
                              callbacks=[common.create_checkpoint_callback("{}/{}".format(training_dir, checkpoint_format), period),
                                         common.create_CSVLogger_callback(training_dir)],
                              validation_data=(test_images, test_labels), shuffle=True)

# Visualize history
common.visualize_history(history)
