#!/usr/bin/python3

import common

# Metadata
batch_size = 128
epochs = 100
checkpoint_dir = "training"
checkpoint_format = "weights.{epoch:04d}-{val_loss:.2f}.h5"
period = 5

# Create checkpoint directory if not exists
common.create_dir_if_not_exists(checkpoint_dir)

# Load data
prefix = "train"
images = common.read_images(prefix)
ori_labels = common.read_labels(prefix)
new_labels = common.category2binary(ori_labels)

# Create model
model = common.create_model()

# Train model
model.fit(images, new_labels, batch_size=batch_size, epochs=epochs,
          callbacks=[common.create_checkpoint_callback(checkpoint_dir+"/"+checkpoint_format, period)],
          validation_split=0.3, shuffle=True)

