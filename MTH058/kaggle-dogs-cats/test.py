import sys, os
import numpy as np

import common
import read_dataset

# Read data
X_test, y_test = read_dataset.read_data('test_new/')
print(f'X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}')

# Create model
# model = common.create_baseline_model() # Baseline model
model = common.create_alexnet_model() # AlexNet model

# Load weights
try:
    checkpoint_filepath = sys.argv[1]
    model.load_weights(checkpoint_filepath)
except IndexError:
    print("Usage: " + os.path.basename(__file__) + " <checkpoint_filepath>")
    sys.exit(1)

# Evaluate
print("=" * 80)
print('Evaluate on test data')
results = model.evaluate(X_test, y_test)
print(f'Test loss = {results[0]:.2f}')
print(f'Test acc = {results[1]*100:.2f}%')
print("=" * 80)

