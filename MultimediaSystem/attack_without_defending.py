"""
  Membership inference attack
  Dataset: CIFAR10
  Model target: CNN
"""

import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from tqdm import tqdm

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3

TARGET_EPOCHS = 12

SHADOW_DATASET_SIZE = 4000
NUM_SHADOWS = 3

ATTACK_TEST_DATASET_SIZE = 4000
ATTACK_EPOCHS = 12

def prepare_attack_data(model, data_in, data_out):
    """
    Prepare the data in the attack model format.

    :param model: Classifier
    :param (X, y) data_in: Data used for training
    :param (X, y) data_out: Data not used for training
    :returns: (X, y) for the attack classifier
    """
    X_in, y_in = data_in
    X_out, y_out = data_out
    y_hat_in = model.predict(X_in)
    y_hat_out = model.predict(X_out)

    labels = np.ones(y_in.shape[0])
    labels = np.hstack([labels, np.zeros(y_out.shape[0])])
    data = np.c_[y_hat_in, y_in]
    data = np.vstack([data, np.c_[y_hat_out, y_out]])
    return data, labels

################################################################################
#                                   GET DATA                                   #
################################################################################

def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)

################################################################################
#                                 TARGET MODEL                                 #
################################################################################

def target_model_fn():
    """The architecture of the target (victim) model.
    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(WIDTH, HEIGHT, CHANNELS)))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

################################################################################
#                                 SHADOW MODEL                                 #
################################################################################

class ShadowModelBundle():
    """
    A bundle of shadow models.

    :param model_fn: Function that builds a new shadow model
    :param shadow_dataset_size: Size of the training data for each shadow model
    :param num_models: Number of shadow models
    """
    def __init__(self, model_fn, shadow_dataset_size, num_models):
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self._prng = np.random.RandomState(int(time.time()))

    def fit_transform(self, X, y, fit_kwargs=None):
        """Train the shadow models and get a dataset for training the attack.

        :param X: Data coming from the same distribution as the target
                  training data
        :param y: Data labels
        :param dict fit_kwargs: Arguments that will be passed to the fit call for
                each shadow model.
        """
        self._fit(X, y, fit_kwargs=fit_kwargs)
        return self._transform()

    def _fit(self, X, y, fit_kwargs=None):
        """Train the shadow models.

        :param X: Data coming from the same distribution as the target
                  training data
        :param y: Data labels
        :param dict fit_kwargs: Arguments that will be passed to the fit call for
                each shadow model.
        """
        self.shadow_train_indices_ = []
        self.shadow_test_indices_ = []

        self.shadow_models_ = []

        fit_kwargs = fit_kwargs or {}
        indices = np.arange(X.shape[0])

        for i in tqdm(range(self.num_models)):
            # Pick indices for this shadow model.
            shadow_indices = self._prng.choice(indices, 2 * self.shadow_dataset_size, replace=False)
            train_indices = shadow_indices[: self.shadow_dataset_size]
            test_indices = shadow_indices[self.shadow_dataset_size :]
            X_train, y_train = X[train_indices], y[train_indices]
            self.shadow_train_indices_.append(train_indices)
            self.shadow_test_indices_.append(test_indices)

            # Train the shadow model.
            shadow_model = self.model_fn()
            shadow_model.fit(X_train, y_train, **fit_kwargs)
            self.shadow_models_.append(shadow_model)

        self.X_fit_ = X
        self.y_fit_ = y
        self._prng = np.random.RandomState(int(time.time()))
        return self

    def _transform(self):
        """Produce in/out data for training the attack model.
        """
        shadow_data_array = []
        shadow_label_array = []

        for i in tqdm(range(self.num_models)):
            shadow_model = self.shadow_models_[i]
            train_indices = self.shadow_train_indices_[i]
            test_indices = self.shadow_test_indices_[i]

            train_data = self.X_fit_[train_indices], self.y_fit_[train_indices]
            test_data = self.X_fit_[test_indices], self.y_fit_[test_indices]
            shadow_data, shadow_labels = prepare_attack_data(shadow_model, train_data, test_data)

            shadow_data_array.append(shadow_data)
            shadow_label_array.append(shadow_labels)

        X_transformed = np.vstack(shadow_data_array).astype("float32")
        y_transformed = np.hstack(shadow_label_array).astype("float32")
        return X_transformed, y_transformed

################################################################################
#                                 ATTACK MODEL                                 #
################################################################################

def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.
    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(NUM_CLASSES)))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

class AttackModelBundle():
    """
    A bundle of attack models, one for each target model class.

    :param model_fn: Function that builds a new shadow model
    :param num_classes: Number of classes
    """
    def __init__(self, model_fn, num_classes):
        self.model_fn = model_fn
        self.num_classes = num_classes

    def fit(self, X, y, fit_kwargs=None):
        """Train the attack models.

        :param X: Shadow predictions coming from
                  :py:func:`ShadowBundle.fit_transform`.
        :param y: In/Out labels
        :param fit_kwargs: Arguments that will be passed to the fit call for
                each attack model.
        """
        X_total = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]

        datasets_by_class = []
        data_indices = np.arange(X_total.shape[0])
        for i in range(self.num_classes):
            class_indices = data_indices[np.argmax(classes, axis=1) == i]
            datasets_by_class.append((X_total[class_indices], y[class_indices]))

        self.attack_models_ = []

        dataset_iter = tqdm(datasets_by_class)
        for i, (X_train, y_train) in enumerate(dataset_iter):
            model = self.model_fn()
            fit_kwargs = fit_kwargs or {}
            model.fit(X_train, y_train, **fit_kwargs)

            self.attack_models_.append(model)

    def predict(self, X):
        result = np.zeros((X.shape[0], 2))
        shadow_preds = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]

        data_indices = np.arange(shadow_preds.shape[0])
        for i in range(self.num_classes):
            model = self.attack_models_[i]
            class_indices = data_indices[np.argmax(classes, axis=1) == i]

            membership_preds = model.predict(shadow_preds[class_indices])
            for j, example_index in enumerate(class_indices):
                prob = np.squeeze(membership_preds[j])
                result[example_index, 1] = prob
                result[example_index, 0] = 1 - prob

        return result[:, 1] > 0.5

################################################################################
#                                     MAIN                                     #
################################################################################

def main():
    (X_train, y_train), (X_test, y_test) = get_data()

    # Train the target model.
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(X_train, y_train, epochs=TARGET_EPOCHS, validation_split=0.1)

    # Train the shadow models.
    smb = ShadowModelBundle(target_model_fn,
                            shadow_dataset_size=SHADOW_DATASET_SIZE,
                            num_models=NUM_SHADOWS)

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(X_test,
                                                                                            y_test,
                                                                                            test_size=0.1)
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(attacker_X_train,
                                           attacker_y_train,
                                           fit_kwargs=dict(epochs=TARGET_EPOCHS,
                                                           validation_data=(attacker_X_test,
                                                                            attacker_y_test)))

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=ATTACK_EPOCHS))

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(target_model, data_in, data_out)

    # Compute the learning task accuracy
    learning_guesses = target_model.predict(X_test)
    learning_accuracy = np.mean(np.argmax(learning_guesses, 1) == np.argmax(y_test, 1))

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print("Learning task accuracy: {}".format(learning_accuracy))
    print("Attack accuracy: {}".format(attack_accuracy))

if __name__ == "__main__":
    main()

