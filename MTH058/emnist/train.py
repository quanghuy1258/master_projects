import common
import resnet
import densenet
import read_dataset as reader

# Default params
batch_size = 128
epochs = 50
checkpoint_format = "weights.{epoch:04d}-{val_loss:.2f}.h5"
period = 5


def Train(model, X, y, dir):
    # Create training dir
    common.create_dir_if_not_exists(dir)
    model.fix(X, y, epochs=epochs, batch_size=batch_size,
              callbacks=[common.create_checkpoint_callback("{}/{}".format(dir, checkpoint_format), period),
                         common.create_CSVLogger_callback(dir)],
              validation_split=0.2, shuffle=True)
    

# modelRN = resnet.ResNet50()
# modelRN.fit(X, y, epochs=100, batch_size=128,
#           callbacks=[common.create_checkpoint_callback("{}/{}".format(training_dir, checkpoint_format), period),
#                      common.create_CSVLogger_callback(training_dir)],
#           validation_split=0.2, shuffle=True)

X, y = reader.getTrainData()
Train(densenet.densenet(), X, y, "densenet_training")
