import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import NumpyEncoder, plot_metrics


parser = argparse.ArgumentParser(description="Training CheXpert multi-label classification model.")
parser.add_argument('--data_dir', type=str, default="dataset/",
                    help="(Required) Path to the CheXpert dataset folder (default: 'dataset/')"
                    )
parser.add_argument('--model_architecture', type=str, default="densenet201", choices=["densenet201", "inceptionresnetv2", "resnet152"],
                    help="The required model architecture for training: ('densenet201','inceptionresnetv2', 'resnet152'), (default: 'densenet201')"
                    )
parser.add_argument('--weighted_classes', default=True, type=bool,
                    help="If set to True, train model with sample weighting; the sample weights per class would be calculated from the training set by the Data Generator (default: True)"
                    )
parser.add_argument('--train_multi_gpu', default=False, type=bool,
                    help="If set to True, train model with multiple GPUs. (default: False)"
                    )
parser.add_argument('--num_gpus', default=1, type=int,
                    help="Set number of available GPUs for multi-gpu training, '--train_multi_gpu' must be also set to True  (default: 1)"
                    )
parser.add_argument('--training_epochs', default=15, type=int,
                    help="Required training epochs (default: 15)"
                    )
parser.add_argument('--resume_train', default=False, type=bool,
                    help="If set to True, resume model training from model_path (default: False)"
                    )
parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adam", "nadam"],
                    help="Required optimizer for training the model: ('sgd','adam','nadam'), (default: 'adam')"
                    )
parser.add_argument('--lr', default=0.0001, type=float,
                    help="Learning rate for the optimizer (default: 0.0001)"
                    )
parser.add_argument('--use_nesterov_sgd', default=False, type=bool,
                    help="Use Nesterov momentum with SGD optimizer: ('True', 'False') (default: False)"
                    )
parser.add_argument('--use_amsgrad_adam', default=False, type=bool,
                    help="Use AMSGrad with adam optimizer: ('True', 'False') (default: False)"
                    )
parser.add_argument('--batch_size', default=16, type=int,
                    help="Input batch size, if --train_multi_gpu then the minimum value must be the number of GPUs (default: 16)"
                    )
parser.add_argument('--image_height', default=512, type=int,
                    help="Input image height (default: 512)"
                    )
parser.add_argument('--image_width', default=512, type=int,
                    help="Input image width (default: 512)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for fit_generator (default: 4)"
                    )
args = parser.parse_args()


def set_tensorflow_mirrored_strategy_gpu_devices_list(num_gpus):
    gpu_devices = [""] * num_gpus

    for i in range(num_gpus):
        gpu_devices[i] = f"/gpu:{i}"  # e.g: devices=["/gpu:0", "/gpu:1"]

    return gpu_devices


def set_model_architecture(model_architecture, image_height, image_width):
    if model_architecture == "densenet201":
        base_model = DenseNet201(
            weights=None,  # Set to None since input will be grayscale images instead of RGB images
            include_top=False,
            input_shape=(image_height, image_width, 1)
        )

    elif model_architecture == "inceptionresnetv2":
        base_model = InceptionResNetV2(
            weights=None,  # Set to None since input will be grayscale images instead of RGB images
            include_top=False,
            input_shape=(image_height, image_width, 1)
        )

    elif model_architecture == "resnet152":
        base_model = ResNet152(
            weights=None,  # Set to None since input will be grayscale images instead of RGB images
            include_top=False,
            input_shape=(image_height, image_width, 1)
        )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(units=14, activation="sigmoid")(x)  # For multi-label classification
    model = Model(inputs=base_model.input, outputs=predictions)

    print("Using {} model architecture.".format(model_architecture))

    return model


def set_optimizer(optimizer, learning_rate, use_nesterov_sgd, use_amsgrad_adam):
    if optimizer == "sgd":
        optimizer = optimizers.SGD(
            lr=learning_rate,
            momentum=0.9,
            nesterov=use_nesterov_sgd
        )

    elif optimizer == "adam":
        optimizer = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.1,
            amsgrad=use_amsgrad_adam
        )

    elif optimizer == "nadam":
        optimizer = optimizers.Nadam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.1
        )

    return optimizer


def main():
    data_dir = args.data_dir
    model_architecture = args.model_architecture
    train_multi_gpu = args.train_multi_gpu
    num_gpus = args.num_gpus
    training_epochs = args.training_epochs
    resume_train = args.resume_train
    optimizer = args.optimizer
    learning_rate = args.lr
    use_nesterov_sgd = args.use_nesterov_sgd
    use_amsgrad_adam = args.use_amsgrad_adam
    batch_size = args.batch_size
    image_height = args.image_height
    image_width = args.image_width
    num_workers = args.num_workers

    train_df = pd.read_csv(
        filepath_or_buffer="labels/train_u-zeroes.csv",
        dtype={  # Setting labels to type np.int32 was necessary for conversion to tf.Tensor object
            "Path": str,
            "Atelectasis": np.int32,
            "Cardiomegaly": np.int32,
            "Consolidation": np.int32,
            "Edema": np.int32,
            "Pleural Effusion": np.int32,
            "Pleural Other": np.int32,
            "Pneumonia": np.int32,
            "Pneumothorax": np.int32,
            "Enlarged Cardiomediastinum": np.int32,
            "Lung Opacity": np.int32,
            "Lung Lesion": np.int32,
            "Fracture": np.int32,
            "Support Devices": np.int32,
            "No Finding": np.int32
        }
    )

    val_df = pd.read_csv(
        filepath_or_buffer="labels/validation_u-zeroes.csv",
        dtype={  # Setting labels to type np.int32 was necessary for conversion to tf.Tensor object
            "Path": str,
            "Atelectasis": np.int32,
            "Cardiomegaly": np.int32,
            "Consolidation": np.int32,
            "Edema": np.int32,
            "Pleural Effusion": np.int32,
            "Pleural Other": np.int32,
            "Pneumonia": np.int32,
            "Pneumothorax": np.int32,
            "Enlarged Cardiomediastinum": np.int32,
            "Lung Opacity": np.int32,
            "Lung Lesion": np.int32,
            "Fracture": np.int32,
            "Support Devices": np.int32,
            "No Finding": np.int32
        }
    )

    list_columns = list(train_df.columns)
    y_cols = list_columns[1::]  # First column is 'Path' column

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1./255.
    )

    train_datagenerator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=data_dir,
        x_col='Path',
        y_col=y_cols,
        weight_col=None,
        target_size=(512, 512),
        color_mode='grayscale',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True,
        validate_filenames=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255.
    )

    val_datagenerator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=data_dir,
        x_col='Path',
        y_col=y_cols,
        weight_col=None,
        target_size=(512, 512),
        color_mode='grayscale',
        class_mode='raw',
        batch_size=batch_size,
        validate_filenames=True
    )

    # Set GPU devices list for Tensorflow MirroredStrategy() 'devices' parameter for Multi-GPU training:
    gpu_devices = set_tensorflow_mirrored_strategy_gpu_devices_list(num_gpus=num_gpus)

    optimizer = set_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        use_nesterov_sgd=use_nesterov_sgd,
        use_amsgrad_adam=use_amsgrad_adam
    )

    auc = tf.keras.metrics.AUC(
        name="auc",
        multi_label=True
    )

    model_path = f"{model_architecture}.h5"

    # Path 1: Resume training from model checkpoint
    if resume_train:
        # Multi GPU training
        if train_multi_gpu:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            with strategy.scope():
                model = load_model(model_path)
                # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
                model.compile(optimizer=model.optimizer, metrics=[auc, "accuracy"], loss="binary_crossentropy")

        # Single-GPU training
        else:
            model = load_model(model_path)
            # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
            model.compile(optimizer=model.optimizer, metrics=[auc, "accuracy"], loss="binary_crossentropy")

        # Change Learning Rate
        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)

    # Path 2: Train from scratch
    else:
        # Multi GPU training
        if train_multi_gpu:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            with strategy.scope():
                model = set_model_architecture(
                    model_architecture=model_architecture,
                    image_height=image_height,
                    image_width=image_width
                )
                model.compile(optimizer=optimizer, metrics=[auc, "accuracy"], loss="binary_crossentropy")
        # Single GPU training
        else:
            model = set_model_architecture(
                model_architecture=model_architecture,
                image_height=image_height,
                image_width=image_width
            )
            model.compile(optimizer=optimizer, metrics=[auc, "accuracy"], loss="binary_crossentropy")

    print(model.summary())

    if train_multi_gpu:
        print("Training on Multi-GPU mode!\n")
    else:
        print("Training on Single-GPU mode!\n")

    reducelronplateau = ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.1,
        patience=2,
        verbose=1,
        mode="max",
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_auc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max'
    )

    fit = model.fit(
        x=train_datagenerator,
        epochs=training_epochs,
        validation_data=val_datagenerator,
        verbose=1,
        callbacks=[reducelronplateau, checkpoint],
        workers=num_workers
    )

    # Modified to fix the 'np.float32 is not JSON serializable issue'
    dumped = json.dumps(fit.history, cls=NumpyEncoder)
    with open(f'{model_architecture}_model_history.txt', 'w') as f:
        json.dump(dumped, f)

    # Plot train losses and validation losses
    plot_metrics(model_history=fit.history, model_architecture=model_architecture, stop=training_epochs)


if __name__ == '__main__':
    main()