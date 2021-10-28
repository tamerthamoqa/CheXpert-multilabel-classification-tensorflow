import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
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
parser.add_argument('--train_multi_gpu', default=False, type=bool,
                    help="If set to True, train model with multiple GPUs. (default: False)"
                    )
parser.add_argument('--num_gpus', default=1, type=int,
                    help="Set number of available GPUs for multi-gpu training, '--train_multi_gpu' must be also set to True  (default: 1)"
                    )
parser.add_argument('--training_epochs', default=25, type=int,
                    help="Required training epochs (default: 25)"
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


def compute_class_weights(labels):
    """
    Note: Imported from the AI for Medicine Specialization course on Coursera: Assignment 1 Week 1.
    Compute positive and negative weights for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)

    Returns:
        positive_weights (np.array): array of positive weights for each
                                         class, size (num_classes)
        negative_weights (np.array): array of negative weights for each
                                         class, size (num_classes)
    """
    # total number of patients (rows).
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = np.sum(labels == 0, axis=0) / N

    positive_weights = negative_frequencies
    negative_weights = positive_frequencies

    return positive_weights, negative_weights


def set_binary_crossentropy_weighted_loss(positive_weights, negative_weights, epsilon=1e-7):
    """
    Note: Imported from the AI for Medicine Specialization course on Coursera: Assignment 1 Week 1.
    Returns weighted binary cross entropy loss function given negative weights and positive weights.

    Args:
      positive_weights (np.array): array of positive weights for each class, size (num_classes)
      negative_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """
    def binary_crossentropy_weighted_loss(y_true, y_pred):
        """
        Returns weighted binary cross entropy loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)

        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(positive_weights)):
            # for each class, add average weighted loss for that class
            loss += -1 * K.mean((positive_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +
                                 negative_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss

    return binary_crossentropy_weighted_loss


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
        dtype={  # Setting labels to type np.float32 was necessary for conversion to tf.Tensor object
            "Path": str,
            "Atelectasis": np.float32,
            "Cardiomegaly": np.float32,
            "Consolidation": np.float32,
            "Edema": np.float32,
            "Pleural Effusion": np.float32,
            "Pleural Other": np.float32,
            "Pneumonia": np.float32,
            "Pneumothorax": np.float32,
            "Enlarged Cardiomediastinum": np.float32,
            "Lung Opacity": np.float32,
            "Lung Lesion": np.float32,
            "Fracture": np.float32,
            "Support Devices": np.float32,
            "No Finding": np.float32
        }
    )

    val_df = pd.read_csv(
        filepath_or_buffer="labels/validation_u-zeroes.csv",
        dtype={  # Setting labels to type np.float32 was necessary for conversion to tf.Tensor object
            "Path": str,
            "Atelectasis": np.float32,
            "Cardiomegaly": np.float32,
            "Consolidation": np.float32,
            "Edema": np.float32,
            "Pleural Effusion": np.float32,
            "Pleural Other": np.float32,
            "Pneumonia": np.float32,
            "Pneumothorax": np.float32,
            "Enlarged Cardiomediastinum": np.float32,
            "Lung Opacity": np.float32,
            "Lung Lesion": np.float32,
            "Fracture": np.float32,
            "Support Devices": np.float32,
            "No Finding": np.float32
        }
    )

    list_columns = list(train_df.columns)
    y_cols = list_columns[1::]  # First column is 'Path' column

    training_dataset_mean = np.load("misc/training_dataset_mean_and_std_values/CheXpert_training_set_mean.npy")
    training_dataset_std = np.load("misc/training_dataset_mean_and_std_values/CheXpert_training_set_std.npy")

    train_datagen = ImageDataGenerator(
        featurewise_center=True,  # Mean and standard deviation values of the training set will be loaded to the object
        featurewise_std_normalization=True,
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=False,  # Some labels would be heavily affected by this change if it is True
        vertical_flip=False,  # Not suitable for Chest X-ray images if it is True
    )

    # Set training dataset mean and std values for feature_wise centering and std normalization
    train_datagen.mean = training_dataset_mean
    train_datagen.std = training_dataset_std

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
        featurewise_center=True,  # Mean and standard deviation values of the training set will be loaded to the object
        featurewise_std_normalization=True
    )

    # Set training dataset mean and std values for feature_wise centering and std normalization
    val_datagen.mean = training_dataset_mean
    val_datagen.std = training_dataset_std

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

    positive_weights, negative_weights = compute_class_weights(labels=train_datagenerator.labels)
    print(f"\nPositive Weights: {positive_weights}")
    print(f"Negative Weights: {negative_weights}\n")

    loss = set_binary_crossentropy_weighted_loss(
        positive_weights=positive_weights,
        negative_weights=negative_weights,
        epsilon=1e-7
    )

    accuracy = tf.keras.metrics.BinaryAccuracy()
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
                model.compile(optimizer=model.optimizer, metrics=[auc, accuracy], loss=loss)

        # Single-GPU training
        else:
            model = load_model(model_path)
            # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
            model.compile(optimizer=model.optimizer, metrics=[auc, accuracy], loss=loss)

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
                model.compile(optimizer=optimizer, metrics=[auc, accuracy], loss=loss)
        # Single GPU training
        else:
            model = set_model_architecture(
                model_architecture=model_architecture,
                image_height=image_height,
                image_width=image_width
            )
            model.compile(optimizer=optimizer, metrics=[auc, accuracy], loss=loss)

    print(f"\n{model.summary()}\n")

    if train_multi_gpu:
        print("Training on Multi-GPU mode!\n")
    else:
        print("Training on Single-GPU mode!\n")

    reducelronplateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        verbose=1,
        mode="min",
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
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
