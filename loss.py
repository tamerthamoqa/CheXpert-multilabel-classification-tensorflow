import tensorflow.keras.backend as K
import numpy as np


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
