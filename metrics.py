from sklearn.model_selection import KFold
import numpy as np

def compute_roc(distances, matches, thresholds, fold_size=10):
    """
    calculate tpr, fpr, accuracy using KFold
    """
    assert(len(distances) == len(matches))

    m = len(distances)
    kf = KFold(n_splits=fold_size, shuffle=False)

    tpr = np.zeros(fold_size, len(thresholds))
    fpr = np.zeros(fold_size, len(thresholds))
    accuracy = np.zeros(fold_size)

    for fold_index, (training_indices, val_indices) in enumerate(kf.split(range(m))):
        training_distances = distances[training_indices]
        training_matches = matches[training_indices]

        # 1. find the best threshold for this fold using training set
        best_threshold_true_predicts = 0
        best_threshold_index = 0
        for threshold_index, threshold in enumerate(thresholds):
            true_predicts = np.sum((training_distances < threshold) == training_matches)
            if true_predicts > best_threshold_true_predicts:
                best_threshold_true_predicts = true_predicts
                best_threshold_index = threshold_index

        # 2. using the best threshold, calculate tpr, fpr, accuracy on validation set
        best_threshold = thresholds[best_threshold_index]
        val_distances = distances[val_indices]
        val_matches = matches[val_indices]
        for threshold_index, threshold in enumerate(thresholds):
            predicts = val_distances < best_threshold

            tp = np.sum(np.logical_and(predicts, matches))
            fp = np.sum(np.logical_and(predicts, np.logical_not(matches)))
            tn = np.sum(np.logical_and(np.logical_not(predicts), np.logical_not(matches)))
            fn = np.sum(np.logical_and(np.logical_not(predicts), matches))

            tpr[fold_index][threshold_index] = tp / (tp + fn)
            fpr[fold_index][threshold_index] = fp / (fp + tn)

        accuracy[fold_index] = best_threshold_true_predicts / m

    # average fold
    tpr = np.mean(tpr, 0)
    fpr = np.mean(fpr, 0)
    accuracy = np.mean(accuracy, 0)

    return tpr, fpr, accuracy

def compute_map(distances, matches, thresholds):
    """
    calculate tpr, fpr, accuracy using KFold
    """
    assert(len(distances) == len(matches))

    m = len(distances)
    kf = KFold(n_splits=fold_size, shuffle=False)

    tpr = np.zeros(fold_size, len(thresholds))
    fpr = np.zeros(fold_size, len(thresholds))
    accuracy = np.zeros(fold_size)

    for fold_index, (training_indices, val_indices) in enumerate(kf.split(range(m))):
        training_distances = distances[training_indices]
        training_matches = matches[training_indices]

        # 1. find the best threshold for this fold using training set
        best_threshold_true_predicts, best_threshold_index = \
                select_best_threshold(training_distances, training_matches, thresholds)
        best_threshold = thresholds[best_threshold_index]

        # 2. using the best threshold, calculate tpr, fpr, accuracy on validation set
        val_distances = distances[val_indices]
        val_matches = matches[val_indices]
        for threshold_index, threshold in enumerate(thresholds):
            predicts = val_distances < best_threshold

            tp = np.sum(np.logical_and(predicts, matches))
            fp = np.sum(np.logical_and(predicts, np.logical_not(matches)))
            tn = np.sum(np.logical_and(np.logical_not(predicts), np.logical_not(matches)))
            fn = np.sum(np.logical_and(np.logical_not(predicts), matches))

            tpr[fold_index][threshold_index] = tp / (tp + fn)
            fpr[fold_index][threshold_index] = fp / (fp + tn)

        accuracy[fold_index] = best_threshold_true_predicts / m

    # average fold
    tpr = np.mean(tpr, 0)
    fpr = np.mean(fpr, 0)
    accuracy = np.mean(accuracy, 0)

    return tpr, fpr, accuracy

def select_best_threshold(distances, matches, thresholds):
    best_threshold_true_predicts = 0
    best_threshold_index = 0
    for threshold_index, threshold in enumerate(thresholds):
        true_predicts = np.sum((training_distances < threshold) == training_matches)
        if true_predicts > best_threshold_true_predicts:
            best_threshold_true_predicts = true_predicts
            best_threshold_index = threshold_index
    return best_threshold_true_predicts, best_threshold_index