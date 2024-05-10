def calculate_precision(y_true, y_pred):
    if isinstance(y_true[0], list) and isinstance(y_pred[0], list):
        true_positives = sum(
            len(set(true) & set(pred)) for true, pred in zip(y_true, y_pred)
        )
        total_predicted = sum(len(pred) for pred in y_pred)
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total_predicted = len(y_pred)
    else:
        raise ValueError(
            "Both y_true and y_pred should be either list of strings or list of list of strings"
        )

    if total_predicted == 0:
        return 0  # to avoid division by zero
    else:
        return true_positives / total_predicted


def calculate_recall(y_true, y_pred):
    if isinstance(y_true[0], list) and isinstance(y_pred[0], list):
        true_positives = sum(
            len(set(true) & set(pred)) for true, pred in zip(y_true, y_pred)
        )
        total_actual = sum(len(true) for true in y_true)
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total_actual = len(y_true)
    else:
        raise ValueError(
            "Both y_true and y_pred should be either list of strings or list of list of strings"
        )

    if total_actual == 0:
        return 0  # to avoid division by zero
    else:
        return true_positives / total_actual


def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0  # to avoid division by zero
    else:
        return 2 * (precision * recall) / (precision + recall)


y_true = [
    ["A", "B", "C"],
    ["A", "B", "C"],
    ["A", "B", "C"],
]
y_pred = [["A", "B", "C"], ["A", "C", "B"], ["A"]]

precision = calculate_precision(y_true, y_pred)
print("Precision:", precision)

recall = calculate_recall(y_true, y_pred)
print("Recall:", recall)

f1_score = calculate_f1_score(y_true, y_pred)
print("F1 Score:", f1_score)


a = "There are two bananas on the table."
b = "On the table are two apples."
