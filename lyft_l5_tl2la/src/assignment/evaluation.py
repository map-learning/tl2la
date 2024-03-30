# ------------------------------------------------------------------------
# Functions to evaluate classification results
# Lyft Lvl 5 Dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics


def get_metrics(y_target: pd.Series, y_pred:  pd.Series) -> Tuple[float,...]:
    """Visualize confusion matrix

    Args:
        y_target (pd.Series): target label of classes
        y_pred (pd.Series): predicted label of clases
        
    Returns:
        (Tuple[flaot, ...]): acc, prec, rec, f1
    """
    accuracy = metrics.accuracy_score(y_target, y_pred)
    precision = metrics.precision_score(y_target, y_pred)
    recall = metrics.recall_score(y_target, y_pred)
    f1_score = metrics.f1_score(y_target, y_pred)
    return accuracy, precision, recall, f1_score


def visualize_confusion_matrix(y_target: pd.Series, y_pred:  pd.Series, title: str, save_path: Optional[str] = None):
    """Visualize confusion matrix

    Args:
        y_target (pd.Series): target label of classes
        y_pred (pd.Series): predicted label of clases
        title (str): Title of plot
        save_path (str): save plot to given path
    """
    confusion_matrix = metrics.confusion_matrix(y_target, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    cm_display.plot()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_metrics(y_target: pd.Series, y_pred:  pd.Series, title: str, save_path: Optional[str] = None):
    """Visualize metrics in a table

    Args:
        y_target (pd.Series): target label of classes
        y_pred (pd.Series): predicted label of clases
        title (str): Title of the table
        save_path (str): save table to file under given path
    """
    accuracy = metrics.accuracy_score(y_target, y_pred)
    precision = metrics.precision_score(y_target, y_pred)
    recall = metrics.recall_score(y_target, y_pred)
    f1_score = metrics.f1_score(y_target, y_pred)

    beta = 8  # beta > 0 focuses more on recall (relevant classes)
    fbeta_score = metrics.fbeta_score(y_target, y_pred, beta=beta)

    table = PrettyTable(field_names=["Metrics", "Results [%]"])
    table.add_row(["ACCURACY", round(accuracy*100, 1)])
    table.add_row(["PRECISION", round(precision*100,1)])
    table.add_row(["RECALL", round(recall*100,1)])

    table.add_row(["F_1", round(f1_score*100, 1)])
    table.add_row(["F_beta", round(fbeta_score*100,1)])

    # print(title)
    print(table)

    if not save_path:
        return

    with open(save_path, 'w') as f:
        print(title)
        print(table, file=f)

