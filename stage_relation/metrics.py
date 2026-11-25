import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p.argmax(1)
    qwk = float(cohen_kappa_score(y, pred, weights='quadratic', labels=[0, 1, 2, 3])) if len(set(y) | set(pred)) > 1 else None
    return {
        'count': len(y), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
        'qwk': qwk if qwk is not None and np.isfinite(qwk) else None,
        'auc_macro_ovr': float(roc_auc_score(y, p, multi_class='ovr', labels=[0, 1, 2, 3])) if len(set(y)) == 4 else None,
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
    }
