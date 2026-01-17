# ml/src/evaluate.py
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, out_path="ml/reports/metrics.json"):
    """Evaluate model and save metrics + confusion matrix."""
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs > 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Plot confusion matrix
    plt.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("ml/reports/confusion_matrix.png")
    plt.close()

    return report, cm
