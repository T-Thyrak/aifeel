import matplotlib.pyplot as plt
import numpy as np
from rich import print
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.model_selection import cross_val_score


def evaluate_model(initialized_model, loaded_model, model_name, X_train, y_train, X_test, y_test):
    cross_score = cross_val_score(initialized_model, X_train, y_train, cv=10)
    
    y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Evaluation:")
    print(f"Cross-Validation Scores: {cross_score}")
    print(f"Cross-Validation Average Score: {cross_score.mean():.2f}")
    print(f"Accuracy Scores: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot()
    disp.ax_.set(title='Confusion Matrix')
    plt.show()

   # Plot ROC curve
    y_scores = loaded_model.predict_proba(X_test)[:, 1]
    
    # Convert string labels to integers
    y_test_int = y_test.astype(int)
    
    fpr, tpr, _ = roc_curve(y_test_int, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
