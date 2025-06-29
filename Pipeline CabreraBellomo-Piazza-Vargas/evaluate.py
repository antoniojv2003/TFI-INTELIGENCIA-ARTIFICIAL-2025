from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from config import COMPETITION_TASKS, THRESHOLDS          # añade THRESHOLDS

def evaluate_model(model, valid_dataset, valid_df):
    y_pred_probs = model.predict(valid_dataset)
    y_true = valid_df[COMPETITION_TASKS].values

    print("\nMétricas del Modelo en el Conjunto de Validación:")
    for i, task in enumerate(COMPETITION_TASKS):
        prob_vec = y_pred_probs[:, i]
        thr      = THRESHOLDS[task]                       # ← umbral propio
        y_pred   = (prob_vec > thr).astype(int)

        print(f"\n----- {task}  (threshold = {thr}) -----")
        try:
            auc = roc_auc_score(y_true[:, i], prob_vec)
            print(f"ROC AUC: {auc:.4f}")
        except ValueError:
            print("ROC AUC no está definido para esta clase.")

        precision = precision_score(y_true[:, i], y_pred, zero_division=0)
        recall    = recall_score(y_true[:, i], y_pred, zero_division=0)
        f1        = f1_score(y_true[:, i], y_pred, zero_division=0)

        print(f"Precisión: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Puntuación F1: {f1:.4f}")
