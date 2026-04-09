import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from datetime import datetime

# ── CONFIG — edit these paths ──────────────────────────────────────────────────
MODEL_PATH    = "alzheimer_cnn_model.h5"
TEST_DIR      = "Data/raw"
IMG_SIZE      = (128, 128)
BATCH_SIZE    = 32
REPORT_SAVE   = "evaluation_report.png"

CLASS_NAMES   = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
STAGE_COLORS  = ["#2ecc71", "#3ab5e6", "#f5a623", "#e53935"]

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model(np.zeros((1, 128, 128, 1), dtype=np.float32), training=False)
print("Model loaded.\n")

# ── Load test images ───────────────────────────────────────────────────────────
def load_test_data(test_dir):
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  WARNING: folder not found — {class_dir}")
            continue
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {class_name}: {len(files)} images")
        for fname in files:
            path = os.path.join(class_dir, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_idx)
    images = np.array(images)[..., np.newaxis]   # (N,128,128,1)
    labels = np.array(labels)
    return images, labels

print("Loading test data...")
X_test, y_true = load_test_data(TEST_DIR)
print(f"\nTotal test samples: {len(X_test)}\n")

if len(X_test) == 0:
    raise ValueError(f"No images found in {TEST_DIR}. Check your folder structure.")

# ── Run predictions ────────────────────────────────────────────────────────────
print("Running predictions...")
y_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
y_pred  = np.argmax(y_probs, axis=1)

# ── Metrics ────────────────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# AUC-ROC (one-vs-rest)
y_bin = label_binarize(y_true, classes=[0,1,2,3])
try:
    auc = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='weighted')
except Exception:
    auc = None

print("\n" + "="*55)
print("  MODEL EVALUATION RESULTS")
print("="*55)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  (weighted)")
print(f"  Recall    : {recall:.4f}  (weighted)")
print(f"  F1 Score  : {f1:.4f}  (weighted)")
if auc: print(f"  AUC-ROC   : {auc:.4f}  (weighted OvR)")
print("="*55)
print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Plot ────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor='#0a1525')
fig.suptitle("Alzheimer's Model — Evaluation Report",
             fontsize=15, color='#e8f1fb', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# ── 1. Summary metrics bar ──
ax_met = fig.add_subplot(gs[0, :2])
ax_met.set_facecolor('#080f1a')
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [accuracy, precision, recall, f1]
metric_colors = ['#2ecc71', '#3ab5e6', '#f5a623', '#e53935']
bars = ax_met.barh(metric_names, metric_values,
                  color=metric_colors, alpha=0.85, height=0.5)
ax_met.set_xlim(0, 1.15)
ax_met.set_xlabel('Score', color='#7a99bb', fontsize=9)
ax_met.set_title('Overall Metrics', color='#e8f1fb', fontsize=11, pad=10)
ax_met.tick_params(colors='#7a99bb', labelsize=9)
for sp in ax_met.spines.values(): sp.set_edgecolor('#1e3050')
ax_met.grid(axis='x', color='#1e3050', linewidth=0.5)
ax_met.set_axisbelow(True)
for bar, val, color in zip(bars, metric_values, metric_colors):
    ax_met.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', color=color,
               fontsize=9, fontweight='600')

# ── 2. AUC score box ──
ax_auc = fig.add_subplot(gs[0, 2])
ax_auc.set_facecolor('#080f1a')
ax_auc.axis('off')
auc_text = f"{auc:.4f}" if auc else "N/A"
ax_auc.text(0.5, 0.65, 'AUC-ROC', ha='center', va='center',
           transform=ax_auc.transAxes, fontsize=11, color='#7a99bb')
ax_auc.text(0.5, 0.42, auc_text, ha='center', va='center',
           transform=ax_auc.transAxes, fontsize=28, fontweight='bold',
           color='#4fc3f7')
ax_auc.text(0.5, 0.22, 'Weighted OvR', ha='center', va='center',
           transform=ax_auc.transAxes, fontsize=9, color='#7a99bb')
for sp in ax_auc.spines.values(): sp.set_edgecolor('#1e3050')

# ── 3. Confusion matrix ──
ax_cm = fig.add_subplot(gs[1, :2])
ax_cm.set_facecolor('#080f1a')
cm = confusion_matrix(y_true, y_pred)
short_names = ['ND', 'VMD', 'MD', 'ModD']
sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm,
            cmap='Blues', linewidths=0.5, linecolor='#1e3050',
            xticklabels=short_names, yticklabels=short_names,
            annot_kws={"size": 11, "color": "white"})
ax_cm.set_title('Confusion Matrix', color='#e8f1fb', fontsize=11, pad=10)
ax_cm.set_xlabel('Predicted', color='#7a99bb', fontsize=9)
ax_cm.set_ylabel('Actual', color='#7a99bb', fontsize=9)
ax_cm.tick_params(colors='#7a99bb', labelsize=9)

# ── 4. Per-class F1 ──
ax_f1 = fig.add_subplot(gs[1, 2])
ax_f1.set_facecolor('#080f1a')
per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
bars2  = ax_f1.bar(short_names, per_f1,
                   color=STAGE_COLORS, alpha=0.85, width=0.55)
ax_f1.set_ylim(0, 1.15)
ax_f1.set_title('F1 per Class', color='#e8f1fb', fontsize=11, pad=10)
ax_f1.set_ylabel('F1 Score', color='#7a99bb', fontsize=9)
ax_f1.tick_params(colors='#7a99bb', labelsize=8)
for sp in ax_f1.spines.values(): sp.set_edgecolor('#1e3050')
ax_f1.yaxis.grid(True, color='#1e3050', linewidth=0.5)
ax_f1.set_axisbelow(True)
for bar, val, color in zip(bars2, per_f1, STAGE_COLORS):
    ax_f1.text(bar.get_x()+bar.get_width()/2, val+0.02,
              f'{val:.2f}', ha='center', fontsize=8,
              color=color, fontweight='600')

# ── 5. ROC curves ──
ax_roc = fig.add_subplot(gs[2, :2])
ax_roc.set_facecolor('#080f1a')
if auc:
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, STAGE_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        cls_auc     = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{cls[:7]}.. (AUC={cls_auc:.2f})')
ax_roc.plot([0,1],[0,1], color='#1e3050', linestyle='--', linewidth=1)
ax_roc.set_xlabel('False Positive Rate', color='#7a99bb', fontsize=9)
ax_roc.set_ylabel('True Positive Rate', color='#7a99bb', fontsize=9)
ax_roc.set_title('ROC Curves (one-vs-rest)', color='#e8f1fb', fontsize=11, pad=10)
ax_roc.tick_params(colors='#7a99bb', labelsize=8)
ax_roc.legend(fontsize=7.5, facecolor='#0e1829', labelcolor='#e8f1fb',
             edgecolor='#1e3050', loc='lower right')
for sp in ax_roc.spines.values(): sp.set_edgecolor('#1e3050')
ax_roc.grid(color='#1e3050', linewidth=0.4)

# ── 6. Sample count per class ──
ax_cnt = fig.add_subplot(gs[2, 2])
ax_cnt.set_facecolor('#080f1a')
counts = [np.sum(y_true == i) for i in range(4)]
bars3  = ax_cnt.bar(short_names, counts,
                   color=STAGE_COLORS, alpha=0.85, width=0.55)
ax_cnt.set_title('Test Samples / Class', color='#e8f1fb', fontsize=11, pad=10)
ax_cnt.set_ylabel('Count', color='#7a99bb', fontsize=9)
ax_cnt.tick_params(colors='#7a99bb', labelsize=8)
for sp in ax_cnt.spines.values(): sp.set_edgecolor('#1e3050')
ax_cnt.yaxis.grid(True, color='#1e3050', linewidth=0.5)
ax_cnt.set_axisbelow(True)
for bar, cnt, color in zip(bars3, counts, STAGE_COLORS):
    ax_cnt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(cnt), ha='center', fontsize=8,
            color=color, fontweight='600')

# ── Save ────────────────────────────────────────────────────────────────────────
plt.savefig(REPORT_SAVE, dpi=150, bbox_inches='tight', facecolor='#0a1525')
plt.close(fig)
print(f"\nEvaluation report saved → {REPORT_SAVE}")
print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")