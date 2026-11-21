#!/usr/bin/env python
"""
Generate publication-quality figures for DMLPFFN manuscript.
All figures saved as SVG and PDF in publication/figures/
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.patheffects as path_effects

# Set publication style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIG_DIR = r"C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\publication\figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Figure 1: Overall Research Workflow (6 stages)
# ============================================================
def create_figure1_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
    stages = [
        ("1. Dataset\nAcquisition",
         "USDA Hyperspectral\nTomato Leaf Dataset\n2 sensors: VIS-NIR (300)\n+ SWIR (168 bands)\nDAI 2-16 post-inoculation"),
        ("2. Preprocessing\nPipeline",
         "Water band removal\n(94-115 SWIR)\nSensor-specific\nIncrementalPCA (128)\nTrain-only StandardScaler\n9×9 patches, stride 4"),
        ("3. Group-Wise\nData Split",
         "Biological unit grouping\n(run number / leaf)\nZero spatial leakage\nTrain/Val/Test: 75.4/13.8/10.8%\nFrozen seed=42 manifest"),
        ("4. DMLPFFN\nArchitecture",
         "3-Branch Dual MLP:\n• Global Perceptron\n• Partition Perceptron\n• Local Perceptron (d=1,2,5)\nElement-wise fusion\n447,780 params"),
        ("5. Training\n& Regularization",
         "AdamW, lr=2e-4, wd=1e-4\nWeightedRandomSampler\nDropout=0.7, LabelSmooth=0.1\nMixUp(0.2)+CutMix(1.0)\nCosineAnnealingWarmRestarts\nEarlyStop(patience=25)"),
        ("6. Conditional Binary\nEvaluation",
         "P(Healthy|H/L)=e^z0/(e^z0+e^z3)\nP(Late|H/L)=e^z3/(e^z0+e^z3)\nExact marginalization from\n4-class logits\nGroup-aware bootstrap CI\nMcNemar's test")
    ]

    box_w, box_h = 2.0, 3.2
    start_x = 0.5
    gap = 0.3

    for i, (title, desc) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        y = 0.3

        # Main box
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=colors[i], edgecolor='white', linewidth=2
        )
        ax.add_patch(box)

        # Title
        ax.text(x + box_w/2, y + box_h - 0.35, title,
                ha='center', va='top', fontsize=10, fontweight='bold', color='white')

        # Description
        ax.text(x + box_w/2, y + box_h - 0.85, desc,
                ha='center', va='top', fontsize=7.5, color='white', linespacing=1.2)

        # Arrow to next
        if i < len(stages) - 1:
            ax.annotate('',
                xy=(x + box_w + gap/2, y + box_h/2),
                xytext=(x + box_w, y + box_h/2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    plt.savefig(os.path.join(FIG_DIR, 'figure1_workflow.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure1_workflow.pdf'), format='pdf')
    plt.close()
    print("Figure 1 saved")

# ============================================================
# Figure 2: Preprocessing Pipeline
# ============================================================
def create_figure2_preprocessing():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Preprocessing Pipeline (Leakage-Free)',
            ha='center', va='top', fontsize=13, fontweight='bold')

    # Steps
    steps = [
        (1, 8.5, "Raw Hyperspectral Cubes", "VIS-NIR: 300 bands\nSWIR: 168 bands\nDAI 2-16 post-inoculation", '#4ECDC4'),
        (5, 8.5, "Water Band Removal", "Remove SWIR bands 94-115\n(1350-1460 nm, 1800-1950 nm)\nVIS-NIR unchanged", '#45B7D1'),
        (9, 8.5, "Group-Wise Split", "Split by biological unit (run #)\nTrain / Val / Test\nNO patch extraction yet\nFrozen seed=42", '#96CEB4'),

        (1, 6.5, "Sensor-Specific PCA", "IncrementalPCA(n=128)\nFitted on TRAIN patches ONLY\nNo data leakage\nApplied to Val & Test", '#FFEAA7'),
        (5, 6.5, "Train-Only Scaling", "StandardScaler\nFitted on TRAIN patches ONLY\nApplied to Val & Test\nZero leakage", '#DDA0DD'),
        (9, 6.5, "Patch Extraction", "9×9 spatial patches\nStride=4\nAFTER split & scaling\nTrain: 9,254 patches", '#98D8C8'),

        (3, 4.5, "Class Balancing", "WeightedRandomSampler\nClass weights: [1.76, 3.06, 1.53, 0.41]\nHealthy/Asympt/Early/Late\nAddress severe imbalance", '#F7DC6F'),
        (7, 4.5, "Data Augmentation", "MixUp (α=0.2, 50%)\nCutMix (α=1.0, 50%)\nLabel Smoothing (0.1)\nOnline during training", '#BB8FCE'),

        (5, 2.5, "Final Input Tensor", "Shape: [Batch, 128, 9, 9]\n128 PCA components\n9×9 spatial context\nReady for DMLPFFN", '#85C1E9'),
    ]

    for x, y, title, desc, color in steps:
        box = FancyBboxPatch((x-1.2, y-0.7), 2.4, 1.5,
                             boxstyle="round,pad=0.05,rounding_size=0.08",
                             facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y+0.2, title, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, y-0.45, desc, ha='center', va='center', fontsize=7.5, linespacing=1.1)

    # Arrows
    arrows = [
        ((2.2, 8.5), (3.8, 8.5)), ((6.2, 8.5), (7.8, 8.5)),
        ((1.2, 7.1), (1.2, 7.1)),  # down from step 1
        ((5, 7.1), (5, 7.1)),      # down from step 2
        ((9, 7.1), (9, 7.1)),      # down from step 3
        ((2.2, 6.5), (3.8, 6.5)),  # horiz step 4->5
        ((6.2, 6.5), (7.8, 6.5)),  # horiz step 5->6
        ((2.2, 5.1), (2.2, 5.1)),  # down from step 4
        ((5, 5.1), (5, 5.1)),      # down from step 5
        ((9, 5.1), (9, 5.1)),      # down from step 6
        ((3.5, 3.8), (3.5, 3.8)),  # down from step 7
        ((8.5, 3.8), (8.5, 3.8)),  # down from step 8
        ((4.2, 3.2), (5, 3.2)),    # horiz step 7->9
        ((6, 3.2), (5.8, 3.2)),    # horiz step 8->9
    ]

    for (x1, y1), (x2, y2) in arrows:
        if x1 != x2 or y1 != y2:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))

    # Key callout
    callout = FancyBboxPatch((0.3, 0.3), 11.4, 1.5,
                             boxstyle="round,pad=0.1,rounding_size=0.1",
                             facecolor='#FFF3CD', edgecolor='#FFC107', linewidth=2)
    ax.add_patch(callout)
    ax.text(6, 1.3, '⚠ CRITICAL LEAKAGE PREVENTION (all implemented in corrected pipeline)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#856404')
    ax.text(6, 0.7,
            '• PCA fitted on TRAIN patches only (not per-image, not on all data)\n'
            '• StandardScaler fitted on TRAIN patches only\n'
            '• Group-wise split BEFORE patch extraction (biological unit = zero spatial leakage)\n'
            '• Frozen split manifest (seed=42) ensures reproducibility',
            ha='center', va='center', fontsize=8.5, color='#856404')

    plt.savefig(os.path.join(FIG_DIR, 'figure2_preprocessing.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure2_preprocessing.pdf'), format='pdf')
    plt.close()
    print("Figure 2 saved")

# ============================================================
# Figure 3: DMLPFFN Architecture from actual code
# ============================================================
def create_figure3_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.6, 'DMLPFFN Architecture (from CODES/dmlpffn_disease.py)',
            ha='center', va='top', fontsize=13, fontweight='bold')

    # Input tensor
    in_box = FancyBboxPatch((5.5, 8.2), 3, 1.2,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(in_box)
    ax.text(7, 8.85, 'Input Tensor', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(7, 8.5, '[Batch, 128, 9, 9]\n(NCHW: 128 PCA components)', ha='center', va='center', fontsize=9, color='#2E7D32')

    # Three branches
    branch_y = 5.5
    branch_h = 2.2

    # Low Branch (left)
    low_box = FancyBboxPatch((0.8, branch_y), 3.8, branch_h,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(low_box)
    ax.text(2.7, 7.45, 'Low Branch', ha='center', va='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(2.7, 7.0, 'Conv2d(128→16, 3×3, p=1)\nBN + ReLU\nDMLPBlock(16→16)\nConv2d(16→64, 1×1)', ha='center', va='center', fontsize=8.5, color='#1565C0')

    # Mid Branch (center)
    mid_box = FancyBboxPatch((5.1, branch_y), 3.8, branch_h,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    ax.add_patch(mid_box)
    ax.text(7, 7.45, 'Mid Branch', ha='center', va='center', fontsize=10, fontweight='bold', color='#E65100')
    ax.text(7, 7.0, 'Conv2d(128→32, 5×5, p=2)\nBN + ReLU\nDMLPBlock(32→32)\nConv2d(32→64, 1×1)', ha='center', va='center', fontsize=8.5, color='#E65100')

    # High Branch (right)
    high_box = FancyBboxPatch((9.4, branch_y), 3.8, branch_h,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
    ax.add_patch(high_box)
    ax.text(11.3, 7.45, 'High Branch', ha='center', va='center', fontsize=10, fontweight='bold', color='#C2185B')
    ax.text(11.3, 7.0, 'DMLPBlock(128→64)\nConv2d(64→64, 1×1)', ha='center', va='center', fontsize=8.5, color='#C2185B')

    # DMLPBlock detail (expanded)
    block_y = 2.5
    block_h = 2.2

    ax.text(7, 5.1, 'DMLPBlock Detail (shared across branches)', ha='center', va='center', fontsize=10, fontweight='bold', color='#333')

    # Global Perceptron
    gp_box = FancyBboxPatch((0.8, block_y), 3.5, block_h,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5)
    ax.add_patch(gp_box)
    ax.text(2.55, 4.45, 'GlobalPerceptron', ha='center', va='center', fontsize=9, fontweight='bold', color='#7B1FA2')
    ax.text(2.55, 4.0, 'Branch 1: BN2d → mean([2,3]) → FC(128→C) → ReLU → FC(C→C)\nBranch 2: AvgPool2d(1) → BN1d → FC(128→C) → ReLU → FC(C→C)\nOutput: x1 + x2  (broadcast to spatial)', ha='center', va='center', fontsize=7.5, color='#7B1FA2')

    # Partition Perceptron
    pp_box = FancyBboxPatch((4.8, block_y), 3.5, block_h,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
    ax.add_patch(pp_box)
    ax.text(6.55, 4.45, 'PartitionPerceptron', ha='center', va='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(6.55, 4.0, 'BN2d(128) → Conv2d(128→C, 3×3, p=1, groups=4)\nOutput: spatially partitioned features', ha='center', va='center', fontsize=8, color='#2E7D32')

    # Local Perceptron
    lp_box = FancyBboxPatch((8.8, block_y), 3.5, block_h,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5)
    ax.add_patch(lp_box)
    ax.text(10.55, 4.45, 'LocalPerceptron (Dilated)', ha='center', va='center', fontsize=9, fontweight='bold', color='#F57F17')
    ax.text(10.55, 4.0, '3 Dilated Convs (sum):\n  d=1: Conv2d(128→C, 3×3, p=1) + BN\n  d=2: Conv2d(128→C, 3×3, p=2) + BN\n  d=5: Conv2d(128→C, 3×3, p=5) + BN\nOutput: multi-scale local context', ha='center', va='center', fontsize=7.5, color='#F57F17')

    # Fusion & Classifier
    fusion_y = 0.3
    fusion_box = FancyBboxPatch((3, fusion_y), 8, 1.8,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(7, 1.7, 'Fusion & Classification Head', ha='center', va='center', fontsize=10, fontweight='bold', color='#3F51B5')
    ax.text(7, 1.2, 'Fused = Low + Mid + High (element-wise, 64 ch each)\nGAP → Flatten → Dropout(0.7) → Linear(64 → 4 classes)', ha='center', va='center', fontsize=9, color='#3F51B5')
    ax.text(7, 0.6, 'Total Parameters: 447,780  |  Output: 4-class logits [Healthy, Asymptomatic, Early, Late]', ha='center', va='center', fontsize=9, fontweight='bold', color='#3F51B5')

    # Arrows
    # Input to branches
    for x in [2.7, 7, 11.3]:
        ax.annotate('', xy=(x, branch_y+branch_h), xytext=(7, 8.2),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))

    # Branches to fusion
    for x in [2.7, 7, 11.3]:
        ax.annotate('', xy=(x, branch_y), xytext=(x, fusion_y+1.8),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))

    # DMLPBlock detail arrows
    for bx in [2.55, 6.55, 10.55]:
        ax.annotate('', xy=(bx, block_y+block_h), xytext=(bx, branch_y),
                    arrowprops=dict(arrowstyle='->', lw=1, color='#999', linestyle='dashed'))

    plt.savefig(os.path.join(FIG_DIR, 'figure3_architecture.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure3_architecture.pdf'), format='pdf')
    plt.close()
    print("Figure 3 saved")

# ============================================================
# Figure 4: Training Curves from saved logs
# ============================================================
def create_figure4_training_curves():
    # Load experiment results
    with open(r'C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\artifacts\experiment_results_recovered.json', 'r') as f:
        data = json.load(f)

    history = data['history']
    # train_acc has 45 entries, val_acc has 46 entries (early stopping at epoch 46)
    # train_loss has 45 entries, val_loss has 44 entries
    epochs_train = list(range(1, len(history['train_acc']) + 1))
    epochs_val = list(range(1, len(history['val_acc']) + 1))
    epochs_train_loss = list(range(1, len(history['train_loss']) + 1))
    epochs_val_loss = list(range(1, len(history['val_loss']) + 1))
    epochs_lr = list(range(1, len(history['lr']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('DMLPFFN Training Dynamics (Corrected Pipeline, Seed=42)', fontsize=13, fontweight='bold', y=0.98)

    # Color scheme
    train_color = '#2E86AB'
    val_color = '#A23B72'

    # 1. Accuracy
    ax = axes[0, 0]
    ax.plot(epochs_train, history['train_acc'], label='Train Acc', color=train_color, linewidth=1.5)
    ax.plot(epochs_val, history['val_acc'], label='Val Acc', color=val_color, linewidth=1.5)
    ax.axvline(x=data['best_val_epoch'], color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.8, label=f"Best Val (Epoch {data['best_val_epoch']})")
    ax.axhline(y=data['best_val_acc'], color='#A23B72', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy Curves (Train-Val Gap: {data["train_val_gap"]:.2%})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    # 2. Loss
    ax = axes[0, 1]
    ax.plot(epochs_train_loss, history['train_loss'], label='Train Loss', color=train_color, linewidth=1.5)
    ax.plot(epochs_val_loss, history['val_loss'], label='Val Loss', color=val_color, linewidth=1.5)
    ax.axvline(x=data['best_val_epoch'], color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs_lr, history['lr'], color='#6A994E', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (CosineAnnealingWarmRestarts, T₀=10)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Key Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')

    # Final metrics box
    metrics_text = (
        f"Final Training Accuracy: {data['final_train_acc']:.4f} ({data['final_train_acc']:.2%})\n"
        f"Best Validation Accuracy: {data['best_val_acc']:.4f} ({data['best_val_acc']:.2%}) at Epoch {data['best_val_epoch']}\n"
        f"Train-Val Gap: {data['train_val_gap']:.4f} ({data['train_val_gap']:.2%})\n"
        f"Test Accuracy (4-class): {data['test_accuracy']:.4f} ({data['test_accuracy']:.2%})\n"
        f"Total Epochs Run: {len(epochs_train)} (Early stopping at epoch {len(epochs_val)})\n"
        f"Model Parameters: {data['model_params']['total']:,}\n"
        f"Training Time: {data['training_time_seconds']:.0f} sec\n\n"
        f"Class Distribution (Train):\n"
        f"  Healthy: {data['class_distributions']['train']['0']}  |  "
        f"Asymptomatic: {data['class_distributions']['train']['1']}  |  "
        f"Early: {data['class_distributions']['train']['2']}  |  "
        f"Late: {data['class_distributions']['train']['3']}\n"
        f"Class Distribution (Test):\n"
        f"  Healthy: {data['class_distributions']['test']['0']}  |  "
        f"Asymptomatic: {data['class_distributions']['test']['1']}  |  "
        f"Early: {data['class_distributions']['test']['2']}  |  "
        f"Late: {data['class_distributions']['test']['3']}\n\n"
        f"⚠ Test set lacks Asymptomatic & Early classes —\n"
        f"conditional binary evaluation used in manuscript."
    )

    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.85, transform=ax.transAxes,
                         boxstyle="round,pad=0.05,rounding_size=0.02",
                         facecolor='#F5F5F5', edgecolor='#999', linewidth=1)
    ax.add_patch(box)
    ax.text(0.5, 0.95, 'Training Summary', ha='center', va='top', transform=ax.transAxes,
            fontsize=10, fontweight='bold')
    ax.text(0.05, 0.88, metrics_text, ha='left', va='top', transform=ax.transAxes,
            fontsize=8.5, linespacing=1.3, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figure4_training_curves.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure4_training_curves.pdf'), format='pdf')
    plt.close()
    print("Figure 4 saved")

# ============================================================
# Figure 5: Conditional Binary Confusion Matrix
# ============================================================
def create_figure5_confusion_matrix():
    # Load the frozen inference results
    with open(r'C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\artifacts\dmlpffn_test_logits_frozen.json', 'r') as f:
        data = json.load(f)

    # Get conditional binary confusion matrix from logits
    cm_binary = np.array(data['confusion_matrix_conditional_binary_logits'])
    # TN, FP, FN, TP
    tn, fp, fn, tp = cm_binary[0,0], cm_binary[0,1], cm_binary[1,0], cm_binary[1,1]

    # Also get 4-class confusion matrix
    cm_4class = np.array(data['confusion_matrix_4class'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DMLPFFN Test Set Confusion Matrices (Frozen Manifest, 1330 patches)',
                 fontsize=12, fontweight='bold', y=1.02)

    # --- Left: 4-class confusion matrix ---
    ax = axes[0]
    im = ax.imshow(cm_4class, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(['Healthy\n(0)', 'Asympt.\n(1)', 'Early\n(2)', 'Late\n(3)'])
    ax.set_yticklabels(['Healthy\n(0)', 'Asympt.\n(1)', 'Early\n(2)', 'Late\n(3)'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('4-Class Confusion Matrix', fontweight='bold')

    # Annotate
    for i in range(4):
        for j in range(4):
            val = cm_4class[i, j]
            color = 'white' if val > cm_4class.max()/2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    # Add note about missing classes
    ax.text(2, -0.5, 'Note: Asymptomatic & Early have 0 test samples',
            ha='center', va='top', fontsize=8, style='italic', color='#C73E1D')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count')

    # --- Right: Conditional Binary (Healthy vs Late) ---
    ax = axes[1]
    im = ax.imshow(cm_binary, cmap='Greens', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Healthy', 'Pred Late'])
    ax.set_yticklabels(['True Healthy', 'True Late'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Conditional Binary: Healthy vs Late\n(Exact marginalization from 4-class logits)', fontweight='bold')

    # Annotate with labels
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = cm_binary[i, j]
            label = labels[i][j]
            color = 'white' if val > cm_binary.max()/2 else 'black'
            ax.text(j, i, f'{label}\n{val}', ha='center', va='center', fontsize=13, fontweight='bold', color=color)

    # Metrics
    acc = data['conditional_binary_metrics']['accuracy']
    macro_f1 = data['conditional_binary_metrics']['macro_f1']
    weighted_f1 = data['conditional_binary_metrics']['weighted_f1']
    auc = data['conditional_binary_metrics']['auc_roc']
    prec_h = data['conditional_binary_metrics']['precision_healthy']
    prec_l = data['conditional_binary_metrics']['precision_late']
    rec_h = data['conditional_binary_metrics']['recall_healthy']
    rec_l = data['conditional_binary_metrics']['recall_late']

    metrics_text = (
        f'Accuracy: {acc:.4f}\n'
        f'Macro F1: {macro_f1:.4f}\n'
        f'Weighted F1: {weighted_f1:.4f}\n'
        f'AUC-ROC: {auc:.4f}\n\n'
        f'Healthy: P={prec_h:.3f}, R={rec_h:.3f}\n'
        f'Late:    P={prec_l:.3f}, R={rec_l:.3f}\n\n'
        f'True Healthy: {tn+fp}  |  True Late: {fn+tp}\n'
        f'P(Late|H/L) = e^z3 / (e^z0 + e^z3)'
    )

    ax.text(1.4, 0.5, metrics_text, ha='left', va='center', fontsize=9,
            transform=ax.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='#999'))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figure5_confusion_matrix.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure5_confusion_matrix.pdf'), format='pdf')
    plt.close()
    print("Figure 5 saved")

# ============================================================
# Figure 6: Baseline Comparison (single-run vs 5-seed)
# ============================================================
def create_figure6_baseline_comparison():
    # Load all baseline results
    baselines = {}

    # 5-seed DL baselines
    for name in ['simple_cnn', 'mlp_mixer_hsi', 'resnet18_hsi']:
        path = f'C:\\Users\\ayush\\OneDrive\\Desktop\\DMLPFFN\\DMLPFFN-TomatoLeafSpot-Detection-main\\artifacts\\baselines\\{name}\\aggregate_results.json'
        with open(path, 'r') as f:
            baselines[name] = json.load(f)

    # Single-run baselines
    for name in ['svm', 'random_forest']:
        path = f'C:\\Users\\ayush\\OneDrive\\Desktop\\DMLPFFN\\DMLPFFN-TomatoLeafSpot-Detection-main\\artifacts\\baselines\\{name}\\aggregate_results.json'
        with open(path, 'r') as f:
            baselines[name] = json.load(f)

    # DMLPFFN reference (from frozen inference)
    with open(r'C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\artifacts\dmlpffn_test_logits_frozen.json', 'r') as f:
        dmlpffn = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 5 Baseline Comparison: DMLPFFN vs Baselines (Conditional Binary Evaluation)',
                 fontsize=12, fontweight='bold', y=0.98)

    # Model info
    models = [
        ('DMLPFFN (ref)', dmlpffn['conditional_binary_metrics']['accuracy'],
         dmlpffn['conditional_binary_metrics']['macro_f1'],
         dmlpffn['conditional_binary_metrics']['weighted_f1'],
         dmlpffn['conditional_binary_metrics']['auc_roc'],
         447780, 'Single Run'),
        ('SimpleCNN', baselines['simple_cnn']['accuracy']['mean'],
         baselines['simple_cnn']['macro_f1']['mean'],
         baselines['simple_cnn']['weighted_f1']['mean'],
         baselines['simple_cnn']['auc_roc_macro']['mean'],
         86156, '5 Seeds'),
        ('MLP-Mixer HSI', baselines['mlp_mixer_hsi']['accuracy']['mean'],
         baselines['mlp_mixer_hsi']['macro_f1']['mean'],
         baselines['mlp_mixer_hsi']['weighted_f1']['mean'],
         baselines['mlp_mixer_hsi']['auc_roc_macro']['mean'],
         2310812, '5 Seeds'),
        ('ResNet-18 HSI', baselines['resnet18_hsi']['accuracy']['mean'],
         baselines['resnet18_hsi']['macro_f1']['mean'],
         baselines['resnet18_hsi']['weighted_f1']['mean'],
         baselines['resnet18_hsi']['auc_roc_macro']['mean'],
         11242884, '5 Seeds'),
        ('SVM (RBF)', baselines['svm']['test_accuracy']['mean'],
         baselines['svm']['macro_f1']['mean'],
         baselines['svm']['weighted_f1']['mean'],
         0,  # no AUC
         'N/A', 'Single Run'),
        ('Random Forest', baselines['random_forest']['test_accuracy']['mean'],
         baselines['random_forest']['macro_f1']['mean'],
         baselines['random_forest']['weighted_f1']['mean'],
         0,  # no AUC
         'N/A', 'Single Run'),
    ]

    names = [m[0] for m in models]
    accs = [m[1] for m in models]
    macro_f1s = [m[2] for m in models]
    weighted_f1s = [m[3] for m in models]
    aucs = [m[4] for m in models]
    params = [m[5] for m in models]
    run_types = [m[6] for m in models]

    # Colors: DMLPFFN = distinct, 5-seed = blue family, single-run = orange family
    colors = ['#2E86AB', '#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#FF6B6B']
    hatches = ['', '///', '///', '///', '\\\\\\', '\\\\\\']
    edgecolors = ['#2E86AB', '#2A8A8C', '#2A8A8C', '#2A8A8C', '#C73E1D', '#C73E1D']

    # 1. Accuracy
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor=edgecolors,
                  hatch=hatches, linewidth=1.5, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy (Conditional Binary)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='DMLPFFN (Reference)'),
        Patch(facecolor='#4ECDC4', hatch='///', edgecolor='#2A8A8C', label='5-Seed DL Baselines (mean ± std)'),
        Patch(facecolor='#FF6B6B', hatch='\\\\\\', edgecolor='#C73E1D', label='Single-Run Classical Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # 2. Macro F1
    ax = axes[0, 1]
    bars = ax.bar(range(len(names)), macro_f1s, color=colors, edgecolor=edgecolors,
                  hatch=hatches, linewidth=1.5, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1-Score (Conditional Binary)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, macro_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. Weighted F1
    ax = axes[1, 0]
    bars = ax.bar(range(len(names)), weighted_f1s, color=colors, edgecolor=edgecolors,
                  hatch=hatches, linewidth=1.5, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Weighted F1')
    ax.set_title('Weighted F1-Score (Conditional Binary)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, weighted_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 4. Parameters vs Accuracy scatter
    ax = axes[1, 1]
    # Only plot models with numeric params
    plot_data = [(p, a, n, c, h, e) for p, a, n, c, h, e in zip(params, accs, names, colors, hatches, edgecolors)
                 if isinstance(p, (int, float))]
    for p, a, n, c, h, e in plot_data:
        marker = 'o' if 'DMLPFFN' in n else ('s' if '5 Seeds' in [rt for nm, rt in zip(names, run_types) if nm == n][0] else '^')
        size = 150 if 'DMLPFFN' in n else 100
        ax.scatter(p, a, s=size, color=c, edgecolor=e, marker=marker, linewidth=2,
                   label=n, zorder=5)
        ax.annotate(n, (p, a), textcoords="offset points", xytext=(5,5), fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (log scale)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Parameter Efficiency: Accuracy vs Model Size')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5e4, 2e7)
    ax.set_ylim(0.3, 1.0)

    # Add reference lines for param counts
    for p_val, label in [(86156, 'SimpleCNN'), (447780, 'DMLPFFN'), (2310812, 'MLP-Mixer'), (11242884, 'ResNet-18')]:
        ax.axvline(x=p_val, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figure6_baseline_comparison.svg'), format='svg')
    plt.savefig(os.path.join(FIG_DIR, 'figure6_baseline_comparison.pdf'), format='pdf')
    plt.close()
    print("Figure 6 saved")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating publication figures...")
    create_figure1_workflow()
    create_figure2_preprocessing()
    create_figure3_architecture()
    create_figure4_training_curves()
    create_figure5_confusion_matrix()
    create_figure6_baseline_comparison()
    print("\nAll 6 figures saved to publication/figures/")