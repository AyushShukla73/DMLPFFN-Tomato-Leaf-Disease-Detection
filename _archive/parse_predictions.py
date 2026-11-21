#!/usr/bin/env python3
import json
import numpy as np
import re

# Read seed_789
with open('./artifacts/baselines/resnet18_hsi/seed_789/test_results.json') as f:
    data = json.load(f)

# Parse predictions and targets using regex to extract numbers from the string
pred_str = data['test_predictions']
targ_str = data['test_targets']

# Extract all numbers from the string
preds = np.array([int(x) for x in re.findall(r'\d+', pred_str)])
targets = np.array([int(x) for x in re.findall(r'\d+', targ_str)])

print(f"SEED_789:")
print(f"  Predictions: {len(preds)}")
print(f"  Targets: {len(targets)}")
print(f"  Test accuracy from data: {data['test_accuracy']}")
print(f"  Calculated accuracy: {(preds == targets).sum() / len(targets):.6f}")
print(f"  Unique targets: {np.unique(targets)}")

# Count by class
for c in [0, 1, 3]:
    mask = targets == c
    print(f"  Class {c}: {mask.sum()} samples")

print()

# Read seed_999
with open('./artifacts/baselines/resnet18_hsi/seed_999/test_results.json') as f:
    data = json.load(f)

pred_str = data['test_predictions']
targ_str = data['test_targets']

preds = np.array([int(x) for x in re.findall(r'\d+', pred_str)])
targets = np.array([int(x) for x in re.findall(r'\d+', targ_str)])

print(f"SEED_999:")
print(f"  Predictions: {len(preds)}")
print(f"  Targets: {len(targets)}")
print(f"  Test accuracy from data: {data['test_accuracy']}")
print(f"  Calculated accuracy: {(preds == targets).sum() / len(targets):.6f}")
print(f"  Unique targets: {np.unique(targets)}")

# Count by class
for c in [0, 1, 3]:
    mask = targets == c
    print(f"  Class {c}: {mask.sum()} samples")

print()

# Parse seed_42, 123, 456 to check if they have different key names
for seed in ['42', '123', '456']:
    with open(f'./artifacts/baselines/resnet18_hsi/seed_{seed}/test_results.json') as f:
        data = json.load(f)
    print(f"SEED_{seed} keys: {list(data.keys())}")
    # Check if predictions are truncated
    pred_str = data.get('test_predictions', '')
    if '...' in pred_str:
        print(f"  WARNING: Predictions are truncated!")
    else:
        preds = np.array([int(x) for x in re.findall(r'\d+', pred_str)])
        print(f"  Predictions: {len(preds)} samples")
        print(f"  Test accuracy: {data.get('test_accuracy') or data.get('test_acc')}")