#!/usr/bin/env python
"""
Recalculate pairwise McNemar tests using saved group predictions from statistical_analysis.json.
Fix the exact p-value bug (capping at 1.0) and update both statistical_analysis.json and PHASE5_BASELINE_RESULTS.md.
"""
import json
import numpy as np
from scipy import stats
import shutil
from datetime import datetime

# Load the statistical analysis JSON
json_path = r"C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\artifacts\baselines\statistical_analysis.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Create backup
backup_path = json_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(json_path, backup_path)
print(f"Backup created: {backup_path}")

# Fixed McNemar calculation function
def mcnemar_group_aware_fixed(group_true, group_pred_a, group_pred_b):
    """Fixed version with p-value capped at 1.0"""
    # Build contingency table for McNemar
    n01 = np.sum((group_pred_a != group_true) & (group_pred_b == group_true))
    n10 = np.sum((group_pred_a == group_true) & (group_pred_b != group_true))
    n_discordant = n01 + n10

    # McNemar's chi-squared statistic (with continuity correction)
    if n_discordant > 0:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n_discordant
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        p_value = 1.0

    # Exact binomial p-value (capped at 1.0)
    if n_discordant > 0:
        p_exact = 2 * min(
            stats.binom.cdf(min(n01, n10), n_discordant, 0.5),
            1 - stats.binom.cdf(max(n01, n10) - 1, n_discordant, 0.5)
        )
        p_exact = min(1.0, p_exact)  # FIX: cap at 1.0
    else:
        p_exact = 1.0

    return {
        'chi2_statistic': float(chi2),
        'p_value_asymptotic': float(p_value),
        'p_value_exact': float(p_exact),
        'n01': int(n01),
        'n10': int(n10),
        'n_discordant': int(n_discordant),
        'n_groups': len(group_true),
        'contingency_table': {
            'A_correct_B_correct': int(np.sum((group_pred_a == group_true) & (group_pred_b == group_true))),
            'A_correct_B_wrong': int(n10),
            'A_wrong_B_correct': int(n01),
            'A_wrong_B_wrong': int(np.sum((group_pred_a != group_true) & (group_pred_b != group_true)))
        },
        'group_predictions': {
            'true': group_true.tolist(),
            'pred_a': group_pred_a.tolist(),
            'pred_b': group_pred_b.tolist()
        },
        'warning': 'Group-aware McNemar with only 4 test groups has very low statistical power. Results are exploratory.'
    }

# Recalculate pairwise McNemar tests
pairwise_mcnemar = data['pairwise_mcnemar']
for comparison_name, result in pairwise_mcnemar.items():
    group_true = np.array(result['group_predictions']['true'])
    group_pred_a = np.array(result['group_predictions']['pred_a'])
    group_pred_b = np.array(result['group_predictions']['pred_b'])

    fixed_result = mcnemar_group_aware_fixed(group_true, group_pred_a, group_pred_b)

    print(f"\n{comparison_name}:")
    print(f"  Old p_value_exact: {result['p_value_exact']}")
    print(f"  New p_value_exact: {fixed_result['p_value_exact']}")
    print(f"  Old p_value_asymptotic: {result['p_value_asymptotic']}")
    print(f"  New p_value_asymptotic: {fixed_result['p_value_asymptotic']}")

    # Update the result
    pairwise_mcnemar[comparison_name] = fixed_result

# Save updated JSON
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nUpdated {json_path}")

# Now update PHASE5_BASELINE_RESULTS.md
md_path = r"C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main\artifacts\baselines\PHASE5_BASELINE_RESULTS.md"
md_backup_path = md_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(md_path, md_backup_path)
print(f"Markdown backup created: {md_backup_path}")

with open(md_path, 'r') as f:
    md_content = f.read()

# Find and update the McNemar results table in the markdown
# The table should be at the end of the file based on the verification report mentioning it
# Let me search for any McNemar references
import re

# The markdown file appears to be 64 lines only, so it may not have the statistical table yet
# Let me check if there's a statistical comparison section
if "McNemar" in md_content or "mcnemar" in md_content.lower():
    print("Found McNemar reference in markdown - updating...")
    # We'll need to find the exact table format
else:
    print("No McNemar table found in markdown - may need to add it")

# Let me read the full markdown file to see its current content
with open(md_path, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print(f"{i+1}: {line.rstrip()}")

print("\n--- Done recalculating ---")