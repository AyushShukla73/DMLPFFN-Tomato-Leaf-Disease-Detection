# Public GitHub Repository Cleanup Assessment Report

**Date**: 2026-11-26  
**Repository**: DMLPFFN-TomatoLeafSpot-Detection-main  
**Purpose**: Assess 8 candidate files/directories for removal while preserving essential research reproducibility files

---

## Executive Summary

After thorough dependency analysis (grep searches across all files, reading key documentation, and examining import chains), here is the categorization:

| Category | Files/Directories | Count |
|----------|------------------|-------|
| **Safe to Remove** | `parameter_counts.py`, `PROJECT_STATUS.md`, `_archive/macos_metadata/__MACOSX/` | 3 |
| **Retain (Essential)** | `PHASE5_BASELINE_RESULTS.md`, `_archive/CODES/baseline_models.py` | 2 |
| **Decision Required** | `generate_figures.py`, `MANUSCRIPT_REVISED.md`, `inference_dmlpffn_frozen.py`, `PHASE5_RESULTS_VERIFICATION.md`, `_archive/phase_docs/`, `_archive/backups/` | 6 |

---

## Detailed Assessment

### ✅ SAFE TO REMOVE (No Dependencies Found)

#### 1. `parameter_counts.py`
- **Location**: Root directory
- **Size**: ~3 KB
- **Dependencies**: Zero grep hits across entire repository
- **Purpose**: Standalone parameter counting utility (historical/debugging)
- **Risk**: None — no imports, no references in any documentation
- **Action**: `rm parameter_counts.py`

#### 2. `PROJECT_STATUS.md`
- **Location**: Root directory
- **Size**: ~15 KB
- **Dependencies**: Referenced only in `_archive/CLEANUP_REPORT.md` (which is itself in _archive and can be reviewed for retention)
- **Purpose**: Internal project tracking document
- **Risk**: None — not referenced in any user-facing documentation (README, REPRODUCIBILITY.md, MANUSCRIPT)
- **Action**: `rm PROJECT_STATUS.md`

#### 3. `_archive/macos_metadata/__MACOSX/` (entire directory)
- **Location**: `_archive/macos_metadata/__MACOSX/`
- **Size**: ~600 files (macOS metadata junk — ._* files, resource forks)
- **Dependencies**: Zero functional references anywhere
- **Purpose**: macOS Finder metadata artifacts from archive creation
- **Risk**: None — pure filesystem noise, zero scientific value
- **Action**: `rm -rf _archive/macos_metadata/__MACOSX/`

---

### 🔒 RETAIN (Essential for Reproducibility)

#### 4. `PHASE5_BASELINE_RESULTS.md`
- **Location**: Root directory (also `artifacts/baselines/PHASE5_BASELINE_RESULTS.md`)
- **Dependencies**: **13 references** including:
  - `REPRODUCIBILITY.md` (line 245: "refer to `PHASE5_RESULTS_VERIFICATION.md` and `PHASE5_BASELINE_RESULTS.md`")
  - `CODES/run_baselines.py` (imports/reads results)
  - `CODES/phase5_baseline_ablation.py` (generates this file)
  - `MANUSCRIPT_REVISED.md` (Table 465: "Phase 5 Summary | `artifacts/baselines/PHASE5_BASELINE_RESULTS.md`")
  - `publication/figures/FIGURE_INDEX.md` (Figure 6 regeneration depends on it)
  - Multiple baseline result directories
- **Purpose**: Verified baseline comparison summary — **core scientific artifact**
- **Risk if removed**: Breaks Phase 5 reproduction, figure generation, and baseline verification
- **Action**: **RETAIN** — move to `artifacts/baselines/` if not already there (it is)

#### 5. `_archive/CODES/baseline_models.py`
- **Location**: `_archive/CODES/baseline_models.py`
- **Dependencies**: **7 critical references**:
  - `REPRODUCIBILITY.md:187` — `from _archive.CODES.baseline_models import get_baseline_model`
  - `CODES/tests/verification_tasks.py:187` — Same import for historical baseline verification
  - `artifacts/frozen/baseline_models_signatures.json` — Documents 11 historical ablations from this file
  - `README.md` (Repository Structure) — Lists as preserved historical source
  - `MANUSCRIPT_REVISED.md` — References historical vs modern ablation distinction
  - `_archive/CLEANUP_REPORT.md` — Documents preservation
  - `_archive/phase_docs/` — Multiple phase docs reference it
- **Purpose**: **Only source code for 11 historical ablation configurations** (no checkpoints exist)
- **Risk if removed**: **Irreversible loss of historical baseline implementations** — these cannot be regenerated
- **Action**: **RETAIN** — essential for historical comparison and exact reproduction of Phase 5 ablation study

---

### ⚠️ DECISION REQUIRED (Weighing Trade-offs)

#### 6. `generate_figures.py`
- **Location**: Root directory
- **Size**: ~25 KB
- **Dependencies**: Referenced in:
  - `MANUSCRIPT_REVISED.md:360, 414-418` (Supplementary Material, Figure Regeneration)
  - `PROJECT_STATUS.md` (internal tracking)
  - `publication/figures/FIGURE_INDEX.md:413-418` (Regeneration command)
- **Purpose**: Reproduces all 6 publication figures (SVG + PDF) from frozen artifacts
- **Consideration**: 
  - **If manuscript is retained**: Essential for figure regeneration (scientific reproducibility)
  - **If manuscript is removed**: Low value (figures already generated and committed to `publication/figures/`)
- **Recommendation**: Retain if keeping manuscript; safe to remove if manuscript removed
- **Action**: Decision tied to MANUSCRIPT_REVISED.md

#### 7. `MANUSCRIPT_REVISED.md`
- **Location**: Root directory
- **Size**: ~55 KB (471 lines)
- **Dependencies**: Referenced in:
  - `REPRODUCIBILITY.md:245` (explicitly listed as supplementary material)
  - `PROJECT_STATUS.md` (internal)
  - `publication/figures/FIGURE_INDEX.md` (figure documentation source)
  - `_archive/phase_docs/` (historical phase documentation)
- **Purpose**: Full revised manuscript for journal submission (Precision Agriculture target)
- **Consideration**:
  - **Scientific value**: Complete methods, results, discussion for paper reproducibility
  - **Repository bloat**: 55 KB is negligible
  - **Public release norm**: Many ML repos include manuscript drafts
- **Recommendation**: **Retain** — minimal size, high scientific value, explicitly referenced in REPRODUCIBILITY.md as supplementary material
- **Action**: **Retain**

#### 8. `inference_dmlpffn_frozen.py`
- **Location**: Root directory
- **Size**: ~15 KB
- **Dependencies**: Referenced **only** in `MANUSCRIPT_REVISED.md:361` (Supplementary Material)
- **Purpose**: Verified inference script using frozen manifest for exact reproduction
- **Consideration**:
  - **If manuscript retained**: Keep as supplementary code
  - **If manuscript removed**: No other references → safe to remove
- **Recommendation**: Retain (small, enables exact frozen inference, complements manuscript)
- **Action**: **Retain**

#### 9. `PHASE5_RESULTS_VERIFICATION.md`
- **Location**: Root directory
- **Size**: ~20 KB
- **Dependencies**: Referenced in:
  - `REPRODUCIBILITY.md:245` (explicitly listed: "refer to `PHASE5_RESULTS_VERIFICATION.md`")
- **Purpose**: Statistical verification of Phase 5 baseline results (bootstrap CI, McNemar's test)
- **Consideration**: Directly referenced in the core reproducibility guide
- **Recommendation**: **Retain** — explicitly required by REPRODUCIBILITY.md
- **Action**: **Retain**

#### 10. `_archive/phase_docs/` (13 files)
- **Location**: `_archive/phase_docs/`
- **Files**: PHASE1_*.md through PHASE5_*.md, PHASE7_*.md, etc.
- **Dependencies**: Referenced in:
  - `MANUSCRIPT_REVISED.md` (methodology traces to phase docs)
  - `_archive/CLEANUP_REPORT.md` (audit trail)
  - Historical context for research evolution
- **Purpose**: Complete research phase documentation (audit trail, methodology evolution)
- **Consideration**:
  - **Historical value**: Documents the 3-leakage discovery, correction evolution, regularization progression
  - **Size**: ~13 files, moderate total size
  - **Reproducibility**: Not strictly required for code reproduction, but valuable for scientific audit
- **Recommendation**: **Retain** — scientific audit trail, referenced by manuscript
- **Action**: **Retain**

#### 11. `_archive/backups/` (1 file)
- **Location**: `_archive/backups/PHASE5_BASELINE_RESULTS.md`
- **Dependencies**: Referenced in `_archive/CLEANUP_REPORT.md` as backup
- **Purpose**: Backup of Phase 5 results during cleanup
- **Consideration**: Duplicate of main `PHASE5_BASELINE_RESULTS.md`
- **Recommendation**: **Remove** — redundant backup, main file retained in artifacts/
- **Action**: `rm _archive/backups/PHASE5_BASELINE_RESULTS.md` (or entire `_archive/backups/` if empty after)

---

## Proposed Deletion Commands

### Safe to Execute (No Risk)
```bash
# 1. Standalone utility with zero dependencies
rm parameter_counts.py

# 2. Internal tracking doc (referenced only in _archive which we're reviewing)
rm PROJECT_STATUS.md

# 3. macOS metadata junk (~600 files, zero scientific value)
rm -rf _archive/macos_metadata/__MACOSX/

# 4. Redundant backup (main file retained at artifacts/baselines/PHASE5_BASELINE_RESULTS.md)
rm _archive/backups/PHASE5_BASELINE_RESULTS.md
# If _archive/backups/ becomes empty:
rmdir _archive/backups/
```

### Retain (Do Not Delete)
```bash
# Essential for historical baseline reproduction (11 configs, source-only)
# _archive/CODES/baseline_models.py

# Core scientific artifacts
# PHASE5_BASELINE_RESULTS.md
# PHASE5_RESULTS_VERIFICATION.md

# Manuscript & supplementary code (explicitly in REPRODUCIBILITY.md)
# MANUSCRIPT_REVISED.md
# generate_figures.py
# inference_dmlpffn_frozen.py

# Scientific audit trail
# _archive/phase_docs/
```

---

## Final Recommendation Summary

| File/Directory | Verdict | Rationale |
|----------------|---------|-----------|
| `parameter_counts.py` | **DELETE** | Zero references, standalone debug script |
| `PROJECT_STATUS.md` | **DELETE** | Internal only, only in _archive/CLEANUP_REPORT.md |
| `_archive/macos_metadata/__MACOSX/` | **DELETE** | 600 macOS metadata files, zero value |
| `_archive/backups/` | **DELETE** | Redundant backup of retained file |
| `PHASE5_BASELINE_RESULTS.md` | **KEEP** | 13 refs, core Phase 5 artifact, figure regeneration |
| `_archive/CODES/baseline_models.py` | **KEEP** | 7 refs, only source for 11 historical ablations |
| `generate_figures.py` | **KEEP** | Manuscript supplementary, figure regeneration |
| `MANUSCRIPT_REVISED.md` | **KEEP** | Explicit in REPRODUCIBILITY.md, 55 KB negligible |
| `inference_dmlpffn_frozen.py` | **KEEP** | Manuscript supplementary, frozen inference |
| `PHASE5_RESULTS_VERIFICATION.md` | **KEEP** | Explicit in REPRODUCIBILITY.md |
| `_archive/phase_docs/` | **KEEP** | Scientific audit trail, manuscript references |

---

## Space Savings Estimate

| Item | Est. Size | Savings |
|------|-----------|---------|
| `parameter_counts.py` | ~3 KB | Negligible |
| `PROJECT_STATUS.md` | ~15 KB | Negligible |
| `_archive/macos_metadata/__MACOSX/` | ~2-5 MB | **Significant** (600 files) |
| `_archive/backups/` | ~20 KB | Negligible |
| **Total** | **~2-5 MB** | **Primarily macOS junk removal** |

**Note**: The meaningful scientific files total <150 KB combined — retaining them has negligible storage impact while preserving full reproducibility.

---

## Next Steps

1. **User confirms** the "Safe to Remove" and "Decision Required = Retain" items
2. **Execute** the 4 safe deletion commands above
3. **Verify** repository still passes:
   - `python CODES/tests/verification_tasks.py` (historical baseline import)
   - `python generate_figures.py` (figure regeneration)
   - `python CODES/phase5_baseline_ablation.py` (Phase 5 reproduction)
4. **Commit** cleanup with message: "chore: remove dead code and macOS metadata junk for public release"