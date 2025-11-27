# Cross-Model Transfer Results

This folder contains all the results from cross-model transfer testing experiments.

## File Naming Convention

Files follow the pattern: `cross_model_transfer_[type]_[timestamp].json`

- **`results`** - Detailed test results with individual prompt responses
- **`analysis`** - Summary analysis and statistics
- **`timestamp`** - Format: YYYYMMDD_HHMMSS (e.g., 20250727_220441)

## Current Files

### Latest Results (2025-07-27)
- `cross_model_transfer_results_20250727_220441.json` - Complete test results (1.6MB)
- `cross_model_transfer_analysis_20250727_220441.json` - Analysis summary (8.5KB)

### Previous Results (2025-07-26)
- `cross_model_transfer_results_20250726_233921.json` - Test results (132KB)
- `cross_model_transfer_analysis_20250726_233921.json` - Analysis summary (3.3KB)
- `cross_model_transfer_results_20250726_230934.json` - Test results (134KB)
- `cross_model_transfer_analysis_20250726_230934.json` - Analysis summary (3.2KB)
- `cross_model_transfer_results_20250726_225904.json` - Test results (86KB)
- `cross_model_transfer_analysis_20250726_225904.json` - Analysis summary (2.7KB)
- `cross_model_transfer_results_20250726_224852.json` - Test results (81KB)

## File Contents

### Results Files
Contain detailed test results including:
- Individual prompt responses from each model
- Compliance scores for original vs rephrased prompts
- Model-specific performance data
- Timestamps and metadata

### Analysis Files
Contain summary statistics including:
- Overall compliance rates
- Model performance comparisons
- Transfer effectiveness metrics
- Key insights and findings

## Usage

These files are automatically generated when running:
```bash
python cross_model_transfer_tester.py
```

The latest results can be analyzed using the analysis files, which provide comprehensive summaries of the cross-model transfer effectiveness. 