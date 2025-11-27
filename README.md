# Mechanical Interpretability Project

A comprehensive toolkit for analyzing and understanding neural network behavior through feature analysis, cross-model transfer testing, and compliance assessment.

## Project Structure

```
mech_interp/
├── src/                          # Main source code
│   ├── core/                     # Core functionality
│   │   ├── main.py              # Main entry point
│   │   ├── interpret_cache.py   # Cache interpretation utilities
│   │   ├── investigate_feature.py # Feature investigation tools
│   │   └── simple_feature_investigation.py # Simplified feature analysis
│   ├── analysis/                 # Analysis modules
│   │   ├── analyze_rhetorical_questions.py # Rhetorical question analysis
│   │   └── gemma_compliance_assessor.py # Compliance assessment tools
│   ├── rewriters/                # Prompt rewriting utilities
│   │   ├── api_based_rewriter.py # API-based prompt rewriting
│   │   └── improved_jailbreak_rewriter.py # Enhanced jailbreak rewriting
│   ├── api/                      # API interfaces
│   │   ├── neuronpedia_api.py   # Neuronpedia API integration
│   │   ├── together_ai_interface.py # Together AI interface
│   │   └── check_hf_models.py   # Hugging Face model utilities
│   └── utils/                    # Utility modules
│       ├── auto_feature_finder/  # Automated feature discovery
│       │   └── feature_finder.py
│       └── model_calling/        # Model calling utilities
│           ├── Gemini.py
│           └── LLM.py
├── scripts/                      # Executable scripts
│   ├── cross_model_transfer_tester.py # Cross-model transfer testing
│   ├── setup_cross_model_testing.py # Test setup utilities
│   ├── run_feature_finder.py    # Feature finder runner
│   └── run_feature_finder_relaxed.py # Relaxed feature finding
├── docs/                         # Documentation
│   ├── guides/                   # User guides and documentation
│   │   ├── README.md            # Main documentation
│   │   ├── CROSS_MODEL_TRANSFER_README.md
│   │   ├── CROSS_MODEL_TRANSFER_SUMMARY.md
│   │   ├── REPRODUCTION_GUIDE.md
│   │   ├── TOGETHER_AI_SETUP.md
│   │   └── HF_MODEL_OPTIONS.md
│   └── results/                  # Analysis results
│       └── cross_model_transfer_results/ # Cross-model transfer results
├── data/                         # Data storage
│   ├── cache/                    # Cached data
│   │   ├── compliance_assessment_cache.json
│   │   └── caches/              # Model-specific caches
│   ├── results/                  # Analysis results
│   │   └── api_rewrite_results.json
│   └── features/                 # Feature analysis data
│       ├── all_features_comprehensive_analysis.json
│       └── feature_18677_detailed_analysis.json
├── tests/                        # Test files
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── poetry.lock                  # Poetry lock file
└── .gitignore                   # Git ignore rules
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   poetry install
   ```

2. **Run main analysis:**
   ```bash
   python src/core/main.py
   ```

3. **Run cross-model transfer testing:**
   ```bash
   python scripts/cross_model_transfer_tester.py
   ```

4. **Find features:**
   ```bash
   python scripts/run_feature_finder.py
   ```

## Key Components

### Core Analysis (`src/core/`)
- **main.py**: Entry point for the main analysis pipeline
- **interpret_cache.py**: Utilities for interpreting cached model responses
- **investigate_feature.py**: Tools for deep feature investigation
- **simple_feature_investigation.py**: Simplified feature analysis workflows

### Analysis Modules (`src/analysis/`)
- **analyze_rhetorical_questions.py**: Specialized analysis for rhetorical questions
- **gemma_compliance_assessor.py**: Compliance assessment for Gemma models

### Rewriters (`src/rewriters/`)
- **api_based_rewriter.py**: API-driven prompt rewriting
- **improved_jailbreak_rewriter.py**: Enhanced jailbreak detection and rewriting

### API Interfaces (`src/api/`)
- **neuronpedia_api.py**: Integration with Neuronpedia
- **together_ai_interface.py**: Together AI API wrapper
- **check_hf_models.py**: Hugging Face model utilities

### Utilities (`src/utils/`)
- **auto_feature_finder/**: Automated feature discovery tools
- **model_calling/**: Model calling abstractions

## Data Organization

- **Cache data** (`data/cache/`): Stored model responses and analysis caches
- **Results** (`data/results/`): Analysis outputs and processed data
- **Features** (`data/features/`): Feature-specific analysis results

## Documentation

- **Guides** (`docs/guides/`): User guides, setup instructions, and methodology
- **Results** (`docs/results/`): Detailed analysis results and summaries

## Development

- **Scripts** (`scripts/`): Executable scripts for various analysis tasks
- **Tests** (`tests/`): Test files for validation

This organization provides a clean separation of concerns, making it easier to navigate, maintain, and extend the project. 