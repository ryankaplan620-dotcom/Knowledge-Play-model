# CLAUDE.md

## Project Overview

Aurora-AI ("Project Nemesis") is an advanced autonomous AI learning system built in Python. It features self-improving AI with multi-modal capabilities, reinforcement learning, and adaptive architecture evolution. The entire system lives in a single monolithic file.

## Structure

```
Aurora-AI/
├── Project Nemesis.py   # All source code (~1,660 lines, 14+ classes)
├── README.md
└── CLAUDE.md
```

## Prerequisites

- Python 3
- PyTorch (`torch`)
- NumPy
- Einops

Install dependencies:

```bash
pip install torch numpy einops
```

## Running

```bash
python3 "Project Nemesis.py"
```

The entry point instantiates `AdvancedAutonomousAIAgent` and begins autonomous learning. Logs are written to `aurora_ai.log` and the console.

## Key Architecture

The system is composed of these major components, all in `Project Nemesis.py`:

- **AdvancedAutonomousAIAgent** — Main orchestrator
- **AdvancedTransformerArchitecture** — Policy/value network with dynamic layer addition
- **MultiModalProcessor** — Vision (CNN), text (Transformer), audio (Conv1d) fusion
- **DifferentiableNeuralComputer** — External memory with content-based addressing
- **WorldModel** — Environment prediction and model-based planning
- **BayesianExploration** — Uncertainty-aware exploration with MC Dropout
- **ModelAgnosticMetaLearning** — MAML for rapid task adaptation
- **SafetyLayer** — Constraint prediction and safe action projection
- **ElasticWeightConsolidation** — Prevents catastrophic forgetting
- **NeuralArchitectureSearch** — Genetic algorithm-based architecture evolution
- **DistributedLearningManager / ParameterServer** — Async multi-worker learning
- **KnowledgeBase** — Persistent experience memory, skill library, concept network

## Coding Conventions

- **Style**: Google-style docstrings, extensive type hints (`Dict`, `List`, `Optional`, `Tuple`, `Union`)
- **Naming**: `PascalCase` for classes, `snake_case` for functions and variables
- **Error handling**: Try/except with `logging.warning`
- **Logging**: `logging` module with rotating file handler + console handler

## Testing & Linting

No test framework, linter, or formatter is currently configured. There is no `pyproject.toml`, `setup.cfg`, `pytest.ini`, or similar tooling.

## Notes

- The file path contains a space (`Project Nemesis.py`) — always quote it in shell commands.
- The project uses pickle for knowledge persistence — be cautious with untrusted data.
