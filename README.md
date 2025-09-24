<div align="center">
<h1>Teaching Transformers to Solve Combinatorial Problems through Efficient Trial & Error</h1>

**Panagiotis Giannoulis<sup>1</sup>, Yorgos Pantis<sup>2,3</sup>, Christos Tzamos<sup>2,3</sup>**

<sup>1</sup>Independent Researcher, Germany<br>
<sup>2</sup>Department of Informatics and Telecommunications, National and Kapodistrian University of Athens, Greece<br>
<sup>3</sup>Archimedes, Athena Research Center, Greece<br>

[![Paper](https://img.shields.io/badge/NeurIPS-2025-red?logo=book&logoColor=white)](https://papers.nips.cc/paper_files/paper/2025/hash/yourpaperid-Abstract.html)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</div>

## Abstract
Despite their proficiency in various language tasks, Large Language Models (LLMs) struggle with combinatorial problems like Satisfiability, Traveling Salesman Problem, or even basic arithmetic. We address this gap through a novel approach for solving problems in the class NP. We focus on the paradigmatic task of Sudoku and achieve state-of-the-art accuracy (99%) compared to prior neuro-symbolic approaches. Unlike prior work that used custom architectures, our method employs a vanilla decoder-only Transformer (GPT-2) without external tools or function calling. Our method integrates imitation learning of simple Sudoku rules with an explicit Depth-First Search (DFS) exploration strategy involving informed guessing and backtracking. Moving beyond imitation learning, we seek to minimize the number of guesses until reaching a solution. We provide a rigorous analysis of this setup, formalizing its connection to a contextual variant of *Min–Sum Set Cover*, a well-studied problem in algorithms and stochastic optimization.

## Citation
If you use this work, please cite it as follows:

```bibtex
@inproceedings{giannouliscombinatorials2025, 
  title={Teaching Transformers to Solve Combinatorial Problems through Efficient Trial & Error}, 
  author={Giannoulis, Panagiotis and Pantis, Yorgos and Tzamos, Christos}, 
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}, 
  year={2025}
}
```

## Sudoku Puzzle Generator
This repository includes code for generating valid Sudoku puzzles. Due to its broader utility, we have extracted this functionality into a separate Python library written in C, named **[SudokuPy](https://github.com/yorgospantis/SudokuPy)**.

You can install it via pip:

```bash
pip install SudokuPy
```

## Project Structure
```
# ReasoningCombinatorials Project Directory

ReasoningCombinatorials/
│
├── dfs_approach/                               # DFS-based trial-and-error solver
│
│ ├── sudoku/                                   # Sudoku model implementation
│ │
│ │ ├── board_enumeration.py                    # Create valid Sudoku boards; part of SudokuPy library
│ │ ├── kaggle_dataset.py                       # Loads and parses Kaggle Sudoku dataset for evaluation
│ │ ├── transformer_model.py                    # Architecture of vanilla decoder-only Transformer by Andrej Karpathy
│ │ ├── optimizer.py                            # Optimizer and learning rate scheduler
│ │ ├── evaluation.py                           # Defines evaluation metrics; cell and board accuracies
│ │ └── training_loop.ipynb                     # Training loop; main notebook that needs all the above
│
│ ├── sat/                                      # SAT model implementation
│ │
│ │ ├── transformer_model.py                    # Architecture of vanilla decoder-only Transformer by Andrej Karpathy
│ │ ├── optimizer.py                            # Optimizer and learning rate scheduler
│ │ ├── evaluation.py                           # Defines evaluation metric; board accuracy
│ │ └── training_loop.ipynb                     # Training loop; main notebook that needs all the above
│
├── sudoku_tools/                               # Additional utilities for DFS-based Sudoku solving
│ │
│ ├── command.txt                               # CLI usage examples
│ ├── main.c                                    # C-based entry point
│ ├── pyproject.toml                            # Python build configuration
│ ├── setup.py                                  # Python setup script
│ └── src/                                      # C source code for Sudoku components
│     ├── enumerate.c                           # Board enumeration logic
│     ├── enumerate.h                           # Header for enumerate.c
│     ├── jcz.c                                 # JCZ solver logic
│     ├── jcz.h                                 # Header for jcz.c
│     ├── sudoku_path.c                         # Sudoku path generation logic
│     └── sudoku_path.h                         # Header for sudoku_path.c
│
├── sat_tools/                                  # Additional utilities for DFS-based SAT solving
│ │
│ ├── command.txt                               # CLI usage examples
│ ├── main.c                                    # C-based entry point
│ ├── pyproject.toml                            # Python build configuration
│ ├── setup.py                                  # Python setup script
│ ├── test_sat_path.c                           # Test cases for SAT path logic
│ └── src/                                      # C source code for SAT components
│     ├── problem_path.c                        # SAT problem path logic
│     ├── problem_path.h                        # Header for problem_path.c
│     ├── sat_path.c                            # SAT-specific logic
│     └── sat_path.h                            # Header for sat_path.c
│
├── one_level_guess_approach/                   # One-level informed guess trial-and-error solver for Sudoku problem
│ │
│ ├── board_enumeration.py                      # Create valid boards; part of SudokuPy library
│ ├── sudoku_generator.py                       # Create valid Sudoku puzzles; part of SudokuPy library
│ ├── single_guess.py                           # Code for making guesses
│ ├── data_generator.py                         # Create valid Sudoku puzzles using guesses
│ ├── transformer_model.py                      # Architecture of vanilla decoder-only Transformer by Andrej Karpathy
│ ├── optimizer.py                              # Optimizer and learning rate scheduler
│ ├── evaluation.py                             # Defines evaluation metrics; cell and board accuracies
│ ├── JCZSolve.c                                # Sudoku solver in C; part of SudokuPy library
│ ├── sudoku1.c                                 # Sudoku logic in C; part of SudokuPy library
│ └── training_loop.ipynb                       # Training loop; main notebook that needs all the above
│
├── README.md                                   # Project documentation
├── MANIFEST.in                                 # Non-Python files to include
├── requirements.txt                            # Python dependencies
└── LICENSE                                     # MIT license
```

## Third-Party Contributions and Acknowledgments
This project uses and modifies the [minGPT](https://github.com/karpathy/minGPT) implementation of a vanilla decoder-only Transformer by Andrej Karpathy, used under the terms of the **MIT License**.
> **Note:** If you are the contributor of the referenced implementation and would prefer changes to this attribution, please contact us.

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program. AWS resources were provided by the National Infrastructures for Research and Technology GRNET and funded by the EU Recovery and Resiliency Facility. 
