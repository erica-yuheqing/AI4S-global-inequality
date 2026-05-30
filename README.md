# Artificial intelligence for science reinforces inequalities in global knowledge production: evidence from AlphaFold

**Authors:** Heqing Yu, Weidong Liu*  
**Affiliation:** Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences  
**Contact:** `yuheqing6428@igsnrr.ac.cn`, `liuwd@igsnrr.ac.cn`  
\*Corresponding author

This repository contains the code and reproducibility materials for the study above. It is designed primarily as a paper-reproduction repository rather than a general-purpose Python package. For most readers, the recommended workflow is to download the processed parquet dataset from Zenodo, extract it into the repository root, install the project environment with `uv`, and run the main `marimo` analysis notebook. A separate notebook is included for users who want to reproduce the OpenAlex data-collection stage from scratch.

## Table of Contents

- [Repository Purpose](#repository-purpose)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Availability and Setup](#data-availability-and-setup)
- [Quick Start Demo](#quick-start-demo)
- [Expected Outputs and Runtime](#expected-outputs-and-runtime)
- [Full Reproduction Workflows](#full-reproduction-workflows)
- [Repository Structure](#repository-structure)
- [Code and Notebook Map](#code-and-notebook-map)
- [Limitations and Running on Your Own Data](#limitations-and-running-on-your-own-data)
- [Abstract](#abstract)
- [License](#license)
- [Citation](#citation)

## Repository Purpose

This repository supports two use cases:

1. Reproduce the paper's main figures, extended data figures, and tables from the processed parquet data package.
2. Reproduce the raw OpenAlex collection workflow and then run the downstream analysis notebook.

For most users, the first path is the intended entry point. It is faster, simpler, and sufficient for reviewing the analytical workflow reported in the manuscript.

## System Requirements

### Hardware Requirements

- A standard desktop or laptop computer is sufficient.
- No non-standard hardware is required.
- Reserve at least 10 GB of free disk space for the processed parquet data package and generated outputs.

### Software Requirements

- Python `>=3.11`
- Dependency management: `uv`
- Notebook interface: `marimo`

### Operating System Notes

- The workflow is designed for standard desktop Python environments.
- macOS and Linux are the recommended targets for reproduction.
- The repository does not require platform-specific accelerators or GPU support.

Project dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Installation Guide

### 1. Install `uv`

Install `uv` using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional on macOS with Homebrew:

```bash
brew install uv
```

### 2. Install the project dependencies

From the repository root, run:

```bash
uv sync
```

This creates a local environment consistent with `pyproject.toml` and `uv.lock`.

### Typical installation time

- Installing `uv` usually takes less than 1 minute.
- `uv sync` typically completes within 5-15 minutes on a normal broadband-connected desktop or laptop, depending on network speed and whether dependencies need to be built locally.

## Data Availability and Setup

The processed dataset used for the paper's main analyses is available from Zenodo:

<https://zenodo.org/records/20442985>

After downloading the archive, extract it into the repository root. The root directory should contain at least:

- `derived_tables_dedup/`
- `derived_tables_dedup_pre2019/`

These directories store the processed parquet tables used by the analysis notebook:

- `derived_tables_dedup/`: main deduplicated analytical package for 2019-2025 publications
- `derived_tables_dedup_pre2019/`: earlier-period deduplicated package for 2015-2018 publications

`code/01_data_analysis.py` reads these directories directly from the repository root, so the archive should not be extracted into a nested subdirectory.

## Quick Start Demo

This is the recommended route for most readers and reviewers.

1. Install `uv`.
2. Install project dependencies with `uv sync`.
3. Download the processed dataset from Zenodo.
4. Extract the dataset into the repository root.
5. Launch the main analysis notebook:

```bash
uv run marimo edit code/01_data_analysis.py
```

The command opens the main interactive notebook used to reproduce the paper's analytical workflow, figures, and tables. If you only want to inspect the study outputs, you do not need to run the raw OpenAlex download notebook.

## Expected Outputs and Runtime

### Expected outputs

Running the main analysis notebook gives access to:

- The notebook overview and analytical workflow used in the paper
- Main-text figures
- Extended Data figures
- Supplementary-note figures
- Extended Data tables
- Exported figure files written to `outputs/figures/600dpi/`

### Expected runtime

- Environment setup: typically 5-15 minutes after `uv` is installed
- Quick-start demo: notebook launch usually takes under a minute once dependencies and processed data are already in place
- Full analysis runtime: depends on whether cells are rerun end to end and on local machine performance
- Raw OpenAlex collection workflow: can take substantially longer, potentially hours, because runtime depends on network conditions, API throughput, and whether authenticated OpenAlex requests are used

## Full Reproduction Workflows

### 1. Reproduce the paper's analysis, figures, and tables

This is the recommended path for most readers.

1. Install `uv` and run `uv sync`.
2. Download the processed parquet dataset from Zenodo.
3. Extract it into the repository root.
4. Run:

```bash
uv run marimo edit code/01_data_analysis.py
```

The notebook contains the analyses used for the paper's main figures, extended data figures, supplementary-note figures, and tables.

### 2. Reproduce the raw data collection pipeline

This path is intended for users who want to start from the OpenAlex collection stage.

1. Install `uv` and run `uv sync`.
2. Launch the raw data notebook:

```bash
uv run marimo edit code/00_data_download.py
```

3. Follow the notebook workflow to download raw data from OpenAlex.
4. Update the `OPENALEX_API_KEY` placeholder in the notebook if your workflow requires authenticated OpenAlex requests.
5. Run the main analysis notebook after data preparation:

```bash
uv run marimo edit code/01_data_analysis.py
```

`code/00_data_download.py` contains the OpenAlex download logic. `code/01_data_analysis.py` contains downstream data processing, analysis, and figure/table generation.

## Repository Structure

```text
code/
  00_data_download.py          # Download raw data from OpenAlex
  01_data_analysis.py          # Main analysis notebook for figures and tables
derived_tables_dedup/          # Processed parquet data for 2019-2025
derived_tables_dedup_pre2019/  # Processed parquet data for 2015-2018
outputs/
  figures/600dpi/              # Exported figures
```

## Code and Notebook Map

| File | Purpose |
| --- | --- |
| `code/00_data_download.py` | Collects AlphaFold-related publication records and related metadata from OpenAlex. |
| `code/01_data_analysis.py` | Reproduces the principal analyses, figures, and tables from the processed parquet package. |
| `pyproject.toml` | Defines the project metadata and Python dependencies. |
| `uv.lock` | Locks dependency versions for reproducible environment setup. |

## Limitations and Running on Your Own Data

- This repository is designed primarily to reproduce the results reported in the manuscript.
- The main analysis notebook expects the processed parquet directories to follow the schema used in `derived_tables_dedup/` and `derived_tables_dedup_pre2019/`.
- Running the analysis on your own data is possible only if you prepare equivalent tables and field structure expected by `code/01_data_analysis.py`.
- The raw collection workflow depends on OpenAlex availability, request limits, and network conditions.

## Abstract

<details>
<summary>Show abstract</summary>

Artificial intelligence for science (AI4S) is widely expected to democratize knowledge production, but whether it reduces global scientific inequality remains unclear. We address this question through investigating AlphaFold research, using a global dataset of 4.07 million life-science publications from 2019 to 2025, including nearly 29,000 AlphaFold-related studies across 152 countries and regions. We demonstrate that AlphaFold has rapidly spread to a near-global research community and is associated with greater international collaboration and larger research teams. However, wider access does not mean a more even distribution of scientific output or influence. Countries with stronger pre-existing research bases adopted earlier, generated more AlphaFold-related publications, and occupied more central positions in collaboration networks, whereas others remained peripheral. We interpret these differences as variation in AI4S capacity across three dimensions - adoption speed, production output, and network influence. Threshold analyses further indicate that gains in adoption speed and scientific output accelerate only beyond a critical level of network influence, suggesting discontinuous rather than gradual convergence. Our findings identify a central tension in AI4S: openness can broaden access, but the capacity to convert access into sustained scientific participation and leadership remains highly uneven.

</details>

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Citation

Citation information will be added upon publication.
