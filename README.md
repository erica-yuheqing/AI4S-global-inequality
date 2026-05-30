# Artificial intelligence for science reinforces inequalities in global knowledge production: evidence from AlphaFold

**Authors:** Heqing Yu, Weidong Liu*  
**Affiliation:** Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences  
**Contact:** `yuheqing6428@igsnrr.ac.cn`, `liuwd@igsnrr.ac.cn`  
\*Corresponding author

This repository contains the code and reproducibility materials for the paper above. The project uses `Python`, `uv`, and `marimo` to reproduce the paper's analysis, figures, and tables on global AlphaFold-related research. For most users, the recommended workflow is to download the processed parquet dataset from Zenodo, extract it into the repository root, and run the main analysis notebook directly.

## Quick Start

1. Install project dependencies:

   ```bash
   uv sync
   ```

2. Download the processed dataset from Zenodo:

   <https://zenodo.org/records/20442985>

3. Extract the downloaded archive into the repository root.

4. Launch the main analysis notebook:

   ```bash
   uv run marimo edit code/01_data_analysis.py
   ```

If you only want to inspect the paper's analytical workflow, figures, and tables, you do not need to run the raw OpenAlex download notebook.

## Data Setup

The analysis notebook expects the processed data to be available in the repository root. After extracting the Zenodo archive, the root directory should contain at least the following folders:

- `derived_tables_dedup/`
- `derived_tables_dedup_pre2019/`

These directories store the processed parquet tables used by the analysis notebook:

- `derived_tables_dedup/`: main deduplicated analytical package for 2019-2025 publications
- `derived_tables_dedup_pre2019/`: earlier-period deduplicated package for 2015-2018 publications

`code/01_data_analysis.py` reads these directories directly from the repository root, so the archive should not be extracted into a nested subdirectory.

## Reproduction Workflows

### 1. Reproduce the paper's analysis, figures, and tables

This is the recommended path for most readers.

1. Download the processed parquet dataset from Zenodo.
2. Extract it into the repository root.
3. Run:

   ```bash
   uv run marimo edit code/01_data_analysis.py
   ```

The notebook contains the analysis used for the paper's main figures, extended data figures, and tables.

### 2. Reproduce the raw data collection pipeline

This path is for users who want to start from the OpenAlex collection stage.

1. Launch the raw data notebook:

   ```bash
   uv run marimo edit code/00_data_download.py
   ```

2. Follow the notebook workflow to download raw data from OpenAlex.
   The notebook includes an `OPENALEX_API_KEY` placeholder; update it if your workflow requires authenticated OpenAlex requests.
3. Run the main analysis notebook after data preparation:

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

## Environment

- Python `>=3.11`
- Dependency management: `uv`
- Notebook interface: `marimo`

Install all dependencies with:

```bash
uv sync
```

## Outputs

- The main analysis notebook includes figures for the main paper body, extended data figures, and extended data tables.
- Exported figures are written to `outputs/figures/600dpi/`.

## Abstract

<details>
<summary>Show abstract</summary>

Artificial intelligence for science (AI4S) is widely expected to democratize knowledge production, but whether it reduces global scientific inequality remains unclear. We address this question through investigating AlphaFold research, using a global dataset of 4.07 million life-science publications from 2019 to 2025, including nearly 29,000 AlphaFold-related studies across 152 countries and regions. We demonstrate that AlphaFold has rapidly spread to a near-global research community and is associated with greater international collaboration and larger research teams. However, wider access does not mean a more even distribution of scientific output or influence. Countries with stronger pre-existing research bases adopted earlier, generated more AlphaFold-related publications, and occupied more central positions in collaboration networks, whereas others remained peripheral. We interpret these differences as variation in AI4S capacity across three dimensions - adoption speed, production output, and network influence. Threshold analyses further indicate that gains in adoption speed and scientific output accelerate only beyond a critical level of network influence, suggesting discontinuous rather than gradual convergence. Our findings identify a central tension in AI4S: openness can broaden access, but the capacity to convert access into sustained scientific participation and leadership remains highly uneven.

</details>

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Citation

Citation information will be added upon publication.
