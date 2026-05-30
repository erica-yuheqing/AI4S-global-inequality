import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import os
    import requests
    from tqdm import tqdm
    import time
    import json
    from typing import Dict, Any, List, Optional
    import pandas as pd

    return Any, Dict, List, Optional, json, os, requests, time, tqdm


@app.cell
def _():
    BASE_URL = "https://api.openalex.org"
    OPENALEX_API_KEY = "xxxxxxxxxxxxxxxxxxx"
    return BASE_URL, OPENALEX_API_KEY


@app.cell(hide_code=True)
def _(Any, Dict, OPENALEX_API_KEY, requests, time):
    def get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap a GET request with simple retry logic.
        """
        _params = dict(params)
        if OPENALEX_API_KEY:
            _params["api_key"] = OPENALEX_API_KEY
        for _ in range(5):
            resp = requests.get(url, params=_params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print("status:", resp.status_code)
            print("url:", resp.url)
            print("body:", resp.text)
            # Simple backoff
            time.sleep(2)
        resp.raise_for_status()
        return {}  # This should never be reached in practice

    return (get_json,)


@app.cell(hide_code=True)
def _(BASE_URL, Optional, get_json):
    def find_alphafold_core_work() -> Optional[str]:
        """
        Use search to find the core AlphaFold2 Nature paper and return its OpenAlex ID
        (e.g. 'https://openalex.org/Wxxxx'). You can also hardcode this ID directly
        to skip this step.
        """
        params = {
            "search": "Highly accurate protein structure prediction with AlphaFold",
            "per-page": 1,
        }
        data = get_json(f"{BASE_URL}/works", params)
        results = data.get("results", [])
        if not results:
            return None
        work = results[0]
        print(
            "Found AlphaFold core paper:", work.get("display_name"), work.get("id")
        )
        return work.get("id")

    return (find_alphafold_core_work,)


@app.cell(hide_code=True)
def _(Dict, List):
    def invert_abstract(inverted: Dict[str, List[int]]) -> str:
        """
        Convert OpenAlex abstract_inverted_index into a normal abstract string.
        `inverted` looks like {"protein": [0, 5], "structure": [1], ...}
        """
        if not inverted:
            return ""
        # First find the abstract length
        max_pos = 0
        for positions in inverted.values():
            max_pos = max(max_pos, max(positions))
        # Positions range from 0 to max_pos
        tokens = [""] * (max_pos + 1)
        for token, positions in inverted.items():
            for pos in positions:
                tokens[pos] = token
        # Join tokens into a string
        return " ".join(tokens)

    return (invert_abstract,)


@app.cell(hide_code=True)
def _(BASE_URL, Optional, get_json, invert_abstract, json, tqdm):
    def fetch_works_citing_alphafold(
        alphafold_work_id: str,
        from_year: int = 2015,
        to_year: int = 2025,
        per_page: int = 200,
        max_works: Optional[int] = None,
        output_path: str = "alphafold_citing_works.jsonl",
    ):
        """
        Fetch all works that cite the core AlphaFold paper within a publication-year
        range, and save metadata + abstract + references to a JSONL file.

        Parameters:
        - alphafold_work_id: OpenAlex ID of the AlphaFold core paper (full URL, for
          example 'https://openalex.org/W123456789')
        - from_year, to_year: publication year window
        - per_page: page size (maximum 200)
        - max_works: optional cap on the number of works to fetch for debugging
        - output_path: output file path
        """
        # OpenAlex work IDs use the short form Wxxx; extract it from the URL tail
        short_id = alphafold_work_id.split("/")[-1]

        # Use referenced_works.id filters to find papers citing it
        filters = [
            # f"referenced_works.id:{short_id}",
            # f"referenced_works:{alphafold_work_id}",
            f"cites:{alphafold_work_id}",
            f"from_publication_date:{from_year}-01-01",
            f"to_publication_date:{to_year}-12-31",
        ]
        params = {
            "filter": ",".join(filters),
            "per-page": per_page,
            # Select only the fields you actually need to reduce payload size
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "publication_year",
                    "publication_date",
                    # "host_venue",
                    "cited_by_count",
                    "referenced_works",
                    "abstract_inverted_index",
                    "authorships",
                    "concepts",
                    "primary_location",
                ]
            ),
        }

        cursor = "*"
        total_fetched = 0

        with open(output_path, "w", encoding="utf-8") as f_out:
            pbar = tqdm(desc="Fetching works citing AlphaFold", unit="work")
            while True:
                params["cursor"] = cursor
                data = get_json(f"{BASE_URL}/works", params)
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    # Basic cleanup or simplification can be done here
                    record = {
                        "id": work.get("id"),  # OpenAlex work ID URL
                        "doi": work.get("doi"),
                        "title": work.get("display_name"),
                        "publication_year": work.get("publication_year"),
                        "publication_date": work.get("publication_date"),
                        # Reconstructed abstract
                        "abstract": invert_abstract(
                            work.get("abstract_inverted_index") or {}
                        ),
                        # Authors and institutions
                        "authorships": work.get("authorships", []),
                        # Journal / venue
                        # "host_venue": work.get("host_venue", {}),
                        # Citation count
                        "cited_by_count": work.get("cited_by_count"),
                        # Referenced works (list of OpenAlex IDs)
                        "referenced_works": work.get("referenced_works", []),
                        # Concepts / subjects
                        "concepts": work.get("concepts", []),
                        "primary_location": work.get("primary_location", {}),
                    }

                    f_out.write(json.dumps(record, ensure_ascii=False) + "\\n")
                    total_fetched += 1
                    pbar.update(1)

                    if max_works is not None and total_fetched >= max_works:
                        pbar.close()
                        print(f"Reached max_works={max_works}, stopping.")
                        return

                # Pagination: cursor-based
                meta = data.get("meta", {})
                cursor = meta.get("next_cursor")
                if not cursor:
                    break

            pbar.close()
        print(f"Total works fetched: {total_fetched}")

    return (fetch_works_citing_alphafold,)


@app.cell(hide_code=True)
def _(fetch_works_citing_alphafold, find_alphafold_core_work):
    def download_data():
        # 1. Find the OpenAlex ID of the AlphaFold core paper
        #    (you can run this once and then hardcode the result)
        core_id = find_alphafold_core_work()
        if core_id is None:
            raise RuntimeError(
                "Cannot find AlphaFold core work in OpenAlex search."
            )

        # 2. Fetch papers that cite it
        #    (example: cap at 400 for easier debugging; remove max_works or raise it for full runs)
        fetch_works_citing_alphafold(
            alphafold_work_id=core_id,
            from_year=2015,
            to_year=2025,
            per_page=100,
            # max_works=400,   # Keep this smaller during testing
            output_path="alphafold_citing_works_sample.jsonl",
        )

    return


@app.cell(hide_code=True)
def _(BASE_URL, get_json, json, tqdm):
    def fetch_domain_works(
        from_year: int = 2015,
        to_year: int = 2025,
        per_page: int = 200,
        max_works: int | None = 5000,
        output_path: str = "domain_works_sample.jsonl",
    ):
        """
        Fetch a large "field + time" sample regardless of whether it is AlphaFold-related.
        This example uses works tagged with Biology or Medicine concepts.
        """

        # You can also switch this to a precise filter like concepts.id:Cxxxxx.
        # For readability, this example uses concept-name search.
        filters = [
            f"from_publication_date:{from_year}-01-01",
            f"to_publication_date:{to_year}-12-31",
            "concepts.display_name.search:Biology|Medicine",
        ]
        params = {
            "filter": ",".join(filters),
            "per-page": per_page,
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "publication_year",
                    "publication_date",
                    # "host_venue",
                    "cited_by_count",
                    "referenced_works",
                    "abstract_inverted_index",
                    "authorships",
                    "concepts",
                    "primary_location",
                ]
            ),
        }

        cursor = "*"
        total_fetched = 0

        with open(output_path, "w", encoding="utf-8") as f_out:
            pbar = tqdm(desc="Fetching domain works", unit="work")
            while True:
                params["cursor"] = cursor
                data = get_json(f"{BASE_URL}/works", params)
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    f_out.write(json.dumps(work, ensure_ascii=False) + "\\n")
                    total_fetched += 1
                    pbar.update(1)

                    if max_works is not None and total_fetched >= max_works:
                        pbar.close()
                        print(f"Reached max_works={max_works}, stopping.")
                        return

                meta = data.get("meta", {})
                cursor = meta.get("next_cursor")
                if not cursor:
                    break

            pbar.close()
        print("Total domain works fetched:", total_fetched)


    def download_domain_works():
        fetch_domain_works(
            from_year=2015,
            to_year=2025,
            per_page=200,
            max_works=5000,  # Sample first; scale up later if needed
            output_path="domain_works_sample.jsonl",
        )


    # Note: `max_works=5000` is just for demonstration here. For production runs,
    # you can scale this to hundreds of thousands or more and consider saving to
    # parquet before analysis.
    return


@app.cell
def _(BASE_URL, List, get_json, invert_abstract, json, os, tqdm):
    def fetch_openalex_subfield_corpus_resume_v1(
        corpus_name: str,
        subfield_ids: List[str],
        from_year: int = 2015,
        to_year: int = 2025,
        per_page: int = 100,
        max_works: int | None = None,
        output_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """
        OpenAlex subfield downloader with resume-from-checkpoint support.

        Notes:
        - Uses OpenAlex subfield IDs for filtering
        - subfield_ids can be full URLs, for example: https://openalex.org/subfields/1315
        - Preserves checkpointing and resume capability
        """
        if not subfield_ids:
            raise ValueError("subfield_ids cannot be empty.")

        _safe_name_subfield = (
            corpus_name.lower()
            .replace(" / ", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        _output_subfield = (
            output_path
            or f"domain_{_safe_name_subfield}_{from_year}_{to_year}.jsonl"
        )
        _checkpoint_subfield = (
            checkpoint_path
            or f"domain_{_safe_name_subfield}_{from_year}_{to_year}.checkpoint.json"
        )

        _normalized_subfield_ids = []
        for _subfield_id in subfield_ids:
            if not _subfield_id:
                continue
            _normalized_subfield_ids.append(
                _subfield_id.rstrip("/").split("/")[-1]
            )

        if not _normalized_subfield_ids:
            raise ValueError("No valid subfield IDs found in subfield_ids.")

        _params_subfield = {
            "filter": ",".join(
                [
                    f"primary_topic.subfield.id:{'|'.join(_normalized_subfield_ids)}",
                    f"from_publication_date:{from_year}-01-01",
                    f"to_publication_date:{to_year}-12-31",
                ]
            ),
            "per-page": per_page,
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "publication_year",
                    "publication_date",
                    "cited_by_count",
                    "referenced_works",
                    "abstract_inverted_index",
                    "authorships",
                    "concepts",
                    "primary_location",
                    "primary_topic",
                    "topics",
                    "type",
                ]
            ),
        }

        _cursor_subfield = "*"
        _total_subfield = 0
        _mode_subfield = "w"

        if os.path.exists(_checkpoint_subfield):
            with open(
                _checkpoint_subfield, "r", encoding="utf-8"
            ) as _f_ckpt_in_subfield:
                _state_subfield = json.load(_f_ckpt_in_subfield)
            _cursor_subfield = _state_subfield.get("next_cursor") or "*"
            _total_subfield = _state_subfield.get("total_fetched", 0)
            _mode_subfield = "a"
            print(
                f"Resuming {corpus_name} from checkpoint: {_checkpoint_subfield}"
            )
            print(f"Already fetched: {_total_subfield}")
        else:
            print(f"Starting new download for {corpus_name}")

        with open(
            _output_subfield, _mode_subfield, encoding="utf-8"
        ) as _f_out_subfield:
            _pbar_subfield = tqdm(
                desc=f"Fetching {corpus_name}",
                unit="work",
                initial=_total_subfield,
            )
            while True:
                _params_subfield["cursor"] = _cursor_subfield
                _data_subfield = get_json(f"{BASE_URL}/works", _params_subfield)
                _results_subfield = _data_subfield.get("results", [])
                if not _results_subfield:
                    break

                for _work_subfield in _results_subfield:
                    _record_subfield = {
                        "id": _work_subfield.get("id"),
                        "doi": _work_subfield.get("doi"),
                        "title": _work_subfield.get("display_name"),
                        "publication_year": _work_subfield.get("publication_year"),
                        "publication_date": _work_subfield.get("publication_date"),
                        "abstract": invert_abstract(
                            _work_subfield.get("abstract_inverted_index") or {}
                        ),
                        "cited_by_count": _work_subfield.get("cited_by_count"),
                        "referenced_works": _work_subfield.get(
                            "referenced_works", []
                        ),
                        "authorships": _work_subfield.get("authorships", []),
                        "concepts": _work_subfield.get("concepts", []),
                        "primary_location": _work_subfield.get(
                            "primary_location", {}
                        ),
                        "primary_topic": _work_subfield.get("primary_topic", {}),
                        "topics": _work_subfield.get("topics", []),
                        "type": _work_subfield.get("type"),
                        "domain_label": corpus_name,
                        "domain_subfield_ids": subfield_ids,
                        "normalized_subfield_ids": _normalized_subfield_ids,
                    }
                    _f_out_subfield.write(
                        json.dumps(_record_subfield, ensure_ascii=False) + "\\n"
                    )
                    _total_subfield += 1
                    _pbar_subfield.update(1)

                    if max_works is not None and _total_subfield >= max_works:
                        _pbar_subfield.close()
                        print(
                            f"Reached max_works={max_works} for {corpus_name}, stopping."
                        )
                        print(f"Saved to: {_output_subfield}")
                        return

                _meta_subfield = _data_subfield.get("meta", {})
                _next_cursor_subfield = _meta_subfield.get("next_cursor")

                _checkpoint_state_subfield = {
                    "corpus_name": corpus_name,
                    "subfield_ids": subfield_ids,
                    "normalized_subfield_ids": _normalized_subfield_ids,
                    "from_year": from_year,
                    "to_year": to_year,
                    "per_page": per_page,
                    "output_path": _output_subfield,
                    "total_fetched": _total_subfield,
                    "next_cursor": _next_cursor_subfield,
                }
                with open(
                    _checkpoint_subfield, "w", encoding="utf-8"
                ) as _f_ckpt_out_subfield:
                    json.dump(
                        _checkpoint_state_subfield,
                        _f_ckpt_out_subfield,
                        ensure_ascii=False,
                        indent=2,
                    )

                if not _next_cursor_subfield:
                    break
                _cursor_subfield = _next_cursor_subfield

            _pbar_subfield.close()

        if os.path.exists(_checkpoint_subfield):
            os.remove(_checkpoint_subfield)

        print(f"Completed download for {corpus_name}")
        print(f"Total works fetched for {corpus_name}: {_total_subfield}")
        print(f"Saved to: {_output_subfield}")


    def download_alphafold_related_field_corpora_resume_v3_subfield(
        corpus_name: str,
        subfield_ids: List[str],
        from_year: int = 2015,
        to_year: int = 2025,
        per_page: int = 200,
        max_works: int | None = None,
        output_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """
        Download a full domain corpus by subfield ID with resume support.

        Example:
        download_alphafold_related_field_corpora_resume_v3_subfield(
            corpus_name="structural_biology",
            subfield_ids=["https://openalex.org/subfields/1315"],
            from_year=2015,
            to_year=2025,
            per_page=200,
            max_works=None,
        )
        """
        fetch_openalex_subfield_corpus_resume_v1(
            corpus_name=corpus_name,
            subfield_ids=subfield_ids,
            from_year=from_year,
            to_year=to_year,
            per_page=per_page,
            max_works=max_works,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Life Sciences
    Includes the following three related fields:
    1. Biochemistry, Genetics and Molecular Biology - https://openalex.org/fields/13
    2. Immunology and Microbiology - https://openalex.org/fields/24
    3. Pharmacology, Toxicology and Pharmaceutics - https://openalex.org/fields/30
    """)
    return


@app.cell
def _():
    # Biochemistry, Genetics and Molecular Biology
    # [OpenAlex](https://openalex.org/fields/13)
    # Works count: 20,700,000
    # Citations count: 299,600,000

    # [Aging,](https://openalex.org/subfields/1302)
    # [Biochemistry,](https://openalex.org/subfields/1303)
    # [Biophysics,](https://openalex.org/subfields/1304)
    # [Biotechnology,](https://openalex.org/subfields/1305)
    # [Cancer Research,](https://openalex.org/subfields/1306)
    # [Cell Biology,](https://openalex.org/subfields/1307)
    # [Clinical Biochemistry,](https://openalex.org/subfields/1308)
    # [Developmental Biology,](https://openalex.org/subfields/1309)
    # [Endocrinology,](https://openalex.org/subfields/1310)
    # [Genetics,](https://openalex.org/subfields/1311)
    # [Molecular Biology,](https://openalex.org/subfields/1312)
    # [Molecular Medicine,](https://openalex.org/subfields/1313)
    # [Physiology,](https://openalex.org/subfields/1314)
    # [Structural Biology](https://openalex.org/subfields/1315)

    # Ranked by direct relevance to AlphaFold research, from strongest to weakest:
    # 1. Structural Biology 15403/204700
    # 2. Molecular Biology 2,771,146/10,730,000
    # 3. Biophysics 189,678/1,467,000
    # 4. Genetics 793,584/3,772,000
    # 5. Molecular Medicine 100,605/227,900
    # 6. Biochemistry 81,169/427,300
    # 7. Cell Biology 313,297/1,137,000
    # 8. Biotechnology 130,766/523,200
    # 9. Cancer Research 392,061/1,046,000
    # 10. Developmental Biology 24,304/97,900
    # 11. Physiology 42,909/240,500
    # 12. Clinical Biochemistry 71,744/321,000
    # 13. Endocrinology 66,628/394,900
    # 14. Aging 48,238/99,590

    # Total: 5,041,532

    # Usage example (do not run automatically)
    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="structural_biology",
    #     subfield_ids=["https://openalex.org/subfields/1315"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for structural_biology
    # Total works fetched for structural_biology: 15403/204700
    # Saved to: domain_structural_biology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="molecular_biology",
    #     subfield_ids=["https://openalex.org/subfields/1312"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for molecular_biology
    # Total works fetched for molecular_biology: 2,771,146/10,730,000
    # Saved to: domain_molecular_biology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="biophysics",
    #     subfield_ids=["https://openalex.org/subfields/1304"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for biophysics
    # Total works fetched for biophysics: 189,678/1,467,000
    # Saved to: domain_biophysics_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="genetics",
    #     subfield_ids=["https://openalex.org/subfields/1311"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for genetics
    # Total works fetched for genetics: 793584
    # Saved to: domain_genetics_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="molecular_medicine",
    #     subfield_ids=["https://openalex.org/subfields/1313"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for molecular_medicine
    # Total works fetched for molecular_medicine: 100605
    # Saved to: domain_molecular_medicine_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="biochemistry",
    #     subfield_ids=["https://openalex.org/subfields/1303"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for biochemistry
    # Total works fetched for biochemistry: 81169
    # Saved to: domain_biochemistry_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="cell_biology",
    #     subfield_ids=["https://openalex.org/subfields/1307"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for cell_biology
    # Total works fetched for cell_biology: 313297
    # Saved to: domain_cell_biology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="biotechnology",
    #     subfield_ids=["https://openalex.org/subfields/1305"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for biotechnology
    # Total works fetched for biotechnology: 130766
    # Saved to: domain_biotechnology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="cancer_research",
    #     subfield_ids=["https://openalex.org/subfields/1306"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for cancer_research
    # Total works fetched for cancer_research: 392061
    # Saved to: domain_cancer_research_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="developmental_biology",
    #     subfield_ids=["https://openalex.org/subfields/1309"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for developmental_biology
    # Total works fetched for developmental_biology: 24304
    # Saved to: domain_developmental_biology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="physiology",
    #     subfield_ids=["https://openalex.org/subfields/1314"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for physiology
    # Total works fetched for physiology: 42909
    # Saved to: domain_physiology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="clinical_biochemistry",
    #     subfield_ids=["https://openalex.org/subfields/1308"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for clinical_biochemistry
    # Total works fetched for clinical_biochemistry: 71744
    # Saved to: domain_clinical_biochemistry_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="endocrinology",
    #     subfield_ids=["https://openalex.org/subfields/1310"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for endocrinology
    # Total works fetched for endocrinology: 66628
    # Saved to: domain_endocrinology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="aging",
    #     subfield_ids=["https://openalex.org/subfields/1302"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for aging
    # Total works fetched for aging: 48238
    # Saved to: domain_aging_2015_2025.jsonl
    return


@app.cell
def _():
    # Immunology and Microbiology
    # https://openalex.org/fields/24
    # Works count: 2,838,000
    # Citations count: 49,990,000

    # Ranked from most to least relevant to AlphaFold:

    # 1. Immunology - 9/10
    # Works count: 498,823/1,603,000
    # Citations count: 34,010,000
    # https://openalex.org/subfields/2403

    # 2. Microbiology - 8/10
    # Works count: 101,406/435,300
    # Citations count: 5,009,000
    # https://openalex.org/subfields/2404

    # 3. Virology - 8/10
    # Works count: 60,224/237,000
    # Citations count: 4,735,000
    # https://openalex.org/subfields/2406

    # 4. Applied Microbiology and Biotechnology - 7/10
    # Works count: 41,842/88,010
    # Citations count: 810,300
    # https://openalex.org/subfields/2402

    # 5. Parasitology - 6/10
    # Works count: 474,400
    # Citations count: 5,421,000
    # https://openalex.org/subfields/2405

    # Total: 702,295

    # Brief ranking rationale:
    # - AlphaFold is directly tied to protein structure prediction, so fields that
    #   depend heavily on protein structure analysis, antigen-antibody interactions,
    #   and receptor binding mechanisms are more relevant.
    # - Immunology ranks highest because it frequently studies the structures of
    #   antibodies, antigens, cytokines, receptors, and related proteins.
    # - Microbiology and Virology also rely heavily on protein function and
    #   structure research, especially for pathogen mechanisms, enzymes, surface
    #   proteins, and host interactions.
    # - Applied Microbiology and Biotechnology uses enzyme engineering, protein
    #   design, and structural biology as well, but the focus is more application-
    #   oriented, so the relevance is slightly lower.
    # - Parasitology also involves parasite proteins, host interactions, and drug
    #   targets, but AlphaFold is usually not the field's central method.

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="immunology",
    #     subfield_ids=["https://openalex.org/subfields/2403"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for immunology
    # Total works fetched for immunology: 498823
    # Saved to: domain_immunology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="microbiology",
    #     subfield_ids=["https://openalex.org/subfields/2404"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for microbiology
    # Total works fetched for microbiology: 101406
    # Saved to: domain_microbiology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="virology",
    #     subfield_ids=["https://openalex.org/subfields/2406"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for virology
    # Total works fetched for virology: 60224
    # Saved to: domain_virology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="applied_microbiology_and_biotechnology",
    #     subfield_ids=["https://openalex.org/subfields/2402"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for applied_microbiology_and_biotechnology
    # Total works fetched for applied_microbiology_and_biotechnology: 41842
    # Saved to: domain_applied_microbiology_and_biotechnology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="parasitology",
    #     subfield_ids=["https://openalex.org/subfields/2405"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    return


@app.cell
def _():
    # Pharmacology, Toxicology and Pharmaceutics
    # https://openalex.org/fields/30
    # Works count: 1,161,000
    # Citations count: 12,360,000

    # Ranked from strongest to weakest:
    # 1. Drug Discovery - 10/10, core relevance
    # https://openalex.org/subfields/3002
    # Works count: 2,106/12,820
    # Citations count: 4,122

    # 2. Pharmacology - 9/10, very strong relevance
    # https://openalex.org/subfields/3004
    # Works count: 171,492/653,400
    # Citations count: 5,486,000

    # 3. Pharmaceutical Science - 7/10, moderately strong relevance
    # https://openalex.org/subfields/3003
    # Works count: 102,357/364,400
    # Citations count: 5,358,000

    # 4. Toxicology - 6/10, moderate relevance
    # https://openalex.org/subfields/3005
    # Works count: 37,693/130,600
    # Citations count: 1,508,000

    # Total: 313,648

    # Categories:
    # - Strong relevance: Drug Discovery, Pharmacology
    # - Moderate relevance: Pharmaceutical Science, Toxicology

    # Brief rationale:
    # Drug Discovery uses AlphaFold most directly for target structures, binding
    # sites, and lead optimization. Pharmacology is also closely related through
    # mechanism-of-action studies and efficacy changes. Pharmaceutical Science is
    # more focused on formulations, ADME, and development workflows, so the link is
    # more indirect. Toxicology is mainly relevant for off-target and toxicity-
    # mechanism inference and is usually not the primary method.

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="drug_discovery",
    #     subfield_ids=["https://openalex.org/subfields/3002"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for drug_discovery
    # Total works fetched for drug_discovery: 2106
    # Saved to: domain_drug_discovery_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="pharmacology",
    #     subfield_ids=["https://openalex.org/subfields/3004"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for pharmacology
    # Total works fetched for pharmacology: 171492
    # Saved to: domain_pharmacology_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="pharmaceutical_science",
    #     subfield_ids=["https://openalex.org/subfields/3003"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for pharmaceutical_science
    # Total works fetched for pharmaceutical_science: 102357
    # Saved to: domain_pharmaceutical_science_2015_2025.jsonl

    # download_alphafold_related_field_corpora_resume_v3_subfield(
    #     corpus_name="toxicology",
    #     subfield_ids=["https://openalex.org/subfields/3005"],
    #     from_year=2015,
    #     to_year=2025,
    #     per_page=200,
    #     max_works=None,
    # )
    # Completed download for toxicology
    # Total works fetched for toxicology: 37693
    # Saved to: domain_toxicology_2015_2025.jsonl
    return


@app.cell
def _():
    # Total: 6,057,475
    return


if __name__ == "__main__":
    app.run()
