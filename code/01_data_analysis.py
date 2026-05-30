import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview

    This notebook reproduces the principal analyses, figures, and tables reported in the study **"Artificial intelligence for science reinforces inequalities in global knowledge production: evidence from AlphaFold"**. The analyses are conducted from a processed parquet data package derived from OpenAlex and organized to support publication-, authorship-, institution-, country-, and network-level examination of AlphaFold-related research.

    ## Notebook Organization

    The notebook is structured in four parts:

    1. Main-text Figures
    2. Extended Data Figures
    3. Supplementary Note Figures
    4. Extended Data Tables

    ## Analytical Scope

    The analysis relies on two processed data directories located at the repository root:

    - `derived_tables_dedup/` for the principal analytical window, **2019-2025**
    - `derived_tables_dedup_pre2019/` for the pre-period reference window, **2015-2018**

    Together, these directories provide a harmonized data resource spanning **2015-2025**. The notebook is intended to be run directly from these processed parquet files.

    ## Outputs

    Figures generated in this notebook are exported to `outputs/figures/600dpi/`. The notebook also assembles the table outputs reported in the main text and Extended Data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Package Description

    The processed data package comprises linked parquet tables derived from OpenAlex. Within each time-window directory, `works.parquet` serves as the principal entry table, while the remaining tables extend the analysis to authorship, institutional affiliation, country attribution, topic assignment, concept annotation, citation linkage, and AlphaFold-label evidence.

    ### Core Tables

    - `works.parquet`: publication-level records, including work identifiers, DOI, title, abstract, publication year and date, document type, citation counts, domain labels, and topic metadata.
    - `authorships.parquet`: author-by-work records, including author order, author identifiers, corresponding-author flags, and raw affiliation strings.
    - `work_institutions.parquet`: institution-level affiliations linked to works and authors, including institution country and Global South indicators.
    - `work_topics.parquet`: OpenAlex topic, subfield, field, and domain assignments for each work.
    - `work_concepts.parquet`: OpenAlex concept annotations and scores.
    - `work_references.parquet`: citation links between source works and referenced works.
    - `af_label_evidence.parquet`: evidence table documenting why a work was labeled as AlphaFold-related, including title, abstract, reference, and keyword-based signals.

    ### Country-Attribution Tables

    - `authorship_countries.parquet`: expanded author-country assignments used in country-level attribution analyses.
    - `first_author_country_credit.parquet`: fractional country credit assigned to first authors.
    - `last_author_country_credit.parquet`: fractional country credit assigned to last authors, where available.
    - `authorship_country_diagnostics.parquet`: diagnostics for first-author country attribution quality and missingness.
    - `last_author_country_diagnostics.parquet`: diagnostics for last-author country attribution quality and missingness, where available.

    ### Time Windows

    The split between `derived_tables_dedup_pre2019/` and `derived_tables_dedup/` is intentional. The former captures the pre-main-period literature for **2015-2018**, whereas the latter contains the principal analytical corpus for **2019-2025**. This separation preserves a common relational structure while distinguishing baseline conditions from the main study period.

    ### AlphaFold Labeling

    AlphaFold-related works are identified through a multi-signal evidence strategy drawing on titles, abstracts, references to core seed papers, and keyword or resource patterns. The table `af_label_evidence.parquet` preserves these labeling signals for auditability and reuse.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import duckdb
    import seaborn as sns
    import geopandas as gpd
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import networkx as nx
    import math
    from scipy import stats
    import json
    from typing import List, Dict, Any
    import pandas as pd
    from pathlib import Path
    import re
    import os

    return (
        Any,
        Dict,
        List,
        Path,
        duckdb,
        gpd,
        json,
        math,
        mo,
        nx,
        pd,
        plt,
        re,
        sns,
        stats,
    )


@app.cell(hide_code=True)
def _(
    Any,
    Dict,
    List,
    extract_raw_affiliation_text,
    extract_work_abstract,
    extract_work_title,
    json,
    pd,
):
    def open_jsonl_text_file(path: str):
        encoding_candidates = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_error = None
        for encoding_name in encoding_candidates:
            try:
                with open(path, "r", encoding=encoding_name) as f:
                    f.readline()
                return open(path, "r", encoding=encoding_name)
            except UnicodeDecodeError as error:
                last_error = error
                continue
        if last_error is not None:
            raise last_error
        return open(path, "r", encoding="utf-8")


    def load_works_jsonl(path: str) -> List[Dict[str, Any]]:
        records = []
        with open_jsonl_text_file(path) as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records


    def iter_works_jsonl(path: str):
        with open_jsonl_text_file(path) as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                yield line_number, json.loads(line)


    def build_basic_tables(
        works: List[Dict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Input: a list of previously downloaded works records (each record is a dict).
        Output:
          - works_df: publication-level information
          - authorships_df: author-work relationship table
          - work_institutions_df: institution-work relationship table

        Note: this is a small-sample inspection helper. The field mapping supports
        two common schemas:
          - native OpenAlex style: display_name / abstract_inverted_index
          - flattened style: title / abstract
        """
        works_rows = []
        authorships_rows = []
        work_institutions_rows = []

        for w in works:
            work_id = w.get("id")
            doi = w.get("doi")
            title = extract_work_title(w)
            year = w.get("publication_year")
            pub_date = w.get("publication_date")
            cited_by_count = w.get("cited_by_count", 0)
            work_type = w.get("type")
            abstract_text = extract_work_abstract(w)

            primary_location = w.get("primary_location") or {}
            source_obj = primary_location.get("source") or {}
            venue_id = source_obj.get("id")
            venue_name = source_obj.get("display_name")
            venue_type = source_obj.get("type")

            primary_topic = w.get("primary_topic") or {}
            concepts = w.get("concepts") or []
            concept_ids = [c.get("id") for c in concepts]
            concept_names = [c.get("display_name") for c in concepts]

            works_rows.append({
                "work_id": work_id,
                "doi": doi,
                "title": title,
                "publication_year": year,
                "publication_date": pub_date,
                "abstract": abstract_text,
                "cited_by_count": cited_by_count,
                "type": work_type,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "venue_type": venue_type,
                "primary_topic_id": primary_topic.get("id"),
                "primary_topic_display_name": primary_topic.get("display_name"),
                "concept_ids": concept_ids,
                "concept_names": concept_names,
            })

            for auth in w.get("authorships", []):
                author = auth.get("author") or {}
                institutions = auth.get("institutions") or []
                authorships_rows.append({
                    "work_id": work_id,
                    "author_position": auth.get("author_position"),
                    "author_id": author.get("id"),
                    "author_name": author.get("display_name"),
                    "is_corresponding": auth.get("is_corresponding"),
                    "raw_affiliation_string": extract_raw_affiliation_text(auth),
                    "institution_ids": [inst.get("id") for inst in institutions],
                    "institution_names": [inst.get("display_name") for inst in institutions],
                    "institution_countries": [inst.get("country_code") for inst in institutions],
                })

                for inst in institutions:
                    work_institutions_rows.append({
                        "work_id": work_id,
                        "author_id": author.get("id"),
                        "institution_id": inst.get("id"),
                        "institution_name": inst.get("display_name"),
                        "country_code": inst.get("country_code"),
                        "institution_type": inst.get("type"),
                    })

        works_df = pd.DataFrame(works_rows)
        authorships_df = pd.DataFrame(authorships_rows)
        work_institutions_df = pd.DataFrame(work_institutions_rows)

        return {
            "works": works_df,
            "authorships": authorships_df,
            "work_institutions": work_institutions_df,
        }

    return iter_works_jsonl, load_works_jsonl


@app.cell(hide_code=True)
def _(Any, Dict, List, Path, re):
    def invert_abstract_index(inverted_index: Dict[str, List[int]]) -> str:
        if not inverted_index:
            return ""

        token_positions = []
        for token, positions in inverted_index.items():
            for pos in positions or []:
                token_positions.append((pos, token))

        if not token_positions:
            return ""

        token_positions.sort(key=lambda x: x[0])
        return " ".join(token for _, token in token_positions)


    def ensure_parent_dir(path_str: str) -> Path:
        path_obj = Path(path_str)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj


    def normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


    def normalize_openalex_id(value: Any) -> str:
        _text = normalize_text(value)
        if not _text:
            return ""
        return _text.upper()


    af_title_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\balphafold\b",
            r"\balphafold2\b",
            r"\balphafold 2\b",
            r"\balphafold3\b",
            r"\balphafold 3\b",
            r"\balphafold db\b",
            r"\balphafold database\b",
            r"\bafdb\b",
            r"\balphafold[- ]multimer\b",
            r"\bcolabfold\b",
        ]
    ]


    af_usage_patterns = {
        "direct_use": [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"using alphafold",
                r"used alphafold",
                r"generated with alphafold",
                r"predicted with alphafold",
                r"predicted by alphafold",
                r"alphafold-predicted",
                r"alphafold predicted",
                r"alphafold-generated",
                r"alphafold generated",
                r"alphafold-derived",
                r"alphafold derived",
                r"alphafold model",
                r"alphafold structure",
                r"alphafold structural model",
                r"alphafold structure model",
                r"alphafold prediction",
                r"alphafold predictions",
                r"predicted structure from alphafold",
                r"structure predicted using alphafold",
                r"structures predicted using alphafold",
                r"structure prediction using alphafold",
                r"alphafold-multimer",
                r"alphafold multimer",
                r"using alphafold-multimer",
                r"used alphafold-multimer",
                r"predicted with alphafold-multimer",
                r"colabfold",
                r"using colabfold",
                r"used colabfold",
                r"predicted with colabfold",
            ]
        ],
        "database_or_resource_use": [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"afdb",
                r"alphafold db",
                r"alphafold-db",
                r"alphafold database",
                r"alphafold protein structure database",
                r"retrieved from alphafold database",
                r"downloaded from alphafold database",
                r"obtained from alphafold database",
                r"structures from alphafold database",
                r"retrieved from afdb",
                r"downloaded from afdb",
                r"obtained from afdb",
            ]
        ],
        "benchmark_or_method_discussion": [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"comparison with alphafold",
                r"compare(?:d)? with alphafold",
                r"comparison against alphafold",
                r"compared against alphafold",
                r"benchmark(?:ing)? alphafold",
                r"benchmark against alphafold",
                r"benchmarked against alphafold",
                r"evaluate alphafold",
                r"evaluated alphafold",
                r"alphafold benchmark",
                r"alphafold baseline",
                r"alphafold as baseline",
                r"alphafold performance",
                r"alphafold accuracy",
                r"versus alphafold",
                r"vs\.? alphafold",
            ]
        ],
    }
    return (
        af_title_patterns,
        af_usage_patterns,
        invert_abstract_index,
        normalize_openalex_id,
        normalize_text,
    )


@app.cell(hide_code=True)
def _(
    Any,
    Dict,
    List,
    af_title_patterns,
    af_usage_patterns,
    invert_abstract_index,
    normalize_openalex_id,
    normalize_text,
    pd,
    re,
):
    def extract_work_title(work: Dict[str, Any]) -> str:
        return normalize_text(work.get("title") or work.get("display_name"))


    def extract_work_abstract(work: Dict[str, Any]) -> str:
        direct_abstract = normalize_text(work.get("abstract"))
        if direct_abstract:
            return direct_abstract
        return invert_abstract_index(work.get("abstract_inverted_index") or {})


    def extract_raw_affiliation_text(auth: Dict[str, Any]) -> str:
        single_value = normalize_text(auth.get("raw_affiliation_string"))
        if single_value:
            return single_value

        multi_values = auth.get("raw_affiliation_strings") or []
        normalized_values = [normalize_text(item) for item in multi_values if normalize_text(item)]
        if normalized_values:
            return " | ".join(normalized_values)

        affiliations = auth.get("affiliations") or []
        fallback_values = []
        for affiliation in affiliations:
            raw_value = normalize_text(affiliation.get("raw_affiliation_string"))
            if raw_value:
                fallback_values.append(raw_value)

        if fallback_values:
            return " | ".join(fallback_values)

        return ""


    def classify_af_signals(
        title: str,
        abstract_text: str,
        referenced_works: List[str],
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        af_seed_set = {normalize_openalex_id(item) for item in af_seed_ids or [] if normalize_openalex_id(item)}
        _title = normalize_text(title)
        _abstract = normalize_text(abstract_text)
        combined_text = f"{_title}\n{_abstract}".strip()
        normalized_references = [normalize_openalex_id(ref) for ref in referenced_works or [] if normalize_openalex_id(ref)]

        title_matches = [p.pattern for p in af_title_patterns if p.search(_title)]
        abstract_matches = [p.pattern for p in af_title_patterns if p.search(_abstract)]
        reference_matches = [ref for ref in normalized_references if ref in af_seed_set]

        _seed_type_map = {
            "HTTPS://OPENALEX.ORG/W3177828909": "alphafold2_core",
            "HTTPS://OPENALEX.ORG/W4396721167": "alphafold3_core",
            "HTTPS://OPENALEX.ORG/W3211795435": "alphafold_db",
            "HTTPS://OPENALEX.ORG/W3202105508": "alphafold_multimer_core",
        }
        _matched_seed_types = {_seed_type_map.get(ref) for ref in reference_matches if _seed_type_map.get(ref)}

        usage_level = "non_af"
        usage_patterns = []
        for level_name, pattern_list in af_usage_patterns.items():
            matched = [p.pattern for p in pattern_list if p.search(combined_text)]
            if matched:
                usage_level = level_name
                usage_patterns.extend(matched)
                break

        is_related = bool(title_matches or abstract_matches or reference_matches or usage_patterns)
        if not is_related:
            usage_level = "non_af"
        elif usage_level == "non_af" and reference_matches:
            usage_level = "citation_only"
        elif usage_level == "non_af":
            usage_level = "mention_only"

        is_af2 = bool(
            re.search(r"alphafold2|alphafold 2", combined_text, re.IGNORECASE)
            or "alphafold2_core" in _matched_seed_types
        )
        is_af3 = bool(
            re.search(r"alphafold3|alphafold 3", combined_text, re.IGNORECASE)
            or "alphafold3_core" in _matched_seed_types
        )
        is_af_db = bool(
            re.search(r"alphafold db|alphafold database|\bafdb\b", combined_text, re.IGNORECASE)
            or "alphafold_db" in _matched_seed_types
        )

        return {
            "is_alphafold_related": is_related,
            "is_alphafold2": is_af2,
            "is_alphafold3": is_af3,
            "is_alphafold_db": is_af_db,
            "af_signal_title": bool(title_matches),
            "af_signal_abstract": bool(abstract_matches),
            "af_signal_reference": bool(reference_matches),
            "af_signal_keyword": bool(usage_patterns),
            "af_usage_level": usage_level,
            "title_matches": title_matches,
            "abstract_matches": abstract_matches,
            "reference_matches": reference_matches,
            "usage_matches": usage_patterns,
        }


    def normalize_openalex_record(
        work: Dict[str, Any],
        domain_label: str = "",
        domain_search_query: str = "",
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        work_id = work.get("id")
        doi = work.get("doi")
        title = extract_work_title(work)
        publication_year = work.get("publication_year")
        publication_date = work.get("publication_date")
        cited_by_count = work.get("cited_by_count", 0)
        work_type = work.get("type")
        referenced_works = work.get("referenced_works") or []

        abstract_text = extract_work_abstract(work)

        primary_location = work.get("primary_location") or {}
        source_obj = primary_location.get("source") or {}
        source_id = source_obj.get("id")
        source_name = source_obj.get("display_name")

        primary_topic = work.get("primary_topic") or {}
        subfield_obj = primary_topic.get("subfield") or {}
        field_obj = primary_topic.get("field") or {}
        domain_obj = primary_topic.get("domain") or {}

        af_flags = classify_af_signals(
            title=title,
            abstract_text=abstract_text,
            referenced_works=referenced_works,
            af_seed_ids=af_seed_ids,
        )

        works_row = {
            "work_id": work_id,
            "doi": doi,
            "title": title,
            "abstract": abstract_text,
            "publication_year": publication_year,
            "publication_date": publication_date,
            "cited_by_count": cited_by_count,
            "type": work_type,
            "domain_label": domain_label,
            "domain_search_query": domain_search_query,
            "primary_topic_id": primary_topic.get("id"),
            "primary_topic_display_name": primary_topic.get("display_name"),
            "primary_field_id": field_obj.get("id"),
            "primary_field_display_name": field_obj.get("display_name"),
            "primary_subfield_id": subfield_obj.get("id"),
            "primary_subfield_display_name": subfield_obj.get("display_name"),
            "primary_domain_id": domain_obj.get("id"),
            "primary_domain_display_name": domain_obj.get("display_name"),
            "journal_or_source_id": source_id,
            "journal_or_source_name": source_name,
            **{k: v for k, v in af_flags.items() if not k.endswith("_matches")},
        }

        reference_rows = [
            {
                "source_work_id": work_id,
                "referenced_work_id": ref_id,
            }
            for ref_id in referenced_works
        ]

        authorship_rows = []
        institution_rows = []
        for auth in work.get("authorships") or []:
            author = auth.get("author") or {}
            author_id = author.get("id")
            author_name = author.get("display_name")
            institutions = auth.get("institutions") or []
            raw_affiliation_text = extract_raw_affiliation_text(auth)

            authorship_rows.append({
                "work_id": work_id,
                "author_position": auth.get("author_position"),
                "author_id": author_id,
                "author_name": author_name,
                "is_corresponding": auth.get("is_corresponding"),
                "raw_affiliation_string": raw_affiliation_text,
            })

            for inst in institutions:
                institution_rows.append({
                    "work_id": work_id,
                    "author_id": author_id,
                    "institution_id": inst.get("id"),
                    "institution_name": inst.get("display_name"),
                    "country_code": inst.get("country_code"),
                    "institution_type": inst.get("type"),
                    "is_global_south": None,
                })

        topic_rows = []
        for topic in work.get("topics") or []:
            subfield_item = topic.get("subfield") or {}
            field_item = topic.get("field") or {}
            domain_item = topic.get("domain") or {}
            topic_rows.append({
                "work_id": work_id,
                "topic_id": topic.get("id"),
                "topic_display_name": topic.get("display_name"),
                "topic_score": topic.get("score"),
                "subfield_id": subfield_item.get("id"),
                "subfield_display_name": subfield_item.get("display_name"),
                "field_id": field_item.get("id"),
                "field_display_name": field_item.get("display_name"),
                "domain_id": domain_item.get("id"),
                "domain_display_name": domain_item.get("display_name"),
            })

        concept_rows = [
            {
                "work_id": work_id,
                "concept_id": concept.get("id"),
                "concept_display_name": concept.get("display_name"),
                "concept_score": concept.get("score"),
            }
            for concept in (work.get("concepts") or [])
        ]

        evidence_rows = []
        for matched_pattern in af_flags["title_matches"]:
            evidence_rows.append({
                "work_id": work_id,
                "evidence_type": "title_match",
                "title_match": True,
                "abstract_match": False,
                "reference_match": False,
                "keyword_match": False,
                "evidence_value": title,
                "matched_pattern": matched_pattern,
            })

        for matched_pattern in af_flags["abstract_matches"]:
            evidence_rows.append({
                "work_id": work_id,
                "evidence_type": "abstract_match",
                "title_match": False,
                "abstract_match": True,
                "reference_match": False,
                "keyword_match": False,
                "evidence_value": abstract_text,
                "matched_pattern": matched_pattern,
            })

        for matched_reference in af_flags["reference_matches"]:
            evidence_rows.append({
                "work_id": work_id,
                "evidence_type": "reference_match",
                "title_match": False,
                "abstract_match": False,
                "reference_match": True,
                "keyword_match": False,
                "evidence_value": matched_reference,
                "matched_pattern": matched_reference,
            })

        for matched_pattern in af_flags["usage_matches"]:
            evidence_rows.append({
                "work_id": work_id,
                "evidence_type": "keyword_match",
                "title_match": False,
                "abstract_match": False,
                "reference_match": False,
                "keyword_match": True,
                "evidence_value": f"{title}\n{abstract_text}".strip(),
                "matched_pattern": matched_pattern,
            })

        return {
            "works": [works_row],
            "work_references": reference_rows,
            "authorships": authorship_rows,
            "work_institutions": institution_rows,
            "work_topics": topic_rows,
            "work_concepts": concept_rows,
            "af_label_evidence": evidence_rows,
        }


    def build_normalized_tables(
        works: List[Dict[str, Any]],
        domain_label: str = "",
        domain_search_query: str = "",
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        table_rows = {
            "works": [],
            "work_references": [],
            "authorships": [],
            "work_institutions": [],
            "work_topics": [],
            "work_concepts": [],
            "af_label_evidence": [],
        }

        for work in works:
            normalized = normalize_openalex_record(
                work=work,
                domain_label=domain_label,
                domain_search_query=domain_search_query,
                af_seed_ids=af_seed_ids,
            )
            for table_name, rows in normalized.items():
                table_rows[table_name].extend(rows)

        return {
            table_name: pd.DataFrame(rows)
            for table_name, rows in table_rows.items()
        }

    return (
        extract_raw_affiliation_text,
        extract_work_abstract,
        extract_work_title,
        normalize_openalex_record,
    )


@app.cell(hide_code=True)
def _(Any, Dict, List, normalize_openalex_record, normalize_text, pd):
    def extract_authorship_country_records(
        work_id: Any,
        auth: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        author = auth.get("author") or {}
        author_id = author.get("id")
        author_name = author.get("display_name")
        author_position = auth.get("author_position")
        is_corresponding = auth.get("is_corresponding")
        is_first_author = author_position == "first"

        direct_countries = [
            normalize_text(country_code).upper()
            for country_code in (auth.get("countries") or [])
            if normalize_text(country_code)
        ]
        direct_countries = list(dict.fromkeys(direct_countries))

        institution_country_records = []
        institution_countries = []
        for institution in auth.get("institutions") or []:
            institution_country = normalize_text(institution.get("country_code")).upper()
            if not institution_country:
                continue
            institution_country_records.append({
                "country_code": institution_country,
                "institution_id": institution.get("id"),
                "institution_name": institution.get("display_name"),
            })
            institution_countries.append(institution_country)

        institution_countries = list(dict.fromkeys(institution_countries))

        if direct_countries:
            selected_countries = direct_countries
            country_source = "authorship_countries_field"
        elif institution_countries:
            selected_countries = institution_countries
            country_source = "institution_country_code"
        else:
            selected_countries = []
            country_source = "missing"

        country_count = len(selected_countries)
        country_weight = 1.0 / country_count if country_count > 0 else None

        authorship_country_rows = []
        for country_code in selected_countries:
            matching_institutions = [
                institution_row
                for institution_row in institution_country_records
                if institution_row["country_code"] == country_code
            ]
            if matching_institutions:
                for institution_row in matching_institutions:
                    authorship_country_rows.append({
                        "work_id": work_id,
                        "author_id": author_id,
                        "author_name": author_name,
                        "author_position": author_position,
                        "is_first_author": is_first_author,
                        "is_corresponding": is_corresponding,
                        "country_code": country_code,
                        "country_source": country_source,
                        "country_weight_within_author": country_weight,
                        "n_countries_for_author": country_count,
                        "institution_id": institution_row["institution_id"],
                        "institution_name": institution_row["institution_name"],
                    })
            else:
                authorship_country_rows.append({
                    "work_id": work_id,
                    "author_id": author_id,
                    "author_name": author_name,
                    "author_position": author_position,
                    "is_first_author": is_first_author,
                    "is_corresponding": is_corresponding,
                    "country_code": country_code,
                    "country_source": country_source,
                    "country_weight_within_author": country_weight,
                    "n_countries_for_author": country_count,
                    "institution_id": None,
                    "institution_name": None,
                })

        return authorship_country_rows


    def build_first_author_country_credit_rows(
        work_id: Any,
        authorship_country_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        first_author_rows = [
            row for row in authorship_country_rows if row.get("is_first_author")
        ]
        if not first_author_rows:
            return []

        first_author_country_map = {}
        for row in first_author_rows:
            country_code = normalize_text(row.get("country_code")).upper()
            if not country_code:
                continue
            if country_code not in first_author_country_map:
                first_author_country_map[country_code] = row.get("country_source")

        ordered_country_codes = list(first_author_country_map.keys())
        country_count = len(ordered_country_codes)
        if country_count == 0:
            return []

        country_weight = 1.0 / country_count
        return [
            {
                "work_id": work_id,
                "country_code": country_code,
                "first_author_fraction": country_weight,
                "n_first_author_countries": country_count,
                "first_author_country_source": first_author_country_map[country_code],
            }
            for country_code in ordered_country_codes
        ]


    def build_authorship_country_diagnostic_rows(
        work_id: Any,
        authorships: List[Dict[str, Any]],
        authorship_country_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        first_author_authorships = [
            auth for auth in authorships if auth.get("author_position") == "first"
        ]
        first_author_country_rows = [
            row for row in authorship_country_rows if row.get("is_first_author")
        ]
        first_author_country_sources = sorted(
            {
                normalize_text(row.get("country_source"))
                for row in first_author_country_rows
                if normalize_text(row.get("country_source"))
            }
        )
        first_author_country_codes = sorted(
            {
                normalize_text(row.get("country_code")).upper()
                for row in first_author_country_rows
                if normalize_text(row.get("country_code"))
            }
        )

        return [{
            "work_id": work_id,
            "has_first_author": bool(first_author_authorships),
            "n_first_author_authorship_rows": len(first_author_authorships),
            "first_author_country_count": len(first_author_country_codes),
            "first_author_country_missing": bool(first_author_authorships) and not bool(first_author_country_codes),
            "first_author_country_sources": first_author_country_sources,
            "first_author_country_codes": first_author_country_codes,
        }]


    def normalize_openalex_record_extended(
        work: Dict[str, Any],
        domain_label: str = "",
        domain_search_query: str = "",
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized_tables = normalize_openalex_record(
            work=work,
            domain_label=domain_label,
            domain_search_query=domain_search_query,
            af_seed_ids=af_seed_ids,
        )

        work_id = work.get("id")
        authorships = work.get("authorships") or []
        authorship_country_rows = []
        for auth in authorships:
            authorship_country_rows.extend(
                extract_authorship_country_records(
                    work_id=work_id,
                    auth=auth,
                )
            )

        normalized_tables["authorship_countries"] = authorship_country_rows
        normalized_tables["first_author_country_credit"] = build_first_author_country_credit_rows(
            work_id=work_id,
            authorship_country_rows=authorship_country_rows,
        )
        normalized_tables["authorship_country_diagnostics"] = build_authorship_country_diagnostic_rows(
            work_id=work_id,
            authorships=authorships,
            authorship_country_rows=authorship_country_rows,
        )
        return normalized_tables


    def build_normalized_tables_extended(
        works: List[Dict[str, Any]],
        domain_label: str = "",
        domain_search_query: str = "",
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        table_rows = {
            "works": [],
            "work_references": [],
            "authorships": [],
            "work_institutions": [],
            "work_topics": [],
            "work_concepts": [],
            "af_label_evidence": [],
            "authorship_countries": [],
            "first_author_country_credit": [],
            "authorship_country_diagnostics": [],
        }

        for work in works:
            normalized = normalize_openalex_record_extended(
                work=work,
                domain_label=domain_label,
                domain_search_query=domain_search_query,
                af_seed_ids=af_seed_ids,
            )
            for table_name, rows in normalized.items():
                table_rows[table_name].extend(rows)

        return {
            table_name: pd.DataFrame(rows)
            for table_name, rows in table_rows.items()
        }

    return (build_normalized_tables_extended,)


@app.cell(hide_code=True)
def _(
    Any,
    Dict,
    List,
    Path,
    build_normalized_tables_extended,
    iter_works_jsonl,
    load_works_jsonl,
    pd,
):
    def write_normalized_tables_to_parquet(
        tables: Dict[str, pd.DataFrame],
        output_dir: str,
        file_stem: str,
    ) -> Dict[str, str]:
        output_paths = {}
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        for table_name, table_df in tables.items():
            output_path = output_base / f"{file_stem}.{table_name}.parquet"
            table_df.to_parquet(output_path, index=False)
            output_paths[table_name] = str(output_path)

        return output_paths


    def append_normalized_tables_to_existing_parquet(
        tables: Dict[str, pd.DataFrame],
        output_dir: str,
        file_stem: str,
    ) -> Dict[str, str]:
        """
        Note: this is a conservative append interface.
        The current implementation still writes separate parquet files per batch,
        which is better suited to downstream glob-based reads in DuckDB, for example:
        derived_tables/works/*.parquet
        """
        output_paths = {}
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        for table_name, table_df in tables.items():
            table_dir = output_base / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            output_path = table_dir / f"{file_stem}.parquet"
            table_df.to_parquet(output_path, index=False)
            output_paths[table_name] = str(output_path)

        return output_paths


    def build_af_seed_works_table(seed_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Example `seed_rows`:
        [
            {
                "seed_work_id": "https://openalex.org/W...",
                "seed_type": "alphafold2_core",
                "title": "Highly accurate protein structure prediction with AlphaFold"
            }
        ]
        """
        return pd.DataFrame(seed_rows)


    def prepare_domain_tables_from_jsonl(
        jsonl_path: str,
        domain_label: str,
        domain_search_query: str,
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Note: this is a manual entry point.
        Defining it does not trigger execution; the JSONL file is read only when you call it explicitly.
        This helper reads all records into memory at once and is therefore suitable only for small-sample inspection.
        For large files, use `stream_normalize_jsonl_to_parquet(...)` instead.
        """
        works_records = load_works_jsonl(jsonl_path)
        return build_normalized_tables_extended(
            works=works_records,
            domain_label=domain_label,
            domain_search_query=domain_search_query,
            af_seed_ids=af_seed_ids,
        )


    def write_batch_tables_to_partitioned_parquet(
        tables: Dict[str, pd.DataFrame],
        output_dir: str,
        file_stem: str,
        batch_index: int,
    ) -> Dict[str, str]:
        output_paths = {}
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        for table_name, table_df in tables.items():
            if table_df.empty:
                continue
            table_dir = output_base / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            output_path = table_dir / f"{file_stem}.part_{batch_index:06d}.parquet"
            table_df.to_parquet(output_path, index=False)
            output_paths[table_name] = str(output_path)

        return output_paths


    def normalize_works_batch(
        works_batch: List[Dict[str, Any]],
        domain_label: str = "",
        domain_search_query: str = "",
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        return build_normalized_tables_extended(
            works=works_batch,
            domain_label=domain_label,
            domain_search_query=domain_search_query,
            af_seed_ids=af_seed_ids,
        )


    def detect_existing_batch_indices(
        output_dir: str,
        file_stem: str,
        table_name: str = "works",
    ) -> List[int]:
        table_dir = Path(output_dir) / table_name
        if not table_dir.exists():
            return []

        batch_indices = []
        prefix = f"{file_stem}.part_"
        suffix = ".parquet"
        for parquet_path in table_dir.glob(f"{file_stem}.part_*.parquet"):
            file_name = parquet_path.name
            if not (file_name.startswith(prefix) and file_name.endswith(suffix)):
                continue
            batch_str = file_name[len(prefix):-len(suffix)]
            if batch_str.isdigit():
                batch_indices.append(int(batch_str))

        return sorted(batch_indices)


    def estimate_resume_position(
        output_dir: str,
        file_stem: str,
        batch_size: int,
        table_name: str = "works",
    ) -> Dict[str, Any]:
        existing_batch_indices = detect_existing_batch_indices(
            output_dir=output_dir,
            file_stem=file_stem,
            table_name=table_name,
        )

        if not existing_batch_indices:
            return {
                "resume_enabled": False,
                "existing_batch_indices": [],
                "next_batch_index": 1,
                "records_to_skip": 0,
            }

        completed_batches = len(existing_batch_indices)
        next_batch_index = max(existing_batch_indices) + 1
        records_to_skip = completed_batches * batch_size
        return {
            "resume_enabled": True,
            "existing_batch_indices": existing_batch_indices,
            "next_batch_index": next_batch_index,
            "records_to_skip": records_to_skip,
        }


    def stream_normalize_jsonl_to_parquet(
        jsonl_path: str,
        domain_label: str,
        domain_search_query: str,
        output_dir: str = "derived_tables",
        file_stem: str | None = None,
        af_seed_ids: List[str] | None = None,
        batch_size: int = 5000,
        start_batch_index: int = 1,
        max_records: int | None = None,
        progress_every_batches: int = 10,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Streaming normalization entry point for large files:
        - read JSONL line by line
        - normalize records in batches
        - write parquet partitions batch by batch
        - resume automatically from previously written batch files

        Suitable for JSONL files on the order of 35 GB without loading the full file into memory at once.
        """
        resolved_file_stem = file_stem or Path(jsonl_path).stem
        seed_ids = af_seed_ids or []
        batch_records = []
        total_records = 0
        total_written_batches = 0
        latest_output_paths = {}
        skipped_records = 0
        table_write_counts = {
            "works": 0,
            "work_references": 0,
            "authorships": 0,
            "work_institutions": 0,
            "work_topics": 0,
            "work_concepts": 0,
            "af_label_evidence": 0,
            "authorship_countries": 0,
            "first_author_country_credit": 0,
            "authorship_country_diagnostics": 0,
        }

        resume_state = {
            "resume_enabled": False,
            "existing_batch_indices": [],
            "next_batch_index": start_batch_index,
            "records_to_skip": 0,
        }
        current_batch_index = start_batch_index

        if resume:
            resume_state = estimate_resume_position(
                output_dir=output_dir,
                file_stem=resolved_file_stem,
                batch_size=batch_size,
            )
            current_batch_index = max(start_batch_index, resume_state["next_batch_index"])

        records_to_skip = 0
        if resume and current_batch_index == resume_state["next_batch_index"]:
            records_to_skip = resume_state["records_to_skip"]

        def _flush_batch(_batch_records: List[Dict[str, Any]], _batch_index: int) -> Dict[str, str]:
            if not _batch_records:
                return {}
            batch_tables = normalize_works_batch(
                works_batch=_batch_records,
                domain_label=domain_label,
                domain_search_query=domain_search_query,
                af_seed_ids=seed_ids,
            )
            batch_paths = write_batch_tables_to_partitioned_parquet(
                tables=batch_tables,
                output_dir=output_dir,
                file_stem=resolved_file_stem,
                batch_index=_batch_index,
            )
            for table_name, table_df in batch_tables.items():
                table_write_counts[table_name] += len(table_df)
            return batch_paths

        for line_number, work in iter_works_jsonl(jsonl_path):
            if skipped_records < records_to_skip:
                skipped_records += 1
                if progress_every_batches > 0 and skipped_records % (batch_size * progress_every_batches) == 0:
                    print(
                        f"Skipped {skipped_records:,} records to resume at batch {current_batch_index:,} "
                        f"from {jsonl_path}"
                    )
                continue

            batch_records.append(work)
            total_records += 1

            if max_records is not None and total_records >= max_records:
                latest_output_paths = _flush_batch(batch_records, current_batch_index)
                total_written_batches += 1 if batch_records else 0
                batch_records = []
                break

            if len(batch_records) >= batch_size:
                latest_output_paths = _flush_batch(batch_records, current_batch_index)
                total_written_batches += 1
                if progress_every_batches > 0 and total_written_batches % progress_every_batches == 0:
                    print(
                        f"Processed {total_records:,} new records across {total_written_batches:,} new batches "
                        f"from {jsonl_path}; latest source line {line_number:,}"
                    )
                batch_records = []
                current_batch_index += 1

        if batch_records:
            latest_output_paths = _flush_batch(batch_records, current_batch_index)
            total_written_batches += 1

        return {
            "jsonl_path": jsonl_path,
            "output_dir": output_dir,
            "file_stem": resolved_file_stem,
            "batch_size": batch_size,
            "start_batch_index": start_batch_index,
            "resume": resume,
            "resume_state": resume_state,
            "records_skipped_for_resume": skipped_records,
            "batches_written": total_written_batches,
            "records_processed": total_records,
            "table_row_counts": table_write_counts,
            "latest_output_paths": latest_output_paths,
        }

    return (
        append_normalized_tables_to_existing_parquet,
        build_af_seed_works_table,
        prepare_domain_tables_from_jsonl,
        stream_normalize_jsonl_to_parquet,
    )


@app.cell(hide_code=True)
def _(
    Any,
    Dict,
    List,
    Path,
    af_seed_ids_example,
    append_normalized_tables_to_existing_parquet,
    prepare_domain_tables_from_jsonl,
    stream_normalize_jsonl_to_parquet,
):
    def run_manual_example(
        jsonl_path: str,
        domain_label: str,
        domain_search_query: str,
        output_dir: str = "derived_tables",
        file_stem: str | None = None,
        af_seed_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        _file_stem = file_stem or Path(jsonl_path).stem

        example_tables = prepare_domain_tables_from_jsonl(
            jsonl_path=jsonl_path,
            domain_label=domain_label,
            domain_search_query=domain_search_query,
            af_seed_ids=af_seed_ids
            if af_seed_ids is not None
            else af_seed_ids_example,
        )

        example_output_paths = append_normalized_tables_to_existing_parquet(
            tables=example_tables,
            output_dir=output_dir,
            file_stem=_file_stem,
        )

        return {
            "tables": example_tables,
            "output_paths": example_output_paths,
        }


    def run_streaming_example(
        jsonl_path: str,
        domain_label: str,
        domain_search_query: str,
        output_dir: str = "derived_tables",
        file_stem: str | None = None,
        af_seed_ids: List[str] | None = None,
        batch_size: int = 5000,
        start_batch_index: int = 1,
        max_records: int | None = None,
        progress_every_batches: int = 10,
        resume: bool = False,
    ) -> Dict[str, Any]:
        return stream_normalize_jsonl_to_parquet(
            jsonl_path=jsonl_path,
            domain_label=domain_label,
            domain_search_query=domain_search_query,
            output_dir=output_dir,
            file_stem=file_stem,
            af_seed_ids=af_seed_ids
            if af_seed_ids is not None
            else af_seed_ids_example,
            batch_size=batch_size,
            start_batch_index=start_batch_index,
            max_records=max_records,
            progress_every_batches=progress_every_batches,
            resume=resume,
        )


    # Full small-sample entry point: suitable for spot checks only, not for 35 GB files
    # run_manual_example(
    #     jsonl_path = "raw_jsonl/domain_structural_biology_2015_2025.jsonl",
    #     domain_label= "structural_biology",
    #     domain_search_query="your original query",
    #     output_dir = "derived_tables",
    #     file_stem = "domain_structural_biology_2015_2025"
    # )

    # Recommended entry point for large files: streaming batch conversion for JSONL files on the order of 35 GB
    # For a first run, `resume=False` is recommended; after interruption, `resume=True` is recommended for recovery
    # run_streaming_example(
    #     jsonl_path="domain_structural_biology_2015_2025.jsonl",
    #     # jsonl_path="data_samples/alphafold_citing_works_sample.jsonl",
    #     domain_label="structural_biology",
    #     domain_search_query='"structural biology" OR protein structure OR cryo-EM OR crystallography OR NMR structure',
    #     output_dir="derived_tables",
    #     file_stem="domain_structural_biology_2015_2025",
    #     batch_size=50000,
    #     # max_records=50000,
    #     progress_every_batches=50,
    #     resume=True,
    # )

    # run_streaming_example(
    #     jsonl_path="raw_jsonl/domain_drug_discovery_medicinal_chemistry_2015_2025.jsonl",
    #     # jsonl_path="data_samples/alphafold_citing_works_sample.jsonl",
    #     domain_label="drug_discovery_medicinal_chemistry",
    #     domain_search_query='https://openalex.org/concepts/C74187038',
    #     output_dir="derived_tables",
    #     file_stem="domain_drug_discovery_medicinal_chemistry_2015_2025",
    #     batch_size=10000,
    #     # max_records=50000,
    #     progress_every_batches=10,
    #     resume=True,
    # )

    # run_streaming_example(
    #     jsonl_path="raw_jsonl/domain_protein_structure_2015_2025.jsonl",
    #     domain_label="protein_structure",
    #     domain_search_query='https://openalex.org/concepts/C47701112',
    #     output_dir="derived_tables",
    #     file_stem="domain_protein_structure_2015_2025",
    #     batch_size=10000,
    #     # max_records=50000,
    #     progress_every_batches=10,
    #     resume=False,
    # )

    # run_streaming_example(
    #     jsonl_path="raw_jsonl/domain_bioinformatics_2015_2025.jsonl",
    #     domain_label="bioinformatics",
    #     domain_search_query='https://openalex.org/concepts/C60644358',
    #     output_dir="derived_tables",
    #     file_stem="domain_bioinformatics_2015_2025",
    #     batch_size=100000,
    #     # max_records=50000,
    #     progress_every_batches=50,
    #     resume=False,
    # )
    return (run_streaming_example,)


@app.cell(hide_code=True)
def _(build_af_seed_works_table):
    af_seed_rows_example = [
        {
            "seed_work_id": "https://openalex.org/W3177828909",
            "seed_type": "alphafold2_core",
            "title": "Highly accurate protein structure prediction with AlphaFold",
        },
        {
            "seed_work_id": "https://openalex.org/W4396721167",
            "seed_type": "alphafold3_core",
            "title": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
        },
        {
            "seed_work_id": "https://openalex.org/W3211795435",
            "seed_type": "alphafold_db",
            "title": "The AlphaFold Protein Structure Database",
        },
        {
            "seed_work_id": "https://openalex.org/W3202105508",
            "seed_type": "alphafold_multimer_core",
            "title": "Protein complex prediction with AlphaFold-Multimer",
        },
    ]

    af_seed_works_example = build_af_seed_works_table(af_seed_rows_example)
    af_seed_ids_example = af_seed_works_example["seed_work_id"].tolist()

    af_seed_works_example
    return (af_seed_ids_example,)


@app.cell(hide_code=True)
def _(Any, Dict, List, Path, af_seed_ids_example, run_streaming_example):
    # Batch JSONL entry point: iterate over all `.jsonl` files in a directory and call `run_streaming_example` on each one (not executed by default)
    # Usage:
    # 1. Set `jsonl_root_dir` to your target directory
    # 2. To run the batch process, set `dry_run=False` or uncomment the function call near the bottom
    # 3. `domain_label` defaults to the filename stem; `domain_search_query` defaults to empty and can be customized as needed

    def run_streaming_example_for_all_jsonl(
        jsonl_root_dir: str,
        output_dir: str = "derived_tables",
        af_seed_ids: List[str] | None = None,
        batch_size: int = 100000,
        start_batch_index: int = 1,
        max_records: int | None = None,
        progress_every_batches: int = 50,
        resume: bool = False,
        recursive: bool = True,
        dry_run: bool = True,
    ) -> List[Dict[str, Any]]:
        _root_path = Path(jsonl_root_dir)
        _glob_pattern = "**/*.jsonl" if recursive else "*.jsonl"
        _jsonl_paths = sorted(_root_path.glob(_glob_pattern))

        _results = []
        for _jsonl_path in _jsonl_paths:
            _file_stem = _jsonl_path.stem
            _domain_label = _file_stem.replace("domain_", "").replace("_2015_2025", "")
            _domain_search_query = ""

            _job_kwargs = {
                "jsonl_path": str(_jsonl_path),
                "domain_label": _domain_label,
                "domain_search_query": _domain_search_query,
                "output_dir": output_dir,
                "file_stem": _file_stem,
                "af_seed_ids": af_seed_ids if af_seed_ids is not None else af_seed_ids_example,
                "batch_size": batch_size,
                "start_batch_index": start_batch_index,
                "max_records": max_records,
                "progress_every_batches": progress_every_batches,
                "resume": resume,
            }

            if dry_run:
                _results.append({
                    "status": "planned",
                    **_job_kwargs,
                })
            else:
                _results.append(run_streaming_example(**_job_kwargs))

        return _results


    # Not executed automatically; to preview which files would be processed, call manually:
    # run_streaming_example_for_all_jsonl(
    #     jsonl_root_dir="raw_jsonl/Pharmacology_Toxicology_and_Pharmaceutics",
    #     output_dir="derived_tables",
    #     batch_size=1000,
    #     progress_every_batches=50,
    #     resume=False,
    #     recursive=True,
    #     dry_run=True,
    # )

    # To execute the batch workflow, uncomment the call below and set `dry_run=False`:
    # run_streaming_example_for_all_jsonl(
    #     # jsonl_root_dir="raw_jsonl/Biochemistry_Genetics_and_Molecular_Biology",
    #     jsonl_root_dir="/Volumes/JadenSSD/DATA/AlphaFoldResearch/raw_jsonl/Biochemistry_Genetics_and_Molecular_Biology",
    #     output_dir="derived_tables",
    #     batch_size=100000,
    #     progress_every_batches=50,
    #     resume=True,
    #     recursive=True,
    #     dry_run=False,
    # )

    # run_streaming_example_for_all_jsonl(
    #     jsonl_root_dir="raw_jsonl/Immunology_and_Microbiology",
    #     # jsonl_root_dir="/Volumes/JadenSSD/DATA/AlphaFoldResearch/raw_jsonl/Immunology_and_Microbiology",
    #     output_dir="derived_tables",
    #     batch_size=100000,
    #     progress_every_batches=50,
    #     resume=False,
    #     recursive=True,
    #     dry_run=False,
    # )

    # run_streaming_example_for_all_jsonl(
    #     jsonl_root_dir="raw_jsonl/Pharmacology_Toxicology_and_Pharmaceutics",
    #     output_dir="derived_tables",
    #     batch_size=100000,
    #     progress_every_batches=50,
    #     resume=False,
    #     recursive=True,
    #     dry_run=False,
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- WITH base AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/works/*.parquet')
        -- ),
        -- text_probe AS (
        --     SELECT 'text_has_alphafold_in_title' AS metric, COUNT(*) AS count
        --     FROM base
        --     WHERE regexp_matches(lower(coalesce(title, '')), 'alphafold|colabfold|afdb')
        --     UNION ALL
        --     SELECT 'text_has_alphafold_in_abstract' AS metric, COUNT(*) AS count
        --     FROM base
        --     WHERE regexp_matches(lower(coalesce(abstract, '')), 'alphafold|colabfold|afdb')
        -- ),
        -- flag_probe AS (
        --     SELECT 'flag_is_alphafold_related' AS metric, SUM(CASE WHEN is_alphafold_related THEN 1 ELSE 0 END) AS count
        --     FROM base
        --     UNION ALL
        --     SELECT 'flag_af_signal_title' AS metric, SUM(CASE WHEN af_signal_title THEN 1 ELSE 0 END) AS count
        --     FROM base
        --     UNION ALL
        --     SELECT 'flag_af_signal_abstract' AS metric, SUM(CASE WHEN af_signal_abstract THEN 1 ELSE 0 END) AS count
        --     FROM base
        --     UNION ALL
        --     SELECT 'flag_af_signal_reference' AS metric, SUM(CASE WHEN af_signal_reference THEN 1 ELSE 0 END) AS count
        --     FROM base
        --     UNION ALL
        --     SELECT 'flag_af_signal_keyword' AS metric, SUM(CASE WHEN af_signal_keyword THEN 1 ELSE 0 END) AS count
        --     FROM base
        -- )
        -- SELECT * FROM text_probe
        -- UNION ALL
        -- SELECT * FROM flag_probe
        -- ORDER BY metric
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- WITH base AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/works/*.parquet')
        -- ),
        -- normalized AS (
        --     SELECT
        --         work_id,
        --         doi,
        --         title,
        --         lower(trim(regexp_replace(coalesce(title, ''), '\\s+', ' ', 'g'))) AS normalized_title
        --     FROM base
        -- ),
        -- duplicate_work_id AS (
        --     SELECT
        --         'duplicate_work_id' AS duplicate_type,
        --         work_id AS duplicate_key,
        --         COUNT(*) AS duplicate_count
        --     FROM normalized
        --     WHERE work_id IS NOT NULL AND trim(work_id) <> ''
        --     GROUP BY work_id
        --     HAVING COUNT(*) > 1
        -- ),

        -- all_duplicates AS (
        --     SELECT * FROM duplicate_work_id
        -- )
        -- SELECT *
        -- FROM all_duplicates
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _af_related_pre2019 = mo.sql(
        f"""
        -- SELECT
        --     work_id,
        --     doi,
        --     publication_year,
        --     publication_date,
        --     title,
        --     abstract,
        --     is_alphafold_related,
        --     is_alphafold2,
        --     is_alphafold3,
        --     is_alphafold_db,
        --     af_signal_title,
        --     af_signal_abstract,
        --     af_signal_reference,
        --     af_signal_keyword,
        --     af_usage_level,
        --     primary_topic_display_name,
        --     primary_field_display_name,
        --     primary_subfield_display_name,
        --     journal_or_source_name,
        --     cited_by_count,
        --     domain_label
        -- FROM read_parquet('derived_tables/works/*.parquet')
        -- WHERE coalesce(is_alphafold_related, false)
        --   AND coalesce(publication_year, 0) < 2019
        -- ORDER BY publication_year, publication_date, cited_by_count DESC, title
        """
    )
    return


@app.cell(hide_code=True)
def _(Path, pd):
    _path_diagnostic = pd.DataFrame([
        {
            "path": "derived_tables",
            "exists": Path("derived_tables").exists(),
            "is_dir": Path("derived_tables").is_dir(),
            "parquet_files": sorted(str(p) for p in Path("derived_tables").glob("*.parquet")) if Path("derived_tables").exists() else [],
            "child_dirs": sorted(str(p) for p in Path("derived_tables").iterdir() if p.is_dir()) if Path("derived_tables").exists() else [],
        },
        {
            "path": "derived_tables/works",
            "exists": Path("derived_tables/works").exists(),
            "is_dir": Path("derived_tables/works").is_dir(),
            "parquet_files": sorted(str(p) for p in Path("derived_tables/works").glob("*.parquet")) if Path("derived_tables/works").exists() else [],
            "child_dirs": sorted(str(p) for p in Path("derived_tables/works").iterdir() if p.is_dir()) if Path("derived_tables/works").exists() and Path("derived_tables/works").is_dir() else [],
        },
        {
            "path": "derived_tables_dedup",
            "exists": Path("derived_tables_dedup").exists(),
            "is_dir": Path("derived_tables_dedup").is_dir(),
            "parquet_files": sorted(str(p) for p in Path("derived_tables_dedup").glob("*.parquet")) if Path("derived_tables_dedup").exists() else [],
            "child_dirs": sorted(str(p) for p in Path("derived_tables_dedup").iterdir() if p.is_dir()) if Path("derived_tables_dedup").exists() else [],
        },
    ])

    _path_diagnostic
    return


@app.cell(hide_code=True)
def dedup_function():
    # _dedup_output_dir = Path("derived_tables_dedup")
    # _dedup_output_dir.mkdir(parents=True, exist_ok=True)

    # def _load_parquet_row_count(_paths):
    #     _row_count = 0
    #     for _path in _paths:
    #         _row_count += len(pd.read_parquet(_path))
    #     return _row_count

    # _works_source_paths = sorted(Path("derived_tables/works").glob("*.parquet"))
    # _works_rows_before = _load_parquet_row_count(_works_source_paths) if _works_source_paths else 0

    # if not _works_source_paths:
    #     _dedup_summary = pd.DataFrame(
    #         [{"table_name": "works", "rows_before": 0, "rows_after": 0, "output_path": str(_dedup_output_dir / "works.parquet")}]
    #     )
    # else:
    #     _works_minimal_frames = []
    #     for _path in _works_source_paths:
    #         _works_part_for_schema = pd.read_parquet(_path)
    #         _available_columns = _works_part_for_schema.columns.tolist()
    #         _needed_columns = [col for col in ["work_id", "publication_year", "publication_date", "cited_by_count"] if col in _available_columns]
    #         _works_minimal_frames.append(_works_part_for_schema[_needed_columns].copy())

    #     _works_all_minimal = pd.concat(_works_minimal_frames, ignore_index=True) if _works_minimal_frames else pd.DataFrame()

    #     if _works_all_minimal.empty:
    #         _dedup_summary = pd.DataFrame(
    #             [{"table_name": "works", "rows_before": 0, "rows_after": 0, "output_path": str(_dedup_output_dir / "works.parquet")}]
    #         )
    #     elif "work_id" not in _works_all_minimal.columns:
    #         raise KeyError("work_id")
    #     else:
    #         _works_ranked = _works_all_minimal.copy()
    #         if "publication_year" in _works_ranked.columns:
    #             _works_ranked = _works_ranked[_works_ranked["publication_year"].fillna(0) >= 2019].copy()
    #         _works_ranked["_publication_date_rank"] = pd.to_datetime(
    #             _works_ranked["publication_date"], errors="coerce"
    #         ) if "publication_date" in _works_ranked.columns else pd.NaT
    #         _works_ranked["_cited_by_count_rank"] = _works_ranked["cited_by_count"] if "cited_by_count" in _works_ranked.columns else 0

    #         _works_dedup_keys = (
    #             _works_ranked
    #             .sort_values(
    #                 by=["work_id", "_publication_date_rank", "_cited_by_count_rank"],
    #                 ascending=[True, False, False],
    #                 na_position="last",
    #             )
    #             .drop_duplicates(subset=["work_id"], keep="first")[["work_id"]]
    #             .reset_index(drop=True)
    #         )

    #         _kept_work_ids = set(_works_dedup_keys["work_id"].dropna().astype(str))
    #         _works_dedup_frames = []

    #         for _path in _works_source_paths:
    #             _works_part = pd.read_parquet(_path)
    #             if "work_id" not in _works_part.columns:
    #                 raise KeyError("work_id")
    #             _works_part_filtered = _works_part[_works_part["work_id"].astype(str).isin(_kept_work_ids)]
    #             if "publication_year" in _works_part_filtered.columns:
    #                 _works_part_filtered = _works_part_filtered[_works_part_filtered["publication_year"].fillna(0) >= 2019]
    #             if not _works_part_filtered.empty:
    #                 _works_dedup_frames.append(_works_part_filtered)

    #         _works_dedup_all = pd.concat(_works_dedup_frames, ignore_index=True) if _works_dedup_frames else pd.DataFrame()

    #         if _works_dedup_all.empty:
    #             _works_dedup = _works_dedup_all.copy()
    #         else:
    #             _works_dedup_ranked = _works_dedup_all.copy()
    #             _works_dedup_ranked["_publication_date_rank"] = pd.to_datetime(
    #                 _works_dedup_ranked["publication_date"], errors="coerce"
    #             ) if "publication_date" in _works_dedup_ranked.columns else pd.NaT
    #             _works_dedup_ranked["_cited_by_count_rank"] = _works_dedup_ranked["cited_by_count"] if "cited_by_count" in _works_dedup_ranked.columns else 0

    #             _works_dedup = (
    #                 _works_dedup_ranked
    #                 .sort_values(
    #                     by=["work_id", "_publication_date_rank", "_cited_by_count_rank"],
    #                     ascending=[True, False, False],
    #                     na_position="last",
    #                 )
    #                 .drop_duplicates(subset=["work_id"], keep="first")
    #                 .drop(columns=["_publication_date_rank", "_cited_by_count_rank"])
    #                 .reset_index(drop=True)
    #             )

    #         _works_output_path = _dedup_output_dir / "works.parquet"
    #         _works_dedup.to_parquet(_works_output_path, index=False)

    #         _dedup_summary_rows = [{
    #             "table_name": "works",
    #             "rows_before": _works_rows_before,
    #             "rows_after": len(_works_dedup),
    #             "output_path": str(_works_output_path),
    #         }]

    #         _related_table_keys = {
    #             "work_references": "source_work_id",
    #             "authorships": "work_id",
    #             "work_institutions": "work_id",
    #             "work_topics": "work_id",
    #             "work_concepts": "work_id",
    #             "af_label_evidence": "work_id",
    #             "authorship_countries": "work_id",
    #             "first_author_country_credit": "work_id",
    #             "authorship_country_diagnostics": "work_id",
    #         }

    #         for _table_name, _link_key in _related_table_keys.items():
    #             _table_dir = Path("derived_tables") / _table_name
    #             _table_paths = sorted(_table_dir.glob("*.parquet"))
    #             _rows_before = _load_parquet_row_count(_table_paths) if _table_paths else 0

    #             _filtered_frames = []
    #             for _path in _table_paths:
    #                 _table_part = pd.read_parquet(_path)
    #                 if _table_part.empty:
    #                     continue
    #                 if _link_key in _table_part.columns:
    #                     _table_part = _table_part[_table_part[_link_key].astype(str).isin(_kept_work_ids)]
    #                 _filtered_frames.append(_table_part)

    #             _table_filtered = pd.concat(_filtered_frames, ignore_index=True) if _filtered_frames else pd.DataFrame()
    #             _output_path = _dedup_output_dir / f"{_table_name}.parquet"
    #             _table_filtered.to_parquet(_output_path, index=False)

    #             _dedup_summary_rows.append({
    #                 "table_name": _table_name,
    #                 "rows_before": _rows_before,
    #                 "rows_after": len(_table_filtered),
    #                 "output_path": str(_output_path),
    #             })

    #         _dedup_summary = pd.DataFrame(_dedup_summary_rows)

    # _dedup_summary
    return


@app.cell(hide_code=True)
def _():
    # _dedup_pre2019_output_dir = Path("derived_tables_dedup_pre2019")
    # _dedup_pre2019_output_dir.mkdir(parents=True, exist_ok=True)

    # def _load_parquet_row_count_pre2019(_paths):
    #     _row_count = 0
    #     for _path in _paths:
    #         _row_count += len(pd.read_parquet(_path))
    #     return _row_count

    # _works_source_paths_pre2019 = sorted(Path("derived_tables/works").glob("*.parquet"))
    # _works_rows_before_pre2019 = _load_parquet_row_count_pre2019(_works_source_paths_pre2019) if _works_source_paths_pre2019 else 0

    # if not _works_source_paths_pre2019:
    #     _dedup_pre2019_summary = pd.DataFrame(
    #         [{"table_name": "works", "rows_before": 0, "rows_after": 0, "output_path": str(_dedup_pre2019_output_dir / "works.parquet")}]
    #     )
    # else:
    #     _works_minimal_frames_pre2019 = []
    #     for _path_pre2019 in _works_source_paths_pre2019:
    #         _works_part_for_schema_pre2019 = pd.read_parquet(_path_pre2019)
    #         _available_columns_pre2019 = _works_part_for_schema_pre2019.columns.tolist()
    #         _needed_columns_pre2019 = [col for col in ["work_id", "publication_year", "publication_date", "cited_by_count"] if col in _available_columns_pre2019]
    #         _works_minimal_frames_pre2019.append(_works_part_for_schema_pre2019[_needed_columns_pre2019].copy())

    #     _works_all_minimal_pre2019 = pd.concat(_works_minimal_frames_pre2019, ignore_index=True) if _works_minimal_frames_pre2019 else pd.DataFrame()

    #     if _works_all_minimal_pre2019.empty:
    #         _dedup_pre2019_summary = pd.DataFrame(
    #             [{"table_name": "works", "rows_before": 0, "rows_after": 0, "output_path": str(_dedup_pre2019_output_dir / "works.parquet")}]
    #         )
    #     elif "work_id" not in _works_all_minimal_pre2019.columns:
    #         raise KeyError("work_id")
    #     else:
    #         _works_ranked_pre2019 = _works_all_minimal_pre2019.copy()
    #         if "publication_year" in _works_ranked_pre2019.columns:
    #             _works_ranked_pre2019 = _works_ranked_pre2019[_works_ranked_pre2019["publication_year"].fillna(9999) < 2019].copy()
    #         _works_ranked_pre2019["_publication_date_rank"] = pd.to_datetime(
    #             _works_ranked_pre2019["publication_date"], errors="coerce"
    #         ) if "publication_date" in _works_ranked_pre2019.columns else pd.NaT
    #         _works_ranked_pre2019["_cited_by_count_rank"] = _works_ranked_pre2019["cited_by_count"] if "cited_by_count" in _works_ranked_pre2019.columns else 0

    #         _works_dedup_keys_pre2019 = (
    #             _works_ranked_pre2019
    #             .sort_values(
    #                 by=["work_id", "_publication_date_rank", "_cited_by_count_rank"],
    #                 ascending=[True, False, False],
    #                 na_position="last",
    #             )
    #             .drop_duplicates(subset=["work_id"], keep="first")[["work_id"]]
    #             .reset_index(drop=True)
    #         )

    #         _kept_work_ids_pre2019 = set(_works_dedup_keys_pre2019["work_id"].dropna().astype(str))
    #         _works_dedup_frames_pre2019 = []

    #         for _path_pre2019 in _works_source_paths_pre2019:
    #             _works_part_pre2019 = pd.read_parquet(_path_pre2019)
    #             if "work_id" not in _works_part_pre2019.columns:
    #                 raise KeyError("work_id")
    #             _works_part_filtered_pre2019 = _works_part_pre2019[_works_part_pre2019["work_id"].astype(str).isin(_kept_work_ids_pre2019)]
    #             if "publication_year" in _works_part_filtered_pre2019.columns:
    #                 _works_part_filtered_pre2019 = _works_part_filtered_pre2019[_works_part_filtered_pre2019["publication_year"].fillna(9999) < 2019]
    #             if not _works_part_filtered_pre2019.empty:
    #                 _works_dedup_frames_pre2019.append(_works_part_filtered_pre2019)

    #         _works_dedup_all_pre2019 = pd.concat(_works_dedup_frames_pre2019, ignore_index=True) if _works_dedup_frames_pre2019 else pd.DataFrame()

    #         if _works_dedup_all_pre2019.empty:
    #             _works_dedup_pre2019 = _works_dedup_all_pre2019.copy()
    #         else:
    #             _works_dedup_ranked_pre2019 = _works_dedup_all_pre2019.copy()
    #             _works_dedup_ranked_pre2019["_publication_date_rank"] = pd.to_datetime(
    #                 _works_dedup_ranked_pre2019["publication_date"], errors="coerce"
    #             ) if "publication_date" in _works_dedup_ranked_pre2019.columns else pd.NaT
    #             _works_dedup_ranked_pre2019["_cited_by_count_rank"] = _works_dedup_ranked_pre2019["cited_by_count"] if "cited_by_count" in _works_dedup_ranked_pre2019.columns else 0

    #             _works_dedup_pre2019 = (
    #                 _works_dedup_ranked_pre2019
    #                 .sort_values(
    #                     by=["work_id", "_publication_date_rank", "_cited_by_count_rank"],
    #                     ascending=[True, False, False],
    #                     na_position="last",
    #                 )
    #                 .drop_duplicates(subset=["work_id"], keep="first")
    #                 .drop(columns=["_publication_date_rank", "_cited_by_count_rank"])
    #                 .reset_index(drop=True)
    #             )

    #         _works_output_path_pre2019 = _dedup_pre2019_output_dir / "works.parquet"
    #         _works_dedup_pre2019.to_parquet(_works_output_path_pre2019, index=False)

    #         _dedup_pre2019_summary_rows = [{
    #             "table_name": "works",
    #             "rows_before": _works_rows_before_pre2019,
    #             "rows_after": len(_works_dedup_pre2019),
    #             "output_path": str(_works_output_path_pre2019),
    #         }]

    #         _related_table_keys_pre2019 = {
    #             "work_references": "source_work_id",
    #             "authorships": "work_id",
    #             "work_institutions": "work_id",
    #             "work_topics": "work_id",
    #             "work_concepts": "work_id",
    #             "af_label_evidence": "work_id",
    #             "authorship_countries": "work_id",
    #             "first_author_country_credit": "work_id",
    #             "authorship_country_diagnostics": "work_id",
    #         }

    #         for _table_name_pre2019, _link_key_pre2019 in _related_table_keys_pre2019.items():
    #             _table_dir_pre2019 = Path("derived_tables") / _table_name_pre2019
    #             _table_paths_pre2019 = sorted(_table_dir_pre2019.glob("*.parquet"))
    #             _rows_before_pre2019 = _load_parquet_row_count_pre2019(_table_paths_pre2019) if _table_paths_pre2019 else 0

    #             _filtered_frames_pre2019 = []
    #             for _path_pre2019 in _table_paths_pre2019:
    #                 _table_part_pre2019 = pd.read_parquet(_path_pre2019)
    #                 if _table_part_pre2019.empty:
    #                     continue
    #                 if _link_key_pre2019 in _table_part_pre2019.columns:
    #                     _table_part_pre2019 = _table_part_pre2019[_table_part_pre2019[_link_key_pre2019].astype(str).isin(_kept_work_ids_pre2019)]
    #                 _filtered_frames_pre2019.append(_table_part_pre2019)

    #             _table_filtered_pre2019 = pd.concat(_filtered_frames_pre2019, ignore_index=True) if _filtered_frames_pre2019 else pd.DataFrame()
    #             _output_path_pre2019 = _dedup_pre2019_output_dir / f"{_table_name_pre2019}.parquet"
    #             _table_filtered_pre2019.to_parquet(_output_path_pre2019, index=False)

    #             _dedup_pre2019_summary_rows.append({
    #                 "table_name": _table_name_pre2019,
    #                 "rows_before": _rows_before_pre2019,
    #                 "rows_after": len(_table_filtered_pre2019),
    #                 "output_path": str(_output_path_pre2019),
    #             })

    #         _dedup_pre2019_summary = pd.DataFrame(_dedup_pre2019_summary_rows)

    # _dedup_pre2019_summary
    return


@app.cell(hide_code=True)
def _():
    # if Path("derived_tables_dedup_pre2019/works.parquet").exists():
    #     _dedup_pre2019_check = pd.DataFrame([
    #         {
    #             "path": "derived_tables_dedup_pre2019/works.parquet",
    #             "exists": True,
    #             "row_count": len(pd.read_parquet("derived_tables_dedup_pre2019/works.parquet")),
    #             "min_publication_year": pd.read_parquet("derived_tables_dedup_pre2019/works.parquet")["publication_year"].min() if "publication_year" in pd.read_parquet("derived_tables_dedup_pre2019/works.parquet").columns else None,
    #             "max_publication_year": pd.read_parquet("derived_tables_dedup_pre2019/works.parquet")["publication_year"].max() if "publication_year" in pd.read_parquet("derived_tables_dedup_pre2019/works.parquet").columns else None,
    #             "all_years_pre2019": (
    #                 pd.read_parquet("derived_tables_dedup_pre2019/works.parquet")["publication_year"].dropna() < 2019
    #             ).all() if "publication_year" in pd.read_parquet("derived_tables_dedup_pre2019/works.parquet").columns else None,
    #         }
    #     ])
    # else:
    #     _dedup_pre2019_check = pd.DataFrame([
    #         {
    #             "path": "derived_tables_dedup_pre2019/works.parquet",
    #             "exists": False,
    #             "row_count": None,
    #             "min_publication_year": None,
    #             "max_publication_year": None,
    #             "all_years_pre2019": None,
    #         }
    #     ])

    # _dedup_pre2019_check
    return


@app.cell(hide_code=True)
def _(Path, mo):
    if Path("derived_tables_dedup/works.parquet").exists():
        _dedup_works_count = mo.sql(
            f"""
            SELECT COUNT(*) AS dedup_works_count
            FROM read_parquet('derived_tables_dedup/works.parquet')
            """
        )
    else:
        mo.md("""The deduplicated file `derived_tables_dedup/works.parquet` is not yet available. Please run the deduplication cell first.""")
    return


@app.cell(hide_code=True)
def _(mo):
    _new_author_country_tables_probe = mo.sql(
        f"""
        -- WITH authorship_countries AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/authorship_countries/*.parquet')
        -- ),
        -- first_author_country_credit AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/first_author_country_credit/*.parquet')
        -- ),
        -- authorship_country_diagnostics AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/authorship_country_diagnostics/*.parquet')
        -- )
        -- SELECT
        --     'authorship_countries' AS table_name,
        --     COUNT(*) AS row_count,
        --     COUNT(DISTINCT work_id) AS distinct_work_id_count,
        --     COUNT(DISTINCT author_id) AS distinct_author_id_count,
        --     SUM(CASE WHEN is_first_author THEN 1 ELSE 0 END) AS first_author_rows,
        --     SUM(CASE WHEN country_code IS NOT NULL AND TRIM(country_code) <> '' THEN 1 ELSE 0 END) AS non_null_country_rows,
        --     COUNT(DISTINCT country_source) AS distinct_country_sources,
        --     MIN(country_source) AS sample_source_1,
        --     MAX(country_source) AS sample_source_2
        -- FROM authorship_countries

        -- UNION ALL

        -- SELECT
        --     'first_author_country_credit' AS table_name,
        --     COUNT(*) AS row_count,
        --     COUNT(DISTINCT work_id) AS distinct_work_id_count,
        --     CAST(NULL AS BIGINT) AS distinct_author_id_count,
        --     CAST(NULL AS BIGINT) AS first_author_rows,
        --     SUM(CASE WHEN country_code IS NOT NULL AND TRIM(country_code) <> '' THEN 1 ELSE 0 END) AS non_null_country_rows,
        --     COUNT(DISTINCT first_author_country_source) AS distinct_country_sources,
        --     MIN(first_author_country_source) AS sample_source_1,
        --     MAX(first_author_country_source) AS sample_source_2
        -- FROM first_author_country_credit

        -- UNION ALL

        -- SELECT
        --     'authorship_country_diagnostics' AS table_name,
        --     COUNT(*) AS row_count,
        --     COUNT(DISTINCT work_id) AS distinct_work_id_count,
        --     CAST(NULL AS BIGINT) AS distinct_author_id_count,
        --     SUM(CASE WHEN has_first_author THEN 1 ELSE 0 END) AS first_author_rows,
        --     SUM(CASE WHEN NOT first_author_country_missing THEN 1 ELSE 0 END) AS non_null_country_rows,
        --     COUNT(DISTINCT first_author_country_count) AS distinct_country_sources,
        --     CAST(MIN(first_author_country_count) AS VARCHAR) AS sample_source_1,
        --     CAST(MAX(first_author_country_count) AS VARCHAR) AS sample_source_2
        -- FROM authorship_country_diagnostics
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _first_author_diagnostics_probe = mo.sql(
        f"""
        -- WITH base AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/authorship_country_diagnostics/*.parquet')
        -- )
        -- SELECT
        --     COUNT(*) AS total_works,
        --     SUM(CASE WHEN has_first_author THEN 1 ELSE 0 END) AS works_with_first_author,
        --     SUM(CASE WHEN NOT has_first_author THEN 1 ELSE 0 END) AS works_without_first_author,
        --     SUM(CASE WHEN first_author_country_missing THEN 1 ELSE 0 END) AS works_with_missing_first_author_country,
        --     SUM(CASE WHEN first_author_country_count = 1 THEN 1 ELSE 0 END) AS works_with_single_first_author_country,
        --     SUM(CASE WHEN first_author_country_count >= 2 THEN 1 ELSE 0 END) AS works_with_multi_first_author_country,
        --     ROUND(100.0 * SUM(CASE WHEN has_first_author THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_with_first_author,
        --     ROUND(100.0 * SUM(CASE WHEN first_author_country_missing THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_missing_first_author_country,
        --     ROUND(100.0 * SUM(CASE WHEN first_author_country_count >= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_multi_first_author_country
        -- FROM base
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _first_author_credit_probe = mo.sql(
        f"""
        -- WITH base AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/first_author_country_credit/*.parquet')
        -- )
        -- SELECT
        --     country_code,
        --     first_author_fraction,
        --     n_first_author_countries,
        --     first_author_country_source,
        --     COUNT(*) AS row_count,
        --     COUNT(DISTINCT work_id) AS distinct_work_id_count
        -- FROM base
        -- GROUP BY
        --     country_code,
        --     first_author_fraction,
        --     n_first_author_countries,
        --     first_author_country_source
        -- ORDER BY row_count DESC, distinct_work_id_count DESC, country_code
        -- LIMIT 20
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _first_author_missing_by_domain_year = mo.sql(
        f"""
        -- WITH diagnostics AS (
        --     SELECT *
        --     FROM read_parquet('derived_tables/authorship_country_diagnostics/*.parquet')
        -- ),
        -- works AS (
        --     SELECT
        --         work_id,
        --         domain_label,
        --         publication_year
        --     FROM read_parquet('derived_tables/works/*.parquet')
        -- )
        -- SELECT
        --     COALESCE(w.domain_label, 'unknown') AS domain_label,
        --     w.publication_year,
        --     COUNT(*) AS total_works,
        --     SUM(CASE WHEN d.has_first_author THEN 1 ELSE 0 END) AS works_with_first_author,
        --     SUM(CASE WHEN d.first_author_country_missing THEN 1 ELSE 0 END) AS works_with_missing_first_author_country,
        --     SUM(CASE WHEN d.first_author_country_count >= 2 THEN 1 ELSE 0 END) AS works_with_multi_first_author_country,
        --     ROUND(
        --         100.0 * SUM(CASE WHEN d.first_author_country_missing THEN 1 ELSE 0 END)
        --         / NULLIF(SUM(CASE WHEN d.has_first_author THEN 1 ELSE 0 END), 0),
        --         2
        --     ) AS pct_missing_among_first_author_works,
        --     ROUND(
        --         100.0 * SUM(CASE WHEN d.first_author_country_count >= 2 THEN 1 ELSE 0 END)
        --         / NULLIF(COUNT(*), 0),
        --         2
        --     ) AS pct_multi_first_author_country
        -- FROM diagnostics AS d
        -- LEFT JOIN works AS w
        --     ON d.work_id = w.work_id
        -- GROUP BY 1, 2
        -- HAVING COUNT(*) >= 100
        -- ORDER BY pct_missing_among_first_author_works DESC NULLS LAST, total_works DESC, domain_label, publication_year
        -- LIMIT 100
        """
    )
    return


@app.cell(hide_code=True)
def _():
    # import json
    # from urllib.parse import urlencode
    # from urllib.request import urlopen

    # _rnd_metadata_url = "https://api.worldbank.org/v2/sources/2/series/gb.xpd.rsdv.gd.zs/metadata?format=json"

    # with urlopen(_rnd_metadata_url) as _rnd_metadata_response:
    #     _rnd_metadata_payload = json.load(_rnd_metadata_response)

    # _rnd_metadata_rows = []
    # if isinstance(_rnd_metadata_payload, list) and len(_rnd_metadata_payload) > 1:
    #     _rnd_metadata_entries = _rnd_metadata_payload[1]
    #     if isinstance(_rnd_metadata_entries, list):
    #         for _entry_rnd_meta in _rnd_metadata_entries:
    #             _metatype = (
    #                 _entry_rnd_meta.get("metatype", {})
    #                 if isinstance(_entry_rnd_meta, dict)
    #                 else {}
    #             )
    #             _meta_value = (
    #                 _entry_rnd_meta.get("value")
    #                 if isinstance(_entry_rnd_meta, dict)
    #                 else None
    #             )
    #             _rnd_metadata_rows.append(
    #                 {
    #                     "id": _metatype.get("id"),
    #                     "name": _metatype.get("value"),
    #                     "value": _meta_value,
    #                 }
    #             )

    # rnd_metadata_df = pd.DataFrame(_rnd_metadata_rows)

    # _rnd_base_url = (
    #     "https://api.worldbank.org/v2/countries/all/indicators/gb.xpd.rsdv.gd.zs"
    # )
    # _rnd_all_rows = []
    # _rnd_page = 1
    # _rnd_pages = 1

    # while _rnd_page <= _rnd_pages:
    #     _rnd_query = urlencode(
    #         {
    #             "date": "1996:2021",
    #             "format": "json",
    #             "per_page": 20000,
    #             "page": _rnd_page,
    #         }
    #     )
    #     _rnd_page_url = f"{_rnd_base_url}?{_rnd_query}"
    #     with urlopen(_rnd_page_url) as _rnd_data_response:
    #         _rnd_data_payload = json.load(_rnd_data_response)

    #     if isinstance(_rnd_data_payload, list) and len(_rnd_data_payload) > 1:
    #         _rnd_meta_block = (
    #             _rnd_data_payload[0] if isinstance(_rnd_data_payload[0], dict) else {}
    #         )
    #         _rnd_pages = int(_rnd_meta_block.get("pages", 1) or 1)
    #         _rnd_data_entries = (
    #             _rnd_data_payload[1] if isinstance(_rnd_data_payload[1], list) else []
    #         )
    #         for _entry_rnd in _rnd_data_entries:
    #             _country_block = (
    #                 _entry_rnd.get("country", {})
    #                 if isinstance(_entry_rnd, dict)
    #                 else {}
    #             )
    #             _countryiso3 = (
    #                 _entry_rnd.get("countryiso3code")
    #                 if isinstance(_entry_rnd, dict)
    #                 else None
    #             )
    #             _value = (
    #                 _entry_rnd.get("value") if isinstance(_entry_rnd, dict) else None
    #             )
    #             _date = _entry_rnd.get("date") if isinstance(_entry_rnd, dict) else None
    #             if (
    #                 _countryiso3
    #                 and str(_countryiso3).strip() != ""
    #                 and _value is not None
    #             ):
    #                 _rnd_all_rows.append(
    #                     {
    #                         "country_name_wb": _country_block.get("value"),
    #                         "country_iso3": str(_countryiso3).strip().upper(),
    #                         "year": int(_date),
    #                         "rnd_gdp_pct": float(_value),
    #                     }
    #                 )
    #     _rnd_page += 1

    # rnd_country_year_df = pd.DataFrame(_rnd_all_rows)

    # _iso2_candidates = [
    #     _col_iso2
    #     for _col_iso2 in ["WB_A2", "ISO_A2_EH", "ISO_A2"]
    #     if _col_iso2 in world_map.columns
    # ]
    # _iso3_candidates = [
    #     _col_iso3
    #     for _col_iso3 in ["WB_A3", "ADM0_A3", "ISO_A3", "SOV_A3"]
    #     if _col_iso3 in world_map.columns
    # ]

    # _country_code_bridge = world_map.copy()
    # if _iso2_candidates:
    #     _country_code_bridge["country_code"] = (
    #         _country_code_bridge[_iso2_candidates[0]]
    #         .astype(str)
    #         .str.strip()
    #         .str.upper()
    #     )
    # elif "country_code" not in _country_code_bridge.columns:
    #     _country_code_bridge["country_code"] = None
    # if _iso3_candidates:
    #     _country_code_bridge["country_iso3"] = (
    #         _country_code_bridge[_iso3_candidates[0]]
    #         .astype(str)
    #         .str.strip()
    #         .str.upper()
    #     )
    # elif "country_iso3" not in _country_code_bridge.columns:
    #     _country_code_bridge["country_iso3"] = None
    # _country_code_bridge = _country_code_bridge[
    #     ["country_code", "country_iso3", "NAME"]
    # ].dropna(subset=["country_iso3"])
    # _country_code_bridge = _country_code_bridge.rename(
    #     columns={"NAME": "map_name"}
    # ).drop_duplicates(subset=["country_iso3"])

    # _rnd_iso_overrides = pd.DataFrame(
    #     {
    #         "country_iso3": ["KSV", "XKX"],
    #         "country_code": ["XK", "XK"],
    #     }
    # )

    # rnd_country_year_df = rnd_country_year_df.merge(
    #     _country_code_bridge,
    #     on="country_iso3",
    #     how="left",
    # )
    # rnd_country_year_df = rnd_country_year_df.merge(
    #     _rnd_iso_overrides,
    #     on="country_iso3",
    #     how="left",
    #     suffixes=("", "_override"),
    # )
    # rnd_country_year_df["country_code"] = rnd_country_year_df[
    #     "country_code_override"
    # ].fillna(rnd_country_year_df["country_code"])
    # rnd_country_year_df = rnd_country_year_df.drop(columns=["country_code_override"])

    # rnd_country_year_df["country_code"] = (
    #     rnd_country_year_df["country_code"].astype(str).str.strip().str.upper()
    # )
    # rnd_country_year_df = rnd_country_year_df[
    #     rnd_country_year_df["country_code"].notna()
    # ]
    # rnd_country_year_df = rnd_country_year_df[
    #     rnd_country_year_df["country_code"] != "NONE"
    # ]

    # rnd_pre_af2_df = (
    #     rnd_country_year_df[
    #         (rnd_country_year_df["year"] >= 2019)
    #         & (rnd_country_year_df["year"] <= 2021)
    #     ]
    #     .groupby(["country_code", "country_name_wb"], as_index=False)
    #     .agg(
    #         pre_af2_rnd_gdp_pct_mean=("rnd_gdp_pct", "mean"),
    #         rnd_years_available=("year", "nunique"),
    #         rnd_year_min=("year", "min"),
    #         rnd_year_max=("year", "max"),
    #     )
    # )

    # rnd_pre_af2_df = rnd_pre_af2_df.merge(
    #     country_income_group_lookup[["country_code", "income_group"]],
    #     on="country_code",
    #     how="left",
    # )

    # rnd_gap_explainer_df = country_gap_explainer_df.merge(
    #     rnd_pre_af2_df[
    #         [
    #             "country_code",
    #             "pre_af2_rnd_gdp_pct_mean",
    #             "rnd_years_available",
    #             "rnd_year_min",
    #             "rnd_year_max",
    #         ]
    #     ],
    #     on="country_code",
    #     how="left",
    # )
    # rnd_gap_explainer_df = rnd_gap_explainer_df.dropna(
    #     subset=[
    #         "pre_af2_rnd_gdp_pct_mean",
    #         "adoption_lag_months",
    #         "af_fractional_count",
    return


@app.cell(hide_code=True)
def _(duckdb):
    collab_work_base = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related IS NOT NULL
        ),
        authorship_counts AS (
            SELECT
                work_id,
                COUNT(*) AS team_size,
                SUM(CASE WHEN is_corresponding THEN 1 ELSE 0 END) AS n_corresponding_authors
            FROM read_parquet('derived_tables_dedup/authorships.parquet')
            WHERE work_id IS NOT NULL
            GROUP BY work_id
        ),
        institution_base AS (
            SELECT DISTINCT
                work_id,
                institution_id,
                country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
        ),
        institution_counts AS (
            SELECT
                work_id,
                COUNT(DISTINCT institution_id) AS n_institutions,
                COUNT(DISTINCT CASE WHEN country_code IS NOT NULL AND TRIM(country_code) <> '' THEN country_code END) AS n_countries
            FROM institution_base
            GROUP BY work_id
        )
        SELECT
            w.work_id,
            w.is_alphafold_related,
            a.team_size,
            a.n_corresponding_authors,
            i.n_institutions,
            i.n_countries,
            CASE WHEN i.n_countries >= 2 THEN TRUE ELSE FALSE END AS is_international_collab,
            CASE WHEN i.n_institutions >= 2 THEN TRUE ELSE FALSE END AS is_multi_institutional,
            CASE WHEN a.team_size >= 10 THEN TRUE ELSE FALSE END AS is_large_team
        FROM works_clean AS w
        LEFT JOIN authorship_counts AS a
            ON w.work_id = a.work_id
        LEFT JOIN institution_counts AS i
            ON w.work_id = i.work_id
        """
    ).df()

    collab_group_summary = collab_work_base.groupby(
        "is_alphafold_related", as_index=False
    ).agg(
        mean_team_size=("team_size", "mean"),
        median_team_size=("team_size", "median"),
        international_collab_share=("is_international_collab", "mean"),
        multi_institution_share=("is_multi_institutional", "mean"),
        large_team_share=("is_large_team", "mean"),
        mean_n_institutions=("n_institutions", "mean"),
        mean_n_countries=("n_countries", "mean"),
    )
    collab_group_summary["group"] = collab_group_summary["is_alphafold_related"].map(
        {True: "AF", False: "non-AF"}
    )

    country_collab_edges = duckdb.sql(
        """
        WITH work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = TRUE
        )
        SELECT
            a.country_code AS source_country,
            b.country_code AS target_country,
            COUNT(DISTINCT a.work_id) AS n_shared_works
        FROM work_country AS a
        INNER JOIN work_country AS b
            ON a.work_id = b.work_id
           AND a.country_code < b.country_code
        INNER JOIN af_works AS w
            ON a.work_id = w.work_id
        GROUP BY a.country_code, b.country_code
        HAVING COUNT(DISTINCT a.work_id) >= 5
        ORDER BY n_shared_works DESC
        """
    ).df()

    # collab_work_base
    return collab_work_base, country_collab_edges


@app.cell(hide_code=True)
def _(Path, duckdb, gpd, pd):
    country_af_output = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                TRY_CAST(publication_date AS DATE) AS publication_date,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND publication_date IS NOT NULL
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                country_code,
                is_global_south
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT
                work_id,
                COUNT(*) AS n_countries
            FROM work_country_dedup
            GROUP BY work_id
        ),
        country_work_base AS (
            SELECT
                w.work_id,
                w.publication_date,
                w.is_alphafold_related,
                c.country_code,
                c.is_global_south,
                cc.n_countries,
                1.0 / cc.n_countries AS fractional_weight
            FROM works_clean AS w
            INNER JOIN work_country_dedup AS c
                ON w.work_id = c.work_id
            INNER JOIN work_country_counts AS cc
                ON w.work_id = cc.work_id
            WHERE cc.n_countries > 0
        )
        SELECT
            country_code,
            SUM(fractional_weight) AS af_fractional_count
        FROM country_work_base
        WHERE is_alphafold_related = TRUE
        GROUP BY country_code
        ORDER BY af_fractional_count DESC
        """
    ).df()

    country_name_overrides = pd.DataFrame(
        {
            "country_code": [
                "US",
                "GB",
                "CN",
                "TW",
                "KR",
                "RU",
                "TR",
                "IR",
                "VN",
                "SY",
                "CZ",
                "AE",
                "HK",
                "KZ",
            ],
            "country_name": [
                "United States of America",
                "United Kingdom",
                "China",
                "Taiwan",
                "South Korea",
                "Russia",
                "Turkey",
                "Iran",
                "Vietnam",
                "Syria",
                "Czechia",
                "United Arab Emirates",
                "Hong Kong S.A.R.",
                "Kazakhstan",
            ],
        }
    )

    _world_map_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    _world_map_local_path = Path("outputs/cache/ne_110m_admin_0_countries.geojson")
    _world_map_local_path.parent.mkdir(parents=True, exist_ok=True)

    if not _world_map_local_path.exists():
        _urllib_request = __import__("urllib.request", fromlist=["urlretrieve"])
        _urllib_request.urlretrieve(_world_map_url, _world_map_local_path)

    world_map = gpd.read_file(_world_map_local_path)
    country_name_lookup = world_map[["ISO_A2", "NAME"]].rename(
        columns={"ISO_A2": "country_code", "NAME": "country_name_map"}
    )
    country_af_output_named = country_af_output.merge(
        country_name_lookup,
        on="country_code",
        how="left",
    )
    country_af_output_named = country_af_output_named.merge(
        country_name_overrides,
        on="country_code",
        how="left",
    )
    country_af_output_named["country_name"] = country_af_output_named[
        "country_name"
    ].fillna(country_af_output_named["country_name_map"])
    country_af_output_named["country_name"] = country_af_output_named[
        "country_name"
    ].fillna(country_af_output_named["country_code"])

    country_af_map = world_map.merge(
        country_af_output_named[
            ["country_code", "country_name", "af_fractional_count"]
        ],
        left_on="NAME",
        right_on="country_name",
        how="left",
    )
    map_match_summary = pd.DataFrame(
        {
            "metric": [
                "AF countries in source table",
                "AF countries matched to map",
                "AF countries unmatched to map",
                "Fraction matched",
            ],
            "value": [
                int(country_af_output["country_code"].nunique()),
                int(country_af_output_named["country_name_map"].notna().sum()),
                int(country_af_output_named["country_name_map"].isna().sum()),
                round(country_af_output_named["country_name_map"].notna().mean(), 4),
            ],
        }
    )

    top_country_af_output = (
        country_af_map[["country_code", "country_name", "af_fractional_count"]]
        .dropna(subset=["af_fractional_count"])
        .sort_values("af_fractional_count", ascending=False)
        .head(20)
    )

    top_country_af_output
    return (
        country_af_output,
        country_af_output_named,
        country_name_lookup,
        country_name_overrides,
        world_map,
    )


@app.cell(hide_code=True)
def _(world_map):
    country_income_group_lookup = world_map[
        ["ISO_A2", "NAME", "INCOME_GRP"]
    ].copy()
    country_income_group_lookup = country_income_group_lookup.rename(
        columns={
            "ISO_A2": "country_code",
            "NAME": "country_name_map",
            "INCOME_GRP": "income_group_raw",
        }
    )
    country_income_group_lookup["country_code"] = (
        country_income_group_lookup["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    country_income_group_lookup["income_group"] = country_income_group_lookup[
        "income_group_raw"
    ].replace(
        {
            "1. High income: OECD": "High income",
            "2. High income: nonOECD": "High income",
            "3. Upper middle income": "Upper-middle income",
            "4. Lower middle income": "Lower-middle income",
            "5. Low income": "Low income",
        }
    )
    country_income_group_lookup = country_income_group_lookup.dropna(
        subset=["country_code", "income_group"]
    )
    country_income_group_lookup = country_income_group_lookup.drop_duplicates(
        subset=["country_code"]
    )

    country_income_group_lookup
    return (country_income_group_lookup,)


@app.cell(hide_code=True)
def _(country_collab_edges, country_income_group_lookup, nx, pd):
    figure_3b_graph = nx.Graph()
    for _row_3b in country_collab_edges.itertuples(index=False):
        figure_3b_graph.add_edge(
            _row_3b.source_country,
            _row_3b.target_country,
            weight=_row_3b.n_shared_works,
        )

    _figure_3b_weighted_degree = dict(figure_3b_graph.degree(weight="weight"))
    _figure_3b_betweenness = nx.betweenness_centrality(
        figure_3b_graph, weight="weight", normalized=True
    )
    _figure_3b_eigenvector = nx.eigenvector_centrality_numpy(
        figure_3b_graph, weight="weight"
    )

    figure_3b_centrality = pd.DataFrame(
        {
            "country_code": list(figure_3b_graph.nodes()),
            "weighted_degree": [
                _figure_3b_weighted_degree[_country_3b]
                for _country_3b in figure_3b_graph.nodes()
            ],
            "betweenness_centrality": [
                _figure_3b_betweenness[_country_3b]
                for _country_3b in figure_3b_graph.nodes()
            ],
            "eigenvector_centrality": [
                _figure_3b_eigenvector[_country_3b]
                for _country_3b in figure_3b_graph.nodes()
            ],
        }
    )
    figure_3b_centrality["country_code"] = (
        figure_3b_centrality["country_code"].astype(str).str.strip().str.upper()
    )
    figure_3b_centrality = figure_3b_centrality.merge(
        country_income_group_lookup[["country_code", "income_group"]],
        on="country_code",
        how="left",
    )
    figure_3b_centrality = figure_3b_centrality.dropna(subset=["income_group"])

    figure_3b_centrality
    return (figure_3b_centrality,)


@app.cell(hide_code=True)
def _(country_name_lookup, country_name_overrides, duckdb):
    country_af_nonaf_compare = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related IS NOT NULL
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT
                work_id,
                COUNT(*) AS n_countries
            FROM work_country_dedup
            GROUP BY work_id
        ),
        country_work_base AS (
            SELECT
                w.work_id,
                w.is_alphafold_related,
                c.country_code,
                1.0 / cc.n_countries AS fractional_weight
            FROM works_clean AS w
            INNER JOIN work_country_dedup AS c
                ON w.work_id = c.work_id
            INNER JOIN work_country_counts AS cc
                ON w.work_id = cc.work_id
            WHERE cc.n_countries > 0
        ),
        country_counts AS (
            SELECT
                country_code,
                SUM(CASE WHEN is_alphafold_related THEN fractional_weight ELSE 0 END) AS af_fractional_count,
                SUM(CASE WHEN NOT is_alphafold_related THEN fractional_weight ELSE 0 END) AS non_af_fractional_count
            FROM country_work_base
            GROUP BY country_code
        )
        SELECT
            country_code,
            af_fractional_count,
            non_af_fractional_count
        FROM country_counts
        """
    ).df()

    country_af_nonaf_compare = country_af_nonaf_compare.merge(
        country_name_lookup,
        on="country_code",
        how="left",
    )
    country_af_nonaf_compare = country_af_nonaf_compare.merge(
        country_name_overrides,
        on="country_code",
        how="left",
    )
    country_af_nonaf_compare["country_name"] = country_af_nonaf_compare[
        "country_name"
    ].fillna(country_af_nonaf_compare["country_name_map"])
    country_af_nonaf_compare["country_name"] = country_af_nonaf_compare[
        "country_name"
    ].fillna(country_af_nonaf_compare["country_code"])
    country_af_nonaf_compare["af_share"] = (
        country_af_nonaf_compare["af_fractional_count"]
        / country_af_nonaf_compare["af_fractional_count"].sum()
    )
    country_af_nonaf_compare["non_af_share"] = (
        country_af_nonaf_compare["non_af_fractional_count"]
        / country_af_nonaf_compare["non_af_fractional_count"].sum()
    )
    country_af_nonaf_compare
    return (country_af_nonaf_compare,)


@app.cell(hide_code=True)
def _(duckdb):
    _f3e_authorship_schema_probe = duckdb.sql(
        """
        SELECT *
        FROM read_parquet('derived_tables_dedup/authorships.parquet')
        LIMIT 5
        """
    ).df()

    _f3e_authorship_schema_probe
    return


@app.cell(hide_code=True)
def _(duckdb):
    figure_3e_country_first_author_base = duckdb.sql(
        """
        WITH af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT
                work_id,
                COUNT(DISTINCT country_code) AS n_countries
            FROM work_country
            GROUP BY work_id
        ),
        country_af_output AS (
            SELECT
                c.country_code,
                SUM(1.0 / cc.n_countries) AS af_output_fractional
            FROM af_works AS w
            INNER JOIN work_country AS c
                ON w.work_id = c.work_id
            INNER JOIN work_country_counts AS cc
                ON w.work_id = cc.work_id
            WHERE cc.n_countries > 0
            GROUP BY c.country_code
        ),
        country_first_author_output AS (
            SELECT
                UPPER(TRIM(country_code)) AS country_code,
                SUM(first_author_fraction) AS first_author_output_fractional
            FROM read_parquet('derived_tables_dedup/first_author_country_credit.parquet')
            WHERE work_id IN (SELECT work_id FROM af_works)
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
            GROUP BY UPPER(TRIM(country_code))
        )
        SELECT
            o.country_code,
            o.af_output_fractional,
            COALESCE(f.first_author_output_fractional, 0.0) AS first_author_output_fractional
        FROM country_af_output AS o
        LEFT JOIN country_first_author_output AS f
            ON o.country_code = f.country_code
        """
    ).df()

    figure_3e_country_first_author_base["country_code"] = (
        figure_3e_country_first_author_base["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    figure_3e_country_first_author_base["first_author_share_pct"] = (
        100
        * figure_3e_country_first_author_base["first_author_output_fractional"]
        / figure_3e_country_first_author_base["af_output_fractional"]
    )
    figure_3e_country_first_author_base
    return (figure_3e_country_first_author_base,)


@app.cell(hide_code=True)
def _(duckdb, pd, world_map):
    # Composite national R&D strength index using a pre-AlphaFold window.
    #
    # X1: Mean R&D expenditure as % of GDP over 2015-2018 (World Bank).
    # X2: Log pre-AlphaFold publication output over 2015-2018, using the
    #     deduplicated pre-2019 country-publication tables.
    #
    # Each component is z-score standardized across countries, and the final index is
    # their equal-weight average.
    _requests = __import__("requests")
    _log10 = __import__("math").log10

    _rnd_response = _requests.get(
        "https://api.worldbank.org/v2/countries/all/indicators/GB.XPD.RSDV.GD.ZS",
        params={"date": "2015:2018", "format": "json", "per_page": 20000},
        timeout=60,
    )
    _rnd_response.raise_for_status()
    _rnd_payload = _rnd_response.json()
    _rnd_rows = (
        _rnd_payload[1]
        if isinstance(_rnd_payload, list) and len(_rnd_payload) > 1
        else []
    )

    _rnd_country_year_rows = []
    for _row in _rnd_rows:
        _value = _row.get("value")
        _country_iso3 = (_row.get("countryiso3code") or "").strip().upper()
        _year = _row.get("date")
        if _value is None or not _country_iso3:
            continue
        try:
            _rnd_country_year_rows.append(
                {
                    "country_iso3": _country_iso3,
                    "year": int(_year),
                    "rnd_gdp_pct": float(_value),
                }
            )
        except (TypeError, ValueError):
            continue

    rnd_country_year_df = pd.DataFrame(_rnd_country_year_rows)

    _country_code_bridge = world_map[["ISO_A2", "ISO_A3", "NAME"]].copy()
    _country_code_bridge = _country_code_bridge.rename(
        columns={
            "ISO_A2": "country_code",
            "ISO_A3": "country_iso3",
            "NAME": "country_name_map",
        }
    )
    _country_code_bridge["country_code"] = (
        _country_code_bridge["country_code"].astype(str).str.strip().str.upper()
    )
    _country_code_bridge["country_iso3"] = (
        _country_code_bridge["country_iso3"].astype(str).str.strip().str.upper()
    )
    _country_code_bridge = _country_code_bridge.dropna(
        subset=["country_iso3"]
    ).drop_duplicates(subset=["country_iso3"])

    rnd_country_year_df = rnd_country_year_df.merge(
        _country_code_bridge,
        on="country_iso3",
        how="left",
    )
    rnd_country_year_df = rnd_country_year_df.dropna(subset=["country_code"])
    rnd_country_year_df = rnd_country_year_df[
        rnd_country_year_df["country_code"] != "-99"
    ]

    rnd_pre_2015_2018 = rnd_country_year_df.groupby(
        ["country_code", "country_name_map"], as_index=False
    ).agg(
        rnd_gdp_pct_mean_2015_2018=("rnd_gdp_pct", "mean"),
        rnd_years_available=("year", "nunique"),
    )

    pre_af_publication_output = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                publication_year,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup_pre2019/works.parquet')
            WHERE work_id IS NOT NULL
              AND publication_year BETWEEN 2015 AND 2018
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup_pre2019/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT
                work_id,
                COUNT(*) AS n_countries
            FROM work_country_dedup
            GROUP BY work_id
        ),
        country_work_base AS (
            SELECT
                w.work_id,
                c.country_code,
                w.is_alphafold_related,
                1.0 / cc.n_countries AS fractional_weight
            FROM works_clean AS w
            INNER JOIN work_country_dedup AS c
                ON w.work_id = c.work_id
            INNER JOIN work_country_counts AS cc
                ON w.work_id = cc.work_id
            WHERE cc.n_countries > 0
        )
        SELECT
            country_code,
            SUM(
                CASE
                    WHEN is_alphafold_related = FALSE OR is_alphafold_related IS NULL
                        THEN fractional_weight
                    ELSE 0
                END
            ) AS pre_af_non_af_fractional_output_2015_2018,
            SUM(fractional_weight) AS pre_af_life_science_fractional_output_2015_2018
        FROM country_work_base
        GROUP BY country_code
        """
    ).df()

    country_rd_strength_lookup = rnd_pre_2015_2018.merge(
        pre_af_publication_output,
        on="country_code",
        how="inner",
    )
    country_rd_strength_lookup["log_pre_af_non_af_fractional_output_2015_2018"] = (
        country_rd_strength_lookup["pre_af_non_af_fractional_output_2015_2018"]
        .fillna(0)
        .map(lambda _value: _log10(_value + 1))
    )

    for _col in [
        "rnd_gdp_pct_mean_2015_2018",
        "log_pre_af_non_af_fractional_output_2015_2018",
    ]:
        _mu = country_rd_strength_lookup[_col].mean()
        _sigma = country_rd_strength_lookup[_col].std(ddof=0)
        country_rd_strength_lookup[f"z_{_col}"] = (
            (country_rd_strength_lookup[_col] - _mu) / _sigma if _sigma else 0.0
        )

    country_rd_strength_lookup["rd_strength_index"] = (
        country_rd_strength_lookup["z_rnd_gdp_pct_mean_2015_2018"]
        + country_rd_strength_lookup[
            "z_log_pre_af_non_af_fractional_output_2015_2018"
        ]
    ) / 2
    country_rd_strength_lookup["rd_strength_tertile"] = pd.qcut(
        country_rd_strength_lookup["rd_strength_index"],
        q=3,
        labels=["Low R&D strength", "Middle R&D strength", "High R&D strength"],
        duplicates="drop",
    )
    country_rd_strength_lookup = country_rd_strength_lookup.sort_values(
        "rd_strength_index", ascending=False
    ).reset_index(drop=True)

    country_rd_strength_diagnostic = pd.DataFrame(
        {
            "metric": [
                "Countries with R&D strength index",
                "Countries with World Bank R&D data",
                "Countries with pre-2019 dedup publication data",
                "R&D window start",
                "R&D window end",
            ],
            "value": [
                int(country_rd_strength_lookup["country_code"].nunique()),
                int(rnd_pre_2015_2018["country_code"].nunique()),
                int(pre_af_publication_output["country_code"].nunique()),
                2015,
                2018,
            ],
        }
    )

    country_rd_strength_diagnostic
    return (country_rd_strength_lookup,)


@app.cell(hide_code=True)
def _(country_af_output, country_rd_strength_lookup, duckdb, pd):
    figure_2_rd_strength_adoption_df = duckdb.sql(
        """
        WITH country_first_af_adoption AS (
            SELECT
                UPPER(TRIM(country_code)) AS country_code,
                MIN(TRY_CAST(publication_date AS DATE)) AS first_af_date
            FROM read_parquet('derived_tables_dedup/works.parquet') AS w
            INNER JOIN read_parquet('derived_tables_dedup/work_institutions.parquet') AS i
                ON w.work_id = i.work_id
            WHERE w.work_id IS NOT NULL
              AND w.is_alphafold_related = TRUE
              AND publication_date IS NOT NULL
              AND i.country_code IS NOT NULL
              AND TRIM(i.country_code) <> ''
            GROUP BY UPPER(TRIM(country_code))
        )
        SELECT
            country_code,
            first_af_date
        FROM country_first_af_adoption
        """
    ).df()

    figure_2_rd_strength_adoption_df["first_af_date"] = pd.to_datetime(
        figure_2_rd_strength_adoption_df["first_af_date"]
    )
    figure_2_rd_strength_adoption_df["adoption_lag_months"] = (
        figure_2_rd_strength_adoption_df["first_af_date"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        figure_2_rd_strength_adoption_df["first_af_date"].dt.month
        - pd.Timestamp("2018-12-02").month
    )
    figure_2_rd_strength_adoption_df = figure_2_rd_strength_adoption_df.merge(
        country_rd_strength_lookup[
            [
                "country_code",
                "country_name_map",
                "rnd_gdp_pct_mean_2015_2018",
                "pre_af_non_af_fractional_output_2015_2018",
                "log_pre_af_non_af_fractional_output_2015_2018",
                "rd_strength_index",
                "rd_strength_tertile",
            ]
        ],
        on="country_code",
        how="inner",
    )
    figure_2_rd_strength_adoption_df = figure_2_rd_strength_adoption_df.merge(
        country_af_output,
        on="country_code",
        how="left",
    )
    figure_2_rd_strength_adoption_df["af_fractional_count"] = (
        figure_2_rd_strength_adoption_df["af_fractional_count"].fillna(0)
    )
    figure_2_rd_strength_adoption_df["bubble_size"] = (
        50
        + 20
        * figure_2_rd_strength_adoption_df["af_fractional_count"]
        .fillna(0)
        .map(lambda _v: max(_v, 0) ** 0.5)
    )
    figure_2_rd_strength_adoption_df = (
        figure_2_rd_strength_adoption_df.sort_values(
            ["rd_strength_index", "af_fractional_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
    )

    figure_2_rd_strength_adoption_df
    return (figure_2_rd_strength_adoption_df,)


@app.cell(hide_code=True)
def _(
    country_af_output,
    country_rd_strength_lookup,
    figure_3b_centrality,
    figure_3e_country_first_author_base,
    math,
):
    figure_3_rd_strength_leadership_df = figure_3b_centrality.merge(
        figure_3e_country_first_author_base[
            [
                "country_code",
                "af_output_fractional",
                "first_author_output_fractional",
                "first_author_share_pct",
            ]
        ],
        on="country_code",
        how="inner",
    )
    figure_3_rd_strength_leadership_df = figure_3_rd_strength_leadership_df.merge(
        country_rd_strength_lookup[
            [
                "country_code",
                "rd_strength_index",
                "rd_strength_tertile",
                "rnd_gdp_pct_mean_2015_2018",
                "log_pre_af_non_af_fractional_output_2015_2018",
            ]
        ],
        on="country_code",
        how="inner",
    )
    figure_3_rd_strength_leadership_df = figure_3_rd_strength_leadership_df.merge(
        country_af_output,
        on="country_code",
        how="left",
    )
    figure_3_rd_strength_leadership_df = figure_3_rd_strength_leadership_df.dropna(
        subset=[
            "eigenvector_centrality",
            "first_author_share_pct",
            "rd_strength_index",
            "af_fractional_count",
        ]
    ).copy()
    figure_3_rd_strength_leadership_df = figure_3_rd_strength_leadership_df[
        figure_3_rd_strength_leadership_df["af_fractional_count"] >= 10
    ].copy()
    _figure_3_rd_floor = max(
        float(figure_3_rd_strength_leadership_df["eigenvector_centrality"].min())
        * 0.5,
        1e-6,
    )
    figure_3_rd_strength_leadership_df["log10_eigenvector_centrality"] = (
        figure_3_rd_strength_leadership_df["eigenvector_centrality"]
        .clip(lower=_figure_3_rd_floor)
        .map(lambda _v: math.log10(_v))
    )
    figure_3_rd_strength_leadership_df["bubble_size"] = (
        60
        + 18
        * figure_3_rd_strength_leadership_df["af_fractional_count"].map(
            lambda _v: max(_v, 0) ** 0.5
        )
    )
    figure_3_rd_strength_leadership_df = (
        figure_3_rd_strength_leadership_df.sort_values(
            ["rd_strength_index", "af_fractional_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
    )

    figure_3_rd_strength_leadership_df
    return


@app.cell(hide_code=True)
def _(duckdb, pd):
    event_calendar = pd.DataFrame(
        {
            "event_name": [
                "AlphaFold 2",
                "AlphaFold-Multimer",
                "AlphaFold DB",
                "AlphaFold 3",
            ],
            "event_date": pd.to_datetime(
                [
                    "2021-07-01",
                    "2021-10-01",
                    "2022-07-01",
                    "2024-05-01",
                ]
            ),
            "event_label": [
                "AF2",
                "Multimer",
                "AF-DB",
                "AF3",
            ],
        }
    )

    country_first_af_adoption = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                TRY_CAST(publication_date AS DATE) AS publication_date,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND publication_date IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                country_code,
                is_global_south
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        )
        SELECT
            c.country_code,
            MIN(w.publication_date) AS first_af_date,
            MIN(c.is_global_south) AS is_global_south
        FROM works_clean AS w
        INNER JOIN work_country_dedup AS c
            ON w.work_id = c.work_id
        GROUP BY c.country_code
        ORDER BY first_af_date, c.country_code
        """
    ).df()

    country_first_af_adoption["first_af_date"] = pd.to_datetime(
        country_first_af_adoption["first_af_date"]
    )
    country_first_af_adoption["first_af_month"] = (
        country_first_af_adoption["first_af_date"].dt.to_period("M").dt.to_timestamp()
    )

    _event_dates = event_calendar["event_date"].tolist()
    _event_names = event_calendar["event_name"].tolist()

    country_first_af_adoption["entry_phase"] = pd.cut(
        country_first_af_adoption["first_af_date"],
        bins=[pd.Timestamp.min, *_event_dates, pd.Timestamp.max],
        labels=[
            "pre_af2",
            "af2_phase",
            "multimer_phase",
            "afdb_phase",
            "af3_phase",
        ],
        right=False,
    )

    country_adoption_timeline = (
        country_first_af_adoption.groupby("first_af_month", as_index=False)
        .agg(new_adopter_countries=("country_code", "nunique"))
        .sort_values("first_af_month")
    )
    country_adoption_timeline["cumulative_adopting_countries"] = (
        country_adoption_timeline["new_adopter_countries"].cumsum()
    )

    country_adoption_timeline
    return country_first_af_adoption, event_calendar


@app.cell(hide_code=True)
def _(country_first_af_adoption):
    country_first_af_adoption["first_af_quarter"] = (
        country_first_af_adoption["first_af_date"].dt.to_period("Q").dt.to_timestamp()
    )

    country_phase_adoption = (
        country_first_af_adoption.groupby(
            ["first_af_quarter", "entry_phase"], as_index=False
        )
        .agg(new_adopter_countries=("country_code", "nunique"))
        .sort_values(["first_af_quarter", "entry_phase"])
    )

    country_quarter_adoption = (
        country_first_af_adoption.groupby("first_af_quarter", as_index=False)
        .agg(new_adopter_countries=("country_code", "nunique"))
        .sort_values("first_af_quarter")
    )

    country_quarter_adoption
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Main-text Figures

    This section presents the figures underpinning the principal empirical results reported in the main text. The analyses trace the global diffusion, production scale, collaboration structure, network position, and distributional concentration of AlphaFold-related research.
    """)
    return


@app.cell(hide_code=True)
def fig_1_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_first_af_adoption,
    pd,
    plt,
):
    _figure_1c_month_range = pd.date_range("2019-01-01", "2025-12-01", freq="MS")

    _figure_1c_monthly_new = (
        country_first_af_adoption.assign(
            first_af_month=lambda _df: (
                _df["first_af_date"].dt.to_period("M").dt.to_timestamp()
            )
        )
        .loc[
            lambda _df: _df["first_af_month"].between(
                pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-01")
            )
        ]
        .groupby("first_af_month", as_index=False)
        .agg(new_adopting_countries=("country_code", "nunique"))
    )

    figure_1c_adoption_curve = pd.DataFrame(
        {"publication_month": _figure_1c_month_range}
    )
    figure_1c_adoption_curve = figure_1c_adoption_curve.merge(
        _figure_1c_monthly_new,
        left_on="publication_month",
        right_on="first_af_month",
        how="left",
    )
    figure_1c_adoption_curve = figure_1c_adoption_curve.drop(
        columns=["first_af_month"]
    )
    figure_1c_adoption_curve["new_adopting_countries"] = (
        figure_1c_adoption_curve["new_adopting_countries"].fillna(0).astype(int)
    )
    figure_1c_adoption_curve["cumulative_adopting_countries"] = (
        figure_1c_adoption_curve["new_adopting_countries"].cumsum()
    )

    _fig_f1c, _ax_f1c = plt.subplots(figsize=(18, 6), dpi=220)
    _ax_f1c_secondary = _ax_f1c.twinx()

    _af2_start = pd.Timestamp("2021-07-01")
    _af3_start = pd.Timestamp("2024-05-01")

    _ax_f1c.axvspan(
        pd.Timestamp("2019-01-01"),
        _af2_start,
        color=AF_PURPLE,
        alpha=0.06,
        zorder=0,
    )
    _ax_f1c.axvspan(_af2_start, _af3_start, color=AF_BLUE, alpha=0.055, zorder=0)
    _ax_f1c.axvspan(
        _af3_start, pd.Timestamp("2026-01-01"), color=AF_CYAN, alpha=0.07, zorder=0
    )

    _ax_f1c_secondary.bar(
        figure_1c_adoption_curve["publication_month"],
        figure_1c_adoption_curve["new_adopting_countries"],
        width=24,
        color="#93c5fd",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.45,
        zorder=1,
    )

    _ax_f1c.plot(
        figure_1c_adoption_curve["publication_month"],
        figure_1c_adoption_curve["cumulative_adopting_countries"],
        color="#1d4e89",
        linewidth=3.0,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    for _event_date, _event_label in [
        (pd.Timestamp("2021-07-01"), "AF2+AFDB"),
        (pd.Timestamp("2024-05-01"), "AF3"),
    ]:
        _ax_f1c.axvline(
            _event_date,
            color=AF_EVENT_NEUTRAL,
            linestyle=(0, (4, 4)),
            linewidth=1.3,
            alpha=0.9,
            zorder=2,
        )
        _ax_f1c.text(
            _event_date + pd.DateOffset(days=10),
            146,
            _event_label,
            rotation=90,
            va="top",
            ha="left",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_EVENT_NEUTRAL,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.8,
            },
            zorder=5,
        )

    _ax_f1c.annotate(
        "Gentle slope\nEarly adopters only\n(US, UK, CN, DE ...)",
        xy=(
            pd.Timestamp("2020-06-01"),
            figure_1c_adoption_curve.loc[
                figure_1c_adoption_curve["publication_month"]
                == pd.Timestamp("2020-06-01"),
                "cumulative_adopting_countries",
            ].iloc[0],
        ),
        xytext=(pd.Timestamp("2019-04-01"), 28),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.7,
        },
        zorder=6,
    )

    _ax_f1c.annotate(
        "Steep inflection\nBulk of new countries enter",
        xy=(
            pd.Timestamp("2023-03-01"),
            figure_1c_adoption_curve.loc[
                figure_1c_adoption_curve["publication_month"]
                == pd.Timestamp("2023-03-01"),
                "cumulative_adopting_countries",
            ].iloc[0],
        ),
        xytext=(pd.Timestamp("2022-10-01"), 80),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.0},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.7},
        zorder=6,
    )

    _ax_f1c.annotate(
        "Curve begins to flatten\nPost-AF3 expansion continues\nbut with slower accumulation",
        xy=(
            pd.Timestamp("2025-03-01"),
            figure_1c_adoption_curve.loc[
                figure_1c_adoption_curve["publication_month"]
                == pd.Timestamp("2025-03-01"),
                "cumulative_adopting_countries",
            ].iloc[0],
        ),
        xytext=(pd.Timestamp("2024-08-01"), 91),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.0},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.7},
        zorder=6,
    )

    _ax_f1c.text(
        pd.Timestamp("2020-04-01"),
        155.5,
        "Pre-AF2 era\n2019-June 2021",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_EVENT_NEUTRAL,
        ha="center",
        va="bottom",
    )
    _ax_f1c.text(
        pd.Timestamp("2022-11-01"),
        155.5,
        "AF2 era\nJuly 2021-April 2024",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_EVENT_NEUTRAL,
        ha="center",
        va="bottom",
    )
    _ax_f1c.text(
        pd.Timestamp("2025-03-01"),
        155.5,
        "AF3 era\nMay 2024 onward",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_EVENT_NEUTRAL,
        ha="center",
        va="bottom",
    )

    _ax_f1c.set_xlim(pd.Timestamp("2018-12-15"), pd.Timestamp("2025-12-15"))
    _ax_f1c.set_xlabel(
        "Publication month", fontsize=AF_LABEL_FONT_SIZE, labelpad=10
    )
    _ax_f1c.set_ylabel(
        "Cumulative adopting countries", fontsize=AF_LABEL_FONT_SIZE, labelpad=10
    )
    _ax_f1c.set_ylim(0, 152)
    _ax_f1c.set_yticks([0, 25, 50, 75, 100, 125, 152])
    _ax_f1c.grid(
        axis="y", linestyle=(0, (3, 3)), linewidth=0.7, color="#d1d5db", alpha=0.95
    )
    _ax_f1c.grid(axis="x", visible=False)
    _ax_f1c.set_axisbelow(True)
    _ax_f1c.spines["top"].set_visible(False)
    _ax_f1c.spines["right"].set_visible(False)
    _ax_f1c.spines["left"].set_color("#374151")
    _ax_f1c.spines["bottom"].set_color("#374151")
    _ax_f1c.spines["left"].set_linewidth(0.8)
    _ax_f1c.spines["bottom"].set_linewidth(0.8)
    _ax_f1c.tick_params(axis="both", labelsize=10, colors="#1f2937")

    _ax_f1c.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    _ax_f1c.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    _ax_f1c_secondary.set_ylabel(
        "New adopting countries per month",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#4b5563",
    )
    _ax_f1c_secondary.set_ylim(
        0, max(8, figure_1c_adoption_curve["new_adopting_countries"].max() * 1.35)
    )
    _ax_f1c_secondary.tick_params(axis="y", labelsize=9.5, colors="#4b5563")
    _ax_f1c_secondary.spines["top"].set_visible(False)
    _ax_f1c_secondary.spines["left"].set_visible(False)
    _ax_f1c_secondary.spines["right"].set_color("#4b5563")
    _ax_f1c_secondary.spines["right"].set_linewidth(0.8)

    _fig_f1c.autofmt_xdate(rotation=0, ha="center")
    _fig_f1c.subplots_adjust(left=0.07, right=0.93, top=0.87, bottom=0.20)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    country_first_af_adoption,
    country_name_lookup,
    country_name_overrides,
    pd,
    world_map,
):
    figure_1a_country_first_adoption_map = country_first_af_adoption[
        ["country_code", "first_af_date"]
    ].copy()
    figure_1a_country_first_adoption_map["first_af_year"] = (
        figure_1a_country_first_adoption_map["first_af_date"].dt.year
    )

    figure_1a_country_first_adoption_map = figure_1a_country_first_adoption_map.merge(
        country_name_lookup,
        on="country_code",
        how="left",
    )
    figure_1a_country_first_adoption_map = figure_1a_country_first_adoption_map.merge(
        country_name_overrides,
        on="country_code",
        how="left",
    )
    figure_1a_country_first_adoption_map["country_name"] = (
        figure_1a_country_first_adoption_map["country_name"].fillna(
            figure_1a_country_first_adoption_map["country_name_map"]
        )
    )
    figure_1a_country_first_adoption_map["country_name"] = (
        figure_1a_country_first_adoption_map["country_name"].fillna(
            figure_1a_country_first_adoption_map["country_code"]
        )
    )

    figure_1a_world_map = world_map.merge(
        figure_1a_country_first_adoption_map[
            ["country_code", "country_name", "first_af_year"]
        ],
        left_on="NAME",
        right_on="country_name",
        how="left",
    )

    figure_1a_match_summary = pd.DataFrame(
        {
            "metric": [
                "countries with AF adoption",
                "countries matched to basemap",
                "countries unmatched to basemap",
                "match_rate",
            ],
            "value": [
                int(figure_1a_country_first_adoption_map["country_code"].nunique()),
                int(
                    figure_1a_country_first_adoption_map["country_name_map"]
                    .notna()
                    .sum()
                ),
                int(
                    figure_1a_country_first_adoption_map["country_name_map"]
                    .isna()
                    .sum()
                ),
                round(
                    figure_1a_country_first_adoption_map["country_name_map"]
                    .notna()
                    .mean(),
                    4,
                ),
            ],
        }
    )

    figure_1a_country_first_adoption_map
    return figure_1a_country_first_adoption_map, figure_1a_world_map


@app.cell(hide_code=True)
def fig_1_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_CYAN,
    AF_PURPLE,
    AF_SEQUENTIAL_CMAP,
    figure_1a_country_first_adoption_map,
    figure_1a_world_map,
    plt,
):
    _fig_f1a, _ax_f1a = plt.subplots(figsize=(16, 9), dpi=220)

    _f1a_map = figure_1a_world_map[
        figure_1a_world_map["NAME"] != "Antarctica"
    ].copy()
    _f1a_year_min = int(_f1a_map["first_af_year"].dropna().min())
    _f1a_year_max = int(_f1a_map["first_af_year"].dropna().max())
    _f1a_cmap = AF_SEQUENTIAL_CMAP.resampled(_f1a_year_max - _f1a_year_min + 1)

    _f1a_map.plot(
        column="first_af_year",
        cmap=_f1a_cmap,
        linewidth=0.35,
        ax=_ax_f1a,
        edgecolor=AF_CYAN,
        missing_kwds={
            "color": AF_CYAN,
            "edgecolor": "white",
            "linewidth": 0.2,
            "label": "No AF adoption observed",
        },
        legend=True,
        legend_kwds={
            "label": "Year of first AlphaFold-related publication",
            "shrink": 0.28,
            "pad": -0.01,
            "aspect": 14,
        },
    )

    _f1a_colorbar_ax = _fig_f1a.axes[-1]
    _f1a_colorbar_ax.tick_params(
        labelsize=AF_ANNOTATION_FONT_SIZE, colors=AF_PURPLE
    )
    _f1a_colorbar_ax.set_ylabel(
        _f1a_colorbar_ax.get_ylabel(),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_PURPLE,
        labelpad=10,
    )

    _f1a_early_adopters = figure_1a_country_first_adoption_map.sort_values(
        ["first_af_year", "country_code"]
    ).head(20)
    _f1a_label_points = _f1a_map.merge(
        _f1a_early_adopters[["country_code", "country_name", "first_af_year"]],
        on=["country_code", "country_name", "first_af_year"],
        how="inner",
    )
    for _row in _f1a_label_points.itertuples(index=False):
        _centroid = _row.geometry.representative_point()
        _ax_f1a.text(
            _centroid.x,
            _centroid.y,
            _row.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            color=AF_PURPLE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.5,
            },
            zorder=5,
        )

    # _ax_f1a.set_title(
    #     "Figure 1A. Global diffusion of AlphaFold-related research by first adoption year",
    #     fontsize=16,
    #     pad=18,
    #     loc="left",
    # )
    # _ax_f1a.text(
    #     0,
    #     1.115,
    #     "Country color indicates the year in which the country first produced an AlphaFold-related publication in the deduplicated dataset. Earlier-adopting countries are shown in darker colors.",
    #     transform=_ax_f1a.transAxes,
    #     fontsize=10,
    #     color=AF_BLUE,
    #     va="bottom",
    # )
    # _ax_f1a.text(
    #     0,
    #     -0.055,
    #     f"Map coverage: {int(figure_1a_match_summary.loc[figure_1a_match_summary['metric'] == 'countries matched to basemap', 'value'].iloc[0])}/"
    #     f"{int(figure_1a_match_summary.loc[figure_1a_match_summary['metric'] == 'countries with AF adoption', 'value'].iloc[0])} adopting countries matched to the basemap. "
    #     "Countries with no observed AF adoption are shown in light gray. Antarctica is omitted for visual clarity.",
    #     transform=_ax_f1a.transAxes,
    #     fontsize=9,
    #     color=AF_BLUE,
    #     va="top",
    # )
    _ax_f1a.set_axis_off()
    _fig_f1a.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.08)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_1_c(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_af_nonaf_compare,
    inequality_yearly_base,
    plt,
):
    _f1b_top10_codes = country_af_nonaf_compare.nlargest(
        10, "af_fractional_count"
    )["country_code"].tolist()

    _f1b_af_yearly_country = (
        inequality_yearly_base[
            inequality_yearly_base["is_alphafold_related"] == True
        ]
        .groupby(["publication_year", "country_code"], as_index=False)[
            "fractional_weight"
        ]
        .sum()
    )

    _f1b_top10_total_by_year = (
        _f1b_af_yearly_country[
            _f1b_af_yearly_country["country_code"].isin(_f1b_top10_codes)
        ]
        .groupby("publication_year", as_index=False)["fractional_weight"]
        .sum()
        .rename(columns={"fractional_weight": "top10_total"})
    )

    _f1b_global_total_by_year = (
        inequality_yearly_base[
            inequality_yearly_base["is_alphafold_related"] == True
        ]
        .groupby("publication_year", as_index=False)["fractional_weight"]
        .sum()
        .rename(columns={"fractional_weight": "global_total"})
    )

    figure_f1b_output_gap = _f1b_global_total_by_year.merge(
        _f1b_top10_total_by_year,
        on="publication_year",
        how="left",
    )
    figure_f1b_output_gap["top10_total"] = figure_f1b_output_gap[
        "top10_total"
    ].fillna(0)
    figure_f1b_output_gap["non_top10_total"] = (
        figure_f1b_output_gap["global_total"]
        - figure_f1b_output_gap["top10_total"]
    )

    _fig_f1b_output_gap, _ax_f1b_output_gap = plt.subplots(
        figsize=(8, 6.4), dpi=220
    )
    _f1b_label_font_size = AF_LABEL_FONT_SIZE * 0.85
    _f1b_annotation_font_size = AF_ANNOTATION_FONT_SIZE * 0.85
    _ax_f1b_output_gap_right = _ax_f1b_output_gap.twinx()

    _f1b_years = figure_f1b_output_gap["publication_year"].to_numpy()
    _f1b_global = figure_f1b_output_gap["global_total"].to_numpy()
    _f1b_top10 = figure_f1b_output_gap["top10_total"].to_numpy()

    _ax_f1b_output_gap.fill_between(
        _f1b_years,
        _f1b_top10,
        _f1b_global,
        color=AF_CYAN,
        alpha=0.22,
        zorder=1,
    )

    _ax_f1b_output_gap.plot(
        _f1b_years,
        _f1b_global,
        color=AF_PURPLE,
        linewidth=3.0,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Global AlphaFold-related publication output",
        zorder=4,
    )

    _ax_f1b_output_gap_right.plot(
        _f1b_years,
        _f1b_top10,
        color=AF_BLUE,
        linewidth=2.6,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Top 10 countries' AlphaFold-related publication output",
        zorder=5,
    )

    _f1b_gap_label_row = figure_f1b_output_gap.iloc[
        max(len(figure_f1b_output_gap) - 2, 0)
    ]
    _f1b_gap_mid = (
        _f1b_gap_label_row["global_total"] + _f1b_gap_label_row["top10_total"]
    ) / 2
    _ax_f1b_output_gap.annotate(
        "Shaded gap = output from countries\noutside the top 10",
        xy=(
            _f1b_gap_label_row["publication_year"],
            _f1b_gap_mid,
        ),
        xytext=(
            _f1b_gap_label_row["publication_year"] - 3.95,
            _f1b_gap_mid + _f1b_global.max() * 0.11,
        ),
        fontsize=_f1b_annotation_font_size,
        color="#1f2937",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.55,
        },
        zorder=6,
    )

    for _year, _global_value in zip(_f1b_years, _f1b_global):
        _ax_f1b_output_gap.text(
            _year,
            _global_value + _f1b_global.max() * 0.018,
            f"{_global_value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=_f1b_annotation_font_size,
            color="#1f2937",
            zorder=6,
        )

    for _year, _top10_value in zip(_f1b_years, _f1b_top10):
        _ax_f1b_output_gap_right.text(
            _year,
            _top10_value - max(_f1b_top10.max() * 0.03, 6),
            f"{_top10_value:,.0f}",
            ha="center",
            va="top",
            fontsize=_f1b_annotation_font_size,
            color=AF_BLUE,
            zorder=6,
        )

    _ax_f1b_output_gap.set_xlim(_f1b_years.min() - 0.2, _f1b_years.max() + 0.2)
    _ax_f1b_output_gap.set_xticks(_f1b_years)
    _ax_f1b_output_gap.set_xlabel(
        "Publication year", fontsize=_f1b_label_font_size, labelpad=9
    )
    _ax_f1b_output_gap.set_ylabel(
        "Global AF-related output",
        fontsize=_f1b_label_font_size,
        color="#1f2937",
        labelpad=10,
    )
    _ax_f1b_output_gap_right.set_ylabel(
        "Top 10 countries' AF-related output",
        fontsize=_f1b_label_font_size,
        color="#1f2937",
        labelpad=12,
    )

    _ax_f1b_output_gap.yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    )
    _ax_f1b_output_gap_right.yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    )
    _f1b_ymax = max(_f1b_global.max(), _f1b_top10.max()) * 1.08
    _ax_f1b_output_gap.set_ylim(0, _f1b_ymax)
    _ax_f1b_output_gap_right.set_ylim(0, _f1b_ymax)
    _f1b_y_ticks = plt.matplotlib.ticker.MaxNLocator(nbins=6).tick_values(
        0, _f1b_ymax
    )
    _ax_f1b_output_gap.set_yticks(_f1b_y_ticks)
    _ax_f1b_output_gap_right.set_yticks(_f1b_y_ticks)


    _ax_f1b_output_gap.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.72,
        color=AF_CYAN,
        alpha=0.9,
    )
    _ax_f1b_output_gap.grid(axis="x", visible=False)
    _ax_f1b_output_gap.set_axisbelow(True)

    _ax_f1b_output_gap.spines["top"].set_visible(False)
    _ax_f1b_output_gap_right.spines["top"].set_visible(False)
    _ax_f1b_output_gap.spines["right"].set_visible(False)
    _ax_f1b_output_gap_right.spines["left"].set_visible(False)
    _ax_f1b_output_gap.spines["left"].set_color("#374151")
    _ax_f1b_output_gap.spines["bottom"].set_color("#374151")
    _ax_f1b_output_gap_right.spines["right"].set_color("#374151")
    _ax_f1b_output_gap.spines["left"].set_linewidth(0.8)
    _ax_f1b_output_gap.spines["bottom"].set_linewidth(0.8)
    _ax_f1b_output_gap_right.spines["right"].set_linewidth(0.8)

    _ax_f1b_output_gap.tick_params(axis="x", labelsize=10, colors="#1f2937")
    _ax_f1b_output_gap.tick_params(axis="y", labelsize=10, colors="#1f2937")
    _ax_f1b_output_gap_right.tick_params(axis="y", labelsize=10, colors="#1f2937")

    _f1b_handles_left, _f1b_labels_left = (
        _ax_f1b_output_gap.get_legend_handles_labels()
    )
    _f1b_handles_right, _f1b_labels_right = (
        _ax_f1b_output_gap_right.get_legend_handles_labels()
    )
    _ax_f1b_output_gap.legend(
        _f1b_handles_left + _f1b_handles_right,
        _f1b_labels_left + _f1b_labels_right,
        frameon=False,
        loc="upper left",
        fontsize=_f1b_annotation_font_size,
    )

    _fig_f1b_output_gap.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _(country_af_nonaf_compare, duckdb, pd):
    inequality_yearly_base = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                publication_year,
                is_alphafold_related,
                TRY_CAST(publication_date AS DATE) AS publication_date
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND publication_year IS NOT NULL
              AND is_alphafold_related IS NOT NULL
              AND publication_date IS NOT NULL
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT work_id, COUNT(*) AS n_countries
            FROM work_country_dedup
            GROUP BY work_id
        )
        SELECT
            w.publication_year,
            w.publication_date,
            w.is_alphafold_related,
            c.country_code,
            1.0 / cc.n_countries AS fractional_weight
        FROM works_clean AS w
        INNER JOIN work_country_dedup AS c
            ON w.work_id = c.work_id
        INNER JOIN work_country_counts AS cc
            ON w.work_id = cc.work_id
        WHERE cc.n_countries > 0
        """
    ).df()

    inequality_yearly_base["publication_date"] = pd.to_datetime(
        inequality_yearly_base["publication_date"]
    )
    inequality_yearly_base["diffusion_phase"] = pd.cut(
        inequality_yearly_base["publication_date"],
        bins=[
            pd.Timestamp.min,
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-10-01"),
            pd.Timestamp("2022-07-01"),
            pd.Timestamp("2024-05-01"),
            pd.Timestamp.max,
        ],
        labels=[
            "pre_af2",
            "af2_phase",
            "multimer_phase",
            "afdb_phase",
            "af3_phase",
        ],
        right=False,
    )

    def _gini_year_group(_series):
        _arr = pd.Series(_series).fillna(0).sort_values().to_numpy()
        if _arr.sum() == 0:
            return 0.0
        _cum = _arr.cumsum() / _arr.sum()
        _cum = pd.Series([0.0, *_cum])
        _pop = pd.Series(range(0, len(_arr) + 1)) / len(_arr)
        return float(
            1
            - 2 * ((_cum.shift(fill_value=0) + _cum) / 2 * _pop.diff().fillna(0)).sum()
        )

    _gini_year_rows = []
    for (_year, _group), _df in inequality_yearly_base.groupby(
        ["publication_year", "is_alphafold_related"]
    ):
        _country_counts = _df.groupby("country_code", as_index=False)[
            "fractional_weight"
        ].sum()
        _gini_year_rows.append(
            {
                "publication_year": int(_year),
                "group": "AF" if bool(_group) else "non-AF",
                "gini": _gini_year_group(_country_counts["fractional_weight"]),
            }
        )

    gini_year_trend = pd.DataFrame(_gini_year_rows).sort_values(
        ["publication_year", "group"]
    )

    _phase_lorenz_rows = []
    for _phase, _df in inequality_yearly_base[
        inequality_yearly_base["is_alphafold_related"] == True
    ].groupby("diffusion_phase"):
        _country_counts = (
            _df.groupby("country_code", as_index=False)["fractional_weight"]
            .sum()
            .sort_values("fractional_weight")
        )
        _country_counts["cum_output_share"] = (
            _country_counts["fractional_weight"].cumsum()
            / _country_counts["fractional_weight"].sum()
        )
        _country_counts["cum_country_share"] = range(1, len(_country_counts) + 1)
        _country_counts["cum_country_share"] = _country_counts[
            "cum_country_share"
        ] / len(_country_counts)
        _country_counts["phase"] = str(_phase)
        _phase_lorenz_rows.append(
            _country_counts[
                ["phase", "country_code", "cum_country_share", "cum_output_share"]
            ]
        )
    phase_lorenz_curve_data = pd.concat(_phase_lorenz_rows, ignore_index=True)

    _top10_codes = country_af_nonaf_compare.nlargest(10, "af_fractional_count")[
        "country_code"
    ].tolist()
    country_rank_trend = (
        inequality_yearly_base[inequality_yearly_base["is_alphafold_related"] == True]
        .groupby(["publication_year", "country_code"], as_index=False)[
            "fractional_weight"
        ]
        .sum()
    )
    country_rank_trend["rank"] = country_rank_trend.groupby("publication_year")[
        "fractional_weight"
    ].rank(
        ascending=False,
        method="first",
    )
    country_rank_trend = country_rank_trend[
        country_rank_trend["country_code"].isin(_top10_codes)
    ]

    top10_share_trend = (
        inequality_yearly_base[inequality_yearly_base["is_alphafold_related"] == True]
        .groupby(["publication_year", "country_code"], as_index=False)[
            "fractional_weight"
        ]
        .sum()
    )
    top10_share_trend = top10_share_trend[
        top10_share_trend["country_code"].isin(_top10_codes)
    ]

    top10_total_by_year = (
        top10_share_trend.groupby("publication_year", as_index=False)[
            "fractional_weight"
        ]
        .sum()
        .rename(columns={"fractional_weight": "top10_total"})
    )
    all_af_total_by_year = (
        inequality_yearly_base[inequality_yearly_base["is_alphafold_related"] == True]
        .groupby("publication_year", as_index=False)["fractional_weight"]
        .sum()
        .rename(columns={"fractional_weight": "af_total"})
    )
    top10_share_trend = top10_share_trend.merge(
        top10_total_by_year, on="publication_year", how="left"
    )
    top10_share_trend = top10_share_trend.merge(
        all_af_total_by_year, on="publication_year", how="left"
    )
    top10_share_trend["country_share_within_top10"] = (
        top10_share_trend["fractional_weight"] / top10_share_trend["top10_total"]
    )
    top10_share_trend["country_share_within_af"] = (
        top10_share_trend["fractional_weight"] / top10_share_trend["af_total"]
    )

    dynamic_center_periphery = country_rank_trend.merge(
        country_af_nonaf_compare[["country_code", "af_share", "non_af_share"]],
        on="country_code",
        how="left",
    )

    gini_year_trend
    return inequality_yearly_base, top10_share_trend


@app.cell(hide_code=True)
def fig_1_d(
    AF_ANNOTATION_FONT_SIZE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    af_color_at,
    pd,
    plt,
    to_rgb,
    top10_share_trend,
):
    _f8d_pivot = top10_share_trend.pivot_table(
        index="publication_year",
        columns="country_code",
        values="country_share_within_af",
        fill_value=0,
    )

    _fig_f8d, _ax_f8d = plt.subplots(figsize=(12, 7), dpi=220)

    _f8d_last_year = _f8d_pivot.index.max()
    _f8d_order = (
        _f8d_pivot.loc[_f8d_last_year].sort_values(ascending=False).index.tolist()
    )
    _f8d_stack_order = list(reversed(_f8d_order))
    _f8d_pivot = _f8d_pivot[_f8d_stack_order]

    _f8d_label_order = list(reversed(_f8d_stack_order))
    _f8d_base_color = plt.matplotlib.colors.to_rgb(AF_PURPLE)
    _f8d_light_color = to_rgb(AF_CYAN)
    _f8d_gradient_steps = [
        _i / max(len(_f8d_label_order) - 1, 1)
        for _i in range(len(_f8d_label_order))
    ]
    _f8d_label_color_map = {
        _country: af_color_at(_step)
        for _country, _step in zip(_f8d_label_order, _f8d_gradient_steps)
    }
    _f8d_colors = [_f8d_label_color_map[_country] for _country in _f8d_stack_order]
    _f8d_years = _f8d_pivot.index.to_numpy()
    _f8d_values = _f8d_pivot.to_numpy().T

    _ax_f8d.set_facecolor("white")
    _fig_f8d.patch.set_facecolor("white")

    _ax_f8d.stackplot(
        _f8d_years,
        _f8d_values,
        labels=_f8d_stack_order,
        colors=_f8d_colors,
        alpha=0.95,
        linewidth=0,
        zorder=2,
    )

    _f8d_cum = _f8d_pivot.cumsum(axis=1)
    for _country in _f8d_stack_order[:-1]:
        _ax_f8d.plot(
            _f8d_years,
            _f8d_cum[_country],
            color="white",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )

    _f8d_event_color = AF_EVENT_NEUTRAL
    _f8d_event_annotations = pd.DataFrame(
        {
            "event_year": [2021, 2024],
            "event_label": ["AF2 + AF-DB\n2021-07", "AF3\n2024-05"],
        }
    )
    for _row in _f8d_event_annotations.itertuples(index=False):
        if _f8d_years.min() <= _row.event_year <= _f8d_years.max():
            _ax_f8d.axvline(
                _row.event_year,
                color=_f8d_event_color,
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
            )
            _ax_f8d.text(
                _row.event_year,
                0.04,
                _row.event_label,
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=_f8d_event_color,
                fontweight="semibold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.80,
                    "pad": 0.8,
                },
                zorder=5,
            )

    _f8d_label_x = _f8d_last_year + 0.22
    _f8d_last_vals = _f8d_pivot.loc[_f8d_last_year]
    _f8d_running_bottom = 0.0
    for _country in _f8d_stack_order:
        _height = float(_f8d_last_vals[_country])
        if _height > 0.01:
            _y = _f8d_running_bottom + _height / 2
            _ax_f8d.text(
                _f8d_label_x,
                _y,
                _country,
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=_f8d_label_color_map[_country],
                fontweight="semibold",
                va="center",
                ha="left",
                zorder=6,
            )
        _f8d_running_bottom += _height

    _ax_f8d.set_xlabel("Publication year", fontsize=AF_LABEL_FONT_SIZE, labelpad=9)
    _ax_f8d.set_ylabel(
        "Share of total AF output", fontsize=AF_LABEL_FONT_SIZE, labelpad=10
    )
    _ax_f8d.set_ylim(0, 1)
    _ax_f8d.set_xlim(_f8d_years.min(), _f8d_years.max() + 1.0)
    _ax_f8d.set_xticks(_f8d_years)
    _ax_f8d.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    _ax_f8d.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.2))
    _ax_f8d.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.75,
        color=AF_CYAN,
        alpha=0.95,
    )
    _ax_f8d.grid(axis="x", visible=False)
    _ax_f8d.set_axisbelow(True)
    _ax_f8d.spines["top"].set_visible(False)
    _ax_f8d.spines["right"].set_visible(False)
    _ax_f8d.spines["left"].set_color("#374151")
    _ax_f8d.spines["bottom"].set_color("#374151")
    _ax_f8d.spines["left"].set_linewidth(0.8)
    _ax_f8d.spines["bottom"].set_linewidth(0.8)
    _ax_f8d.tick_params(
        axis="both", labelsize=10, colors="#1f2937", length=3.2, width=0.8
    )
    if _ax_f8d.get_legend() is not None:
        _ax_f8d.get_legend().remove()
    _fig_f8d.subplots_adjust(left=0.10, right=0.84, top=0.88, bottom=0.12)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(duckdb, pd):
    discipline_af_base = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                TRY_CAST(publication_date AS DATE) AS publication_date,
                is_alphafold_related,
                primary_field_display_name,
                primary_subfield_display_name,
                primary_topic_display_name
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND publication_date IS NOT NULL
              AND primary_field_display_name IS NOT NULL
              AND primary_subfield_display_name IS NOT NULL
        )
        SELECT
            work_id,
            publication_date,
            is_alphafold_related,
            primary_field_display_name,
            primary_subfield_display_name,
            primary_topic_display_name
        FROM works_clean
        """
    ).df()

    discipline_af_base["publication_date"] = pd.to_datetime(
        discipline_af_base["publication_date"]
    )

    discipline_af_base["adoption_lag_months"] = (
        discipline_af_base["publication_date"].dt.year - pd.Timestamp("2021-07-01").year
    ) * 12 + (
        discipline_af_base["publication_date"].dt.month
        - pd.Timestamp("2021-07-01").month
    )

    discipline_field_summary = discipline_af_base.groupby(
        "primary_field_display_name", as_index=False
    ).agg(
        total_papers=("work_id", "count"),
        af_papers=("is_alphafold_related", "sum"),
    )
    discipline_field_summary["af_share"] = (
        discipline_field_summary["af_papers"] / discipline_field_summary["total_papers"]
    )
    discipline_field_summary = discipline_field_summary.sort_values(
        ["af_share", "af_papers"], ascending=[False, False]
    )

    discipline_subfield_summary = discipline_af_base.groupby(
        "primary_subfield_display_name", as_index=False
    ).agg(
        total_papers=("work_id", "count"),
        af_papers=("is_alphafold_related", "sum"),
    )
    discipline_subfield_summary["af_share"] = (
        discipline_subfield_summary["af_papers"]
        / discipline_subfield_summary["total_papers"]
    )
    discipline_subfield_summary = discipline_subfield_summary.sort_values(
        ["af_share", "af_papers"], ascending=[False, False]
    )

    discipline_adoption_lag = (
        discipline_af_base[discipline_af_base["is_alphafold_related"] == True]
        .groupby("primary_subfield_display_name", as_index=False)
        .agg(first_af_date=("publication_date", "min"))
    )
    discipline_adoption_lag["adoption_lag_months"] = (
        discipline_adoption_lag["first_af_date"].dt.year
        - pd.Timestamp("2021-07-01").year
    ) * 12 + (
        discipline_adoption_lag["first_af_date"].dt.month
        - pd.Timestamp("2021-07-01").month
    )

    discipline_cumulative_adoption = (
        discipline_af_base[discipline_af_base["is_alphafold_related"] == True]
        .assign(
            publication_month=lambda _df: (
                _df["publication_date"].dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby(["publication_month", "primary_subfield_display_name"], as_index=False)
        .agg(af_papers=("work_id", "count"))
        .sort_values(["primary_subfield_display_name", "publication_month"])
    )
    discipline_cumulative_adoption["cumulative_af_papers"] = (
        discipline_cumulative_adoption.groupby("primary_subfield_display_name")[
            "af_papers"
        ].cumsum()
    )

    discipline_heatmap_data = duckdb.sql(
        """
        SELECT
            field_display_name,
            subfield_display_name,
            COUNT(*) AS topic_rows,
            AVG(CASE WHEN is_alphafold_related THEN 1.0 ELSE 0.0 END) AS af_share
        FROM (
            SELECT
                t.field_display_name,
                t.subfield_display_name,
                t.topic_display_name,
                w.is_alphafold_related
            FROM read_parquet('derived_tables_dedup/work_topics.parquet') AS t
            INNER JOIN read_parquet('derived_tables_dedup/works.parquet') AS w
                ON t.work_id = w.work_id
            WHERE t.field_display_name IS NOT NULL
              AND t.subfield_display_name IS NOT NULL
              AND w.is_alphafold_related IS NOT NULL
        )
        GROUP BY field_display_name, subfield_display_name
        """
    ).df()

    discipline_phase_data = discipline_af_base.copy()
    discipline_phase_data["diffusion_phase"] = pd.cut(
        discipline_phase_data["publication_date"],
        bins=[
            pd.Timestamp.min,
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-10-01"),
            pd.Timestamp("2022-07-01"),
            pd.Timestamp("2024-05-01"),
            pd.Timestamp.max,
        ],
        labels=[
            "pre_af2",
            "af2_phase",
            "multimer_phase",
            "afdb_phase",
            "af3_phase",
        ],
        right=False,
    )

    discipline_phase_summary = (
        discipline_phase_data[discipline_phase_data["is_alphafold_related"] == True]
        .groupby(["diffusion_phase", "primary_subfield_display_name"], as_index=False)
        .agg(af_papers=("work_id", "count"))
    )

    discipline_field_summary
    return (discipline_af_base,)


@app.cell(hide_code=True)
def _(country_first_af_adoption, pd):
    country_adoption_lag = country_first_af_adoption.copy()
    _country_base_date = pd.Timestamp("2018-12-02")
    country_adoption_lag["adoption_lag_months"] = (
        country_adoption_lag["first_af_date"].dt.year - _country_base_date.year
    ) * 12 + (country_adoption_lag["first_af_date"].dt.month - _country_base_date.month)
    country_adoption_lag
    return (country_adoption_lag,)


@app.cell(hide_code=True)
def _(pd):
    country_global_south_lookup = pd.DataFrame(
        {
            "country_code": [
                "AE",
                "AG",
                "AL",
                "AM",
                "AR",
                "AT",
                "AU",
                "AZ",
                "BA",
                "BD",
                "BE",
                "BG",
                "BH",
                "BJ",
                "BO",
                "BR",
                "BW",
                "BY",
                "CA",
                "CH",
                "CI",
                "CL",
                "CM",
                "CN",
                "CO",
                "CR",
                "CU",
                "CY",
                "CZ",
                "DE",
                "DK",
                "DO",
                "DZ",
                "EC",
                "EE",
                "EG",
                "ES",
                "ET",
                "FI",
                "FR",
                "GB",
                "GE",
                "GH",
                "GR",
                "GT",
                "HK",
                "HN",
                "HR",
                "HU",
                "ID",
                "IE",
                "IL",
                "IN",
                "IQ",
                "IR",
                "IS",
                "IT",
                "JM",
                "JO",
                "JP",
                "KE",
                "KG",
                "KH",
                "KR",
                "KW",
                "KZ",
                "LA",
                "LB",
                "LK",
                "LT",
                "LU",
                "LV",
                "MA",
                "MD",
                "ME",
                "MG",
                "MK",
                "MM",
                "MN",
                "MO",
                "MT",
                "MU",
                "MX",
                "MY",
                "MZ",
                "NA",
                "NG",
                "NI",
                "NL",
                "NO",
                "NP",
                "NZ",
                "OM",
                "PA",
                "PE",
                "PH",
                "PK",
                "PL",
                "PR",
                "PT",
                "PY",
                "QA",
                "RO",
                "RS",
                "RU",
                "RW",
                "SA",
                "SD",
                "SE",
                "SG",
                "SI",
                "SK",
                "SN",
                "SV",
                "SY",
                "TH",
                "TN",
                "TR",
                "TW",
                "TZ",
                "UA",
                "UG",
                "US",
                "UY",
                "UZ",
                "VE",
                "VN",
                "ZA",
                "ZM",
                "ZW",
            ],
            "south_group": [
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Non-Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
                "Global South",
            ],
        }
    )
    country_global_south_lookup
    return (country_global_south_lookup,)


@app.cell(hide_code=True)
def _(country_adoption_lag, country_global_south_lookup):
    country_adoption_lag_plot = country_adoption_lag.copy()
    country_adoption_lag_plot["country_code"] = (
        country_adoption_lag_plot["country_code"].astype(str).str.strip().str.upper()
    )
    country_adoption_lag_plot = country_adoption_lag_plot.drop(
        columns=["is_global_south"]
    )
    country_adoption_lag_plot = country_adoption_lag_plot.merge(
        country_global_south_lookup,
        on="country_code",
        how="left",
    )
    country_adoption_lag_plot = country_adoption_lag_plot.dropna(subset=["south_group"])
    country_adoption_lag_plot
    return (country_adoption_lag_plot,)


@app.cell(hide_code=True)
def _(
    country_adoption_lag_plot,
    country_name_lookup,
    country_name_overrides,
    discipline_af_base,
    duckdb,
    pd,
):
    research_type_lookup = pd.DataFrame(
        {
            "primary_subfield_display_name": [
                "Structural Biology",
                "Molecular Biology",
                "Biophysics",
                "Genetics",
                "Biochemistry",
                "Cell Biology",
                "Developmental Biology",
                "Aging",
                "Physiology",
                "Immunology",
                "Microbiology",
                "Virology",
                "Applied Microbiology and Biotechnology",
                "Drug Discovery",
                "Pharmacology",
                "Pharmaceutical Science",
                "Toxicology",
                "Molecular Medicine",
                "Clinical Biochemistry",
                "Biotechnology",
                "Cancer Research",
                "Endocrinology",
            ],
            "research_type": [
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "basic",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
                "applied",
            ],
        }
    )

    basic_applied_base = discipline_af_base.merge(
        research_type_lookup,
        on="primary_subfield_display_name",
        how="inner",
    )

    basic_applied_adoption_lag = (
        basic_applied_base[basic_applied_base["is_alphafold_related"] == True]
        .groupby("research_type", as_index=False)
        .agg(first_af_date=("publication_date", "min"))
    )
    basic_applied_adoption_lag["adoption_lag_months"] = (
        basic_applied_adoption_lag["first_af_date"].dt.year
        - pd.Timestamp("2021-07-01").year
    ) * 12 + (
        basic_applied_adoption_lag["first_af_date"].dt.month
        - pd.Timestamp("2021-07-01").month
    )

    basic_applied_cumulative = (
        basic_applied_base[basic_applied_base["is_alphafold_related"] == True]
        .assign(
            publication_month=lambda _df: (
                _df["publication_date"].dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby(["publication_month", "research_type"], as_index=False)
        .agg(af_papers=("work_id", "count"))
        .sort_values(["research_type", "publication_month"])
    )
    basic_applied_cumulative["cumulative_af_papers"] = basic_applied_cumulative.groupby(
        "research_type"
    )["af_papers"].cumsum()

    country_research_type_output = duckdb.sql(
        """
        WITH works_clean AS (
            SELECT
                work_id,
                primary_subfield_display_name,
                is_alphafold_related
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND primary_subfield_display_name IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        type_lookup AS (
            SELECT *
            FROM research_type_lookup
        ),
        work_country_dedup AS (
            SELECT DISTINCT
                work_id,
                country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT work_id, COUNT(*) AS n_countries
            FROM work_country_dedup
            GROUP BY work_id
        )
        SELECT
            t.research_type,
            c.country_code,
            SUM(1.0 / cc.n_countries) AS af_fractional_count
        FROM works_clean AS w
        INNER JOIN type_lookup AS t
            ON w.primary_subfield_display_name = t.primary_subfield_display_name
        INNER JOIN work_country_dedup AS c
            ON w.work_id = c.work_id
        INNER JOIN work_country_counts AS cc
            ON w.work_id = cc.work_id
        WHERE cc.n_countries > 0
        GROUP BY t.research_type, c.country_code
        """
    ).df()

    country_research_type_compare = country_research_type_output.pivot_table(
        index="country_code",
        columns="research_type",
        values="af_fractional_count",
        fill_value=0,
    ).reset_index()
    country_research_type_compare.columns.name = None
    country_research_type_compare = country_research_type_compare.merge(
        country_name_lookup,
        on="country_code",
        how="left",
    )
    country_research_type_compare = country_research_type_compare.merge(
        country_name_overrides,
        on="country_code",
        how="left",
    )
    country_research_type_compare["country_name"] = country_research_type_compare[
        "country_name"
    ].fillna(country_research_type_compare["country_name_map"])
    country_research_type_compare["country_name"] = country_research_type_compare[
        "country_name"
    ].fillna(country_research_type_compare["country_code"])

    basic_applied_country_group = country_adoption_lag_plot[
        ["country_code", "south_group"]
    ].drop_duplicates()
    country_research_type_compare = country_research_type_compare.merge(
        basic_applied_country_group,
        on="country_code",
        how="left",
    )

    basic_applied_lorenz_input = country_research_type_compare[
        ["country_code", "basic", "applied"]
    ].copy()

    basic_applied_base[:5]
    return basic_applied_lorenz_input, research_type_lookup


@app.cell(hide_code=True)
def _(basic_applied_lorenz_input, pd):
    _lorenz_basic = basic_applied_lorenz_input[["country_code", "basic"]].copy()
    _lorenz_basic = _lorenz_basic.fillna({"basic": 0}).sort_values("basic")
    _lorenz_basic["cum_output_share"] = (
        _lorenz_basic["basic"].cumsum() / _lorenz_basic["basic"].sum()
    )
    _lorenz_basic["cum_country_share"] = range(1, len(_lorenz_basic) + 1)
    _lorenz_basic["cum_country_share"] = _lorenz_basic["cum_country_share"] / len(
        _lorenz_basic
    )
    _lorenz_basic["group"] = "basic"

    _lorenz_applied = basic_applied_lorenz_input[["country_code", "applied"]].copy()
    _lorenz_applied = _lorenz_applied.fillna({"applied": 0}).sort_values("applied")
    _lorenz_applied["cum_output_share"] = (
        _lorenz_applied["applied"].cumsum() / _lorenz_applied["applied"].sum()
    )
    _lorenz_applied["cum_country_share"] = range(1, len(_lorenz_applied) + 1)
    _lorenz_applied["cum_country_share"] = _lorenz_applied["cum_country_share"] / len(
        _lorenz_applied
    )
    _lorenz_applied["group"] = "applied"

    basic_applied_lorenz_curve_data = pd.concat(
        [
            _lorenz_basic[
                ["group", "country_code", "cum_country_share", "cum_output_share"]
            ],
            _lorenz_applied[
                ["group", "country_code", "cum_country_share", "cum_output_share"]
            ],
        ],
        ignore_index=True,
    )

    basic_applied_lorenz_curve_data
    return


@app.cell(hide_code=True)
def _(country_af_nonaf_compare, pd):
    _lorenz_af = country_af_nonaf_compare[
        ["country_code", "af_fractional_count"]
    ].copy()
    _lorenz_af = _lorenz_af.fillna({"af_fractional_count": 0}).sort_values(
        "af_fractional_count"
    )
    _lorenz_af["cum_output_share"] = (
        _lorenz_af["af_fractional_count"].cumsum()
        / _lorenz_af["af_fractional_count"].sum()
    )
    _lorenz_af["cum_country_share"] = range(1, len(_lorenz_af) + 1)
    _lorenz_af["cum_country_share"] = _lorenz_af["cum_country_share"] / len(_lorenz_af)
    _lorenz_af["group"] = "AF"

    _lorenz_nonaf = country_af_nonaf_compare[
        ["country_code", "non_af_fractional_count"]
    ].copy()
    _lorenz_nonaf = _lorenz_nonaf.fillna({"non_af_fractional_count": 0}).sort_values(
        "non_af_fractional_count"
    )
    _lorenz_nonaf["cum_output_share"] = (
        _lorenz_nonaf["non_af_fractional_count"].cumsum()
        / _lorenz_nonaf["non_af_fractional_count"].sum()
    )
    _lorenz_nonaf["cum_country_share"] = range(1, len(_lorenz_nonaf) + 1)
    _lorenz_nonaf["cum_country_share"] = _lorenz_nonaf["cum_country_share"] / len(
        _lorenz_nonaf
    )
    _lorenz_nonaf["group"] = "non-AF"

    lorenz_curve_data = pd.concat(
        [
            _lorenz_af[
                ["group", "country_code", "cum_country_share", "cum_output_share"]
            ],
            _lorenz_nonaf[
                ["group", "country_code", "cum_country_share", "cum_output_share"]
            ],
        ],
        ignore_index=True,
    )

    def _gini_from_values(_values):
        _arr = pd.Series(_values).fillna(0).sort_values().to_numpy()
        if _arr.sum() == 0:
            return 0.0
        _cum = _arr.cumsum() / _arr.sum()
        _cum = pd.Series([0.0, *_cum])
        _pop = pd.Series(range(0, len(_arr) + 1)) / len(_arr)
        return float(
            1
            - 2 * ((_cum.shift(fill_value=0) + _cum) / 2 * _pop.diff().fillna(0)).sum()
        )

    lorenz_gini_summary = pd.DataFrame(
        {
            "group": ["AF", "non-AF"],
            "gini": [
                _gini_from_values(country_af_nonaf_compare["af_fractional_count"]),
                _gini_from_values(country_af_nonaf_compare["non_af_fractional_count"]),
            ],
        }
    )

    lorenz_gini_summary
    return lorenz_curve_data, lorenz_gini_summary


@app.cell(hide_code=True)
def fig_1_e(
    AF_ANNOTATION_FONT_SIZE,
    AF_CORAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    lorenz_curve_data,
    lorenz_gini_summary,
    plt,
):
    _fig_f8, _ax_f8 = plt.subplots(figsize=(9, 8))

    _ax_f8.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.2,
        label="Line of equality",
    )

    _f8_af = lorenz_curve_data[lorenz_curve_data["group"] == "AF"]
    _f8_nonaf = lorenz_curve_data[lorenz_curve_data["group"] == "non-AF"]

    _ax_f8.plot(
        _f8_af["cum_country_share"],
        _f8_af["cum_output_share"],
        color=AF_CORAL,
        linewidth=3.6,
        label=f"AF (Gini = {lorenz_gini_summary.loc[lorenz_gini_summary['group'] == 'AF', 'gini'].iloc[0]:.3f})",
    )
    _ax_f8.plot(
        _f8_nonaf["cum_country_share"],
        _f8_nonaf["cum_output_share"],
        color=AF_PURPLE,
        linewidth=2.4,
        label=f"non-AF (Gini = {lorenz_gini_summary.loc[lorenz_gini_summary['group'] == 'non-AF', 'gini'].iloc[0]:.3f})",
    )

    # _ax_f8.set_title(
    #     "Figure 8. Lorenz curves of cross-country concentration in AlphaFold vs non-AlphaFold research",
    #     fontsize=15,
    #     pad=14,
    #     loc="left",
    # )
    # _ax_f8.text(
    #     0,
    #     1.01,
    #     "More bowed curves indicate stronger concentration of publication output in a smaller set of countries.",
    #     transform=_ax_f8.transAxes,
    #     fontsize=10,
    #     color=AF_BLUE,
    #     va="bottom",
    # )
    _ax_f8.set_xlabel("Cumulative share of countries", fontsize=AF_LABEL_FONT_SIZE)
    _ax_f8.set_ylabel(
        "Cumulative share of publication output", fontsize=AF_LABEL_FONT_SIZE
    )
    _ax_f8.set_xlim(0, 1)
    _ax_f8.set_ylim(0, 1)
    _ax_f8.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
    _ax_f8.set_axisbelow(True)
    _ax_f8.spines["top"].set_visible(False)
    _ax_f8.spines["right"].set_visible(False)
    _ax_f8.legend(
        frameon=False, loc="upper left", fontsize=AF_ANNOTATION_FONT_SIZE
    )
    _fig_f8.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.11)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    country_first_af_adoption,
    country_income_group_lookup,
    inequality_yearly_base,
    pd,
):
    country_first_af_adoption_income = country_first_af_adoption.copy()
    country_first_af_adoption_income["country_code"] = (
        country_first_af_adoption_income["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    country_first_af_adoption_income = country_first_af_adoption_income.merge(
        country_income_group_lookup[["country_code", "income_group"]],
        on="country_code",
        how="left",
    )
    country_first_af_adoption_income = country_first_af_adoption_income.dropna(
        subset=["income_group"]
    )
    country_first_af_adoption_income["adoption_lag_months"] = (
        country_first_af_adoption_income["first_af_date"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        country_first_af_adoption_income["first_af_date"].dt.month
        - pd.Timestamp("2018-12-02").month
    )

    income_group_country_counts = country_first_af_adoption_income.groupby(
        "income_group", as_index=False
    ).agg(total_countries_in_group=("country_code", "nunique"))

    income_group_adoption_timeline = (
        country_first_af_adoption_income.groupby(
            ["first_af_month", "income_group"], as_index=False
        )
        .agg(new_adopter_countries=("country_code", "nunique"))
        .sort_values(["income_group", "first_af_month"])
    )
    income_group_adoption_timeline = income_group_adoption_timeline.merge(
        income_group_country_counts,
        on="income_group",
        how="left",
    )
    income_group_adoption_timeline["cumulative_adopting_countries"] = (
        income_group_adoption_timeline.groupby("income_group")[
            "new_adopter_countries"
        ].cumsum()
    )
    income_group_adoption_timeline["adoption_share_within_group"] = (
        income_group_adoption_timeline["cumulative_adopting_countries"]
        / income_group_adoption_timeline["total_countries_in_group"]
    )
    income_group_adoption_timeline["event_time_months"] = (
        income_group_adoption_timeline["first_af_month"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        income_group_adoption_timeline["first_af_month"].dt.month
        - pd.Timestamp("2018-12-02").month
    )

    income_group_adoption_lag = country_first_af_adoption_income[
        ["country_code", "income_group", "first_af_date", "adoption_lag_months"]
    ].copy()

    income_group_output_time = inequality_yearly_base[
        inequality_yearly_base["is_alphafold_related"] == True
    ].copy()
    income_group_output_time["country_code"] = (
        income_group_output_time["country_code"].astype(str).str.strip().str.upper()
    )
    income_group_output_time = income_group_output_time.merge(
        country_income_group_lookup[["country_code", "income_group"]],
        on="country_code",
        how="left",
    )
    income_group_output_time = income_group_output_time.dropna(subset=["income_group"])
    income_group_output_time = income_group_output_time.groupby(
        ["publication_year", "income_group"], as_index=False
    ).agg(af_fractional_output=("fractional_weight", "sum"))
    _income_group_year_totals = income_group_output_time.groupby(
        "publication_year", as_index=False
    ).agg(total_af_output=("af_fractional_output", "sum"))
    income_group_output_time = income_group_output_time.merge(
        _income_group_year_totals,
        on="publication_year",
        how="left",
    )
    income_group_output_time["global_af_output_share"] = (
        income_group_output_time["af_fractional_output"]
        / income_group_output_time["total_af_output"]
    )

    income_group_adoption_timeline
    return (
        country_first_af_adoption_income,
        income_group_adoption_lag,
        income_group_adoption_timeline,
        income_group_output_time,
    )


@app.cell(hide_code=True)
def supplementary_fig_4_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    event_calendar,
    income_group_adoption_timeline,
    pd,
    plt,
):
    _fig_f2a, _ax_f2a = plt.subplots(figsize=(11.8, 7.2), dpi=220)
    _f2a_label_font_size = AF_LABEL_FONT_SIZE * 0.8

    _f2a_plot = income_group_adoption_timeline.copy()
    _f2a_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    _f2a_color_map = {
        "High income": AF_PURPLE,
        "Upper-middle income": AF_BLUE,
        "Lower-middle income": AF_CYAN,
        "Low income": AF_CORAL,
    }
    _f2a_legend_label_map = {
        "High income": "High",
        "Upper-middle income": "Upper-middle",
        "Lower-middle income": "Lower-middle",
        "Low income": "Low",
    }
    _f2a_event_time = event_calendar.copy()
    _f2a_event_time["event_time_months"] = (
        _f2a_event_time["event_date"].dt.year - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        _f2a_event_time["event_date"].dt.month - pd.Timestamp("2018-12-02").month
    )
    _f2a_event_time = _f2a_event_time.sort_values("event_time_months").reset_index(
        drop=True
    )
    _f2a_selected_events = _f2a_event_time[
        _f2a_event_time["event_label"].isin(["AF2", "AF3"])
    ].copy()
    _f2a_af2_month = int(
        _f2a_selected_events.loc[
            _f2a_selected_events["event_label"] == "AF2", "event_time_months"
        ].iloc[0]
    )
    _f2a_af3_month = int(
        _f2a_selected_events.loc[
            _f2a_selected_events["event_label"] == "AF3", "event_time_months"
        ].iloc[0]
    )
    _f2a_event_color = AF_EVENT_NEUTRAL
    _f2a_event_text_map = {"AF2": "AF2+AFDB", "AF3": "AF3"}
    _f2a_event_label_y = {"AF2": 0.04, "AF3": 0.12}
    _f2a_x_min = int(_f2a_plot["event_time_months"].min())
    _f2a_x_max = int(_f2a_plot["event_time_months"].max())

    for _group in _f2a_order:
        _group_df = _f2a_plot[_f2a_plot["income_group"] == _group].sort_values(
            "event_time_months"
        )
        if not _group_df.empty:
            _ax_f2a.step(
                _group_df["event_time_months"],
                _group_df["adoption_share_within_group"],
                where="post",
                linewidth=2.6,
                color=_f2a_color_map[_group],
                label=_f2a_legend_label_map[_group],
            )
            _ax_f2a.scatter(
                _group_df["event_time_months"],
                _group_df["adoption_share_within_group"],
                s=18,
                color=_f2a_color_map[_group],
                edgecolors="white",
                linewidth=0.5,
                zorder=3,
            )

    for _row in _f2a_selected_events.itertuples(index=False):
        _ax_f2a.axvline(
            _row.event_time_months,
            color=_f2a_event_color,
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        _ax_f2a.text(
            _row.event_time_months,
            _f2a_event_label_y[_row.event_label],
            _f2a_event_text_map[_row.event_label],
            color=_f2a_event_color,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.9,
            },
            zorder=4,
        )

    _ax_f2a.text(
        (_f2a_x_min + _f2a_af2_month) / 2,
        1.01,
        "Pre-AF2 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )
    _ax_f2a.text(
        (_f2a_af2_month + _f2a_af3_month) / 2,
        1.01,
        "AF2 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )
    _ax_f2a.text(
        (_f2a_af3_month + _f2a_x_max) / 2,
        1.01,
        "AF3 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )

    _ax_f2a.set_xlabel(
        "Months relative to AlphaFold public release at CASP13 (2018-12)",
        fontsize=_f2a_label_font_size,
    )
    _ax_f2a.set_ylabel(
        "Cumulative adopting share within income group",
        fontsize=_f2a_label_font_size,
    )
    _ax_f2a.set_ylim(0, 1.02)
    _ax_f2a.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, color="0.84")
    _ax_f2a.grid(axis="x", visible=False)
    _ax_f2a.set_axisbelow(True)
    _ax_f2a.spines["top"].set_visible(False)
    _ax_f2a.spines["right"].set_visible(False)
    _ax_f2a.legend(
        frameon=False,
        title="Income group",
        loc="lower right",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _fig_f2a.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_2_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    event_calendar,
    pd,
    plt,
    rd_strength_adoption_timeline,
):
    _fig_f2a_rd, _ax_f2a_rd = plt.subplots(figsize=(11.8, 7.2), dpi=220)
    _f2a_rd_label_font_size = AF_LABEL_FONT_SIZE * 0.8

    _f2a_rd_plot = rd_strength_adoption_timeline.copy()
    _f2a_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f2a_rd_color_map = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _f2a_rd_legend_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _f2a_rd_event_time = event_calendar.copy()
    _f2a_rd_event_time["event_time_months"] = (
        _f2a_rd_event_time["event_date"].dt.year - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        _f2a_rd_event_time["event_date"].dt.month
        - pd.Timestamp("2018-12-02").month
    )
    _f2a_rd_event_time = _f2a_rd_event_time.sort_values(
        "event_time_months"
    ).reset_index(drop=True)
    _f2a_rd_selected_events = _f2a_rd_event_time[
        _f2a_rd_event_time["event_label"].isin(["AF2", "AF3"])
    ].copy()
    _f2a_rd_af2_month = int(
        _f2a_rd_selected_events.loc[
            _f2a_rd_selected_events["event_label"] == "AF2", "event_time_months"
        ].iloc[0]
    )
    _f2a_rd_af3_month = int(
        _f2a_rd_selected_events.loc[
            _f2a_rd_selected_events["event_label"] == "AF3", "event_time_months"
        ].iloc[0]
    )
    _f2a_rd_event_color = AF_EVENT_NEUTRAL
    _f2a_rd_event_text_map = {"AF2": "AF2+AFDB", "AF3": "AF3"}
    _f2a_rd_event_label_y = {"AF2": 0.04, "AF3": 0.12}
    _f2a_rd_x_min = int(_f2a_rd_plot["event_time_months"].min())
    _f2a_rd_x_max = int(_f2a_rd_plot["event_time_months"].max())

    for _group in _f2a_rd_order:
        _group_df = _f2a_rd_plot[
            _f2a_rd_plot["rd_strength_tertile"] == _group
        ].sort_values("event_time_months")
        if not _group_df.empty:
            _ax_f2a_rd.step(
                _group_df["event_time_months"],
                _group_df["adoption_share_within_group"],
                where="post",
                linewidth=2.6,
                color=_f2a_rd_color_map[_group],
                label=_f2a_rd_legend_label_map[_group],
            )
            _ax_f2a_rd.scatter(
                _group_df["event_time_months"],
                _group_df["adoption_share_within_group"],
                s=18,
                color=_f2a_rd_color_map[_group],
                edgecolors="white",
                linewidth=0.5,
                zorder=3,
            )

    for _row in _f2a_rd_selected_events.itertuples(index=False):
        _ax_f2a_rd.axvline(
            _row.event_time_months,
            color=_f2a_rd_event_color,
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        _ax_f2a_rd.text(
            _row.event_time_months,
            _f2a_rd_event_label_y[_row.event_label],
            _f2a_rd_event_text_map[_row.event_label],
            color=_f2a_rd_event_color,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.9,
            },
            zorder=4,
        )

    _ax_f2a_rd.text(
        (_f2a_rd_x_min + _f2a_rd_af2_month) / 2,
        1.01,
        "Pre-AF2 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )
    _ax_f2a_rd.text(
        (_f2a_rd_af2_month + _f2a_rd_af3_month) / 2,
        1.01,
        "AF2 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )
    _ax_f2a_rd.text(
        (_f2a_rd_af3_month + _f2a_rd_x_max) / 2,
        1.01,
        "AF3 era",
        ha="center",
        va="bottom",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
    )

    _ax_f2a_rd.set_xlabel(
        "Months relative to AlphaFold public release at CASP13 (2018-12)",
        fontsize=_f2a_rd_label_font_size,
    )
    _ax_f2a_rd.set_ylabel(
        "Cumulative adopting share within R&D-strength group",
        fontsize=_f2a_rd_label_font_size,
    )
    _ax_f2a_rd.set_ylim(0, 1.02)
    _ax_f2a_rd.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, color="0.84")
    _ax_f2a_rd.grid(axis="x", visible=False)
    _ax_f2a_rd.set_axisbelow(True)
    _ax_f2a_rd.spines["top"].set_visible(False)
    _ax_f2a_rd.spines["right"].set_visible(False)
    _ax_f2a_rd.legend(
        frameon=False,
        title="R&D strength",
        loc="lower right",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _fig_f2a_rd.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)

    plt.gca()
    return


@app.cell(hide_code=True)
def supplementary_fig_4_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    income_group_adoption_lag,
    pd,
    plt,
    sns,
):
    _fig_f2b, _ax_f2b = plt.subplots(figsize=(11.2, 7.0), dpi=300)

    _f2b_plot = income_group_adoption_lag.copy()
    _f2b_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    _f2b_plot = _f2b_plot[_f2b_plot["income_group"].isin(_f2b_order)].copy()
    _f2b_plot["income_group"] = pd.Categorical(
        _f2b_plot["income_group"],
        categories=_f2b_order,
        ordered=True,
    )
    _f2b_palette = {
        "High income": AF_PURPLE,
        "Upper-middle income": AF_BLUE,
        "Lower-middle income": AF_CYAN,
        "Low income": AF_GUIDE_NEUTRAL,
    }
    _f2b_tick_labels = ["High", "Upper-middle", "Lower-middle", "Low"]

    sns.boxplot(
        data=_f2b_plot,
        x="income_group",
        y="adoption_lag_months",
        order=_f2b_order,
        width=0.26,
        showcaps=True,
        showfliers=False,
        boxprops={
            "facecolor": "white",
            "alpha": 0.98,
            "edgecolor": AF_PURPLE,
            "linewidth": 1.0,
        },
        whiskerprops={"color": AF_PURPLE, "linewidth": 1.0},
        capprops={"color": AF_PURPLE, "linewidth": 1.0},
        medianprops={"color": AF_GUIDE_NEUTRAL, "linewidth": 1.8},
        ax=_ax_f2b,
    )

    for _box_patch, _group_name in zip(
        _ax_f2b.patches[: len(_f2b_order)], _f2b_order
    ):
        _box_patch.set_facecolor(
            plt.matplotlib.colors.to_rgba(_f2b_palette[_group_name], 0.22)
        )

    sns.stripplot(
        data=_f2b_plot,
        x="income_group",
        y="adoption_lag_months",
        order=_f2b_order,
        hue="income_group",
        palette=_f2b_palette,
        dodge=False,
        size=2.5,
        alpha=0.32,
        jitter=0.16,
        ax=_ax_f2b,
        legend=False,
    )

    _f2b_group_medians = (
        _f2b_plot.groupby("income_group", observed=False)["adoption_lag_months"]
        .median()
        .reindex(_f2b_order)
    )
    _f2b_ymin = float(_f2b_plot["adoption_lag_months"].min())
    _f2b_ymax = float(_f2b_plot["adoption_lag_months"].max())
    _f2b_span = _f2b_ymax - _f2b_ymin
    _f2b_upper_pad = max(4.0, _f2b_span * 0.12)
    _f2b_lower_pad = max(2.0, _f2b_span * 0.06)

    for _idx, (_group, _median) in enumerate(_f2b_group_medians.items()):
        if pd.notna(_median):
            _ax_f2b.text(
                _idx,
                float(_median) + max(0.9, _f2b_span * 0.025),
                f"Median = {float(_median):.1f}",
                ha="center",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=AF_GUIDE_NEUTRAL,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.7,
                },
                zorder=5,
            )

    _f2b_high_median = float(_f2b_group_medians.loc["High income"])
    _f2b_low_median = float(_f2b_group_medians.loc["Low income"])
    _f2b_gap_months = _f2b_low_median - _f2b_high_median
    _f2b_right_x_start = 0.34
    _f2b_right_x_end = 3.88
    _f2b_gap_y = (_f2b_high_median + _f2b_low_median) / 2

    _ax_f2b.hlines(
        y=_f2b_high_median,
        xmin=-0.33,
        xmax=0.33,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=2.6,
        zorder=4,
    )

    _ax_f2b.hlines(
        y=_f2b_high_median,
        xmin=_f2b_right_x_start,
        xmax=_f2b_right_x_end,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=1.4,
        linestyles=(0, (4, 3)),
        zorder=3,
    )

    _ax_f2b.hlines(
        y=_f2b_low_median,
        xmin=2.68,
        xmax=_f2b_right_x_end,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=1.4,
        linestyles=(0, (4, 3)),
        zorder=3,
    )

    _ax_f2b.text(
        3.28,
        _f2b_gap_y,
        f"a {int(round(_f2b_gap_months))}-month\nmedian gap between\nhigh and low-income countries",
        ha="center",
        va="center",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.4,
            "alpha": 0.92,
            "pad": 0.7,
        },
        zorder=6,
    )

    _ax_f2b.axhline(
        0,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
    )
    _ax_f2b.text(
        1.5,
        0.9,
        "Reference point: AlphaFold first public release at CASP13 (2018-12)",
        color=AF_GUIDE_NEUTRAL,
        fontsize=AF_ANNOTATION_FONT_SIZE,
        ha="center",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.4,
            "alpha": 0.9,
            "pad": 0.55,
        },
        zorder=6,
    )

    _ax_f2b.set_xlabel("Income group", fontsize=AF_LABEL_FONT_SIZE)
    _ax_f2b.set_ylabel(
        "Adoption lag (months since 2018-12)", fontsize=AF_LABEL_FONT_SIZE
    )
    _ax_f2b.set_xticks(range(len(_f2b_tick_labels)))
    _ax_f2b.set_xticklabels(_f2b_tick_labels)
    _ax_f2b.set_xlim(-0.55, 3.95)
    _ax_f2b.set_ylim(_f2b_ymin - _f2b_lower_pad, _f2b_ymax + _f2b_upper_pad)
    _ax_f2b.grid(
        axis="y", linestyle=(0, (2, 3)), linewidth=0.7, color=AF_CYAN, alpha=0.9
    )
    _ax_f2b.grid(axis="x", visible=False)
    _ax_f2b.set_axisbelow(True)
    _ax_f2b.spines["top"].set_visible(False)
    _ax_f2b.spines["right"].set_visible(False)
    _ax_f2b.spines["left"].set_linewidth(0.8)
    _ax_f2b.spines["bottom"].set_linewidth(0.8)
    _ax_f2b.spines["left"].set_color("#374151")
    _ax_f2b.spines["bottom"].set_color("#374151")
    _ax_f2b.tick_params(axis="x", rotation=0, labelsize=10.2, colors="#1f2937")
    _ax_f2b.tick_params(axis="y", labelsize=10.0, colors="#1f2937")

    _fig_f2b.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.14)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_2_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    pd,
    plt,
    rd_strength_adoption_lag,
    sns,
):
    _fig_f2b_rd, _ax_f2b_rd = plt.subplots(figsize=(11.2, 7.0), dpi=300)

    _f2b_rd_plot = rd_strength_adoption_lag.copy()
    _f2b_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f2b_rd_plot = _f2b_rd_plot[
        _f2b_rd_plot["rd_strength_tertile"].isin(_f2b_rd_order)
    ].copy()
    _f2b_rd_plot["rd_strength_tertile"] = pd.Categorical(
        _f2b_rd_plot["rd_strength_tertile"],
        categories=_f2b_rd_order,
        ordered=True,
    )
    _f2b_rd_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _f2b_rd_tick_labels = ["High", "Middle", "Low"]

    sns.boxplot(
        data=_f2b_rd_plot,
        x="rd_strength_tertile",
        y="adoption_lag_months",
        order=_f2b_rd_order,
        width=0.26,
        showcaps=True,
        showfliers=False,
        boxprops={
            "facecolor": "white",
            "alpha": 0.98,
            "edgecolor": AF_PURPLE,
            "linewidth": 1.0,
        },
        whiskerprops={"color": AF_PURPLE, "linewidth": 1.0},
        capprops={"color": AF_PURPLE, "linewidth": 1.0},
        medianprops={"color": AF_GUIDE_NEUTRAL, "linewidth": 1.8},
        ax=_ax_f2b_rd,
    )

    for _box_patch, _group_name in zip(
        _ax_f2b_rd.patches[: len(_f2b_rd_order)], _f2b_rd_order
    ):
        _box_patch.set_facecolor(
            plt.matplotlib.colors.to_rgba(_f2b_rd_palette[_group_name], 0.22)
        )

    sns.stripplot(
        data=_f2b_rd_plot,
        x="rd_strength_tertile",
        y="adoption_lag_months",
        order=_f2b_rd_order,
        hue="rd_strength_tertile",
        palette=_f2b_rd_palette,
        dodge=False,
        size=2.5,
        alpha=0.32,
        jitter=0.16,
        ax=_ax_f2b_rd,
        legend=False,
    )

    _f2b_rd_group_medians = (
        _f2b_rd_plot.groupby("rd_strength_tertile", observed=False)[
            "adoption_lag_months"
        ]
        .median()
        .reindex(_f2b_rd_order)
    )
    _f2b_rd_ymin = float(_f2b_rd_plot["adoption_lag_months"].min())
    _f2b_rd_ymax = float(_f2b_rd_plot["adoption_lag_months"].max())
    _f2b_rd_span = _f2b_rd_ymax - _f2b_rd_ymin
    _f2b_rd_upper_pad = max(4.0, _f2b_rd_span * 0.12)
    _f2b_rd_lower_pad = max(2.0, _f2b_rd_span * 0.06)

    for _idx, (_group, _median) in enumerate(_f2b_rd_group_medians.items()):
        if pd.notna(_median):
            _ax_f2b_rd.text(
                _idx,
                float(_median) + max(0.9, _f2b_rd_span * 0.025),
                f"Median = {float(_median):.1f}",
                ha="center",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=AF_GUIDE_NEUTRAL,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.7,
                },
                zorder=5,
            )

    _f2b_rd_high_median = float(_f2b_rd_group_medians.loc["High R&D strength"])
    _f2b_rd_low_median = float(_f2b_rd_group_medians.loc["Low R&D strength"])
    _f2b_rd_gap_months = _f2b_rd_low_median - _f2b_rd_high_median
    _f2b_rd_right_x_start = 0.34
    _f2b_rd_right_x_end = 2.88
    _f2b_rd_gap_y = (_f2b_rd_high_median + _f2b_rd_low_median) / 2

    _ax_f2b_rd.hlines(
        y=_f2b_rd_high_median,
        xmin=-0.33,
        xmax=0.33,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=2.6,
        zorder=4,
    )

    _ax_f2b_rd.hlines(
        y=_f2b_rd_high_median,
        xmin=_f2b_rd_right_x_start,
        xmax=_f2b_rd_right_x_end,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=1.4,
        linestyles=(0, (4, 3)),
        zorder=3,
    )

    _ax_f2b_rd.hlines(
        y=_f2b_rd_low_median,
        xmin=1.68,
        xmax=_f2b_rd_right_x_end,
        colors=AF_GUIDE_NEUTRAL,
        linewidth=1.4,
        linestyles=(0, (4, 3)),
        zorder=3,
    )

    _ax_f2b_rd.text(
        2.48,
        _f2b_rd_gap_y,
        f"a {int(round(_f2b_rd_gap_months))}-month\nmedian gap between\nhigh and low R&D-strength countries",
        ha="center",
        va="center",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.4,
            "alpha": 0.92,
            "pad": 0.7,
        },
        zorder=6,
    )

    _ax_f2b_rd.axhline(
        0,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
    )
    _ax_f2b_rd.text(
        1.0,
        0.9,
        "Reference point: AlphaFold first public release at CASP13 (2018-12)",
        color=AF_GUIDE_NEUTRAL,
        fontsize=AF_ANNOTATION_FONT_SIZE,
        ha="center",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.4,
            "alpha": 0.9,
            "pad": 0.55,
        },
        zorder=6,
    )

    # _ax_f2b_rd.text(
    #     0.01,
    #     1.035,
    #     "R&D strength version",
    #     transform=_ax_f2b_rd.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=11.0,
    #     color=AF_PURPLE,
    #     fontweight="semibold",
    # )

    _ax_f2b_rd.set_xlabel("R&D strength group", fontsize=AF_LABEL_FONT_SIZE)
    _ax_f2b_rd.set_ylabel(
        "Adoption lag (months since 2018-12)", fontsize=AF_LABEL_FONT_SIZE
    )
    _ax_f2b_rd.set_xticks(range(len(_f2b_rd_tick_labels)))
    _ax_f2b_rd.set_xticklabels(_f2b_rd_tick_labels)
    _ax_f2b_rd.set_xlim(-0.55, 2.95)
    _ax_f2b_rd.set_ylim(
        _f2b_rd_ymin - _f2b_rd_lower_pad, _f2b_rd_ymax + _f2b_rd_upper_pad
    )
    _ax_f2b_rd.grid(
        axis="y", linestyle=(0, (2, 3)), linewidth=0.7, color=AF_CYAN, alpha=0.9
    )
    _ax_f2b_rd.grid(axis="x", visible=False)
    _ax_f2b_rd.set_axisbelow(True)
    _ax_f2b_rd.spines["top"].set_visible(False)
    _ax_f2b_rd.spines["right"].set_visible(False)
    _ax_f2b_rd.spines["left"].set_linewidth(0.8)
    _ax_f2b_rd.spines["bottom"].set_linewidth(0.8)
    _ax_f2b_rd.spines["left"].set_color("#374151")
    _ax_f2b_rd.spines["bottom"].set_color("#374151")
    _ax_f2b_rd.tick_params(axis="x", rotation=0, labelsize=10.2, colors="#1f2937")
    _ax_f2b_rd.tick_params(axis="y", labelsize=10.0, colors="#1f2937")

    _fig_f2b_rd.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.14)

    plt.gca()
    return


@app.cell(hide_code=True)
def supplementary_fig_4_c(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    income_group_output_time,
    plt,
):
    _f2c2_income_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    _f2c2_income_color_map = {
        "High income": AF_PURPLE,
        "Upper-middle income": AF_BLUE,
        "Lower-middle income": AF_CYAN,
        "Low income": AF_CORAL,
    }
    _f2c2_income_label_map = {
        "High income": "High",
        "Upper-middle income": "Upper-middle",
        "Lower-middle income": "Lower-middle",
        "Low income": "Low",
    }

    figure_f2c2_income_growth = income_group_output_time[
        income_group_output_time["income_group"].isin(_f2c2_income_order)
    ].copy()
    figure_f2c2_income_growth = (
        figure_f2c2_income_growth.groupby(
            ["publication_year", "income_group"],
            as_index=False,
            observed=False,
        )["af_fractional_output"]
        .sum()
        .rename(columns={"af_fractional_output": "annual_af_output"})
    )

    _f2c2_income_years = sorted(
        figure_f2c2_income_growth["publication_year"].unique().tolist()
    )
    figure_f2c2_income_growth = (
        figure_f2c2_income_growth.pivot_table(
            index="publication_year",
            columns="income_group",
            values="annual_af_output",
            fill_value=0,
        )
        .reindex(
            index=_f2c2_income_years, columns=_f2c2_income_order, fill_value=0
        )
        .sort_index()
        .reset_index()
    )
    figure_f2c2_income_growth.columns.name = None
    for _group in _f2c2_income_order:
        figure_f2c2_income_growth[f"{_group} cumulative"] = (
            figure_f2c2_income_growth[_group].cumsum()
        )

    _fig_f2c2_income, _ax_f2c2_income = plt.subplots(figsize=(11.8, 7.2), dpi=220)

    _f2c2_income_af2_year = 2021
    _f2c2_income_af3_year = 2024
    _f2c2_income_year_values = figure_f2c2_income_growth[
        "publication_year"
    ].to_numpy()

    for _group in _f2c2_income_order:
        _ax_f2c2_income.plot(
            _f2c2_income_year_values,
            figure_f2c2_income_growth[f"{_group} cumulative"],
            color=_f2c2_income_color_map[_group],
            linewidth=2.8,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=_f2c2_income_label_map[_group],
            zorder=4,
        )

    _f2c2_income_ymax = max(
        float(figure_f2c2_income_growth["High income cumulative"].max()),
        float(figure_f2c2_income_growth["Upper-middle income cumulative"].max()),
        float(figure_f2c2_income_growth["Lower-middle income cumulative"].max()),
        float(figure_f2c2_income_growth["Low income cumulative"].max()),
    )
    _ax_f2c2_income.set_ylim(0, _f2c2_income_ymax * 1.12)

    for _event_year, _event_label in [
        (_f2c2_income_af2_year, "AF2+AFDB"),
        (_f2c2_income_af3_year, "AF3"),
    ]:
        if _event_year in _f2c2_income_year_values:
            _ax_f2c2_income.axvline(
                _event_year,
                color=AF_EVENT_NEUTRAL,
                linestyle=(0, (4, 4)),
                linewidth=1.15,
                alpha=0.75,
                zorder=1,
            )
            _ax_f2c2_income.text(
                _event_year,
                _f2c2_income_ymax * 1.03,
                _event_label,
                ha="center",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=AF_EVENT_NEUTRAL,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.45,
                },
                clip_on=False,
                zorder=5,
            )

    _ax_f2c2_income.annotate(
        "High-income countries remain far above the others",
        xy=(
            _f2c2_income_year_values[-1],
            figure_f2c2_income_growth["High income cumulative"].iloc[-1],
        ),
        xytext=(
            _f2c2_income_year_values[max(len(_f2c2_income_year_values) - 3, 0)]
            - 0.15,
            figure_f2c2_income_growth["High income cumulative"].iloc[-1] * 0.77,
        ),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.45,
        },
        zorder=6,
    )

    _ax_f2c2_income.annotate(
        "Upper-middle-income countries account for most of the catch-up after AF2",
        xy=(
            2022,
            float(
                figure_f2c2_income_growth.loc[
                    figure_f2c2_income_growth["publication_year"] == 2022,
                    "Upper-middle income cumulative",
                ].iloc[0]
            ),
        ),
        xytext=(2019.55, _f2c2_income_ymax * 0.32),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.45,
        },
        zorder=6,
    )

    _ax_f2c2_income.set_xlim(
        _f2c2_income_year_values.min() - 0.15,
        _f2c2_income_year_values.max() + 0.25,
    )
    _ax_f2c2_income.set_xticks(_f2c2_income_year_values)
    _ax_f2c2_income.set_xlabel(
        "Publication year",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=9,
        color="#1f2937",
    )
    _ax_f2c2_income.set_ylabel(
        "Cumulative AF-related publications",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2c2_income.yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    )
    _ax_f2c2_income.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#d1d5db",
        alpha=0.9,
    )
    _ax_f2c2_income.grid(axis="x", visible=False)
    _ax_f2c2_income.set_axisbelow(True)
    _ax_f2c2_income.spines["top"].set_visible(False)
    _ax_f2c2_income.spines["right"].set_visible(False)
    _ax_f2c2_income.spines["left"].set_color("#374151")
    _ax_f2c2_income.spines["bottom"].set_color("#374151")
    _ax_f2c2_income.spines["left"].set_linewidth(0.8)
    _ax_f2c2_income.spines["bottom"].set_linewidth(0.8)
    _ax_f2c2_income.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_f2c2_income.legend(
        frameon=False,
        title="Income group",
        loc="upper left",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _fig_f2c2_income.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.14)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_2_c(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_rd_strength_lookup,
    inequality_yearly_base,
    plt,
):
    _f2c2_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f2c2_rd_color_map = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _f2c2_rd_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }

    figure_f2c2_rd_growth = inequality_yearly_base[
        inequality_yearly_base["is_alphafold_related"] == True
    ].merge(
        country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
        on="country_code",
        how="left",
    )
    figure_f2c2_rd_growth = figure_f2c2_rd_growth[
        figure_f2c2_rd_growth["rd_strength_tertile"].isin(_f2c2_rd_order)
    ].copy()
    figure_f2c2_rd_growth = (
        figure_f2c2_rd_growth.groupby(
            ["publication_year", "rd_strength_tertile"],
            as_index=False,
            observed=False,
        )["fractional_weight"]
        .sum()
        .rename(columns={"fractional_weight": "annual_af_output"})
    )

    _f2c2_rd_years = sorted(
        figure_f2c2_rd_growth["publication_year"].unique().tolist()
    )
    figure_f2c2_rd_growth = (
        figure_f2c2_rd_growth.pivot_table(
            index="publication_year",
            columns="rd_strength_tertile",
            values="annual_af_output",
            fill_value=0,
        )
        .reindex(index=_f2c2_rd_years, columns=_f2c2_rd_order, fill_value=0)
        .sort_index()
        .reset_index()
    )
    figure_f2c2_rd_growth.columns.name = None
    for _group in _f2c2_rd_order:
        figure_f2c2_rd_growth[f"{_group} cumulative"] = figure_f2c2_rd_growth[
            _group
        ].cumsum()

    _fig_f2c2_rd, _ax_f2c2_rd = plt.subplots(figsize=(11.8, 7.2), dpi=220)

    _f2c2_rd_af2_year = 2021
    _f2c2_rd_af3_year = 2024
    _f2c2_rd_year_values = figure_f2c2_rd_growth["publication_year"].to_numpy()

    for _group in _f2c2_rd_order:
        _ax_f2c2_rd.plot(
            _f2c2_rd_year_values,
            figure_f2c2_rd_growth[f"{_group} cumulative"],
            color=_f2c2_rd_color_map[_group],
            linewidth=2.8,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=_f2c2_rd_label_map[_group],
            zorder=4,
        )

    _f2c2_rd_ymax = max(
        float(figure_f2c2_rd_growth["High R&D strength cumulative"].max()),
        float(figure_f2c2_rd_growth["Middle R&D strength cumulative"].max()),
        float(figure_f2c2_rd_growth["Low R&D strength cumulative"].max()),
    )
    _ax_f2c2_rd.set_ylim(0, _f2c2_rd_ymax * 1.12)

    for _event_year, _event_label in [
        (_f2c2_rd_af2_year, "AF2+AFDB"),
        (_f2c2_rd_af3_year, "AF3"),
    ]:
        if _event_year in _f2c2_rd_year_values:
            _ax_f2c2_rd.axvline(
                _event_year,
                color=AF_EVENT_NEUTRAL,
                linestyle=(0, (4, 4)),
                linewidth=1.15,
                alpha=0.75,
                zorder=1,
            )
            _ax_f2c2_rd.text(
                _event_year,
                _f2c2_rd_ymax * 1.03,
                _event_label,
                ha="center",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=AF_EVENT_NEUTRAL,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.45,
                },
                clip_on=False,
                zorder=5,
            )

    _ax_f2c2_rd.annotate(
        "High group stays far above the others",
        xy=(
            _f2c2_rd_year_values[-1],
            figure_f2c2_rd_growth["High R&D strength cumulative"].iloc[-1],
        ),
        xytext=(
            _f2c2_rd_year_values[max(len(_f2c2_rd_year_values) - 3, 0)] - 0.15,
            figure_f2c2_rd_growth["High R&D strength cumulative"].iloc[-1] * 0.78,
        ),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.45,
        },
        zorder=6,
    )

    _ax_f2c2_rd.annotate(
        "Post-AF2 slope is steepest for the high-R&D group",
        xy=(
            2022,
            float(
                figure_f2c2_rd_growth.loc[
                    figure_f2c2_rd_growth["publication_year"] == 2022,
                    "High R&D strength cumulative",
                ].iloc[0]
            ),
        ),
        xytext=(2019.55, _f2c2_rd_ymax * 0.40),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.45,
        },
        zorder=6,
    )

    _ax_f2c2_rd.set_xlim(
        _f2c2_rd_year_values.min() - 0.15, _f2c2_rd_year_values.max() + 0.25
    )
    _ax_f2c2_rd.set_xticks(_f2c2_rd_year_values)
    _ax_f2c2_rd.set_xlabel(
        "Publication year",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=9,
        color="#1f2937",
    )
    _ax_f2c2_rd.set_ylabel(
        "Cumulative AF-related publications",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2c2_rd.yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    )
    _ax_f2c2_rd.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#d1d5db",
        alpha=0.9,
    )
    _ax_f2c2_rd.grid(axis="x", visible=False)
    _ax_f2c2_rd.set_axisbelow(True)
    _ax_f2c2_rd.spines["top"].set_visible(False)
    _ax_f2c2_rd.spines["right"].set_visible(False)
    _ax_f2c2_rd.spines["left"].set_color("#374151")
    _ax_f2c2_rd.spines["bottom"].set_color("#374151")
    _ax_f2c2_rd.spines["left"].set_linewidth(0.8)
    _ax_f2c2_rd.spines["bottom"].set_linewidth(0.8)
    _ax_f2c2_rd.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_f2c2_rd.legend(
        frameon=False,
        title="R&D strength",
        loc="upper left",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _fig_f2c2_rd.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.14)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    country_af_nonaf_compare,
    country_first_af_adoption_income,
    country_income_group_lookup,
    inequality_yearly_base,
    pd,
):
    country_gap_explainer_output = (
        inequality_yearly_base[
            (inequality_yearly_base["publication_date"] >= pd.Timestamp("2019-01-01"))
            & (inequality_yearly_base["publication_date"] < pd.Timestamp("2021-07-01"))
            & (inequality_yearly_base["is_alphafold_related"] == False)
        ]
        .groupby("country_code", as_index=False)
        .agg(pre_af2_non_af_fractional_output=("fractional_weight", "sum"))
    )

    country_gap_explainer_output["log_pre_af2_non_af_fractional_output"] = (
        country_gap_explainer_output["pre_af2_non_af_fractional_output"] + 1
    ).map(lambda _v: __import__("math").log10(_v))

    country_gap_explainer_df = country_first_af_adoption_income[
        ["country_code", "income_group", "first_af_date", "adoption_lag_months"]
    ].copy()
    country_gap_explainer_df["country_code"] = (
        country_gap_explainer_df["country_code"].astype(str).str.strip().str.upper()
    )

    country_gap_explainer_df = country_gap_explainer_df.merge(
        country_gap_explainer_output,
        on="country_code",
        how="left",
    )
    country_gap_explainer_df = country_gap_explainer_df.merge(
        country_af_nonaf_compare[
            ["country_code", "country_name", "af_fractional_count"]
        ],
        on="country_code",
        how="left",
    )
    country_gap_explainer_df = country_gap_explainer_df.merge(
        country_income_group_lookup[["country_code", "income_group"]].rename(
            columns={"income_group": "income_group_lookup"}
        ),
        on="country_code",
        how="left",
    )
    country_gap_explainer_df["income_group"] = country_gap_explainer_df[
        "income_group"
    ].fillna(country_gap_explainer_df["income_group_lookup"])
    country_gap_explainer_df = country_gap_explainer_df.drop(
        columns=["income_group_lookup"]
    )
    country_gap_explainer_df = country_gap_explainer_df.dropna(
        subset=[
            "income_group",
            "adoption_lag_months",
            "pre_af2_non_af_fractional_output",
            "af_fractional_count",
        ]
    )
    country_gap_explainer_df["bubble_size"] = 40 + 18 * country_gap_explainer_df[
        "af_fractional_count"
    ].map(lambda _v: __import__("math").sqrt(_v) if _v >= 0 else 0)
    country_gap_explainer_df
    return country_gap_explainer_df, country_gap_explainer_output


@app.cell(hide_code=True)
def supplementary_fig_4_d(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_gap_explainer_df,
    plt,
):
    figure_f2d2_income_conversion_df = country_gap_explainer_df.copy()

    _f2d2_income_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    _f2d2_income_palette = {
        "High income": AF_PURPLE,
        "Upper-middle income": AF_BLUE,
        "Lower-middle income": AF_CYAN,
        "Low income": AF_CORAL,
    }
    _f2d2_income_label_map = {
        "High income": "High",
        "Upper-middle income": "Upper-middle",
        "Lower-middle income": "Lower-middle",
        "Low income": "Low",
    }
    _f2d2_income_plot = figure_f2d2_income_conversion_df[
        figure_f2d2_income_conversion_df["income_group"].isin(_f2d2_income_order)
    ].copy()
    _f2d2_income_plot["base_bubble_size"] = 40 + 18 * _f2d2_income_plot[
        "pre_af2_non_af_fractional_output"
    ].clip(lower=0).map(lambda _v: __import__("math").sqrt(_v / 1000))
    _f2d2_income_plot["log_af_output"] = __import__("numpy").log10(
        _f2d2_income_plot["af_fractional_count"].clip(lower=1)
    )

    _fig_f2d2_income, _ax_f2d2_income = plt.subplots(figsize=(11.8, 7.5), dpi=260)

    for _group_f2d2 in _f2d2_income_order:
        _group_df_f2d2 = _f2d2_income_plot[
            _f2d2_income_plot["income_group"] == _group_f2d2
        ]
        if not _group_df_f2d2.empty:
            _ax_f2d2_income.scatter(
                _group_df_f2d2["adoption_lag_months"],
                _group_df_f2d2["af_fractional_count"],
                s=_group_df_f2d2["base_bubble_size"],
                color=_f2d2_income_palette[_group_f2d2],
                alpha=0.84,
                edgecolors="white",
                linewidth=0.75,
                label=_f2d2_income_label_map[_group_f2d2],
                zorder=3,
            )

    _f2d2_income_x = (
        _f2d2_income_plot["adoption_lag_months"].astype(float).to_numpy()
    )
    _f2d2_income_y_log = (
        _f2d2_income_plot["log_af_output"].astype(float).to_numpy()
    )
    _f2d2_income_coef = __import__("numpy").polyfit(
        _f2d2_income_x, _f2d2_income_y_log, 1
    )
    _f2d2_income_x_line = __import__("numpy").linspace(
        float(_f2d2_income_x.min()), float(_f2d2_income_x.max()), 240
    )
    _f2d2_income_y_line = 10 ** (
        _f2d2_income_coef[0] * _f2d2_income_x_line + _f2d2_income_coef[1]
    )
    _ax_f2d2_income.plot(
        _f2d2_income_x_line,
        _f2d2_income_y_line,
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.7,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    _f2d2_income_corr = __import__("scipy.stats").stats.pearsonr(
        _f2d2_income_x, _f2d2_income_y_log
    )
    _f2d2_income_r = float(_f2d2_income_corr.statistic)
    _f2d2_income_p = float(_f2d2_income_corr.pvalue)
    _f2d2_income_r2 = _f2d2_income_r**2
    _f2d2_income_slope = float(_f2d2_income_coef[0])

    _f2d2_income_label_pool = _f2d2_income_plot.copy()
    _f2d2_income_label_pool["label_priority"] = 0
    _f2d2_income_label_pool.loc[
        _f2d2_income_label_pool["income_group"] == "High income",
        "label_priority",
    ] += 1
    _f2d2_income_label_pool.loc[
        _f2d2_income_label_pool["af_fractional_count"].rank(
            method="min", ascending=False
        )
        <= 12,
        "label_priority",
    ] += 2
    _f2d2_income_label_pool.loc[
        _f2d2_income_label_pool["adoption_lag_months"].rank(
            method="min", ascending=True
        )
        <= 12,
        "label_priority",
    ] += 1
    _f2d2_income_label_pool = _f2d2_income_label_pool.sort_values(
        [
            "label_priority",
            "af_fractional_count",
            "pre_af2_non_af_fractional_output",
        ],
        ascending=[False, False, False],
    ).head(14)

    for _row_f2d2 in _f2d2_income_label_pool.itertuples(index=False):
        _dx_f2d2 = (
            -0.8
            if _row_f2d2.adoption_lag_months
            >= _f2d2_income_plot["adoption_lag_months"].median()
            else 0.8
        )
        _dy_f2d2 = (
            1.09
            if _row_f2d2.af_fractional_count
            <= _f2d2_income_plot["af_fractional_count"].median()
            else 0.93
        )
        _ax_f2d2_income.text(
            float(_row_f2d2.adoption_lag_months) + _dx_f2d2,
            float(_row_f2d2.af_fractional_count) * _dy_f2d2,
            _row_f2d2.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            ha="right" if _dx_f2d2 < 0 else "left",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 0.2,
            },
            zorder=5,
        )

    _ax_f2d2_income.text(
        0.985,
        0.985,
        f"log-fit slope = {_f2d2_income_slope:.3f}; R^2 = {_f2d2_income_r2:.3f}; p = {_f2d2_income_p:.3g}",
        transform=_ax_f2d2_income.transAxes,
        ha="right",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.6,
            "alpha": 0.92,
            "pad": 0.9,
        },
        zorder=6,
    )

    _f2d2_income_cn_row = _f2d2_income_plot[
        _f2d2_income_plot["country_code"] == "CN"
    ]
    if not _f2d2_income_cn_row.empty:
        _f2d2_income_cn_x = float(
            _f2d2_income_cn_row["adoption_lag_months"].iloc[0]
        )
        _f2d2_income_cn_y = float(
            _f2d2_income_cn_row["af_fractional_count"].iloc[0]
        )
        _ax_f2d2_income.annotate(
            "Earlier adopters tend to convert faster\ninto larger AlphaFold-related output",
            xy=(_f2d2_income_cn_x, _f2d2_income_cn_y),
            xytext=(
                7.2,
                float(_f2d2_income_plot["af_fractional_count"].quantile(0.54)),
            ),
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 0.45,
            },
            zorder=6,
        )

    _ax_f2d2_income.set_xlabel(
        "Adoption lag (months since 2018-12)",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2d2_income.set_ylabel(
        "Post-adoption AF-related output",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2d2_income.set_yscale("log")
    _ax_f2d2_income.grid(
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color="#d1d5db",
        alpha=0.9,
    )
    _ax_f2d2_income.set_axisbelow(True)
    _ax_f2d2_income.spines["top"].set_visible(False)
    _ax_f2d2_income.spines["right"].set_visible(False)
    _ax_f2d2_income.spines["left"].set_color("#374151")
    _ax_f2d2_income.spines["bottom"].set_color("#374151")
    _ax_f2d2_income.spines["left"].set_linewidth(0.8)
    _ax_f2d2_income.spines["bottom"].set_linewidth(0.8)
    _ax_f2d2_income.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _f2d2_income_color_legend = _ax_f2d2_income.legend(
        frameon=False,
        title="Income group",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.86),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        title_fontsize=AF_ANNOTATION_FONT_SIZE,
    )

    _f2d2_income_size_handles = [
        plt.scatter(
            [], [], s=_size, color="#9ca3af", alpha=0.55, edgecolors="none"
        )
        for _size in [90, 180, 320]
    ]
    _f2d2_income_size_legend = _ax_f2d2_income.legend(
        _f2d2_income_size_handles,
        ["Smaller base", "Mid base", "Larger base"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        title="Pre-AF2 research base",
        title_fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _ax_f2d2_income.add_artist(_f2d2_income_color_legend)
    _ax_f2d2_income.add_artist(_f2d2_income_size_legend)

    _fig_f2d2_income.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.12)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_2_d(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    figure_2_rd_strength_adoption_df,
    plt,
):
    figure_f2d2_rd_conversion_df = figure_2_rd_strength_adoption_df.copy()

    _f2d2_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f2d2_rd_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _f2d2_rd_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _f2d2_rd_plot = figure_f2d2_rd_conversion_df[
        figure_f2d2_rd_conversion_df["rd_strength_tertile"].isin(_f2d2_rd_order)
    ].copy()
    _f2d2_rd_plot["base_bubble_size"] = 40 + 18 * _f2d2_rd_plot[
        "pre_af_non_af_fractional_output_2015_2018"
    ].clip(lower=0).map(lambda _v: __import__("math").sqrt(_v / 1000))
    _f2d2_rd_plot["log_af_output"] = __import__("numpy").log10(
        _f2d2_rd_plot["af_fractional_count"].clip(lower=1)
    )

    _fig_f2d2_rd, _ax_f2d2_rd = plt.subplots(figsize=(11.8, 7.5), dpi=260)

    for _group_f2d2 in _f2d2_rd_order:
        _group_df_f2d2 = _f2d2_rd_plot[
            _f2d2_rd_plot["rd_strength_tertile"] == _group_f2d2
        ]
        if not _group_df_f2d2.empty:
            _ax_f2d2_rd.scatter(
                _group_df_f2d2["adoption_lag_months"],
                _group_df_f2d2["af_fractional_count"],
                s=_group_df_f2d2["base_bubble_size"],
                color=_f2d2_rd_palette[_group_f2d2],
                alpha=0.84,
                edgecolors="white",
                linewidth=0.75,
                label=_f2d2_rd_label_map[_group_f2d2],
                zorder=3,
            )

    _f2d2_x = _f2d2_rd_plot["adoption_lag_months"].astype(float).to_numpy()
    _f2d2_y_log = _f2d2_rd_plot["log_af_output"].astype(float).to_numpy()
    _f2d2_coef = __import__("numpy").polyfit(_f2d2_x, _f2d2_y_log, 1)
    _f2d2_x_line = __import__("numpy").linspace(
        float(_f2d2_x.min()), float(_f2d2_x.max()), 240
    )
    _f2d2_y_line = 10 ** (_f2d2_coef[0] * _f2d2_x_line + _f2d2_coef[1])
    _ax_f2d2_rd.plot(
        _f2d2_x_line,
        _f2d2_y_line,
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.7,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    _f2d2_corr = __import__("scipy.stats").stats.pearsonr(_f2d2_x, _f2d2_y_log)
    _f2d2_r = float(_f2d2_corr.statistic)
    _f2d2_p = float(_f2d2_corr.pvalue)
    _f2d2_r2 = _f2d2_r**2
    _f2d2_slope = float(_f2d2_coef[0])

    _f2d2_label_pool = _f2d2_rd_plot.copy()
    _f2d2_label_pool["label_priority"] = 0
    _f2d2_label_pool.loc[
        _f2d2_label_pool["rd_strength_tertile"] == "High R&D strength",
        "label_priority",
    ] += 1
    _f2d2_label_pool.loc[
        _f2d2_label_pool["af_fractional_count"].rank(method="min", ascending=False)
        <= 10,
        "label_priority",
    ] += 2
    _f2d2_label_pool.loc[
        _f2d2_label_pool["adoption_lag_months"].rank(method="min", ascending=True)
        <= 10,
        "label_priority",
    ] += 1
    _f2d2_label_pool = _f2d2_label_pool.sort_values(
        [
            "label_priority",
            "af_fractional_count",
            "pre_af_non_af_fractional_output_2015_2018",
        ],
        ascending=[False, False, False],
    ).head(12)

    for _row_f2d2 in _f2d2_label_pool.itertuples(index=False):
        _dx_f2d2 = (
            -0.8
            if _row_f2d2.adoption_lag_months
            >= _f2d2_rd_plot["adoption_lag_months"].median()
            else 0.8
        )
        _dy_f2d2 = (
            1.09
            if _row_f2d2.af_fractional_count
            <= _f2d2_rd_plot["af_fractional_count"].median()
            else 0.93
        )
        _ax_f2d2_rd.text(
            float(_row_f2d2.adoption_lag_months) + _dx_f2d2,
            float(_row_f2d2.af_fractional_count) * _dy_f2d2,
            _row_f2d2.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            ha="right" if _dx_f2d2 < 0 else "left",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 0.2,
            },
            zorder=5,
        )

    _ax_f2d2_rd.text(
        0.985,
        0.985,
        f"log-fit slope = {_f2d2_slope:.3f}; R² = {_f2d2_r2:.3f}; p = {_f2d2_p:.3g}",
        transform=_ax_f2d2_rd.transAxes,
        ha="right",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#1f2937",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.6,
            "alpha": 0.92,
            "pad": 0.9,
        },
        zorder=6,
    )

    _f2d2_cn_row = _f2d2_rd_plot[_f2d2_rd_plot["country_code"] == "CN"]
    if not _f2d2_cn_row.empty:
        _f2d2_cn_x = float(_f2d2_cn_row["adoption_lag_months"].iloc[0])
        _f2d2_cn_y = float(_f2d2_cn_row["af_fractional_count"].iloc[0])
        _ax_f2d2_rd.annotate(
            "Earlier adopters tend to convert faster\ninto larger AlphaFold-related output",
            xy=(_f2d2_cn_x, _f2d2_cn_y),
            xytext=(
                7.1,
                float(_f2d2_rd_plot["af_fractional_count"].quantile(0.52)),
            ),
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            arrowprops={"arrowstyle": "->", "color": AF_GUIDE_NEUTRAL, "lw": 1.0},
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 0.45,
            },
            zorder=6,
        )

    _ax_f2d2_rd.set_xlabel(
        "Adoption lag (months since 2018-12)",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2d2_rd.set_ylabel(
        "Post-adoption AF-related output",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f2d2_rd.set_yscale("log")
    _ax_f2d2_rd.grid(
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color="#d1d5db",
        alpha=0.9,
    )
    _ax_f2d2_rd.set_axisbelow(True)
    _ax_f2d2_rd.spines["top"].set_visible(False)
    _ax_f2d2_rd.spines["right"].set_visible(False)
    _ax_f2d2_rd.spines["left"].set_color("#374151")
    _ax_f2d2_rd.spines["bottom"].set_color("#374151")
    _ax_f2d2_rd.spines["left"].set_linewidth(0.8)
    _ax_f2d2_rd.spines["bottom"].set_linewidth(0.8)
    _ax_f2d2_rd.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _f2d2_color_legend = _ax_f2d2_rd.legend(
        frameon=False,
        title="R&D strength",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.86),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        title_fontsize=AF_ANNOTATION_FONT_SIZE,
    )

    _f2d2_size_handles = [
        plt.scatter(
            [], [], s=_size, color="#9ca3af", alpha=0.55, edgecolors="none"
        )
        for _size in [90, 180, 320]
    ]
    _f2d2_size_legend = _ax_f2d2_rd.legend(
        _f2d2_size_handles,
        ["Smaller base", "Mid base", "Larger base"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        title="Pre-AF2 research base",
        title_fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _ax_f2d2_rd.add_artist(_f2d2_color_legend)
    _ax_f2d2_rd.add_artist(_f2d2_size_legend)

    _fig_f2d2_rd.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.12)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(country_first_af_adoption, country_rd_strength_lookup, pd):
    country_first_af_adoption_rd = country_first_af_adoption.copy()
    country_first_af_adoption_rd["country_code"] = (
        country_first_af_adoption_rd["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    country_first_af_adoption_rd = country_first_af_adoption_rd.merge(
        country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
        on="country_code",
        how="left",
    )
    country_first_af_adoption_rd = country_first_af_adoption_rd.dropna(
        subset=["rd_strength_tertile"]
    ).copy()

    rd_strength_country_counts = country_first_af_adoption_rd.groupby(
        "rd_strength_tertile", as_index=False
    ).agg(total_countries_in_group=("country_code", "nunique"))

    rd_strength_adoption_timeline = (
        country_first_af_adoption_rd.groupby(
            ["first_af_month", "rd_strength_tertile"], as_index=False
        )
        .agg(new_adopter_countries=("country_code", "nunique"))
        .sort_values(["rd_strength_tertile", "first_af_month"])
    )
    rd_strength_adoption_timeline = rd_strength_adoption_timeline.merge(
        rd_strength_country_counts,
        on="rd_strength_tertile",
        how="left",
    )
    rd_strength_adoption_timeline["cumulative_adopting_countries"] = (
        rd_strength_adoption_timeline.groupby("rd_strength_tertile")[
            "new_adopter_countries"
        ].cumsum()
    )
    rd_strength_adoption_timeline["adoption_share_within_group"] = (
        rd_strength_adoption_timeline["cumulative_adopting_countries"]
        / rd_strength_adoption_timeline["total_countries_in_group"]
    )
    rd_strength_adoption_timeline["event_time_months"] = (
        rd_strength_adoption_timeline["first_af_month"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        rd_strength_adoption_timeline["first_af_month"].dt.month
        - pd.Timestamp("2018-12-02").month
    )
    rd_strength_adoption_timeline = rd_strength_adoption_timeline.sort_values(
        ["rd_strength_tertile", "event_time_months"]
    ).reset_index(drop=True)

    rd_strength_adoption_timeline
    return (
        country_first_af_adoption_rd,
        rd_strength_adoption_timeline,
        rd_strength_country_counts,
    )


@app.cell(hide_code=True)
def _(country_first_af_adoption, country_rd_strength_lookup, pd):
    rd_strength_adoption_lag = country_first_af_adoption.copy()
    rd_strength_adoption_lag["country_code"] = (
        rd_strength_adoption_lag["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    rd_strength_adoption_lag = rd_strength_adoption_lag.merge(
        country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
        on="country_code",
        how="left",
    )
    rd_strength_adoption_lag = rd_strength_adoption_lag.dropna(
        subset=["rd_strength_tertile", "first_af_date"]
    ).copy()
    rd_strength_adoption_lag["adoption_lag_months"] = (
        rd_strength_adoption_lag["first_af_date"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        rd_strength_adoption_lag["first_af_date"].dt.month
        - pd.Timestamp("2018-12-02").month
    )

    rd_strength_adoption_lag
    return (rd_strength_adoption_lag,)


@app.cell(hide_code=True)
def _(country_rd_strength_lookup, inequality_yearly_base):
    rd_strength_output_time = inequality_yearly_base[
        inequality_yearly_base["is_alphafold_related"] == True
    ].copy()
    rd_strength_output_time["country_code"] = (
        rd_strength_output_time["country_code"].astype(str).str.strip().str.upper()
    )
    rd_strength_output_time = rd_strength_output_time.merge(
        country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
        on="country_code",
        how="left",
    )
    rd_strength_output_time = rd_strength_output_time.dropna(
        subset=["rd_strength_tertile"]
    )
    rd_strength_output_time = rd_strength_output_time.groupby(
        ["publication_year", "rd_strength_tertile"], as_index=False
    ).agg(af_fractional_output=("fractional_weight", "sum"))
    _rd_strength_year_totals = rd_strength_output_time.groupby(
        "publication_year", as_index=False
    ).agg(total_af_output=("af_fractional_output", "sum"))
    rd_strength_output_time = rd_strength_output_time.merge(
        _rd_strength_year_totals,
        on="publication_year",
        how="left",
    )
    rd_strength_output_time["global_af_output_share"] = (
        rd_strength_output_time["af_fractional_output"]
        / rd_strength_output_time["total_af_output"]
    )
    rd_strength_output_time = rd_strength_output_time.sort_values(
        ["publication_year", "rd_strength_tertile"]
    ).reset_index(drop=True)

    rd_strength_output_time
    return


@app.cell(hide_code=True)
def _(
    country_af_nonaf_compare,
    country_first_af_adoption,
    country_gap_explainer_output,
    country_rd_strength_lookup,
    pd,
):
    country_gap_explainer_rd_df = country_first_af_adoption[
        [
            "country_code",
            "first_af_date",
        ]
    ].copy()
    country_gap_explainer_rd_df["country_code"] = (
        country_gap_explainer_rd_df["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    country_gap_explainer_rd_df["adoption_lag_months"] = (
        country_gap_explainer_rd_df["first_af_date"].dt.year
        - pd.Timestamp("2018-12-02").year
    ) * 12 + (
        country_gap_explainer_rd_df["first_af_date"].dt.month
        - pd.Timestamp("2018-12-02").month
    )
    country_gap_explainer_rd_df = country_gap_explainer_rd_df.merge(
        country_gap_explainer_output,
        on="country_code",
        how="left",
    )
    country_gap_explainer_rd_df = country_gap_explainer_rd_df.merge(
        country_af_nonaf_compare[
            ["country_code", "country_name", "af_fractional_count"]
        ],
        on="country_code",
        how="left",
    )
    country_gap_explainer_rd_df = country_gap_explainer_rd_df.merge(
        country_rd_strength_lookup[
            ["country_code", "rd_strength_tertile", "rd_strength_index"]
        ],
        on="country_code",
        how="left",
    )
    country_gap_explainer_rd_df = country_gap_explainer_rd_df.dropna(
        subset=[
            "rd_strength_tertile",
            "adoption_lag_months",
            "pre_af2_non_af_fractional_output",
            "af_fractional_count",
        ]
    ).copy()
    country_gap_explainer_rd_df["bubble_size"] = (
        40
        + 18
        * country_gap_explainer_rd_df["af_fractional_count"].map(
            lambda _v: __import__("math").sqrt(_v) if _v >= 0 else 0
        )
    )

    country_gap_explainer_rd_df
    return


@app.cell(hide_code=True)
def fig_3_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_CORAL,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    collab_work_base,
    duckdb,
    pd,
    plt,
):
    _f3c2_quarter_map = duckdb.sql(
        """
        SELECT
            work_id,
            publication_date
        FROM read_parquet('derived_tables_dedup/works.parquet')
        WHERE work_id IS NOT NULL
          AND publication_date IS NOT NULL
          AND publication_year BETWEEN 2019 AND 2025
        """
    ).df()

    figure_f3c2_international = collab_work_base.merge(
        _f3c2_quarter_map,
        on="work_id",
        how="inner",
    )
    figure_f3c2_international["publication_date"] = pd.to_datetime(
        figure_f3c2_international["publication_date"]
    )
    figure_f3c2_international["publication_quarter"] = (
        figure_f3c2_international["publication_date"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )
    figure_f3c2_international["research_type"] = figure_f3c2_international[
        "is_alphafold_related"
    ].map({True: "AF", False: "non-AF"})

    figure_f3c2_international = figure_f3c2_international.groupby(
        ["publication_quarter", "research_type"],
        as_index=False,
    ).agg(
        total_papers=("work_id", "nunique"),
        international_papers=("is_international_collab", "sum"),
    )
    figure_f3c2_international["international_collab_share_pct"] = (
        100
        * figure_f3c2_international["international_papers"]
        / figure_f3c2_international["total_papers"]
    )

    _f3c2_plot = figure_f3c2_international.copy()
    _f3c2_plot = _f3c2_plot.sort_values(["research_type", "publication_quarter"])
    _f3c2_order = ["AF", "non-AF"]
    _f3c2_color_map = {"AF": AF_CORAL, "non-AF": AF_PURPLE}

    _fig_f3c2_international, _ax_f3c2_international = plt.subplots(
        figsize=(7.7, 7.2), dpi=260
    )
    _f3c2i_label_font_size = AF_LABEL_FONT_SIZE * 0.9
    _f3c2i_annotation_font_size = AF_ANNOTATION_FONT_SIZE * 0.9

    _f3c2_af2_start = pd.Timestamp("2021-07-01")
    _f3c2_af3_start = pd.Timestamp("2024-05-01")
    _f3c2_x_min = _f3c2_plot["publication_quarter"].min()
    _f3c2_x_max = _f3c2_plot["publication_quarter"].max()

    for _group_f3c2 in _f3c2_order:
        _group_df_f3c2 = _f3c2_plot[_f3c2_plot["research_type"] == _group_f3c2]
        _ax_f3c2_international.plot(
            _group_df_f3c2["publication_quarter"],
            _group_df_f3c2["international_collab_share_pct"],
            color=_f3c2_color_map[_group_f3c2],
            linewidth=2.6,
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=_group_f3c2,
            zorder=3,
        )

    for _event_x_f3c2, _event_label_f3c2 in [
        (pd.Timestamp("2021-07-01"), "AF2+AFDB"),
        (pd.Timestamp("2024-05-01"), "AF3"),
    ]:
        _ax_f3c2_international.axvline(
            _event_x_f3c2,
            color=AF_EVENT_NEUTRAL,
            linestyle=(0, (4, 4)),
            linewidth=1.15,
            alpha=0.78,
            zorder=1,
        )
        _ax_f3c2_international.text(
            _event_x_f3c2,
            1.01,
            _event_label_f3c2,
            transform=_ax_f3c2_international.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_f3c2i_annotation_font_size,
            color=AF_EVENT_NEUTRAL,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 0.4,
            },
            clip_on=False,
            zorder=5,
        )

    _f3c2_af_pre = _f3c2_plot[
        (_f3c2_plot["research_type"] == "AF")
        & (_f3c2_plot["publication_quarter"] < _f3c2_af2_start)
    ]["international_collab_share_pct"]
    _f3c2_af_af2 = _f3c2_plot[
        (_f3c2_plot["research_type"] == "AF")
        & (_f3c2_plot["publication_quarter"] >= _f3c2_af2_start)
        & (_f3c2_plot["publication_quarter"] < _f3c2_af3_start)
    ]["international_collab_share_pct"]
    _f3c2_pre_mean = float(_f3c2_af_pre.mean())
    _f3c2_af2_mean = float(_f3c2_af_af2.mean())
    _f3c2_pp_diff = _f3c2_af2_mean - _f3c2_pre_mean

    _ax_f3c2_international.axhline(
        _f3c2_pre_mean,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=1.1,
        alpha=0.95,
        zorder=2,
    )
    _ax_f3c2_international.axhline(
        _f3c2_af2_mean,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=1.1,
        alpha=0.95,
        zorder=2,
    )

    _f3c2_label_x = _f3c2_x_max - pd.offsets.QuarterBegin(startingMonth=1)
    _ax_f3c2_international.text(
        _f3c2_label_x,
        _f3c2_pre_mean,
        f"Pre-AF2 mean: {_f3c2_pre_mean:.1f}%",
        fontsize=_f3c2i_annotation_font_size,
        color=AF_GUIDE_NEUTRAL,
        ha="right",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.3,
        },
        zorder=6,
    )
    _ax_f3c2_international.text(
        _f3c2_label_x,
        _f3c2_af2_mean,
        f"AF2 era mean: {_f3c2_af2_mean:.1f}%",
        fontsize=_f3c2i_annotation_font_size,
        color=AF_GUIDE_NEUTRAL,
        ha="right",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.3,
        },
        zorder=6,
    )

    _f3c2_mid_x = _f3c2_x_min + (_f3c2_x_max - _f3c2_x_min) * 0.66
    _f3c2_mid_y = (_f3c2_pre_mean + _f3c2_af2_mean) / 2
    _ax_f3c2_international.annotate(
        f"{_f3c2_pp_diff:+.1f} pp",
        xy=(_f3c2_mid_x, _f3c2_af2_mean),
        xytext=(_f3c2_mid_x, _f3c2_pre_mean),
        ha="left",
        va="center",
        fontsize=_f3c2i_annotation_font_size,
        color=AF_GUIDE_NEUTRAL,
        arrowprops={
            "arrowstyle": "<->",
            "color": AF_GUIDE_NEUTRAL,
            "lw": 1.0,
            "shrinkA": 4,
            "shrinkB": 4,
        },
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
            "pad": 0.35,
        },
        zorder=6,
    )

    _ax_f3c2_international.set_xlim(
        _f3c2_x_min,
        _f3c2_x_max + pd.offsets.QuarterBegin(startingMonth=1),
    )
    _ax_f3c2_international.margins(x=0)

    _ax_f3c2_international.set_xlabel(
        "Publication quarter",
        fontsize=_f3c2i_label_font_size,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3c2_international.set_ylabel(
        "Internationally coauthored papers (%)",
        fontsize=_f3c2i_label_font_size,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3c2_international.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color="#d1d5db",
        alpha=0.92,
    )
    _ax_f3c2_international.grid(axis="x", visible=False)
    _ax_f3c2_international.set_axisbelow(True)
    _ax_f3c2_international.spines["top"].set_visible(False)
    _ax_f3c2_international.spines["right"].set_visible(False)
    _ax_f3c2_international.spines["left"].set_color("#374151")
    _ax_f3c2_international.spines["bottom"].set_color("#374151")
    _ax_f3c2_international.spines["left"].set_linewidth(0.8)
    _ax_f3c2_international.spines["bottom"].set_linewidth(0.8)
    _ax_f3c2_international.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_f3c2_international.legend(
        frameon=False,
        loc="upper left",
        fontsize=_f3c2i_annotation_font_size,
    )

    _ax_f3c2_international.xaxis.set_major_locator(
        plt.matplotlib.dates.YearLocator()
    )
    _ax_f3c2_international.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%Y")
    )

    _fig_f3c2_international.subplots_adjust(
        left=0.11, right=0.97, top=0.88, bottom=0.16
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_3_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    collab_work_base,
    duckdb,
    pd,
    plt,
):
    _fig_f3c_teamshare, _ax_f3c_teamshare = plt.subplots(
        figsize=(11.2, 7.0),
        dpi=320,
    )

    _f3c_teamshare_year_map = duckdb.sql(
        """
        SELECT
            work_id,
            publication_year
        FROM read_parquet('derived_tables_dedup/works.parquet')
        WHERE work_id IS NOT NULL
          AND publication_year BETWEEN 2019 AND 2025
        """
    ).df()

    _f3c_teamshare_base = collab_work_base.merge(
        _f3c_teamshare_year_map,
        on="work_id",
        how="inner",
    )
    _f3c_teamshare_base = _f3c_teamshare_base.dropna(
        subset=["team_size", "publication_year"]
    ).copy()
    _f3c_teamshare_base["publication_year"] = _f3c_teamshare_base[
        "publication_year"
    ].astype(int)
    _f3c_teamshare_base["group"] = _f3c_teamshare_base["is_alphafold_related"].map(
        {True: "AF", False: "non-AF"}
    )

    _f3c_teamshare_mean = _f3c_teamshare_base.groupby(
        ["group", "publication_year"],
        as_index=False,
    ).agg(mean_team_size=("team_size", "mean"))

    _f3c_teamshare_pivot = (
        _f3c_teamshare_mean.pivot_table(
            index="publication_year",
            columns="group",
            values="mean_team_size",
        )
        .reindex(columns=["AF", "non-AF"])
        .sort_index()
    )

    _f3c_teamshare_large = _f3c_teamshare_base.copy()
    _f3c_teamshare_large["large_team_bin"] = pd.cut(
        _f3c_teamshare_large["team_size"],
        bins=[9, 19, float("inf")],
        labels=["10-19", "20+"],
        right=True,
    )
    _f3c_teamshare_large = _f3c_teamshare_large.dropna(subset=["large_team_bin"])
    _f3c_teamshare_large_counts = (
        _f3c_teamshare_large.groupby(
            ["publication_year", "group", "large_team_bin"],
            as_index=False,
            observed=False,
        )
        .size()
        .rename(columns={"size": "paper_count"})
    )
    _f3c_teamshare_group_totals = (
        _f3c_teamshare_base.groupby(
            ["publication_year", "group"],
            as_index=False,
            observed=False,
        )
        .size()
        .rename(columns={"size": "group_total"})
    )
    _f3c_teamshare_large_counts = _f3c_teamshare_large_counts.merge(
        _f3c_teamshare_group_totals,
        on=["publication_year", "group"],
        how="left",
    )
    _f3c_teamshare_large_counts["share_pct"] = (
        100
        * _f3c_teamshare_large_counts["paper_count"]
        / _f3c_teamshare_large_counts["group_total"]
    )
    _f3c_teamshare_large_gap = _f3c_teamshare_large_counts.pivot_table(
        index=["publication_year", "large_team_bin"],
        columns="group",
        values="share_pct",
    ).reset_index()
    _f3c_teamshare_large_gap["diff_pct_points"] = (
        _f3c_teamshare_large_gap["AF"] - _f3c_teamshare_large_gap["non-AF"]
    )
    _f3c_teamshare_large_gap = _f3c_teamshare_large_gap[
        _f3c_teamshare_large_gap["publication_year"] >= 2021
    ].copy()

    _f3c_teamshare_years = _f3c_teamshare_pivot.index.to_numpy()
    _f3c_teamshare_color_map = {"AF": AF_CORAL, "non-AF": AF_PURPLE}
    _f3c_teamshare_large_colors = {"10-19": AF_BLUE, "20+": AF_GUIDE_NEUTRAL}

    for _group_f3c_teamshare in ["AF", "non-AF"]:
        _ax_f3c_teamshare.plot(
            _f3c_teamshare_years,
            _f3c_teamshare_pivot[_group_f3c_teamshare],
            color=_f3c_teamshare_color_map[_group_f3c_teamshare],
            linewidth=3.0 if _group_f3c_teamshare == "AF" else 2.6,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=_group_f3c_teamshare,
            zorder=4,
        )

    for _event_x_f3c, _event_label_f3c in [(2021.5, "AF2+AFDB"), (2024.4, "AF3")]:
        _ax_f3c_teamshare.axvline(
            _event_x_f3c,
            color=AF_EVENT_NEUTRAL,
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            alpha=0.82,
            zorder=1,
        )
        _ax_f3c_teamshare.text(
            _event_x_f3c,
            1.01,
            _event_label_f3c,
            transform=_ax_f3c_teamshare.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_EVENT_NEUTRAL,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 0.35,
            },
            clip_on=False,
            zorder=5,
        )

    _f3c_teamshare_ymin = float(_f3c_teamshare_pivot[["AF", "non-AF"]].min().min())
    _f3c_teamshare_ymax = float(_f3c_teamshare_pivot[["AF", "non-AF"]].max().max())
    _f3c_teamshare_pad = max(
        0.35, (_f3c_teamshare_ymax - _f3c_teamshare_ymin) * 0.18
    )

    _ax_f3c_teamshare.set_xlim(2018.8, 2025.2)
    _ax_f3c_teamshare.set_ylim(
        _f3c_teamshare_ymin - _f3c_teamshare_pad * 0.35,
        _f3c_teamshare_ymax + _f3c_teamshare_pad,
    )
    _ax_f3c_teamshare.set_xticks(_f3c_teamshare_years)
    _ax_f3c_teamshare.set_xlabel(
        "Publication year",
        fontsize=AF_LABEL_FONT_SIZE * 0.9,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3c_teamshare.set_ylabel(
        "Mean team size (authors per paper)",
        fontsize=AF_LABEL_FONT_SIZE * 0.9,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3c_teamshare.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#d1d5db",
        alpha=0.95,
        zorder=0,
    )
    _ax_f3c_teamshare.grid(axis="x", visible=False)
    _ax_f3c_teamshare.set_axisbelow(True)
    _ax_f3c_teamshare.spines["top"].set_visible(False)
    _ax_f3c_teamshare.spines["right"].set_visible(False)
    _ax_f3c_teamshare.spines["left"].set_color("#374151")
    _ax_f3c_teamshare.spines["bottom"].set_color("#374151")
    _ax_f3c_teamshare.spines["left"].set_linewidth(0.8)
    _ax_f3c_teamshare.spines["bottom"].set_linewidth(0.8)
    _ax_f3c_teamshare.tick_params(axis="both", labelsize=9.8, colors="#1f2937")
    _ax_f3c_teamshare.legend(
        frameon=False, loc="upper left", fontsize=AF_ANNOTATION_FONT_SIZE
    )

    _ax_f3c_teamshare_inset = _fig_f3c_teamshare.add_axes([0.64, 0.18, 0.30, 0.28])
    for _bin_label in ["10-19", "20+"]:
        _bin_df = _f3c_teamshare_large_gap[
            _f3c_teamshare_large_gap["large_team_bin"] == _bin_label
        ].sort_values("publication_year")
        _ax_f3c_teamshare_inset.plot(
            _bin_df["publication_year"],
            _bin_df["diff_pct_points"],
            color=_f3c_teamshare_large_colors[_bin_label],
            linewidth=1.8,
            linestyle=(0, (4, 3)),
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=_bin_label,
            zorder=3,
        )
    _ax_f3c_teamshare_inset.axhline(
        0,
        color=AF_GUIDE_NEUTRAL,
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        alpha=0.85,
        zorder=1,
    )
    _ax_f3c_teamshare_inset.set_xlim(2020.85, 2025.15)
    _ax_f3c_teamshare_inset.set_xticks([2021, 2022, 2023, 2024, 2025])
    _ax_f3c_teamshare_inset.set_xticklabels(["21", "22", "23", "24", "25"])
    _ax_f3c_teamshare_inset.tick_params(
        axis="x", labelsize=7.1, pad=1, colors="#1f2937"
    )
    _ax_f3c_teamshare_inset.tick_params(axis="y", labelsize=7.0, colors="#1f2937")
    _ax_f3c_teamshare_inset.yaxis.set_major_locator(
        plt.matplotlib.ticker.MaxNLocator(nbins=4)
    )
    _ax_f3c_teamshare_inset.yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:+.1f}")
    )
    _ax_f3c_teamshare_inset.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.55,
        color="#d1d5db",
        alpha=0.85,
    )
    _ax_f3c_teamshare_inset.grid(axis="x", visible=False)
    _ax_f3c_teamshare_inset.spines["top"].set_visible(False)
    _ax_f3c_teamshare_inset.spines["right"].set_visible(False)
    _ax_f3c_teamshare_inset.spines["left"].set_color("#6b7280")
    _ax_f3c_teamshare_inset.spines["bottom"].set_color("#6b7280")
    _ax_f3c_teamshare_inset.spines["left"].set_linewidth(0.7)
    _ax_f3c_teamshare_inset.spines["bottom"].set_linewidth(0.7)
    _ax_f3c_teamshare_inset.set_title(
        "AF - non-AF in large-team share (pp)",
        fontsize=11.8,
        color="#1f2937",
        pad=3,
    )
    _ax_f3c_teamshare_inset.legend(
        frameon=False,
        loc="upper left",
        fontsize=11.4,
        handlelength=1.5,
        borderpad=0.15,
        labelspacing=0.25,
    )

    _fig_f3c_teamshare.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.88,
        bottom=0.14,
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def _(country_collab_edges, country_income_group_lookup, duckdb, pd):
    figure_3d_income_edges = country_collab_edges.copy()
    figure_3d_income_edges["source_country"] = (
        figure_3d_income_edges["source_country"].astype(str).str.strip().str.upper()
    )
    figure_3d_income_edges["target_country"] = (
        figure_3d_income_edges["target_country"].astype(str).str.strip().str.upper()
    )

    _figure_3d_income_lookup = country_income_group_lookup[
        ["country_code", "income_group"]
    ].copy()

    figure_3d_income_edges = figure_3d_income_edges.merge(
        _figure_3d_income_lookup.rename(
            columns={
                "country_code": "source_country",
                "income_group": "source_income_group",
            }
        ),
        on="source_country",
        how="left",
    )
    figure_3d_income_edges = figure_3d_income_edges.merge(
        _figure_3d_income_lookup.rename(
            columns={
                "country_code": "target_country",
                "income_group": "target_income_group",
            }
        ),
        on="target_country",
        how="left",
    )
    figure_3d_income_edges = figure_3d_income_edges.dropna(
        subset=["source_income_group", "target_income_group"]
    )

    figure_3d_income_pair_counts = figure_3d_income_edges.groupby(
        ["source_income_group", "target_income_group"],
        as_index=False,
    ).agg(collaboration_weight=("n_shared_works", "sum"))

    _figure_3d_income_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    figure_3d_income_pair_matrix = figure_3d_income_pair_counts.pivot_table(
        index="source_income_group",
        columns="target_income_group",
        values="collaboration_weight",
        fill_value=0,
    ).reindex(
        index=_figure_3d_income_order, columns=_figure_3d_income_order, fill_value=0
    )
    figure_3d_income_pair_share = figure_3d_income_pair_matrix.div(
        figure_3d_income_pair_matrix.sum(axis=1).replace(0, pd.NA),
        axis=0,
    )

    _figure_3d_nonaf_edges = duckdb.sql(
        """
        WITH non_af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = FALSE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        )
        SELECT
            a.country_code AS source_country,
            b.country_code AS target_country,
            COUNT(DISTINCT a.work_id) AS n_shared_works
        FROM work_country AS a
        INNER JOIN work_country AS b
            ON a.work_id = b.work_id
           AND a.country_code < b.country_code
        INNER JOIN non_af_works AS w
            ON a.work_id = w.work_id
        GROUP BY a.country_code, b.country_code
        HAVING COUNT(DISTINCT a.work_id) >= 5
        ORDER BY n_shared_works DESC
        """
    ).df()

    _figure_3d_nonaf_edges["source_country"] = (
        _figure_3d_nonaf_edges["source_country"].astype(str).str.strip().str.upper()
    )
    _figure_3d_nonaf_edges["target_country"] = (
        _figure_3d_nonaf_edges["target_country"].astype(str).str.strip().str.upper()
    )

    figure_3d_nonaf_income_edges = _figure_3d_nonaf_edges.merge(
        _figure_3d_income_lookup.rename(
            columns={
                "country_code": "source_country",
                "income_group": "source_income_group",
            }
        ),
        on="source_country",
        how="left",
    )
    figure_3d_nonaf_income_edges = figure_3d_nonaf_income_edges.merge(
        _figure_3d_income_lookup.rename(
            columns={
                "country_code": "target_country",
                "income_group": "target_income_group",
            }
        ),
        on="target_country",
        how="left",
    )
    figure_3d_nonaf_income_edges = figure_3d_nonaf_income_edges.dropna(
        subset=["source_income_group", "target_income_group"]
    )

    figure_3d_nonaf_pair_counts = figure_3d_nonaf_income_edges.groupby(
        ["source_income_group", "target_income_group"],
        as_index=False,
    ).agg(collaboration_weight=("n_shared_works", "sum"))

    figure_3d_nonaf_pair_matrix = figure_3d_nonaf_pair_counts.pivot_table(
        index="source_income_group",
        columns="target_income_group",
        values="collaboration_weight",
        fill_value=0,
    ).reindex(
        index=_figure_3d_income_order, columns=_figure_3d_income_order, fill_value=0
    )
    figure_3d_nonaf_pair_share = figure_3d_nonaf_pair_matrix.div(
        figure_3d_nonaf_pair_matrix.sum(axis=1).replace(0, pd.NA),
        axis=0,
    )

    figure_3d_af_nonaf_share_diff = figure_3d_income_pair_share.subtract(
        figure_3d_nonaf_pair_share,
        fill_value=0,
    )

    figure_3d_af_nonaf_share_diff
    return figure_3d_af_nonaf_share_diff, figure_3d_income_pair_share


@app.cell(hide_code=True)
def fig_3_c_income_group(
    AF_ANNOTATION_FONT_SIZE,
    AF_LABEL_FONT_SIZE,
    figure_3d_af_nonaf_share_diff,
    figure_3d_income_pair_share,
    plt,
    sns,
):
    _fig_f3d, _ax_f3d_a = plt.subplots(
        1,
        1,
        figsize=(8.2, 7.2),
        dpi=220,
    )
    # _fig_f3d, (_ax_f3d_a, _ax_f3d_b) = plt.subplots(
    #     1,
    #     2,
    #     figsize=(15.8, 7.2),
    #     dpi=220,
    #     gridspec_kw={"width_ratios": [1.0, 1.03]},
    # )

    _f3d_af_heat_pct = figure_3d_income_pair_share.copy() * 100
    _f3d_diff_heat_pct = figure_3d_af_nonaf_share_diff.copy() * 100
    _f3d_diff_absmax = float(_f3d_diff_heat_pct.abs().max().max())
    _f3d_diff_absmax = _f3d_diff_absmax if _f3d_diff_absmax > 0 else 1.0
    _f3d_tick_labels = ["High", "Upper-middle", "Lower-middle", "Low"]
    _f3d_label_font_size = AF_LABEL_FONT_SIZE * 0.9

    sns.heatmap(
        _f3d_af_heat_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "AF collaboration share within source income group (%)"
        },
        ax=_ax_f3d_a,
    )

    # sns.heatmap(
    #     _f3d_diff_heat_pct,
    #     annot=True,
    #     fmt="+.1f",
    #     cmap="RdBu_r",
    #     center=0,
    #     vmin=-_f3d_diff_absmax,
    #     vmax=_f3d_diff_absmax,
    #     linewidths=0.8,
    #     linecolor="white",
    #     cbar_kws={"label": "AF minus non-AF collaboration share (percentage points)"},
    #     ax=_ax_f3d_b,
    # )

    # _ax_f3d_a.text(
    #     0.0,
    #     1.05,
    #     "A",
    #     transform=_ax_f3d_a.transAxes,
    #     fontsize=15,
    #     fontweight="bold",
    #     color=AF_PURPLE,
    # )
    # _ax_f3d_b.text(
    #     0.0,
    #     1.05,
    #     "B",
    #     transform=_ax_f3d_b.transAxes,
    #     fontsize=15,
    #     fontweight="bold",
    #     color=AF_PURPLE,
    # )

    _ax_f3d_a.set_title(
        "AlphaFold-related collaboration structure", fontsize=12.5, pad=10
    )
    # _ax_f3d_b.set_title("Difference from non-AlphaFold baseline", fontsize=12.5, pad=10)

    _ax_f3d_a.set_xlabel("Partner income group", fontsize=_f3d_label_font_size)
    _ax_f3d_a.set_ylabel("Source income group", fontsize=_f3d_label_font_size)
    # _ax_f3d_b.set_xlabel("")
    # _ax_f3d_b.set_ylabel("")

    _ax_f3d_a.set_xticklabels(_f3d_tick_labels)
    _ax_f3d_a.set_yticklabels(_f3d_tick_labels)
    # _ax_f3d_b.set_xticklabels(_f3d_tick_labels)
    # _ax_f3d_b.set_yticklabels(_f3d_tick_labels)

    plt.setp(_ax_f3d_a.get_xticklabels(), rotation=25, ha="right")
    # plt.setp(_ax_f3d_b.get_xticklabels(), rotation=25, ha="right")
    plt.setp(_ax_f3d_a.get_yticklabels(), rotation=0)
    # plt.setp(_ax_f3d_b.get_yticklabels(), rotation=0)

    _f3d_colorbar_ax = _fig_f3d.axes[-1]
    _f3d_colorbar_ax.tick_params(
        labelsize=AF_ANNOTATION_FONT_SIZE, colors="#1f2937"
    )

    _fig_f3d.subplots_adjust(left=0.16, right=0.92, top=0.88, bottom=0.16)
    # _fig_f3d.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.16, wspace=0.28)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(country_collab_edges, country_rd_strength_lookup, duckdb, pd):
    figure_3d_rd_edges = country_collab_edges.copy()
    figure_3d_rd_edges["source_country"] = (
        figure_3d_rd_edges["source_country"].astype(str).str.strip().str.upper()
    )
    figure_3d_rd_edges["target_country"] = (
        figure_3d_rd_edges["target_country"].astype(str).str.strip().str.upper()
    )

    _figure_3d_rd_lookup = country_rd_strength_lookup[
        ["country_code", "rd_strength_tertile"]
    ].copy()

    figure_3d_rd_edges = figure_3d_rd_edges.merge(
        _figure_3d_rd_lookup.rename(
            columns={
                "country_code": "source_country",
                "rd_strength_tertile": "source_rd_strength_tertile",
            }
        ),
        on="source_country",
        how="left",
    )
    figure_3d_rd_edges = figure_3d_rd_edges.merge(
        _figure_3d_rd_lookup.rename(
            columns={
                "country_code": "target_country",
                "rd_strength_tertile": "target_rd_strength_tertile",
            }
        ),
        on="target_country",
        how="left",
    )
    figure_3d_rd_edges = figure_3d_rd_edges.dropna(
        subset=["source_rd_strength_tertile", "target_rd_strength_tertile"]
    )

    figure_3d_rd_pair_counts = figure_3d_rd_edges.groupby(
        ["source_rd_strength_tertile", "target_rd_strength_tertile"],
        as_index=False,
    ).agg(collaboration_weight=("n_shared_works", "sum"))

    _figure_3d_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    figure_3d_rd_pair_matrix = figure_3d_rd_pair_counts.pivot_table(
        index="source_rd_strength_tertile",
        columns="target_rd_strength_tertile",
        values="collaboration_weight",
        fill_value=0,
    ).reindex(index=_figure_3d_rd_order, columns=_figure_3d_rd_order, fill_value=0)
    figure_3d_rd_pair_share = figure_3d_rd_pair_matrix.div(
        figure_3d_rd_pair_matrix.sum(axis=1).replace(0, pd.NA),
        axis=0,
    )

    _figure_3d_nonaf_edges = duckdb.sql(
        """
        WITH non_af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = FALSE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        )
        SELECT
            a.country_code AS source_country,
            b.country_code AS target_country,
            COUNT(DISTINCT a.work_id) AS n_shared_works
        FROM work_country AS a
        INNER JOIN work_country AS b
            ON a.work_id = b.work_id
           AND a.country_code < b.country_code
        INNER JOIN non_af_works AS w
            ON a.work_id = w.work_id
        GROUP BY a.country_code, b.country_code
        HAVING COUNT(DISTINCT a.work_id) >= 5
        ORDER BY n_shared_works DESC
        """
    ).df()

    _figure_3d_nonaf_edges["source_country"] = (
        _figure_3d_nonaf_edges["source_country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    _figure_3d_nonaf_edges["target_country"] = (
        _figure_3d_nonaf_edges["target_country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    figure_3d_nonaf_rd_edges = _figure_3d_nonaf_edges.merge(
        _figure_3d_rd_lookup.rename(
            columns={
                "country_code": "source_country",
                "rd_strength_tertile": "source_rd_strength_tertile",
            }
        ),
        on="source_country",
        how="left",
    )
    figure_3d_nonaf_rd_edges = figure_3d_nonaf_rd_edges.merge(
        _figure_3d_rd_lookup.rename(
            columns={
                "country_code": "target_country",
                "rd_strength_tertile": "target_rd_strength_tertile",
            }
        ),
        on="target_country",
        how="left",
    )
    figure_3d_nonaf_rd_edges = figure_3d_nonaf_rd_edges.dropna(
        subset=["source_rd_strength_tertile", "target_rd_strength_tertile"]
    )

    figure_3d_nonaf_rd_pair_counts = figure_3d_nonaf_rd_edges.groupby(
        ["source_rd_strength_tertile", "target_rd_strength_tertile"],
        as_index=False,
    ).agg(collaboration_weight=("n_shared_works", "sum"))

    figure_3d_nonaf_rd_pair_matrix = figure_3d_nonaf_rd_pair_counts.pivot_table(
        index="source_rd_strength_tertile",
        columns="target_rd_strength_tertile",
        values="collaboration_weight",
        fill_value=0,
    ).reindex(index=_figure_3d_rd_order, columns=_figure_3d_rd_order, fill_value=0)
    figure_3d_nonaf_rd_pair_share = figure_3d_nonaf_rd_pair_matrix.div(
        figure_3d_nonaf_rd_pair_matrix.sum(axis=1).replace(0, pd.NA),
        axis=0,
    )

    figure_3d_rd_nonaf_share_diff = figure_3d_rd_pair_share.subtract(
        figure_3d_nonaf_rd_pair_share,
        fill_value=0,
    )

    figure_3d_rd_nonaf_share_diff, figure_3d_rd_pair_share
    return figure_3d_rd_pair_matrix, figure_3d_rd_pair_share


@app.cell(hide_code=True)
def fig_3_c(
    AF_ANNOTATION_FONT_SIZE,
    AF_LABEL_FONT_SIZE,
    AF_SEQUENTIAL_CMAP,
    figure_3d_rd_pair_share,
    plt,
    sns,
):
    _fig_f3d_rd, _ax_f3d_rd = plt.subplots(
        1,
        1,
        figsize=(8.2, 7.2),
        dpi=220,
    )

    _f3d_rd_heat_pct = figure_3d_rd_pair_share.copy() * 100
    _f3d_rd_tick_labels = ["High", "Middle", "Low"]
    _f3d_rd_label_font_size = AF_LABEL_FONT_SIZE * 0.9

    sns.heatmap(
        _f3d_rd_heat_pct,
        annot=True,
        fmt=".1f",
        cmap=AF_SEQUENTIAL_CMAP.reversed(),
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "AF collaboration share within source R&D-strength group (%)"
        },
        ax=_ax_f3d_rd,
    )

    _ax_f3d_rd.set_title(
        "AlphaFold-related collaboration structure by R&D strength",
        fontsize=12.5,
        pad=10,
    )

    _ax_f3d_rd.set_xlabel(
        "Partner R&D strength group", fontsize=_f3d_rd_label_font_size
    )
    _ax_f3d_rd.set_ylabel(
        "Source R&D strength group", fontsize=_f3d_rd_label_font_size
    )

    _ax_f3d_rd.set_xticklabels(_f3d_rd_tick_labels)
    _ax_f3d_rd.set_yticklabels(_f3d_rd_tick_labels)

    plt.setp(_ax_f3d_rd.get_xticklabels(), rotation=25, ha="right")
    plt.setp(_ax_f3d_rd.get_yticklabels(), rotation=0)

    _f3d_rd_colorbar_ax = _fig_f3d_rd.axes[-1]
    _f3d_rd_colorbar_ax.tick_params(
        labelsize=AF_ANNOTATION_FONT_SIZE, colors="#1f2937"
    )

    _fig_f3d_rd.subplots_adjust(left=0.16, right=0.92, top=0.88, bottom=0.16)

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_3_d(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_rd_strength_lookup,
    figure_3b_centrality,
    pd,
    plt,
    sns,
):
    figure_f3d2_rd_centrality_distribution = figure_3b_centrality.merge(
        country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
        on="country_code",
        how="inner",
    ).dropna(subset=["rd_strength_tertile", "eigenvector_centrality"])

    _f3d2_rd_plot = figure_f3d2_rd_centrality_distribution.copy()
    _f3d2_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f3d2_rd_plot = _f3d2_rd_plot[
        _f3d2_rd_plot["rd_strength_tertile"].isin(_f3d2_rd_order)
    ].copy()
    _f3d2_rd_plot["rd_strength_tertile"] = pd.Categorical(
        _f3d2_rd_plot["rd_strength_tertile"],
        categories=_f3d2_rd_order,
        ordered=True,
    )
    _f3d2_rd_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _f3d2_rd_labels = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _f3d2_rd_floor = max(
        float(_f3d2_rd_plot["eigenvector_centrality"].min()) * 0.5, 1e-4
    )
    _f3d2_rd_plot["eigenvector_centrality_log10"] = (
        _f3d2_rd_plot["eigenvector_centrality"]
        .clip(lower=_f3d2_rd_floor)
        .map(lambda _v: __import__("math").log10(_v))
    )

    _f3d2_rd_group_counts = (
        _f3d2_rd_plot.groupby("rd_strength_tertile", observed=False)[
            "country_code"
        ]
        .nunique()
        .reindex(_f3d2_rd_order)
    )
    _f3d2_rd_group_medians = (
        _f3d2_rd_plot.groupby("rd_strength_tertile", observed=False)[
            "eigenvector_centrality_log10"
        ]
        .median()
        .reindex(_f3d2_rd_order)
    )
    _f3d2_rd_ymin = float(_f3d2_rd_plot["eigenvector_centrality_log10"].min())
    _f3d2_rd_ymax = float(_f3d2_rd_plot["eigenvector_centrality_log10"].max())
    _f3d2_rd_span = _f3d2_rd_ymax - _f3d2_rd_ymin
    _f3d2_rd_main_top = float(
        _f3d2_rd_plot["eigenvector_centrality_log10"].quantile(0.995)
        + max(0.03, _f3d2_rd_span * 0.05)
    )
    _f3d2_rd_lower_pad = max(0.03, _f3d2_rd_span * 0.08)
    _f3d2_rd_upper_pad = max(0.04, _f3d2_rd_span * 0.07)

    _f3d2_rd_stats = __import__("scipy.stats").stats
    _f3d2_rd_samples = [
        _f3d2_rd_plot.loc[
            _f3d2_rd_plot["rd_strength_tertile"] == _group,
            "eigenvector_centrality",
        ]
        .dropna()
        .to_numpy()
        for _group in _f3d2_rd_order
    ]
    _f3d2_rd_valid_samples = [
        _sample for _sample in _f3d2_rd_samples if len(_sample) > 0
    ]
    _f3d2_rd_kw = _f3d2_rd_stats.kruskal(*_f3d2_rd_valid_samples)
    _f3d2_rd_hi = (
        _f3d2_rd_plot.loc[
            _f3d2_rd_plot["rd_strength_tertile"] == "High R&D strength",
            "eigenvector_centrality",
        ]
        .dropna()
        .to_numpy()
    )
    _f3d2_rd_non_hi = (
        _f3d2_rd_plot.loc[
            _f3d2_rd_plot["rd_strength_tertile"] != "High R&D strength",
            "eigenvector_centrality",
        ]
        .dropna()
        .to_numpy()
    )
    _f3d2_rd_mw = _f3d2_rd_stats.mannwhitneyu(
        _f3d2_rd_hi, _f3d2_rd_non_hi, alternative="two-sided"
    )
    _f3d2_rd_u = float(_f3d2_rd_mw.statistic)
    _f3d2_rd_hi_n = len(_f3d2_rd_hi)
    _f3d2_rd_non_hi_n = len(_f3d2_rd_non_hi)
    _f3d2_rd_delta = (
        ((2 * _f3d2_rd_u) / (_f3d2_rd_hi_n * _f3d2_rd_non_hi_n)) - 1
        if _f3d2_rd_hi_n and _f3d2_rd_non_hi_n
        else float("nan")
    )

    _fig_f3d2_rd, _ax_f3d2_rd = plt.subplots(figsize=(11.8, 7.6), dpi=320)
    _fig_f3d2_rd.patch.set_facecolor("white")
    _ax_f3d2_rd.set_facecolor("white")

    _f3d2_rd_positions = list(range(len(_f3d2_rd_order)))

    sns.violinplot(
        data=_f3d2_rd_plot,
        x="rd_strength_tertile",
        y="eigenvector_centrality_log10",
        order=_f3d2_rd_order,
        hue="rd_strength_tertile",
        palette=_f3d2_rd_palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=1,
        width=0.82,
        bw_adjust=0.7,
        dodge=False,
        legend=False,
        ax=_ax_f3d2_rd,
    )

    _f3d2_rd_violin_bodies = _ax_f3d2_rd.collections[: len(_f3d2_rd_order)]
    for _idx_f3d2_rd, _body_f3d2_rd in enumerate(_f3d2_rd_violin_bodies):
        _body_f3d2_rd.set_alpha(0.92)
        _body_f3d2_rd.set_edgecolor("#334155")
        _body_f3d2_rd.set_linewidth(1.0)
        _body_f3d2_rd.set_zorder(1)
        _paths_f3d2_rd = _body_f3d2_rd.get_paths()
        if _paths_f3d2_rd:
            _vertices_f3d2_rd = _paths_f3d2_rd[0].vertices
            _center_f3d2_rd = _f3d2_rd_positions[_idx_f3d2_rd]
            _vertices_f3d2_rd[:, 0] = _vertices_f3d2_rd[:, 0].clip(
                max=_center_f3d2_rd
            )

    _f3d2_rd_box_data = [
        _f3d2_rd_plot.loc[
            _f3d2_rd_plot["rd_strength_tertile"] == _group,
            "eigenvector_centrality_log10",
        ]
        .dropna()
        .to_numpy()
        for _group in _f3d2_rd_order
    ]
    _f3d2_rd_box = _ax_f3d2_rd.boxplot(
        _f3d2_rd_box_data,
        positions=_f3d2_rd_positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": AF_GUIDE_NEUTRAL, "linewidth": 2.1},
        boxprops={"facecolor": "white", "edgecolor": "#1f2937", "linewidth": 1.0},
        whiskerprops={"color": "#1f2937", "linewidth": 1.0},
        capprops={"color": "#1f2937", "linewidth": 1.0},
    )
    for _patch_f3d2_rd in _f3d2_rd_box["boxes"]:
        _patch_f3d2_rd.set_zorder(4)
    for _line_group_f3d2_rd in [
        _f3d2_rd_box["whiskers"],
        _f3d2_rd_box["caps"],
        _f3d2_rd_box["medians"],
    ]:
        for _line_f3d2_rd in _line_group_f3d2_rd:
            _line_f3d2_rd.set_zorder(4)

    for _idx_f3d2_rd, _group_f3d2_rd in enumerate(_f3d2_rd_order):
        _group_points_f3d2_rd = (
            _f3d2_rd_plot.loc[
                _f3d2_rd_plot["rd_strength_tertile"] == _group_f3d2_rd,
                "eigenvector_centrality_log10",
            ]
            .dropna()
            .to_numpy()
        )
        if _group_points_f3d2_rd.size > 0:
            _group_jitter_f3d2_rd = (
                pd.Series(range(_group_points_f3d2_rd.size))
                .map(lambda _i: 0.10 + ((_i % 11) - 5) * 0.018)
                .to_numpy()
            )
            _ax_f3d2_rd.scatter(
                _idx_f3d2_rd + _group_jitter_f3d2_rd,
                _group_points_f3d2_rd,
                s=15,
                color=_f3d2_rd_palette[_group_f3d2_rd],
                alpha=0.35,
                edgecolors="white",
                linewidth=0.35,
                zorder=3,
            )

    for _idx_f3d2_rd, _group_f3d2_rd in enumerate(_f3d2_rd_order):
        _median_f3d2_rd = float(_f3d2_rd_group_medians.loc[_group_f3d2_rd])
        _count_f3d2_rd = (
            int(_f3d2_rd_group_counts.loc[_group_f3d2_rd])
            if pd.notna(_f3d2_rd_group_counts.loc[_group_f3d2_rd])
            else 0
        )
        _median_label_y_f3d2_rd = _median_f3d2_rd * 1.015
        if pd.notna(_median_f3d2_rd):
            _ax_f3d2_rd.text(
                _idx_f3d2_rd + 0.13,
                _median_label_y_f3d2_rd,
                f"Median = {_median_f3d2_rd:.3f}",
                ha="left",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color=AF_GUIDE_NEUTRAL,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                    "pad": 0.55,
                },
                zorder=5,
            )
            _ax_f3d2_rd.text(
                _idx_f3d2_rd,
                _f3d2_rd_ymin - _f3d2_rd_lower_pad * 0.45,
                f"n={_count_f3d2_rd}",
                ha="center",
                va="top",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color="#475569",
                zorder=5,
            )

    _ax_f3d2_rd.text(
        0.015,
        0.985,
        (
            f"Kruskal–Wallis: H={_f3d2_rd_kw.statistic:.2f}, p={_f3d2_rd_kw.pvalue:.3g}\n"
            f"High R&D vs others: U={_f3d2_rd_u:.1f}, Cliff's δ={_f3d2_rd_delta:.2f}, p={_f3d2_rd_mw.pvalue:.3g}"
        ),
        transform=_ax_f3d2_rd.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#374151",
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "linewidth": 0.6,
            "alpha": 0.94,
            "pad": 0.85,
        },
        zorder=6,
    )

    _ax_f3d2_rd.set_xlabel(
        "R&D strength group",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d2_rd.set_ylabel(
        "log10(Eigenvector centrality)",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d2_rd.set_xticks(_f3d2_rd_positions)
    _ax_f3d2_rd.set_xticklabels(
        [_f3d2_rd_labels[_group_f3d2_rd] for _group_f3d2_rd in _f3d2_rd_order],
        fontsize=10,
    )
    _ax_f3d2_rd.set_ylim(
        _f3d2_rd_ymin - _f3d2_rd_lower_pad,
        _f3d2_rd_main_top + _f3d2_rd_upper_pad,
    )
    _ax_f3d2_rd.grid(
        axis="y", linestyle=(0, (2, 3)), linewidth=0.65, color="#d0d0d0", alpha=0.9
    )
    _ax_f3d2_rd.grid(axis="x", visible=False)
    _ax_f3d2_rd.set_axisbelow(True)
    _ax_f3d2_rd.spines["top"].set_visible(False)
    _ax_f3d2_rd.spines["right"].set_visible(False)
    _ax_f3d2_rd.spines["left"].set_color("#6b7280")
    _ax_f3d2_rd.spines["bottom"].set_color("#6b7280")
    _ax_f3d2_rd.spines["left"].set_linewidth(0.8)
    _ax_f3d2_rd.spines["bottom"].set_linewidth(0.8)
    _ax_f3d2_rd.tick_params(axis="x", length=0, pad=8, colors="#1f2937")
    _ax_f3d2_rd.tick_params(axis="y", labelsize=9.8, colors="#2f2f2f")

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_3_d_income_group(
    AF_ANNOTATION_FONT_SIZE,
    AF_LABEL_FONT_SIZE,
    figure_3b_centrality,
    pd,
    plt,
    sns,
    stats,
):
    _fig_f3b_clean, _ax_f3b_clean = plt.subplots(figsize=(11.8, 7.6), dpi=320)

    _f3b_clean_plot = figure_3b_centrality.copy()
    _f3b_clean_order = [
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Low income",
    ]
    _f3b_clean_plot = _f3b_clean_plot[
        _f3b_clean_plot["income_group"].isin(_f3b_clean_order)
    ].copy()
    _f3b_clean_plot["income_group"] = pd.Categorical(
        _f3b_clean_plot["income_group"],
        categories=_f3b_clean_order,
        ordered=True,
    )
    _f3b_clean_palette = {
        "High income": "#163a70",
        "Upper-middle income": "#3d6fb6",
        "Lower-middle income": "#7ea6d8",
        "Low income": "#d6e3f3",
    }
    _f3b_clean_labels = {
        "High income": "High",
        "Upper-middle income": "Upper-middle",
        "Lower-middle income": "Lower-middle",
        "Low income": "Low",
    }
    _f3b_clean_floor = max(
        float(_f3b_clean_plot["eigenvector_centrality"].min()) * 0.5, 1e-4
    )
    _f3b_clean_plot["eigenvector_centrality_log10"] = (
        _f3b_clean_plot["eigenvector_centrality"]
        .clip(lower=_f3b_clean_floor)
        .map(lambda _v: __import__("math").log10(_v))
    )

    _f3b_clean_group_counts = (
        _f3b_clean_plot.groupby("income_group", observed=False)["country_code"]
        .nunique()
        .reindex(_f3b_clean_order)
    )
    _f3b_clean_group_medians = (
        _f3b_clean_plot.groupby("income_group", observed=False)[
            "eigenvector_centrality_log10"
        ]
        .median()
        .reindex(_f3b_clean_order)
    )
    _f3b_clean_group_q1 = (
        _f3b_clean_plot.groupby("income_group", observed=False)[
            "eigenvector_centrality_log10"
        ]
        .quantile(0.25)
        .reindex(_f3b_clean_order)
    )
    _f3b_clean_group_q3 = (
        _f3b_clean_plot.groupby("income_group", observed=False)[
            "eigenvector_centrality_log10"
        ]
        .quantile(0.75)
        .reindex(_f3b_clean_order)
    )
    _f3b_clean_ymin = float(_f3b_clean_plot["eigenvector_centrality_log10"].min())
    _f3b_clean_ymax = float(_f3b_clean_plot["eigenvector_centrality_log10"].max())
    _f3b_clean_span = _f3b_clean_ymax - _f3b_clean_ymin
    _f3b_clean_main_top = float(
        _f3b_clean_plot["eigenvector_centrality_log10"].quantile(0.995)
        + max(0.03, _f3b_clean_span * 0.05)
    )
    _f3b_clean_lower_pad = max(0.03, _f3b_clean_span * 0.08)
    _f3b_clean_upper_pad = max(0.04, _f3b_clean_span * 0.07)

    _f3b_clean_samples = [
        _f3b_clean_plot.loc[
            _f3b_clean_plot["income_group"] == _group, "eigenvector_centrality"
        ]
        .dropna()
        .to_numpy()
        for _group in _f3b_clean_order
    ]
    _f3b_clean_valid_samples = [
        _sample for _sample in _f3b_clean_samples if len(_sample) > 0
    ]
    _f3b_clean_kw = stats.kruskal(*_f3b_clean_valid_samples)
    _f3b_clean_hi = (
        _f3b_clean_plot.loc[
            _f3b_clean_plot["income_group"] == "High income",
            "eigenvector_centrality",
        ]
        .dropna()
        .to_numpy()
    )
    _f3b_clean_non_hi = (
        _f3b_clean_plot.loc[
            _f3b_clean_plot["income_group"] != "High income",
            "eigenvector_centrality",
        ]
        .dropna()
        .to_numpy()
    )
    _f3b_clean_mw = stats.mannwhitneyu(
        _f3b_clean_hi, _f3b_clean_non_hi, alternative="two-sided"
    )
    _f3b_clean_u = float(_f3b_clean_mw.statistic)
    _f3b_clean_hi_n = len(_f3b_clean_hi)
    _f3b_clean_non_hi_n = len(_f3b_clean_non_hi)
    _f3b_clean_delta = (
        ((2 * _f3b_clean_u) / (_f3b_clean_hi_n * _f3b_clean_non_hi_n)) - 1
        if _f3b_clean_hi_n and _f3b_clean_non_hi_n
        else float("nan")
    )

    _fig_f3b_clean.patch.set_facecolor("white")
    _ax_f3b_clean.set_facecolor("white")

    _f3b_clean_positions = list(range(len(_f3b_clean_order)))

    sns.violinplot(
        data=_f3b_clean_plot,
        x="income_group",
        y="eigenvector_centrality_log10",
        order=_f3b_clean_order,
        hue="income_group",
        palette=_f3b_clean_palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=1,
        width=0.82,
        bw_adjust=0.7,
        dodge=False,
        legend=False,
        ax=_ax_f3b_clean,
    )

    _f3b_clean_violin_bodies = _ax_f3b_clean.collections[: len(_f3b_clean_order)]
    for _idx_f3b_clean, _body_f3b_clean in enumerate(_f3b_clean_violin_bodies):
        _body_f3b_clean.set_alpha(0.92)
        _body_f3b_clean.set_edgecolor("#334155")
        _body_f3b_clean.set_linewidth(1.0)
        _body_f3b_clean.set_zorder(1)
        _paths_f3b_clean = _body_f3b_clean.get_paths()
        if _paths_f3b_clean:
            _vertices_f3b_clean = _paths_f3b_clean[0].vertices
            _center_f3b_clean = _f3b_clean_positions[_idx_f3b_clean]
            _vertices_f3b_clean[:, 0] = _vertices_f3b_clean[:, 0].clip(
                max=_center_f3b_clean
            )

    _f3b_clean_box_data = [
        _f3b_clean_plot.loc[
            _f3b_clean_plot["income_group"] == _group,
            "eigenvector_centrality_log10",
        ]
        .dropna()
        .to_numpy()
        for _group in _f3b_clean_order
    ]
    _f3b_clean_box = _ax_f3b_clean.boxplot(
        _f3b_clean_box_data,
        positions=_f3b_clean_positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#7f1d1d", "linewidth": 2.1},
        boxprops={"facecolor": "white", "edgecolor": "#1f2937", "linewidth": 1.0},
        whiskerprops={"color": "#1f2937", "linewidth": 1.0},
        capprops={"color": "#1f2937", "linewidth": 1.0},
    )
    for _patch_f3b_clean in _f3b_clean_box["boxes"]:
        _patch_f3b_clean.set_zorder(4)
    for _line_group_f3b_clean in [
        _f3b_clean_box["whiskers"],
        _f3b_clean_box["caps"],
        _f3b_clean_box["medians"],
    ]:
        for _line_f3b_clean in _line_group_f3b_clean:
            _line_f3b_clean.set_zorder(4)

    for _idx_f3b_clean, _group_f3b_clean in enumerate(_f3b_clean_order):
        _group_points_f3b_clean = (
            _f3b_clean_plot.loc[
                _f3b_clean_plot["income_group"] == _group_f3b_clean,
                "eigenvector_centrality_log10",
            ]
            .dropna()
            .to_numpy()
        )
        if _group_points_f3b_clean.size > 0:
            _group_jitter_f3b_clean = (
                pd.Series(range(_group_points_f3b_clean.size))
                .map(lambda _i: 0.10 + ((_i % 11) - 5) * 0.018)
                .to_numpy()
            )
            _ax_f3b_clean.scatter(
                _idx_f3b_clean + _group_jitter_f3b_clean,
                _group_points_f3b_clean,
                s=15,
                color=_f3b_clean_palette[_group_f3b_clean],
                alpha=0.35,
                edgecolors="white",
                linewidth=0.35,
                zorder=3,
            )

    for _idx_f3b_clean, _group_f3b_clean in enumerate(_f3b_clean_order):
        _median_f3b_clean = float(_f3b_clean_group_medians.loc[_group_f3b_clean])
        _count_f3b_clean = (
            int(_f3b_clean_group_counts.loc[_group_f3b_clean])
            if pd.notna(_f3b_clean_group_counts.loc[_group_f3b_clean])
            else 0
        )
        _median_label_y_f3b_clean = _median_f3b_clean * 1.015
        if pd.notna(_median_f3b_clean):
            _ax_f3b_clean.text(
                _idx_f3b_clean + 0.13,
                _median_label_y_f3b_clean,
                f"Median = {_median_f3b_clean:.3f}",
                ha="left",
                va="bottom",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color="#7f1d1d",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                    "pad": 0.55,
                },
                zorder=5,
            )
            _ax_f3b_clean.text(
                _idx_f3b_clean,
                _f3b_clean_ymin - _f3b_clean_lower_pad * 0.45,
                f"n={_count_f3b_clean}",
                ha="center",
                va="top",
                fontsize=AF_ANNOTATION_FONT_SIZE,
                color="#475569",
                zorder=5,
            )

    _ax_f3b_clean.text(
        0.015,
        0.985,
        (
            f"Kruskal–Wallis: H={_f3b_clean_kw.statistic:.2f}, p={_f3b_clean_kw.pvalue:.3g}\n"
            f"High income vs others: U={_f3b_clean_u:.1f}, Cliff's δ={_f3b_clean_delta:.2f}, p={_f3b_clean_mw.pvalue:.3g}"
        ),
        transform=_ax_f3b_clean.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color="#374151",
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "linewidth": 0.6,
            "alpha": 0.94,
            "pad": 0.85,
        },
        zorder=6,
    )

    _ax_f3b_clean.set_xlabel("Income group", fontsize=AF_LABEL_FONT_SIZE)
    _ax_f3b_clean.set_ylabel(
        "log10(Eigenvector centrality)", fontsize=AF_LABEL_FONT_SIZE, labelpad=10
    )
    _ax_f3b_clean.set_xticks(_f3b_clean_positions)
    _ax_f3b_clean.set_xticklabels(
        [
            _f3b_clean_labels[_group_f3b_clean]
            for _group_f3b_clean in _f3b_clean_order
        ],
        fontsize=10,
    )
    _ax_f3b_clean.set_ylim(
        _f3b_clean_ymin - _f3b_clean_lower_pad,
        _f3b_clean_main_top + _f3b_clean_upper_pad,
    )
    _ax_f3b_clean.grid(
        axis="y", linestyle=(0, (2, 3)), linewidth=0.65, color="#d0d0d0", alpha=0.9
    )
    _ax_f3b_clean.grid(axis="x", visible=False)
    _ax_f3b_clean.set_axisbelow(True)
    _ax_f3b_clean.spines["top"].set_visible(False)
    _ax_f3b_clean.spines["right"].set_visible(False)
    _ax_f3b_clean.spines["left"].set_color("#6b7280")
    _ax_f3b_clean.spines["bottom"].set_color("#6b7280")
    _ax_f3b_clean.spines["left"].set_linewidth(0.8)
    _ax_f3b_clean.spines["bottom"].set_linewidth(0.8)
    _ax_f3b_clean.tick_params(axis="x", length=0, pad=8)
    _ax_f3b_clean.tick_params(axis="y", labelsize=9.8, colors="#2f2f2f")

    plt.gca()
    return


@app.cell(hide_code=True)
def f3d3_rd_share_print(
    AF_CYAN,
    AF_PURPLE,
    figure_3c2_rd_leadership_distribution,
):
    figure_f3d3_rd_authorship = figure_3c2_rd_leadership_distribution[
        figure_3c2_rd_leadership_distribution["metric"].isin(
            [
                "Participation",
                "First author",
                "Last author",
            ]
        )
    ].copy()

    _f3d3_rd_group_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f3d3_rd_metric_order = ["First author", "Last author"]
    _f3d3_rd_group_labels = ["High", "Middle", "Low"]
    _f3d3_rd_metric_palette = {
        "First author": AF_CYAN,
        "Last author": AF_PURPLE,
    }
    _f3d3_rd_metric_markers = {
        "First author": "o",
        "Last author": "D",
    }


    _f3d3_rd_ratio = (
        figure_f3d3_rd_authorship.pivot(
            index="rd_strength_tertile", columns="metric", values="share_pct"
        )
        .reindex(_f3d3_rd_group_order)
        .reset_index()
    )
    _f3d3_rd_ratio["first_author_ratio"] = (
        _f3d3_rd_ratio["First author"] / _f3d3_rd_ratio["Participation"]
    )
    _f3d3_rd_ratio["last_author_ratio"] = (
        _f3d3_rd_ratio["Last author"] / _f3d3_rd_ratio["Participation"]
    )

    figure_f3d3_rd_authorship = _f3d3_rd_ratio[
        [
            "rd_strength_tertile",
            "Participation",
            "First author",
            "Last author",
            "first_author_ratio",
            "last_author_ratio",
        ]
    ].copy()


    _f3d3_rd_share_print = (
        figure_f3d3_rd_authorship[
            [
                "rd_strength_tertile",
                "Participation",
                "First author",
                "Last author",
            ]
        ]
        .copy()
        .rename(
            columns={
                "Participation": "total_output_share_pct",
                "First author": "first_author_share_pct",
                "Last author": "last_author_share_pct",
            }
        )
    )

    _f3d3_rd_share_print = _f3d3_rd_share_print.set_index(
        "rd_strength_tertile"
    ).loc[["High R&D strength", "Middle R&D strength", "Low R&D strength"]]

    print("Total AF publication share (%)")
    for _group, _row in _f3d3_rd_share_print.iterrows():
        print(f"  {_group}: {_row['total_output_share_pct']:.6f}%")

    print("\nFirst-author share (%)")
    for _group, _row in _f3d3_rd_share_print.iterrows():
        print(f"  {_group}: {_row['first_author_share_pct']:.6f}%")

    print("\nLast-author share (%)")
    for _group, _row in _f3d3_rd_share_print.iterrows():
        print(f"  {_group}: {_row['last_author_share_pct']:.6f}%")
    return (figure_f3d3_rd_authorship,)


@app.cell(hide_code=True)
def fig_3_e(
    AF_ANNOTATION_FONT_SIZE,
    AF_CYAN,
    AF_PURPLE,
    figure_f3d3_rd_authorship,
    plt,
):
    _f3d3_rd_grouped_bar = figure_f3d3_rd_authorship[
        [
            "rd_strength_tertile",
            "first_author_ratio",
            "last_author_ratio",
        ]
    ].copy()

    _f3d3_rd_grouped_bar = _f3d3_rd_grouped_bar.set_index(
        "rd_strength_tertile"
    ).loc[["High R&D strength", "Middle R&D strength", "Low R&D strength"]]

    _f3d3_rd_bar_groups = ["High", "Middle", "Low"]
    _f3d3_rd_bar_x = list(range(len(_f3d3_rd_bar_groups)))
    _f3d3_rd_bar_width = 0.34
    _f3d3_rd_bar_palette = {
        "First author": AF_CYAN,
        "Last author": AF_PURPLE,
    }

    _fig_f3d3_rd_grouped_bar, _ax_f3d3_rd_grouped_bar = plt.subplots(
        figsize=(9.0, 6.9), dpi=260
    )
    _fig_f3d3_rd_grouped_bar.patch.set_facecolor("white")
    _ax_f3d3_rd_grouped_bar.set_facecolor("white")

    _f3d3_rd_first_vals = (
        _f3d3_rd_grouped_bar["first_author_ratio"].astype(float).tolist()
    )
    _f3d3_rd_last_vals = (
        _f3d3_rd_grouped_bar["last_author_ratio"].astype(float).tolist()
    )
    _f3d3_rd_first_pos = [_x - _f3d3_rd_bar_width / 2 for _x in _f3d3_rd_bar_x]
    _f3d3_rd_last_pos = [_x + _f3d3_rd_bar_width / 2 for _x in _f3d3_rd_bar_x]

    _ax_f3d3_rd_grouped_bar.axhline(
        1.0,
        color="#6b7280",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    _ax_f3d3_rd_grouped_bar.bar(
        _f3d3_rd_first_pos,
        _f3d3_rd_first_vals,
        width=_f3d3_rd_bar_width,
        color=_f3d3_rd_bar_palette["First author"],
        edgecolor="white",
        linewidth=0.9,
        label="First author",
        zorder=3,
    )
    _ax_f3d3_rd_grouped_bar.bar(
        _f3d3_rd_last_pos,
        _f3d3_rd_last_vals,
        width=_f3d3_rd_bar_width,
        color=_f3d3_rd_bar_palette["Last author"],
        edgecolor="white",
        linewidth=0.9,
        label="Last author",
        zorder=3,
    )

    for _pos, _val in zip(_f3d3_rd_first_pos, _f3d3_rd_first_vals):
        _ax_f3d3_rd_grouped_bar.text(
            _pos,
            _val + 0.018,
            f"{_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            zorder=5,
        )

    for _pos, _val in zip(_f3d3_rd_last_pos, _f3d3_rd_last_vals):
        _ax_f3d3_rd_grouped_bar.text(
            _pos,
            _val + 0.018,
            f"{_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            zorder=5,
        )

    # _ax_f3d3_rd_grouped_bar.text(
    #     0.015,
    #     0.985,
    #     "Dashed line marks parity with each group's share of total AF publication output.",
    #     transform=_ax_f3d3_rd_grouped_bar.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=AF_ANNOTATION_FONT_SIZE,
    #     color="#374151",
    #     bbox={
    #         "facecolor": "white",
    #         "edgecolor": "#d1d5db",
    #         "linewidth": 0.6,
    #         "alpha": 0.94,
    #         "pad": 0.7,
    #     },
    #     zorder=6,
    # )

    _ax_f3d3_rd_grouped_bar.set_xticks(_f3d3_rd_bar_x)
    _ax_f3d3_rd_grouped_bar.set_xticklabels(_f3d3_rd_bar_groups)
    _ax_f3d3_rd_grouped_bar.set_xlabel(
        "Country R&D-strength group",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d3_rd_grouped_bar.set_ylabel(
        "Representation ratio\n(authorship share / total AF output share)",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d3_rd_grouped_bar.set_ylim(0, 1.12)
    _ax_f3d3_rd_grouped_bar.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _ax_f3d3_rd_grouped_bar.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#d1d5db",
        alpha=0.92,
    )
    _ax_f3d3_rd_grouped_bar.grid(axis="x", visible=False)
    _ax_f3d3_rd_grouped_bar.set_axisbelow(True)
    _ax_f3d3_rd_grouped_bar.spines["top"].set_visible(False)
    _ax_f3d3_rd_grouped_bar.spines["right"].set_visible(False)
    _ax_f3d3_rd_grouped_bar.spines["left"].set_color("#374151")
    _ax_f3d3_rd_grouped_bar.spines["bottom"].set_color("#374151")
    _ax_f3d3_rd_grouped_bar.spines["left"].set_linewidth(0.8)
    _ax_f3d3_rd_grouped_bar.spines["bottom"].set_linewidth(0.8)
    _ax_f3d3_rd_grouped_bar.tick_params(
        axis="both", labelsize=10, colors="#1f2937"
    )
    _ax_f3d3_rd_grouped_bar.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.70, 1.01),
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0,
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )

    _fig_f3d3_rd_grouped_bar.subplots_adjust(
        left=0.17, right=0.84, top=0.84, bottom=0.15
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_3_f(
    AF_ANNOTATION_FONT_SIZE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_PURPLE,
    figure_3c3_rd_country_metrics,
    pd,
    plt,
):
    figure_f3d4_rd_country_authorship = figure_3c3_rd_country_metrics[
        [
            "country_code",
            "rd_strength_tertile",
            "first_author_share_pct",
            "last_author_share_pct",
        ]
    ].copy()

    _f3d4_rd_order = [
        "Low R&D strength",
        "Middle R&D strength",
        "High R&D strength",
    ]
    _f3d4_rd_group_labels = ["Low", "Middle", "High"]
    _f3d4_rd_metric_palette = {
        "First author": AF_CYAN,
        "Last author": AF_PURPLE,
    }
    _f3d4_rd_hue_order = ["First author", "Last author"]

    figure_f3d4_rd_country_authorship = figure_f3d4_rd_country_authorship.melt(
        id_vars=["country_code", "rd_strength_tertile"],
        value_vars=["first_author_share_pct", "last_author_share_pct"],
        var_name="metric",
        value_name="share_pct",
    )
    figure_f3d4_rd_country_authorship["metric"] = (
        figure_f3d4_rd_country_authorship["metric"].map(
            {
                "first_author_share_pct": "First author",
                "last_author_share_pct": "Last author",
            }
        )
    )
    figure_f3d4_rd_country_authorship = figure_f3d4_rd_country_authorship.dropna(
        subset=["rd_strength_tertile", "metric", "share_pct"]
    ).copy()
    figure_f3d4_rd_country_authorship["rd_strength_tertile"] = pd.Categorical(
        figure_f3d4_rd_country_authorship["rd_strength_tertile"],
        categories=_f3d4_rd_order,
        ordered=True,
    )
    figure_f3d4_rd_country_authorship["metric"] = pd.Categorical(
        figure_f3d4_rd_country_authorship["metric"],
        categories=_f3d4_rd_hue_order,
        ordered=True,
    )
    figure_f3d4_rd_country_authorship = (
        figure_f3d4_rd_country_authorship.sort_values(
            ["rd_strength_tertile", "metric", "share_pct"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    )

    _f3d4_rd_summary = (
        figure_f3d4_rd_country_authorship.groupby(
            ["rd_strength_tertile", "metric"], observed=False
        )["share_pct"]
        .median()
        .reset_index()
    )

    _fig_f3d4_rd_country_authorship, _ax_f3d4_rd_country_authorship = plt.subplots(
        figsize=(10.4, 7.4), dpi=240
    )
    _fig_f3d4_rd_country_authorship.patch.set_facecolor("white")
    _ax_f3d4_rd_country_authorship.set_facecolor("white")

    _f3d4_rd_x = list(range(len(_f3d4_rd_order)))
    _f3d4_rd_offsets = {"First author": -0.19, "Last author": 0.19}
    _f3d4_rd_box_width = 0.24

    for _metric_f3d4 in _f3d4_rd_hue_order:
        _metric_series = []
        _metric_positions = []
        for _idx_f3d4, _group_f3d4 in enumerate(_f3d4_rd_order):
            _metric_values = (
                figure_f3d4_rd_country_authorship.loc[
                    (
                        figure_f3d4_rd_country_authorship["rd_strength_tertile"]
                        == _group_f3d4
                    )
                    & (
                        figure_f3d4_rd_country_authorship["metric"] == _metric_f3d4
                    ),
                    "share_pct",
                ]
                .dropna()
                .to_numpy()
            )
            _metric_series.append(_metric_values)
            _metric_positions.append(
                _f3d4_rd_x[_idx_f3d4] + _f3d4_rd_offsets[_metric_f3d4]
            )

        _box_f3d4 = _ax_f3d4_rd_country_authorship.boxplot(
            _metric_series,
            positions=_metric_positions,
            widths=_f3d4_rd_box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1f2937", "linewidth": 1.8},
            boxprops={
                "facecolor": _f3d4_rd_metric_palette[_metric_f3d4],
                "alpha": 0.82,
                "edgecolor": _f3d4_rd_metric_palette[_metric_f3d4],
                "linewidth": 1.0,
            },
            whiskerprops={"color": "#475569", "linewidth": 0.9},
            capprops={"color": "#475569", "linewidth": 0.9},
        )
        for _patch_f3d4 in _box_f3d4["boxes"]:
            _patch_f3d4.set_zorder(3)
        for _line_group_f3d4 in [
            _box_f3d4["whiskers"],
            _box_f3d4["caps"],
            _box_f3d4["medians"],
        ]:
            for _line_f3d4 in _line_group_f3d4:
                _line_f3d4.set_zorder(4)

        for _idx_f3d4, _group_f3d4 in enumerate(_f3d4_rd_order):
            _metric_points = (
                figure_f3d4_rd_country_authorship.loc[
                    (
                        figure_f3d4_rd_country_authorship["rd_strength_tertile"]
                        == _group_f3d4
                    )
                    & (
                        figure_f3d4_rd_country_authorship["metric"] == _metric_f3d4
                    ),
                    "share_pct",
                ]
                .dropna()
                .to_numpy()
            )
            if _metric_points.size:
                _jitter_f3d4 = (
                    pd.Series(range(_metric_points.size))
                    .map(lambda _i: ((_i % 9) - 4) * 0.012)
                    .to_numpy()
                )
                _ax_f3d4_rd_country_authorship.scatter(
                    _metric_positions[_idx_f3d4] + _jitter_f3d4,
                    _metric_points,
                    s=22,
                    color=_f3d4_rd_metric_palette[_metric_f3d4],
                    alpha=0.42,
                    edgecolors="white",
                    linewidth=0.35,
                    zorder=5,
                )

    for _row_f3d4 in _f3d4_rd_summary.itertuples(index=False):
        _group_idx_f3d4 = _f3d4_rd_order.index(_row_f3d4.rd_strength_tertile)
        _x_f3d4 = _group_idx_f3d4 + _f3d4_rd_offsets[_row_f3d4.metric]
        _ax_f3d4_rd_country_authorship.text(
            _x_f3d4,
            float(_row_f3d4.share_pct) + 2.0,
            f"Median {float(_row_f3d4.share_pct):.1f}%",
            ha="center",
            va="bottom",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_GUIDE_NEUTRAL,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.28,
            },
            zorder=6,
        )

    # _ax_f3d4_rd_country_authorship.text(
    #     0.015,
    #     0.985,
    #     "Country-level distributions reveal substantial within-group heterogeneity in authorship ratios.",
    #     transform=_ax_f3d4_rd_country_authorship.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=AF_ANNOTATION_FONT_SIZE,
    #     color="#1f2937",
    #     bbox={
    #         "facecolor": "white",
    #         "edgecolor": "#d1d5db",
    #         "linewidth": 0.6,
    #         "alpha": 0.94,
    #         "pad": 0.75,
    #     },
    #     zorder=6,
    # )

    _f3d4_rd_legend_handles = [
        plt.matplotlib.patches.Patch(
            color=_f3d4_rd_metric_palette[_metric], label=_metric
        )
        for _metric in _f3d4_rd_hue_order
    ]
    _ax_f3d4_rd_country_authorship.legend(
        _f3d4_rd_legend_handles,
        _f3d4_rd_hue_order,
        frameon=False,
        loc="upper right",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )

    _ax_f3d4_rd_country_authorship.set_xticks(_f3d4_rd_x)
    _ax_f3d4_rd_country_authorship.set_xticklabels(_f3d4_rd_group_labels)
    _ax_f3d4_rd_country_authorship.set_xlabel(
        "Country R&D-strength group",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d4_rd_country_authorship.set_ylabel(
        "Country-level authorship ratio (%)",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_f3d4_rd_country_authorship.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#d1d5db",
        alpha=0.92,
    )
    _ax_f3d4_rd_country_authorship.grid(axis="x", visible=False)
    _ax_f3d4_rd_country_authorship.set_axisbelow(True)
    _ax_f3d4_rd_country_authorship.spines["top"].set_visible(False)
    _ax_f3d4_rd_country_authorship.spines["right"].set_visible(False)
    _ax_f3d4_rd_country_authorship.spines["left"].set_color("#374151")
    _ax_f3d4_rd_country_authorship.spines["bottom"].set_color("#374151")
    _ax_f3d4_rd_country_authorship.spines["left"].set_linewidth(0.8)
    _ax_f3d4_rd_country_authorship.spines["bottom"].set_linewidth(0.8)
    _ax_f3d4_rd_country_authorship.tick_params(
        axis="both", labelsize=10, colors="#1f2937"
    )

    _fig_f3d4_rd_country_authorship.subplots_adjust(
        left=0.13, right=0.97, top=0.9, bottom=0.14
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    # _last_author_output_dir = __import__("pathlib").Path("derived_tables_dedup")
    # _last_author_output_dir.mkdir(parents=True, exist_ok=True)

    # _last_author_country_credit_df = duckdb.sql(
    #     """
    #     WITH last_author_rows AS (
    #         SELECT DISTINCT
    #             work_id,
    #             author_id
    #         FROM read_parquet('derived_tables_dedup/authorships.parquet')
    #         WHERE work_id IS NOT NULL
    #           AND author_id IS NOT NULL
    #           AND author_position = 'last'
    #     ),
    #     last_author_countries AS (
    #         SELECT DISTINCT
    #             lac.work_id,
    #             UPPER(TRIM(lac.country_code)) AS country_code,
    #             lac.country_source
    #         FROM read_parquet('derived_tables_dedup/authorship_countries.parquet') AS lac
    #         INNER JOIN last_author_rows AS lar
    #             ON lac.work_id = lar.work_id
    #            AND lac.author_id = lar.author_id
    #         WHERE lac.country_code IS NOT NULL
    #           AND TRIM(lac.country_code) <> ''
    #     ),
    #     last_author_country_counts AS (
    #         SELECT
    #             work_id,
    #             COUNT(DISTINCT country_code) AS n_last_author_countries
    #         FROM last_author_countries
    #         GROUP BY work_id
    #     ),
    #     last_author_country_source_ranked AS (
    #         SELECT
    #             work_id,
    #             country_code,
    #             country_source,
    #             ROW_NUMBER() OVER (
    #                 PARTITION BY work_id, country_code
    #                 ORDER BY CASE country_source
    #                     WHEN 'authorship_countries_field' THEN 1
    #                     WHEN 'institution_country_code' THEN 2
    #                     ELSE 3
    #                 END,
    #                 country_source
    #             ) AS source_rank
    #         FROM last_author_countries
    #     )
    #     SELECT
    #         s.work_id,
    #         s.country_code,
    #         1.0 / c.n_last_author_countries AS last_author_fraction,
    #         c.n_last_author_countries,
    #         s.country_source AS last_author_country_source
    #     FROM last_author_country_source_ranked AS s
    #     INNER JOIN last_author_country_counts AS c
    #         ON s.work_id = c.work_id
    #     WHERE s.source_rank = 1
    #     ORDER BY s.work_id, s.country_code
    #     """
    # ).df()

    # _last_author_diagnostics_df = duckdb.sql(
    #     """
    #     WITH last_author_authorships AS (
    #         SELECT
    #             work_id,
    #             author_id
    #         FROM read_parquet('derived_tables_dedup/authorships.parquet')
    #         WHERE work_id IS NOT NULL
    #           AND author_id IS NOT NULL
    #           AND author_position = 'last'
    #     ),
    #     last_author_country_rows AS (
    #         SELECT DISTINCT
    #             lac.work_id,
    #             lar.author_id,
    #             UPPER(TRIM(lac.country_code)) AS country_code,
    #             lac.country_source
    #         FROM read_parquet('derived_tables_dedup/authorship_countries.parquet') AS lac
    #         INNER JOIN last_author_authorships AS lar
    #             ON lac.work_id = lar.work_id
    #            AND lac.author_id = lar.author_id
    #         WHERE lac.country_code IS NOT NULL
    #           AND TRIM(lac.country_code) <> ''
    #     ),
    #     last_author_authorship_counts AS (
    #         SELECT
    #             work_id,
    #             COUNT(*) AS n_last_author_authorship_rows
    #         FROM last_author_authorships
    #         GROUP BY work_id
    #     ),
    #     last_author_country_agg AS (
    #         SELECT
    #             work_id,
    #             COUNT(DISTINCT country_code) AS last_author_country_count,
    #             ARRAY_SORT(ARRAY_DISTINCT(LIST(country_source))) AS last_author_country_sources,
    #             ARRAY_SORT(ARRAY_DISTINCT(LIST(country_code))) AS last_author_country_codes
    #         FROM last_author_country_rows
    #         GROUP BY work_id
    #     )
    #     SELECT
    #         w.work_id,
    #         COALESCE(a.n_last_author_authorship_rows, 0) > 0 AS has_last_author,
    #         COALESCE(a.n_last_author_authorship_rows, 0) AS n_last_author_authorship_rows,
    #         COALESCE(c.last_author_country_count, 0) AS last_author_country_count,
    #         COALESCE(a.n_last_author_authorship_rows, 0) > 0
    #             AND COALESCE(c.last_author_country_count, 0) = 0 AS last_author_country_missing,
    #         COALESCE(c.last_author_country_sources, []) AS last_author_country_sources,
    #         COALESCE(c.last_author_country_codes, []) AS last_author_country_codes
    #     FROM (
    #         SELECT DISTINCT work_id
    #         FROM read_parquet('derived_tables_dedup/authorships.parquet')
    #         WHERE work_id IS NOT NULL
    #     ) AS w
    #     LEFT JOIN last_author_authorship_counts AS a
    #         ON w.work_id = a.work_id
    #     LEFT JOIN last_author_country_agg AS c
    #         ON w.work_id = c.work_id
    #     ORDER BY w.work_id
    #     """
    # ).df()

    # _last_author_country_credit_path = (
    #     _last_author_output_dir / "last_author_country_credit.parquet"
    # )
    # _last_author_diagnostics_path = (
    #     _last_author_output_dir / "last_author_country_diagnostics.parquet"
    # )
    # _last_author_country_credit_df.to_parquet(
    #     _last_author_country_credit_path, index=False
    # )
    # _last_author_diagnostics_df.to_parquet(
    #     _last_author_diagnostics_path, index=False
    # )

    # last_author_parquet_summary = pd.DataFrame(
    #     [
    #         {
    #             "table_name": "last_author_country_credit",
    #             "rows": len(_last_author_country_credit_df),
    #             "path": str(_last_author_country_credit_path),
    #         },
    #         {
    #             "table_name": "last_author_country_diagnostics",
    #             "rows": len(_last_author_diagnostics_df),
    #             "path": str(_last_author_diagnostics_path),
    #         },
    #     ]
    # )

    # last_author_parquet_summary
    return


@app.cell(hide_code=True)
def _(country_af_output, country_rd_strength_lookup, duckdb, pd):
    figure_3c2_rd_leadership_distribution = duckdb.sql(
        """
        WITH af_participation AS (
            SELECT
                r.rd_strength_tertile,
                SUM(c.af_fractional_count) AS metric_weight
            FROM country_af_output AS c
            INNER JOIN country_rd_strength_lookup AS r
                ON c.country_code = r.country_code
            GROUP BY r.rd_strength_tertile
        ),
        af_first_author AS (
            SELECT
                r.rd_strength_tertile,
                SUM(f.first_author_fraction) AS metric_weight
            FROM read_parquet('derived_tables_dedup/first_author_country_credit.parquet') AS f
            INNER JOIN read_parquet('derived_tables_dedup/works.parquet') AS w
                ON f.work_id = w.work_id
            INNER JOIN country_rd_strength_lookup AS r
                ON UPPER(TRIM(f.country_code)) = r.country_code
            WHERE w.work_id IS NOT NULL
              AND w.is_alphafold_related = TRUE
              AND f.country_code IS NOT NULL
              AND TRIM(f.country_code) <> ''
            GROUP BY r.rd_strength_tertile
        ),
        af_last_author AS (
            SELECT
                r.rd_strength_tertile,
                SUM(l.last_author_fraction) AS metric_weight
            FROM read_parquet('derived_tables_dedup/last_author_country_credit.parquet') AS l
            INNER JOIN read_parquet('derived_tables_dedup/works.parquet') AS w
                ON l.work_id = w.work_id
            INNER JOIN country_rd_strength_lookup AS r
                ON UPPER(TRIM(l.country_code)) = r.country_code
            WHERE w.work_id IS NOT NULL
              AND w.is_alphafold_related = TRUE
              AND l.country_code IS NOT NULL
              AND TRIM(l.country_code) <> ''
            GROUP BY r.rd_strength_tertile
        ),
        combined AS (
            SELECT 'Participation' AS metric, rd_strength_tertile, metric_weight FROM af_participation
            UNION ALL
            SELECT 'First author' AS metric, rd_strength_tertile, metric_weight FROM af_first_author
            UNION ALL
            SELECT 'Last author' AS metric, rd_strength_tertile, metric_weight FROM af_last_author
        ),
        metric_totals AS (
            SELECT metric, SUM(metric_weight) AS total_metric_weight
            FROM combined
            GROUP BY metric
        )
        SELECT
            c.metric,
            c.rd_strength_tertile,
            c.metric_weight,
            t.total_metric_weight,
            100.0 * c.metric_weight / NULLIF(t.total_metric_weight, 0) AS share_pct
        FROM combined AS c
        INNER JOIN metric_totals AS t
            ON c.metric = t.metric
        """
    ).df()

    _figure_3c2_metric_order = ["Participation", "First author", "Last author"]
    _figure_3c2_group_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    figure_3c2_rd_leadership_distribution["metric"] = pd.Categorical(
        figure_3c2_rd_leadership_distribution["metric"],
        categories=_figure_3c2_metric_order,
        ordered=True,
    )
    figure_3c2_rd_leadership_distribution["rd_strength_tertile"] = pd.Categorical(
        figure_3c2_rd_leadership_distribution["rd_strength_tertile"],
        categories=_figure_3c2_group_order,
        ordered=True,
    )
    figure_3c2_rd_leadership_distribution = (
        figure_3c2_rd_leadership_distribution.sort_values(
            ["rd_strength_tertile", "metric"]
        ).reset_index(drop=True)
    )

    figure_3c2_rd_leadership_distribution
    return (figure_3c2_rd_leadership_distribution,)


@app.cell(hide_code=True)
def _(figure_3c2_rd_leadership_distribution, pd):
    _f3c2_wide = figure_3c2_rd_leadership_distribution.pivot_table(
        index="rd_strength_tertile",
        columns="metric",
        values="share_pct",
    )
    figure_3c2_rd_leadership_premium = _f3c2_wide.reset_index().rename_axis(
        columns=None
    )
    figure_3c2_rd_leadership_premium = figure_3c2_rd_leadership_premium.rename(
        columns={
            "Participation": "participation_share_pct",
            "First author": "first_author_share_pct",
            "Last author": "last_author_share_pct",
        }
    )
    figure_3c2_rd_leadership_premium["first_author_premium_pp"] = (
        figure_3c2_rd_leadership_premium["first_author_share_pct"]
        - figure_3c2_rd_leadership_premium["participation_share_pct"]
    )
    figure_3c2_rd_leadership_premium["last_author_premium_pp"] = (
        figure_3c2_rd_leadership_premium["last_author_share_pct"]
        - figure_3c2_rd_leadership_premium["participation_share_pct"]
    )
    figure_3c2_rd_leadership_premium = figure_3c2_rd_leadership_premium[
        [
            "rd_strength_tertile",
            "participation_share_pct",
            "first_author_share_pct",
            "last_author_share_pct",
            "first_author_premium_pp",
            "last_author_premium_pp",
        ]
    ].copy()
    figure_3c2_rd_leadership_premium["rd_strength_tertile"] = pd.Categorical(
        figure_3c2_rd_leadership_premium["rd_strength_tertile"],
        categories=[
            "High R&D strength",
            "Middle R&D strength",
            "Low R&D strength",
        ],
        ordered=True,
    )
    figure_3c2_rd_leadership_premium = (
        figure_3c2_rd_leadership_premium.sort_values(
            "rd_strength_tertile"
        ).reset_index(drop=True)
    )

    figure_3c2_rd_leadership_premium
    return


@app.cell(hide_code=True)
def _(duckdb):
    figure_3e_country_last_author_base = duckdb.sql(
        """
        WITH af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT work_id, COUNT(DISTINCT country_code) AS n_countries
            FROM work_country
            GROUP BY work_id
        ),
        country_af_output AS (
            SELECT
                c.country_code,
                SUM(1.0 / cc.n_countries) AS af_output_fractional
            FROM af_works AS w
            INNER JOIN work_country AS c
                ON w.work_id = c.work_id
            INNER JOIN work_country_counts AS cc
                ON w.work_id = cc.work_id
            WHERE cc.n_countries > 0
            GROUP BY c.country_code
        ),
        country_last_author_output AS (
            SELECT
                UPPER(TRIM(country_code)) AS country_code,
                SUM(last_author_fraction) AS last_author_output_fractional
            FROM read_parquet('derived_tables_dedup/last_author_country_credit.parquet')
            WHERE work_id IN (SELECT work_id FROM af_works)
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
            GROUP BY UPPER(TRIM(country_code))
        )
        SELECT
            o.country_code,
            o.af_output_fractional,
            COALESCE(l.last_author_output_fractional, 0.0) AS last_author_output_fractional
        FROM country_af_output AS o
        LEFT JOIN country_last_author_output AS l
            ON o.country_code = l.country_code
        """
    ).df()

    figure_3e_country_last_author_base["country_code"] = (
        figure_3e_country_last_author_base["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    figure_3e_country_last_author_base["last_author_share_pct"] = (
        100
        * figure_3e_country_last_author_base["last_author_output_fractional"]
        / figure_3e_country_last_author_base["af_output_fractional"]
    )

    figure_3e_country_last_author_base
    return (figure_3e_country_last_author_base,)


@app.cell(hide_code=True)
def _(
    country_af_output,
    country_rd_strength_lookup,
    figure_3b_centrality,
    figure_3e_country_first_author_binary_base,
    math,
):
    figure_3e_rd_strength_binary_plot = (
        figure_3e_country_first_author_binary_base.merge(
            figure_3b_centrality[["country_code", "eigenvector_centrality"]],
            on="country_code",
            how="inner",
        )
        .merge(
            country_rd_strength_lookup[
                ["country_code", "rd_strength_tertile", "rd_strength_index"]
            ],
            on="country_code",
            how="inner",
        )
        .merge(
            country_af_output,
            on="country_code",
            how="left",
        )
        .dropna(
            subset=[
                "eigenvector_centrality",
                "first_author_binary_share_pct",
                "rd_strength_tertile",
                "af_fractional_count",
            ]
        )
        .copy()
    )
    figure_3e_rd_strength_binary_plot = figure_3e_rd_strength_binary_plot[
        figure_3e_rd_strength_binary_plot["af_fractional_count"] >= 5
    ].copy()
    _figure_3e_rd_binary_floor = max(
        float(figure_3e_rd_strength_binary_plot["eigenvector_centrality"].min())
        * 0.5,
        1e-6,
    )
    figure_3e_rd_strength_binary_plot["log10_eigenvector_centrality"] = (
        figure_3e_rd_strength_binary_plot["eigenvector_centrality"]
        .clip(lower=_figure_3e_rd_binary_floor)
        .map(lambda _v: math.log10(_v))
    )
    figure_3e_rd_strength_binary_plot["bubble_size"] = (
        28
        + 12
        * figure_3e_rd_strength_binary_plot["af_fractional_count"].map(
            lambda _v: math.sqrt(_v) if _v >= 0 else 0
        )
    ).clip(upper=240)
    figure_3e_rd_strength_binary_plot["reg_weight"] = (
        figure_3e_rd_strength_binary_plot["af_fractional_count"]
        .clip(lower=1)
        .map(lambda _v: math.sqrt(_v))
    )
    figure_3e_rd_strength_binary_plot = (
        figure_3e_rd_strength_binary_plot.sort_values(
            ["rd_strength_index", "af_fractional_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
    )

    figure_3e_rd_strength_binary_plot
    return


@app.cell(hide_code=True)
def _(
    country_adoption_lag_plot,
    country_af_nonaf_compare,
    figure_3b_centrality,
    figure_3e_country_first_author_base,
):
    aiac_country_3d = country_adoption_lag_plot[
        ["country_code", "adoption_lag_months"]
    ].copy()
    aiac_country_3d["country_code"] = (
        aiac_country_3d["country_code"].astype(str).str.strip().str.upper()
    )
    aiac_country_3d = aiac_country_3d.merge(
        figure_3b_centrality[
            ["country_code", "eigenvector_centrality", "income_group"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_3d = aiac_country_3d.merge(
        country_af_nonaf_compare[
            ["country_code", "af_fractional_count", "country_name"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_3d = aiac_country_3d.merge(
        figure_3e_country_first_author_base[
            ["country_code", "first_author_share_pct"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_3d = aiac_country_3d.dropna(
        subset=[
            "adoption_lag_months",
            "eigenvector_centrality",
            "af_fractional_count",
            "income_group",
        ]
    ).copy()
    aiac_country_3d["adoption_speed_norm"] = 1 - (
        (
            aiac_country_3d["adoption_lag_months"]
            - aiac_country_3d["adoption_lag_months"].min()
        )
        / (
            aiac_country_3d["adoption_lag_months"].max()
            - aiac_country_3d["adoption_lag_months"].min()
        )
    )
    aiac_country_3d["output_log10"] = (
        aiac_country_3d["af_fractional_count"] + 1
    ).map(lambda _v: __import__("math").log10(_v))
    aiac_country_3d["output_norm"] = (
        (
            aiac_country_3d["af_fractional_count"]
            - aiac_country_3d["af_fractional_count"].min()
        )
        / (
            aiac_country_3d["af_fractional_count"].max()
            - aiac_country_3d["af_fractional_count"].min()
        )
    )
    aiac_country_3d["centrality_log10"] = aiac_country_3d[
        "eigenvector_centrality"
    ].clip(
        lower=max(
            float(aiac_country_3d["eigenvector_centrality"].min()) * 0.5,
            1e-6,
        )
    ).map(lambda _v: __import__("math").log10(_v))
    aiac_country_3d["eigenvector_norm"] = (
        (
            aiac_country_3d["eigenvector_centrality"]
            - aiac_country_3d["eigenvector_centrality"].min()
        )
        / (
            aiac_country_3d["eigenvector_centrality"].max()
            - aiac_country_3d["eigenvector_centrality"].min()
        )
    )
    if aiac_country_3d["first_author_share_pct"].notna().any():
        _aiac_first_author_norm = (
            (
                aiac_country_3d["first_author_share_pct"]
                - aiac_country_3d["first_author_share_pct"].min()
            )
            / (
                aiac_country_3d["first_author_share_pct"].max()
                - aiac_country_3d["first_author_share_pct"].min()
            )
        )
        aiac_country_3d["network_norm"] = (
            aiac_country_3d["eigenvector_norm"]
            + _aiac_first_author_norm.fillna(aiac_country_3d["eigenvector_norm"])
        ) / 2
    else:
        aiac_country_3d["network_norm"] = aiac_country_3d["eigenvector_norm"]
    aiac_country_3d["bubble_size"] = 28 + 240 * aiac_country_3d["output_norm"]**0.9
    aiac_country_3d
    return (aiac_country_3d,)


@app.cell(hide_code=True)
def _(aiac_country_3d, mo, pd, stats):
    _aiac_metrics_plot_df = aiac_country_3d.copy()
    _aiac_metrics_plot_df = _aiac_metrics_plot_df[
        _aiac_metrics_plot_df["income_group"].isin(
            [
                "High income",
                "Upper-middle income",
                "Lower-middle income",
                "Low income",
            ]
        )
    ].copy()

    _aiac_metrics_x = _aiac_metrics_plot_df["centrality_log10"].astype(float).to_numpy()
    _aiac_metrics_y = _aiac_metrics_plot_df["adoption_speed_norm"].astype(float).to_numpy()
    _aiac_metrics_w = _aiac_metrics_plot_df["af_fractional_count"].clip(lower=1).map(
        lambda _v: __import__("math").sqrt(_v)
    ).to_numpy()
    _aiac_metrics_x_design = __import__("numpy").column_stack(
        [__import__("numpy").ones_like(_aiac_metrics_x), _aiac_metrics_x]
    )
    _aiac_metrics_w_matrix = __import__("numpy").diag(_aiac_metrics_w)
    _aiac_metrics_beta = __import__("numpy").linalg.solve(
        _aiac_metrics_x_design.T @ _aiac_metrics_w_matrix @ _aiac_metrics_x_design,
        _aiac_metrics_x_design.T @ _aiac_metrics_w_matrix @ _aiac_metrics_y,
    )
    _aiac_metrics_fit = _aiac_metrics_x_design @ _aiac_metrics_beta
    _aiac_metrics_resid = _aiac_metrics_y - _aiac_metrics_fit
    _aiac_metrics_df = max(len(_aiac_metrics_x) - 2, 1)
    _aiac_metrics_sigma2 = float(
        (_aiac_metrics_w * (_aiac_metrics_resid**2)).sum() / _aiac_metrics_df
    )
    _aiac_metrics_cov = _aiac_metrics_sigma2 * __import__("numpy").linalg.inv(
        _aiac_metrics_x_design.T @ _aiac_metrics_w_matrix @ _aiac_metrics_x_design
    )
    _aiac_metrics_slope = float(_aiac_metrics_beta[1])
    _aiac_metrics_slope_se = float(
        __import__("numpy").sqrt(max(_aiac_metrics_cov[1, 1], 0))
    )
    _aiac_metrics_t = (
        _aiac_metrics_slope / _aiac_metrics_slope_se
        if _aiac_metrics_slope_se > 0
        else float("nan")
    )
    _aiac_metrics_pval = (
        2 * (1 - stats.norm.cdf(abs(_aiac_metrics_t)))
        if _aiac_metrics_slope_se > 0
        else float("nan")
    )
    _aiac_metrics_r2 = 1 - float((_aiac_metrics_resid**2).sum()) / max(
        float(((_aiac_metrics_y - _aiac_metrics_y.mean()) ** 2).sum()),
        1e-9,
    )
    _aiac_metrics_x_med = float(_aiac_metrics_plot_df["centrality_log10"].median())
    _aiac_metrics_y_med = float(_aiac_metrics_plot_df["adoption_speed_norm"].median())
    _aiac_metrics_output_min = float(_aiac_metrics_plot_df["af_fractional_count"].min())
    _aiac_metrics_output_max = float(_aiac_metrics_plot_df["af_fractional_count"].max())
    _aiac_metrics_output_range = max(
        _aiac_metrics_output_max - _aiac_metrics_output_min,
        1e-9,
    )

    _aiac_metrics_table = pd.DataFrame(
        {
            "metric": [
                "slope",
                "R_squared",
                "p_value",
                "x_median_log10_eigenvector",
                "y_median_adoption_speed_norm",
                "n_countries",
            ],
            "value": [
                _aiac_metrics_slope,
                _aiac_metrics_r2,
                _aiac_metrics_pval,
                _aiac_metrics_x_med,
                _aiac_metrics_y_med,
                int(len(_aiac_metrics_plot_df)),
            ],
        }
    )

    _aiac_size_reference_table = pd.DataFrame(
        {"AF_papers_reference": [100, 1000, 5000]}
    )
    _aiac_size_reference_table["clipped_reference"] = _aiac_size_reference_table[
        "AF_papers_reference"
    ].clip(lower=_aiac_metrics_output_min, upper=_aiac_metrics_output_max)
    _aiac_size_reference_table["output_norm"] = (
        _aiac_size_reference_table["clipped_reference"] - _aiac_metrics_output_min
    ) / _aiac_metrics_output_range
    _aiac_size_reference_table["scatter_area_s"] = 28 + 240 * (
        _aiac_size_reference_table["output_norm"] ** 0.9
    )
    _aiac_size_reference_table["legend_markersize"] = _aiac_size_reference_table[
        "scatter_area_s"
    ].map(lambda _v: float(_v) ** 0.5)

    mo.vstack(
        [
            _aiac_metrics_table,
            _aiac_size_reference_table,
        ]
    )
    return


@app.cell(hide_code=True)
def _(aiac_country_3d, pd):
    aiac_radar_base = aiac_country_3d.copy()
    aiac_radar_base["income_group"] = pd.Categorical(
        aiac_radar_base["income_group"],
        categories=[
            "High income",
            "Upper-middle income",
            "Lower-middle income",
            "Low income",
        ],
        ordered=True,
    )

    aiac_radar_group_raw = (
        aiac_radar_base.groupby("income_group", as_index=False)
        .agg(
            output_mean=("af_fractional_count", "mean"),
            timing_mean=("adoption_speed_norm", "mean"),
            network_mean=("network_norm", "mean"),
        )
        .sort_values("income_group")
    )

    aiac_radar_group_raw["output_log10_mean"] = aiac_radar_group_raw[
        "output_mean"
    ].map(lambda _v: __import__("math").log10(_v + 1))

    aiac_radar_summary = aiac_radar_group_raw[["income_group"]].copy()
    for _col_aiac_radar_src, _col_aiac_radar_dst in [
        ("output_log10_mean", "Output"),
        ("timing_mean", "Timing"),
        ("network_mean", "Network"),
    ]:
        _vals_aiac_radar = aiac_radar_group_raw[_col_aiac_radar_src].astype(float)
        aiac_radar_summary[_col_aiac_radar_dst] = (
            (_vals_aiac_radar - _vals_aiac_radar.min())
            / max(_vals_aiac_radar.max() - _vals_aiac_radar.min(), 1e-9)
        )

    aiac_radar_summary
    return (aiac_radar_group_raw,)


@app.cell(hide_code=True)
def _(aiac_radar_group_raw):
    aiac_radar_raw_table = aiac_radar_group_raw[
        [
            "income_group",
            "output_mean",
            "output_log10_mean",
            "timing_mean",
            "network_mean",
        ]
    ].copy()
    aiac_radar_raw_table = aiac_radar_raw_table.rename(
        columns={
            "income_group": "Income group",
            "output_mean": "AF output mean",
            "output_log10_mean": "log10(AF output mean + 1)",
            "timing_mean": "Adoption speed mean",
            "network_mean": "Network influence mean",
        }
    )
    aiac_radar_raw_table
    return


@app.cell(hide_code=True)
def _(
    country_adoption_lag,
    country_af_nonaf_compare,
    country_rd_strength_lookup,
    duckdb,
    figure_3b_centrality,
    figure_3e_country_first_author_base,
    figure_3e_country_last_author_base,
    pd,
):
    _aiac_rd_citations = duckdb.sql(
        """
        WITH af_works AS (
            SELECT
                work_id,
                COALESCE(cited_by_count, 0) AS cited_by_count
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        work_country_counts AS (
            SELECT
                work_id,
                COUNT(DISTINCT country_code) AS n_countries
            FROM work_country
            GROUP BY work_id
        )
        SELECT
            c.country_code,
            SUM(w.cited_by_count * 1.0 / cc.n_countries) AS af_fractional_citations
        FROM af_works AS w
        INNER JOIN work_country AS c
            ON w.work_id = c.work_id
        INNER JOIN work_country_counts AS cc
            ON w.work_id = cc.work_id
        WHERE cc.n_countries > 0
        GROUP BY c.country_code
        """
    ).df()


    def _zscore(_series):
        _vals = _series.astype(float)
        _sigma = float(_vals.std(ddof=0))
        if not _sigma:
            return pd.Series(0.0, index=_series.index)
        return (_vals - float(_vals.mean())) / _sigma


    aiac_country_rd_composite = country_adoption_lag[
        ["country_code", "adoption_lag_months"]
    ].copy()
    aiac_country_rd_composite["country_code"] = (
        aiac_country_rd_composite["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        figure_3b_centrality[["country_code", "eigenvector_centrality"]],
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        figure_3e_country_first_author_base[
            ["country_code", "first_author_share_pct"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        figure_3e_country_last_author_base[
            ["country_code", "last_author_share_pct"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        country_af_nonaf_compare[
            ["country_code", "af_fractional_count", "country_name"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        _aiac_rd_citations,
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite = aiac_country_rd_composite.merge(
        country_rd_strength_lookup[
            ["country_code", "rd_strength_tertile", "rd_strength_index"]
        ],
        on="country_code",
        how="left",
    )
    aiac_country_rd_composite["first_author_share_pct"] = (
        aiac_country_rd_composite["first_author_share_pct"].fillna(0)
    )
    aiac_country_rd_composite["last_author_share_pct"] = aiac_country_rd_composite[
        "last_author_share_pct"
    ].fillna(0)
    aiac_country_rd_composite["af_fractional_citations"] = (
        aiac_country_rd_composite["af_fractional_citations"].fillna(0)
    )
    aiac_country_rd_composite = aiac_country_rd_composite.dropna(
        subset=[
            "adoption_lag_months",
            "eigenvector_centrality",
            "af_fractional_count",
            "rd_strength_tertile",
        ]
    ).copy()
    aiac_country_rd_composite["rd_strength_tertile"] = pd.Categorical(
        aiac_country_rd_composite["rd_strength_tertile"],
        categories=[
            "High R&D strength",
            "Middle R&D strength",
            "Low R&D strength",
        ],
        ordered=True,
    )
    _aiac_centrality_floor = max(
        float(aiac_country_rd_composite["eigenvector_centrality"].min()) * 0.5,
        1e-6,
    )
    aiac_country_rd_composite["centrality_log10"] = (
        aiac_country_rd_composite["eigenvector_centrality"]
        .clip(lower=_aiac_centrality_floor)
        .map(lambda _v: __import__("math").log10(_v))
    )
    aiac_country_rd_composite["adoption_speed_raw"] = aiac_country_rd_composite[
        "adoption_lag_months"
    ].map(lambda _v: -__import__("math").log1p(max(float(_v), 0.0)))
    aiac_country_rd_composite["paper_log1p"] = aiac_country_rd_composite[
        "af_fractional_count"
    ].map(lambda _v: __import__("math").log1p(_v))
    aiac_country_rd_composite["citation_log1p"] = aiac_country_rd_composite[
        "af_fractional_citations"
    ].map(lambda _v: __import__("math").log1p(_v))
    aiac_country_rd_composite["centrality_z"] = _zscore(
        aiac_country_rd_composite["centrality_log10"]
    )
    aiac_country_rd_composite["first_author_z"] = _zscore(
        aiac_country_rd_composite["first_author_share_pct"]
    )
    aiac_country_rd_composite["last_author_z"] = _zscore(
        aiac_country_rd_composite["last_author_share_pct"]
    )
    aiac_country_rd_composite["adoption_speed_z"] = _zscore(
        aiac_country_rd_composite["adoption_speed_raw"]
    )
    aiac_country_rd_composite["paper_z"] = _zscore(
        aiac_country_rd_composite["paper_log1p"]
    )
    aiac_country_rd_composite["citation_z"] = _zscore(
        aiac_country_rd_composite["citation_log1p"]
    )
    aiac_country_rd_composite["network_influence_score"] = (
        aiac_country_rd_composite["centrality_z"]
        + aiac_country_rd_composite["first_author_z"]
        + aiac_country_rd_composite["last_author_z"]
    ) / 3.0
    aiac_country_rd_composite["production_scale_impact_score"] = (
        aiac_country_rd_composite["paper_z"]
        + aiac_country_rd_composite["citation_z"]
    ) / 2.0
    _aiac_size_min = float(
        aiac_country_rd_composite["production_scale_impact_score"].min()
    )
    _aiac_size_max = float(
        aiac_country_rd_composite["production_scale_impact_score"].max()
    )
    _aiac_size_range = max(_aiac_size_max - _aiac_size_min, 1e-9)
    aiac_country_rd_composite["production_scale_impact_norm"] = (
        aiac_country_rd_composite["production_scale_impact_score"] - _aiac_size_min
    ) / _aiac_size_range
    aiac_country_rd_composite["bubble_size"] = (
        55 + 345 * aiac_country_rd_composite["production_scale_impact_norm"] ** 0.9
    )
    aiac_country_rd_composite = aiac_country_rd_composite.sort_values(
        [
            "rd_strength_tertile",
            "network_influence_score",
            "production_scale_impact_score",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    aiac_country_rd_composite
    return (aiac_country_rd_composite,)


@app.cell(hide_code=True)
def _(aiac_country_rd_composite):
    aiac_radar_rd_raw = (
        aiac_country_rd_composite.groupby(
            "rd_strength_tertile", as_index=False, observed=False
        )
        .agg(
            network_influence_mean=("network_influence_score", "mean"),
            adoption_speed_mean=("adoption_speed_z", "mean"),
            production_scale_impact_mean=("production_scale_impact_score", "mean"),
            adoption_speed_raw_mean=("adoption_speed_raw", "mean"),
            n_countries=("country_code", "nunique"),
        )
        .sort_values("rd_strength_tertile")
    )


    def _radar_global_scale(_value, _series):
        _lower = float(_series.quantile(0.05))
        _upper = float(_series.quantile(0.95))
        _span = max(_upper - _lower, 1e-9)
        return min(max((float(_value) - _lower) / _span, 0.0), 1.0)


    aiac_radar_rd_summary = aiac_radar_rd_raw[["rd_strength_tertile"]].copy()
    for _src_aiac_rd, _dst_aiac_rd, _country_col_aiac_rd in [
        ("adoption_speed_mean", "Adoption speed", "adoption_speed_z"),
        ("network_influence_mean", "Network influence", "network_influence_score"),
        (
            "production_scale_impact_mean",
            "Production scale and impact",
            "production_scale_impact_score",
        ),
    ]:
        _country_series_aiac_rd = aiac_country_rd_composite[_country_col_aiac_rd]
        aiac_radar_rd_summary[_dst_aiac_rd] = aiac_radar_rd_raw[_src_aiac_rd].map(
            lambda _v: _radar_global_scale(_v, _country_series_aiac_rd)
        )

    aiac_radar_rd_table = aiac_radar_rd_raw.rename(
        columns={
            "rd_strength_tertile": "R&D strength group",
            "network_influence_mean": "Network influence mean",
            "adoption_speed_mean": "Adoption speed mean (z)",
            "adoption_speed_raw_mean": "Adoption speed mean (raw)",
            "production_scale_impact_mean": "Production scale-impact mean",
            "n_countries": "Countries",
        }
    )

    aiac_radar_rd_raw
    return (aiac_radar_rd_summary,)


@app.cell(hide_code=True)
def fig_4(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_PURPLE,
    aiac_radar_rd_summary,
    plt,
):
    fig_aiac_radar_rd = plt.figure(figsize=(8.9, 8.9), dpi=320)
    ax_aiac_radar_rd = plt.subplot(111, polar=True)

    _aiac_radar_rd_categories = [
        "Adoption speed",
        "Network influence",
        "Production output",
    ]
    _aiac_radar_rd_angles = (
        __import__("numpy")
        .linspace(
            0,
            2 * __import__("numpy").pi,
            len(_aiac_radar_rd_categories),
            endpoint=False,
        )
        .tolist()
    )
    _aiac_radar_rd_angles += _aiac_radar_rd_angles[:1]
    _aiac_radar_rd_colors = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _aiac_radar_rd_line_colors = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_GUIDE_NEUTRAL,
    }
    _aiac_radar_rd_legend_labels = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }

    for _, _row_aiac_radar_rd in aiac_radar_rd_summary.iterrows():
        _vals_aiac_radar_rd = [
            float(_row_aiac_radar_rd["Adoption speed"]),
            float(_row_aiac_radar_rd["Network influence"]),
            float(_row_aiac_radar_rd["Production scale and impact"]),
        ]
        _vals_aiac_radar_rd += _vals_aiac_radar_rd[:1]
        _group_aiac_radar_rd = _row_aiac_radar_rd["rd_strength_tertile"]
        ax_aiac_radar_rd.plot(
            _aiac_radar_rd_angles,
            _vals_aiac_radar_rd,
            linewidth=3.0 if _group_aiac_radar_rd == "High R&D strength" else 2.25,
            color=_aiac_radar_rd_line_colors[_group_aiac_radar_rd],
            label=_aiac_radar_rd_legend_labels[_group_aiac_radar_rd],
            zorder=4 if _group_aiac_radar_rd == "High R&D strength" else 3,
        )
        ax_aiac_radar_rd.fill(
            _aiac_radar_rd_angles,
            _vals_aiac_radar_rd,
            color=_aiac_radar_rd_colors[_group_aiac_radar_rd],
            alpha=0.14 if _group_aiac_radar_rd == "High R&D strength" else 0.10,
            zorder=2,
        )

    ax_aiac_radar_rd.set_theta_offset(__import__("numpy").pi / 2)
    ax_aiac_radar_rd.set_theta_direction(-1)
    ax_aiac_radar_rd.set_xticks(_aiac_radar_rd_angles[:-1])
    ax_aiac_radar_rd.set_xticklabels(
        [""] * len(_aiac_radar_rd_categories),
        fontsize=30.6,
        color="#1f2937",
    )
    ax_aiac_radar_rd.set_ylim(0, 1)
    ax_aiac_radar_rd.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_aiac_radar_rd.set_yticklabels(["", "", "", ""], fontsize=8.8, color=AF_BLUE)
    ax_aiac_radar_rd.set_rlabel_position(92)
    ax_aiac_radar_rd.yaxis.grid(
        True, linestyle=(0, (2, 3)), linewidth=0.8, color=AF_CYAN
    )
    ax_aiac_radar_rd.xaxis.grid(
        True, linestyle=(0, (2, 3)), linewidth=0.8, color=AF_CYAN
    )
    ax_aiac_radar_rd.spines["polar"].set_color(AF_CYAN)
    ax_aiac_radar_rd.spines["polar"].set_linewidth(1.0)
    ax_aiac_radar_rd.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.24, 1.12),
        title="R&D strength group",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    fig_aiac_radar_rd.subplots_adjust(left=0.06, right=0.84, top=0.96, bottom=0.06)

    plt.gca()
    fig_aiac_radar_rd
    return


@app.cell(hide_code=True)
def fig_5_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    aiac_country_rd_composite,
    plt,
    sns,
):
    fig_aiac_bubble_rd = plt.figure(figsize=(12.2, 8.9), dpi=320)
    _aiac_rd_grid = fig_aiac_bubble_rd.add_gridspec(
        2,
        2,
        width_ratios=[4.9, 0.48],
        height_ratios=[0.52, 4.9],
        wspace=0.03,
        hspace=0.07,
    )
    ax_aiac_bubble_rd_top = fig_aiac_bubble_rd.add_subplot(_aiac_rd_grid[0, 0])
    ax_aiac_bubble_rd = fig_aiac_bubble_rd.add_subplot(_aiac_rd_grid[1, 0])
    ax_aiac_bubble_rd_right = fig_aiac_bubble_rd.add_subplot(_aiac_rd_grid[1, 1])

    _aiac_rd_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _aiac_rd_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _aiac_rd_legend_labels = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _aiac_rd_quadrant_font_size = AF_ANNOTATION_FONT_SIZE * 0.8
    _aiac_rd_plot = aiac_country_rd_composite[
        aiac_country_rd_composite["rd_strength_tertile"].isin(_aiac_rd_order)
    ].copy()
    _aiac_rd_x_med = float(_aiac_rd_plot["network_influence_score"].median())
    _aiac_rd_y_med = float(_aiac_rd_plot["adoption_speed_z"].median())
    _aiac_rd_xmin = float(_aiac_rd_plot["network_influence_score"].min())
    _aiac_rd_xmax = float(_aiac_rd_plot["network_influence_score"].max())
    _aiac_rd_ymin = float(_aiac_rd_plot["adoption_speed_z"].min())
    _aiac_rd_ymax = float(_aiac_rd_plot["adoption_speed_z"].max())

    for _group_aiac_rd in _aiac_rd_order:
        _group_df_aiac_rd = _aiac_rd_plot[
            _aiac_rd_plot["rd_strength_tertile"] == _group_aiac_rd
        ]
        if not _group_df_aiac_rd.empty:
            ax_aiac_bubble_rd.scatter(
                _group_df_aiac_rd["network_influence_score"],
                _group_df_aiac_rd["adoption_speed_z"],
                s=_group_df_aiac_rd["bubble_size"],
                color=_aiac_rd_palette[_group_aiac_rd],
                alpha=0.82,
                edgecolors="white",
                linewidth=0.9,
                zorder=3,
            )

    _aiac_rd_label_pool = _aiac_rd_plot.sort_values(
        ["production_scale_impact_score", "network_influence_score"],
        ascending=[False, False],
    ).head(14)
    for _row_aiac_rd in _aiac_rd_label_pool.itertuples(index=False):
        _dx_aiac_rd = (
            0.07
            if _row_aiac_rd.network_influence_score <= _aiac_rd_x_med
            else -0.07
        )
        _dy_aiac_rd = (
            0.08 if _row_aiac_rd.adoption_speed_z <= _aiac_rd_y_med else -0.08
        )
        ax_aiac_bubble_rd.text(
            _row_aiac_rd.network_influence_score + _dx_aiac_rd,
            _row_aiac_rd.adoption_speed_z + _dy_aiac_rd,
            _row_aiac_rd.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_PURPLE,
            ha="left" if _dx_aiac_rd > 0 else "right",
            va="bottom" if _dy_aiac_rd > 0 else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 0.32,
            },
            zorder=5,
        )

    ax_aiac_bubble_rd.axvline(
        _aiac_rd_x_med,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.9,
        zorder=1,
    )
    ax_aiac_bubble_rd.axhline(
        _aiac_rd_y_med,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.9,
        zorder=1,
    )

    for _group_aiac_rd in _aiac_rd_order:
        _vals_aiac_rd_top = _aiac_rd_plot.loc[
            _aiac_rd_plot["rd_strength_tertile"] == _group_aiac_rd,
            "network_influence_score",
        ].dropna()
        if len(_vals_aiac_rd_top) > 1:
            sns.kdeplot(
                x=_vals_aiac_rd_top,
                ax=ax_aiac_bubble_rd_top,
                color=_aiac_rd_palette[_group_aiac_rd],
                linewidth=1.7,
                fill=False,
                bw_adjust=1.0,
                clip=(_aiac_rd_xmin - 0.2, _aiac_rd_xmax + 0.2),
            )

    for _group_aiac_rd in _aiac_rd_order:
        _vals_aiac_rd_right = _aiac_rd_plot.loc[
            _aiac_rd_plot["rd_strength_tertile"] == _group_aiac_rd,
            "adoption_speed_z",
        ].dropna()
        if len(_vals_aiac_rd_right) > 1:
            sns.kdeplot(
                y=_vals_aiac_rd_right,
                ax=ax_aiac_bubble_rd_right,
                color=_aiac_rd_palette[_group_aiac_rd],
                linewidth=1.7,
                fill=False,
                bw_adjust=1.0,
                clip=(_aiac_rd_ymin - 0.2, _aiac_rd_ymax + 0.2),
            )

    ax_aiac_bubble_rd_top.axvline(
        _aiac_rd_x_med,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        alpha=0.8,
    )
    ax_aiac_bubble_rd_right.axhline(
        _aiac_rd_y_med,
        color=AF_GUIDE_NEUTRAL,
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        alpha=0.8,
    )
    for _ax_aiac_rd_marginal in [ax_aiac_bubble_rd_top, ax_aiac_bubble_rd_right]:
        _ax_aiac_rd_marginal.set_facecolor("white")
        _ax_aiac_rd_marginal.grid(False)
        _ax_aiac_rd_marginal.tick_params(pad=2)
        for _spine_aiac_rd in _ax_aiac_rd_marginal.spines.values():
            _spine_aiac_rd.set_visible(False)
    ax_aiac_bubble_rd_top.set_xlim(_aiac_rd_xmin - 0.2, _aiac_rd_xmax + 0.2)
    ax_aiac_bubble_rd_top.margins(y=0.06)
    ax_aiac_bubble_rd_top.set_xticks([])
    ax_aiac_bubble_rd_top.set_yticks([])
    ax_aiac_bubble_rd_top.set_xlabel("")
    ax_aiac_bubble_rd_top.set_ylabel("")
    ax_aiac_bubble_rd_right.set_ylim(_aiac_rd_ymin - 0.18, _aiac_rd_ymax + 0.18)
    ax_aiac_bubble_rd_right.margins(x=0.06)
    ax_aiac_bubble_rd_right.set_xticks([])
    ax_aiac_bubble_rd_right.set_yticks([])
    ax_aiac_bubble_rd_right.set_xlabel("")
    ax_aiac_bubble_rd_right.set_ylabel("")

    _aiac_rd_size_levels = [0.2, 0.5, 0.8]
    _aiac_rd_size_labels = ["Lower", "Median", "Higher"]
    _aiac_rd_size_handles = []
    for _size_level in _aiac_rd_size_levels:
        _size_area = 55 + 345 * (_size_level**0.9)
        _aiac_rd_size_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=AF_BLUE,
                markeredgecolor="white",
                markeredgewidth=0.75,
                alpha=0.42,
                markersize=_size_area**0.5,
                linestyle="None",
            )
        )
    _aiac_rd_size_legend = ax_aiac_bubble_rd.legend(
        _aiac_rd_size_handles,
        _aiac_rd_size_labels,
        title="Production output",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.80),
        labelspacing=0.95,
        handletextpad=0.9,
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    ax_aiac_bubble_rd.add_artist(_aiac_rd_size_legend)
    _aiac_rd_group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_aiac_rd_palette[_group_aiac_rd],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.8,
            label=_aiac_rd_legend_labels[_group_aiac_rd],
            linestyle="None",
        )
        for _group_aiac_rd in _aiac_rd_order
    ]
    _aiac_rd_group_legend = ax_aiac_bubble_rd.legend(
        handles=_aiac_rd_group_handles,
        frameon=False,
        title="R&D strength group",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.56),
        labelspacing=0.8,
        handletextpad=0.7,
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _aiac_rd_group_legend._legend_box.align = "left"
    ax_aiac_bubble_rd.add_artist(_aiac_rd_group_legend)

    ax_aiac_bubble_rd.text(
        0.94,
        0.89,
        "Core: fast adoption +\nhigh network influence",
        transform=ax_aiac_bubble_rd.transAxes,
        ha="right",
        va="top",
        fontsize=_aiac_rd_quadrant_font_size,
        color=AF_PURPLE,
        fontweight="semibold",
    )
    ax_aiac_bubble_rd.text(
        0.03,
        0.08,
        "Periphery: slow adoption + low network influence",
        transform=ax_aiac_bubble_rd.transAxes,
        ha="left",
        va="bottom",
        fontsize=_aiac_rd_quadrant_font_size,
        color=AF_PURPLE,
        fontweight="semibold",
    )
    ax_aiac_bubble_rd.text(
        0.98,
        0.08,
        "Slow adoption + high influence",
        transform=ax_aiac_bubble_rd.transAxes,
        ha="right",
        va="bottom",
        fontsize=_aiac_rd_quadrant_font_size,
        color=AF_BLUE,
    )
    ax_aiac_bubble_rd.text(
        0.03,
        0.84,
        "Fast adoption + low influence",
        transform=ax_aiac_bubble_rd.transAxes,
        ha="left",
        va="center",
        fontsize=_aiac_rd_quadrant_font_size,
        color=AF_BLUE,
    )

    ax_aiac_bubble_rd.set_xlabel(
        "Network influence composite score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    ax_aiac_bubble_rd.set_ylabel(
        "Adoption speed composite score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    ax_aiac_bubble_rd.set_xlim(_aiac_rd_xmin - 0.2, _aiac_rd_xmax + 0.2)
    ax_aiac_bubble_rd.set_ylim(_aiac_rd_ymin - 0.18, _aiac_rd_ymax + 0.18)
    ax_aiac_bubble_rd.grid(
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color=AF_CYAN,
        alpha=0.9,
    )
    ax_aiac_bubble_rd.set_axisbelow(True)
    ax_aiac_bubble_rd.spines["top"].set_visible(False)
    ax_aiac_bubble_rd.spines["right"].set_visible(False)
    ax_aiac_bubble_rd.spines["left"].set_color("#374151")
    ax_aiac_bubble_rd.spines["bottom"].set_color("#374151")
    ax_aiac_bubble_rd.spines["left"].set_linewidth(0.8)
    ax_aiac_bubble_rd.spines["bottom"].set_linewidth(0.8)
    ax_aiac_bubble_rd.tick_params(axis="both", labelsize=10, colors="#1f2937")
    fig_aiac_bubble_rd.subplots_adjust(
        left=0.10, right=0.97, top=0.965, bottom=0.12
    )

    plt.gca()
    fig_aiac_bubble_rd
    return


@app.cell(hide_code=True)
def fig_5_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    aiac_radar_rd_summary,
    pd,
    plt,
):
    _aiac_rd_bar_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _aiac_rd_bar_color_map = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _aiac_rd_bar_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _aiac_rd_bar_metric_map = {
        "Adoption": "Adoption speed",
        "Output": "Production scale and impact",
        "Network": "Network influence",
    }

    _aiac_rd_bar_plot = aiac_radar_rd_summary.copy()
    _aiac_rd_bar_plot["rd_strength_tertile"] = pd.Categorical(
        _aiac_rd_bar_plot["rd_strength_tertile"],
        categories=_aiac_rd_bar_order,
        ordered=True,
    )
    _aiac_rd_bar_plot = _aiac_rd_bar_plot.sort_values("rd_strength_tertile")

    _fig_aiac_bar_rd, _ax_aiac_bar_rd = plt.subplots(figsize=(7.4, 7.4), dpi=320)
    _aiac_rd_bar_label_font_size = AF_LABEL_FONT_SIZE * 0.8
    _aiac_rd_bar_annotation_font_size = AF_ANNOTATION_FONT_SIZE * 0.8

    _aiac_rd_bar_categories = list(_aiac_rd_bar_metric_map.keys())
    _aiac_rd_bar_x = __import__("numpy").arange(len(_aiac_rd_bar_categories)) * (
        4 / 9
    )
    _aiac_rd_bar_width = 0.12
    _aiac_rd_bar_offsets = [-_aiac_rd_bar_width, 0, _aiac_rd_bar_width]

    for _offset, _group in zip(_aiac_rd_bar_offsets, _aiac_rd_bar_order):
        _group_row = _aiac_rd_bar_plot[
            _aiac_rd_bar_plot["rd_strength_tertile"] == _group
        ].iloc[0]
        _values = [
            float(_group_row[_aiac_rd_bar_metric_map[_category]])
            for _category in _aiac_rd_bar_categories
        ]
        _bars = _ax_aiac_bar_rd.bar(
            _aiac_rd_bar_x + _offset,
            _values,
            width=_aiac_rd_bar_width,
            color=_aiac_rd_bar_color_map[_group],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
            label=_aiac_rd_bar_label_map[_group],
            zorder=3,
        )
        for _bar, _value in zip(_bars, _values):
            _ax_aiac_bar_rd.text(
                _bar.get_x() + _bar.get_width() / 2,
                _value + 0.02,
                f"{_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=_aiac_rd_bar_annotation_font_size,
                color=AF_GUIDE_NEUTRAL,
                zorder=5,
            )

    _ax_aiac_bar_rd.set_xticks(_aiac_rd_bar_x)
    _ax_aiac_bar_rd.set_xticklabels(_aiac_rd_bar_categories)
    _ax_aiac_bar_rd.set_xlabel(
        "Composite dimension",
        fontsize=_aiac_rd_bar_label_font_size,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_bar_rd.set_ylim(0, 1.02)
    _ax_aiac_bar_rd.set_ylabel(
        "Average score",
        fontsize=_aiac_rd_bar_label_font_size,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_bar_rd.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color="#d1d5db",
        alpha=0.88,
    )
    _ax_aiac_bar_rd.grid(axis="x", visible=False)
    _ax_aiac_bar_rd.set_axisbelow(True)
    _ax_aiac_bar_rd.spines["top"].set_visible(False)
    _ax_aiac_bar_rd.spines["right"].set_visible(False)
    _ax_aiac_bar_rd.spines["left"].set_color("#374151")
    _ax_aiac_bar_rd.spines["bottom"].set_color("#374151")
    _ax_aiac_bar_rd.spines["left"].set_linewidth(0.8)
    _ax_aiac_bar_rd.spines["bottom"].set_linewidth(0.8)
    _ax_aiac_bar_rd.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_aiac_bar_rd.legend(
        frameon=False,
        title="R&D strength group",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=_aiac_rd_bar_annotation_font_size,
    )
    _fig_aiac_bar_rd.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.16)

    plt.gca()
    fig_aiac_bubble_rd_bar = _fig_aiac_bar_rd
    fig_aiac_bubble_rd_bar
    return


@app.cell(hide_code=True)
def _(aiac_country_rd_composite, mo, pd):
    aiac_gam_base = aiac_country_rd_composite.copy()
    aiac_gam_base = aiac_gam_base.dropna(
        subset=[
            "network_influence_score",
            "production_scale_impact_score",
            "af_fractional_count",
            "rd_strength_tertile",
        ]
    ).copy()

    _aiac_gam_plot = aiac_gam_base.sort_values("network_influence_score").copy()
    _aiac_gam_plot["output_weight"] = (
        _aiac_gam_plot["af_fractional_count"]
        .clip(lower=1)
        .map(lambda _v: __import__("math").sqrt(_v))
    )

    _aiac_gam_x = (
        _aiac_gam_plot["network_influence_score"].astype(float).to_numpy()
    )
    _aiac_gam_y = (
        _aiac_gam_plot["production_scale_impact_score"].astype(float).to_numpy()
    )
    _aiac_gam_weights = _aiac_gam_plot["output_weight"].astype(float).to_numpy()

    _aiac_gam_frac = 0.35
    _aiac_gam_n = len(_aiac_gam_x)
    _aiac_gam_k = max(
        8, int(__import__("math").ceil(_aiac_gam_frac * _aiac_gam_n))
    )
    _aiac_gam_x_grid = __import__("numpy").linspace(
        float(_aiac_gam_x.min()), float(_aiac_gam_x.max()), 220
    )

    _aiac_gam_y_smooth = []
    _aiac_gam_local_slope = []
    for _x0_gam in _aiac_gam_x_grid:
        _dist_gam = __import__("numpy").abs(_aiac_gam_x - _x0_gam)
        _bandwidth_gam = __import__("numpy").partition(_dist_gam, _aiac_gam_k - 1)[
            _aiac_gam_k - 1
        ]
        _bandwidth_gam = max(float(_bandwidth_gam), 1e-6)
        _u_gam = _dist_gam / _bandwidth_gam
        _kernel_gam = (1 - __import__("numpy").clip(_u_gam, 0, 1) ** 3) ** 3
        _kernel_gam[_u_gam >= 1] = 0
        _w_gam = _kernel_gam * _aiac_gam_weights
        _x_local_gam = _aiac_gam_x - _x0_gam
        _design_local_gam = __import__("numpy").column_stack(
            [__import__("numpy").ones_like(_x_local_gam), _x_local_gam]
        )
        _xtwx_local_gam = (
            _design_local_gam.T
            @ (__import__("numpy").diag(_w_gam))
            @ _design_local_gam
        )
        _xtwy_local_gam = (
            _design_local_gam.T @ (__import__("numpy").diag(_w_gam)) @ _aiac_gam_y
        )
        _beta_local_gam = (
            __import__("numpy").linalg.pinv(_xtwx_local_gam) @ _xtwy_local_gam
        )
        _aiac_gam_y_smooth.append(float(_beta_local_gam[0]))
        _aiac_gam_local_slope.append(float(_beta_local_gam[1]))

    _aiac_gam_curve = pd.DataFrame(
        {
            "network_influence_score": _aiac_gam_x_grid,
            "production_scale_impact_smooth": _aiac_gam_y_smooth,
            "local_slope": _aiac_gam_local_slope,
        }
    )
    _aiac_gam_curve["slope_change"] = _aiac_gam_curve["local_slope"].diff()
    _aiac_gam_curve["acceleration_proxy"] = (
        _aiac_gam_curve["local_slope"].diff().rolling(7, center=True).mean()
    )

    _aiac_gam_inner_curve = _aiac_gam_curve.iloc[12:-12].copy()
    if _aiac_gam_inner_curve["acceleration_proxy"].notna().any():
        _aiac_gam_turning_idx = int(
            _aiac_gam_inner_curve["acceleration_proxy"].idxmax()
        )
    else:
        _aiac_gam_turning_idx = int(
            _aiac_gam_curve["local_slope"].iloc[12:-12].idxmax()
        )
    _aiac_gam_turning_point = _aiac_gam_curve.loc[_aiac_gam_turning_idx].to_dict()

    aiac_gam_summary = pd.DataFrame(
        {
            "metric": [
                "n_countries",
                "lowess_fraction",
                "candidate_turning_x_network_influence",
                "candidate_turning_y_scale_impact",
                "local_slope_at_turning_point",
                "acceleration_proxy_at_turning_point",
            ],
            "value": [
                int(_aiac_gam_n),
                float(_aiac_gam_frac),
                float(_aiac_gam_turning_point["network_influence_score"]),
                float(_aiac_gam_turning_point["production_scale_impact_smooth"]),
                float(_aiac_gam_turning_point["local_slope"]),
                float(_aiac_gam_turning_point["acceleration_proxy"]),
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                "### Composite AI absorptive capacity nonlinear turning-point summary"
            ),
            aiac_gam_summary,
            _aiac_gam_curve,
        ]
    )
    return (aiac_gam_base,)


@app.cell(hide_code=True)
def _(aiac_country_rd_composite, mo, pd, stats):
    aiac_piecewise_base = aiac_country_rd_composite.copy()
    aiac_piecewise_base = aiac_piecewise_base.dropna(
        subset=[
            "network_influence_score",
            "adoption_speed_z",
            "af_fractional_count",
            "rd_strength_tertile",
        ]
    ).copy()

    _aiac_pw_x = (
        aiac_piecewise_base["network_influence_score"].astype(float).to_numpy()
    )
    _aiac_pw_y = aiac_piecewise_base["adoption_speed_z"].astype(float).to_numpy()
    _aiac_pw_w = (
        aiac_piecewise_base["af_fractional_count"]
        .clip(lower=1)
        .map(lambda _v: __import__("math").sqrt(_v))
        .to_numpy()
    )

    _aiac_pw_candidates = sorted(
        pd.Series(_aiac_pw_x)
        .quantile(
            [
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
            ]
        )
        .tolist()
    )

    _aiac_pw_rows = []
    for _c_pw in _aiac_pw_candidates:
        _pw_term = __import__("numpy").maximum(_aiac_pw_x - _c_pw, 0)
        _x_design_pw = __import__("numpy").column_stack(
            [
                __import__("numpy").ones_like(_aiac_pw_x),
                _aiac_pw_x,
                _pw_term,
            ]
        )
        _w_matrix_pw = __import__("numpy").diag(_aiac_pw_w)
        _beta_pw = __import__("numpy").linalg.solve(
            _x_design_pw.T @ _w_matrix_pw @ _x_design_pw,
            _x_design_pw.T @ _w_matrix_pw @ _aiac_pw_y,
        )
        _fit_pw = _x_design_pw @ _beta_pw
        _resid_pw = _aiac_pw_y - _fit_pw
        _sse_pw = float((_aiac_pw_w * (_resid_pw**2)).sum())
        _aiac_pw_rows.append(
            {
                "threshold_c": float(_c_pw),
                "sse": _sse_pw,
                "intercept": float(_beta_pw[0]),
                "slope_pre": float(_beta_pw[1]),
                "slope_change": float(_beta_pw[2]),
                "slope_post": float(_beta_pw[1] + _beta_pw[2]),
            }
        )

    aiac_piecewise_search = pd.DataFrame(_aiac_pw_rows).sort_values("sse")
    aiac_piecewise_best = aiac_piecewise_search.iloc[0].to_dict()

    _aiac_pw_best_c = float(aiac_piecewise_best["threshold_c"])
    _aiac_pw_best_pw_term = __import__("numpy").maximum(
        _aiac_pw_x - _aiac_pw_best_c, 0
    )
    _aiac_pw_best_design = __import__("numpy").column_stack(
        [
            __import__("numpy").ones_like(_aiac_pw_x),
            _aiac_pw_x,
            _aiac_pw_best_pw_term,
        ]
    )
    _aiac_pw_best_w_matrix = __import__("numpy").diag(_aiac_pw_w)
    _aiac_pw_best_beta = __import__("numpy").linalg.solve(
        _aiac_pw_best_design.T @ _aiac_pw_best_w_matrix @ _aiac_pw_best_design,
        _aiac_pw_best_design.T @ _aiac_pw_best_w_matrix @ _aiac_pw_y,
    )
    _aiac_pw_best_fit = _aiac_pw_best_design @ _aiac_pw_best_beta
    _aiac_pw_best_resid = _aiac_pw_y - _aiac_pw_best_fit
    _aiac_pw_best_df = max(len(_aiac_pw_x) - _aiac_pw_best_design.shape[1], 1)
    _aiac_pw_best_sigma2 = float(
        (_aiac_pw_w * (_aiac_pw_best_resid**2)).sum() / _aiac_pw_best_df
    )
    _aiac_pw_best_cov = _aiac_pw_best_sigma2 * __import__("numpy").linalg.inv(
        _aiac_pw_best_design.T @ _aiac_pw_best_w_matrix @ _aiac_pw_best_design
    )
    _aiac_pw_best_t = __import__("numpy").divide(
        _aiac_pw_best_beta,
        __import__("numpy").sqrt(
            __import__("numpy").clip(
                __import__("numpy").diag(_aiac_pw_best_cov), 0, None
            )
        ),
    )
    _aiac_pw_best_p = 2 * (1 - stats.norm.cdf(abs(_aiac_pw_best_t)))
    _aiac_pw_best_r2 = 1 - float((_aiac_pw_best_resid**2).sum()) / max(
        float(((_aiac_pw_y - _aiac_pw_y.mean()) ** 2).sum()),
        1e-9,
    )

    aiac_piecewise_summary = pd.DataFrame(
        {
            "metric": [
                "best_threshold_network_influence",
                "slope_pre_threshold",
                "slope_change_after_threshold",
                "slope_post_threshold",
                "piecewise_R_squared",
                "p_value_slope_pre",
                "p_value_slope_change",
                "n_countries",
            ],
            "value": [
                _aiac_pw_best_c,
                float(_aiac_pw_best_beta[1]),
                float(_aiac_pw_best_beta[2]),
                float(_aiac_pw_best_beta[1] + _aiac_pw_best_beta[2]),
                _aiac_pw_best_r2,
                float(_aiac_pw_best_p[1]),
                float(_aiac_pw_best_p[2]),
                int(len(aiac_piecewise_base)),
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                "### Composite AI absorptive capacity piecewise threshold search"
            ),
            aiac_piecewise_summary,
            aiac_piecewise_search,
        ]
    )
    return aiac_piecewise_base, aiac_piecewise_search


@app.cell(hide_code=True)
def fig_6_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    aiac_piecewise_base,
    aiac_piecewise_search,
    plt,
    stats,
):
    _fig_aiac_piecewise, _ax_aiac_piecewise = plt.subplots(
        figsize=(11.8, 7.8), dpi=260
    )

    _aiac_piecewise_plot = aiac_piecewise_base.copy().sort_values(
        "network_influence_score"
    )
    _aiac_piecewise_colors = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }

    _aiac_pw_best_row = aiac_piecewise_search.iloc[0]
    _aiac_pw_plot_best_c = float(_aiac_pw_best_row["threshold_c"])
    _aiac_pw_plot_x = (
        aiac_piecewise_base["network_influence_score"].astype(float).to_numpy()
    )
    _aiac_pw_plot_y = (
        aiac_piecewise_base["adoption_speed_z"].astype(float).to_numpy()
    )
    _aiac_pw_plot_w = (
        aiac_piecewise_base["af_fractional_count"]
        .clip(lower=1)
        .map(lambda _v: __import__("math").sqrt(_v))
        .to_numpy()
    )
    _aiac_pw_plot_term = __import__("numpy").maximum(
        _aiac_pw_plot_x - _aiac_pw_plot_best_c, 0
    )
    _aiac_pw_plot_design = __import__("numpy").column_stack(
        [
            __import__("numpy").ones_like(_aiac_pw_plot_x),
            _aiac_pw_plot_x,
            _aiac_pw_plot_term,
        ]
    )
    _aiac_pw_plot_w_matrix = __import__("numpy").diag(_aiac_pw_plot_w)
    _aiac_pw_plot_beta = __import__("numpy").linalg.solve(
        _aiac_pw_plot_design.T @ _aiac_pw_plot_w_matrix @ _aiac_pw_plot_design,
        _aiac_pw_plot_design.T @ _aiac_pw_plot_w_matrix @ _aiac_pw_plot_y,
    )
    _aiac_pw_plot_fit = _aiac_pw_plot_design @ _aiac_pw_plot_beta
    _aiac_pw_plot_resid = _aiac_pw_plot_y - _aiac_pw_plot_fit
    _aiac_pw_plot_df = max(len(_aiac_pw_plot_x) - _aiac_pw_plot_design.shape[1], 1)
    _aiac_pw_plot_sigma2 = float(
        (_aiac_pw_plot_w * (_aiac_pw_plot_resid**2)).sum() / _aiac_pw_plot_df
    )
    _aiac_pw_plot_cov = _aiac_pw_plot_sigma2 * __import__("numpy").linalg.inv(
        _aiac_pw_plot_design.T @ _aiac_pw_plot_w_matrix @ _aiac_pw_plot_design
    )
    _aiac_pw_plot_p = 2 * (
        1
        - stats.norm.cdf(
            abs(
                __import__("numpy").divide(
                    _aiac_pw_plot_beta,
                    __import__("numpy").sqrt(
                        __import__("numpy").clip(
                            __import__("numpy").diag(_aiac_pw_plot_cov), 0, None
                        )
                    ),
                )
            )
        )
    )

    for _group_piecewise in [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]:
        _group_df_piecewise = _aiac_piecewise_plot[
            _aiac_piecewise_plot["rd_strength_tertile"] == _group_piecewise
        ]
        if not _group_df_piecewise.empty:
            _ax_aiac_piecewise.scatter(
                _group_df_piecewise["network_influence_score"],
                _group_df_piecewise["adoption_speed_z"],
                s=_group_df_piecewise["bubble_size"],
                color=_aiac_piecewise_colors[_group_piecewise],
                alpha=0.72,
                edgecolors="white",
                linewidth=0.7,
                zorder=3,
            )

    _aiac_pw_top_labels = _aiac_piecewise_plot.sort_values(
        "bubble_size", ascending=False
    ).head(10)
    for _row_pw in _aiac_pw_top_labels.itertuples(index=False):
        _dx_pw = (
            0.045
            if _row_pw.network_influence_score <= _aiac_pw_plot_best_c
            else -0.045
        )
        _dy_pw = (
            0.06
            if _row_pw.adoption_speed_z
            <= aiac_piecewise_base["adoption_speed_z"].median()
            else -0.06
        )
        _ax_aiac_piecewise.text(
            _row_pw.network_influence_score + _dx_pw,
            _row_pw.adoption_speed_z + _dy_pw,
            _row_pw.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_PURPLE,
            ha="left" if _dx_pw > 0 else "right",
            va="bottom" if _dy_pw > 0 else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 0.28,
            },
            zorder=6,
        )

    _aiac_pw_grid = __import__("numpy").linspace(
        float(_aiac_piecewise_plot["network_influence_score"].min()),
        float(_aiac_piecewise_plot["network_influence_score"].max()),
        300,
    )
    _aiac_pw_grid_term = __import__("numpy").maximum(
        _aiac_pw_grid - _aiac_pw_plot_best_c, 0
    )
    _aiac_pw_grid_design = __import__("numpy").column_stack(
        [
            __import__("numpy").ones_like(_aiac_pw_grid),
            _aiac_pw_grid,
            _aiac_pw_grid_term,
        ]
    )
    _aiac_pw_grid_fit = _aiac_pw_grid_design @ _aiac_pw_plot_beta

    _aiac_pw_pre_mask = _aiac_pw_grid <= _aiac_pw_plot_best_c
    _aiac_pw_post_mask = _aiac_pw_grid >= _aiac_pw_plot_best_c

    _ax_aiac_piecewise.plot(
        _aiac_pw_grid[_aiac_pw_pre_mask],
        _aiac_pw_grid_fit[_aiac_pw_pre_mask],
        color=AF_GUIDE_NEUTRAL,
        linewidth=2.8,
        linestyle=(0, (4, 3)),
        zorder=4,
        label="Pre-threshold fit",
    )
    _ax_aiac_piecewise.plot(
        _aiac_pw_grid[_aiac_pw_post_mask],
        _aiac_pw_grid_fit[_aiac_pw_post_mask],
        color=AF_CORAL,
        linewidth=2.8,
        zorder=4,
        label="Post-threshold fit",
    )

    _ax_aiac_piecewise.axvline(
        _aiac_pw_plot_best_c,
        color=AF_CORAL,
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    _ax_aiac_piecewise.text(
        _aiac_pw_plot_best_c + 0.03,
        float(_aiac_piecewise_plot["adoption_speed_z"].min()) + 0.1,
        f"threshold = {_aiac_pw_plot_best_c:.2f}",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_CORAL,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CORAL,
            "linewidth": 0.6,
            "alpha": 0.92,
            "pad": 0.7,
        },
        zorder=5,
    )

    _ax_aiac_piecewise.text(
        0.015,
        0.985,
        (
            f"Pre-threshold slope = {float(_aiac_pw_plot_beta[1]):.2f}\n"
            f"Post-threshold slope = {float(_aiac_pw_plot_beta[1] + _aiac_pw_plot_beta[2]):.2f}\n"
            f"Slope-change p = {float(_aiac_pw_plot_p[2]):.3g}"
        ),
        transform=_ax_aiac_piecewise.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_PURPLE,
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.6,
            "alpha": 0.94,
            "pad": 0.85,
        },
        zorder=6,
    )

    _ax_aiac_piecewise.set_xlabel(
        "Network influence composite score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_piecewise.set_ylabel(
        "Adoption speed composite score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_piecewise.grid(
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color=AF_CYAN,
        alpha=0.9,
    )
    _ax_aiac_piecewise.set_axisbelow(True)
    _ax_aiac_piecewise.spines["top"].set_visible(False)
    _ax_aiac_piecewise.spines["right"].set_visible(False)
    _ax_aiac_piecewise.spines["left"].set_color("#374151")
    _ax_aiac_piecewise.spines["bottom"].set_color("#374151")
    _ax_aiac_piecewise.spines["left"].set_linewidth(0.8)
    _ax_aiac_piecewise.spines["bottom"].set_linewidth(0.8)
    _ax_aiac_piecewise.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_aiac_piecewise.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.79),
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )
    _fig_aiac_piecewise.subplots_adjust(
        left=0.11, right=0.97, top=0.9, bottom=0.13
    )

    plt.gca()
    return


@app.cell(hide_code=True)
def fig_6_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    aiac_gam_base,
    pd,
    plt,
):
    _fig_aiac_gam, _ax_aiac_gam = plt.subplots(figsize=(11.8, 7.8), dpi=260)

    _aiac_gam_plot_local = aiac_gam_base.sort_values(
        "network_influence_score"
    ).copy()
    _aiac_gam_colors = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }

    _aiac_gam_x_local = (
        _aiac_gam_plot_local["network_influence_score"].astype(float).to_numpy()
    )
    _aiac_gam_y_local = (
        _aiac_gam_plot_local["production_scale_impact_score"]
        .astype(float)
        .to_numpy()
    )
    _aiac_gam_weights_local = (
        _aiac_gam_plot_local["af_fractional_count"]
        .clip(lower=1)
        .map(lambda _v: __import__("math").sqrt(_v))
        .astype(float)
        .to_numpy()
    )

    _aiac_gam_frac_local = 0.35
    _aiac_gam_n_local = len(_aiac_gam_x_local)
    _aiac_gam_k_local = max(
        8, int(__import__("math").ceil(_aiac_gam_frac_local * _aiac_gam_n_local))
    )
    _aiac_gam_x_grid_local = __import__("numpy").linspace(
        float(_aiac_gam_x_local.min()), float(_aiac_gam_x_local.max()), 220
    )

    _aiac_gam_y_smooth_local = []
    _aiac_gam_local_slope_local = []
    for _x0_gam_local in _aiac_gam_x_grid_local:
        _dist_gam_local = __import__("numpy").abs(
            _aiac_gam_x_local - _x0_gam_local
        )
        _bandwidth_gam_local = __import__("numpy").partition(
            _dist_gam_local, _aiac_gam_k_local - 1
        )[_aiac_gam_k_local - 1]
        _bandwidth_gam_local = max(float(_bandwidth_gam_local), 1e-6)
        _u_gam_local = _dist_gam_local / _bandwidth_gam_local
        _kernel_gam_local = (
            1 - __import__("numpy").clip(_u_gam_local, 0, 1) ** 3
        ) ** 3
        _kernel_gam_local[_u_gam_local >= 1] = 0
        _w_gam_local = _kernel_gam_local * _aiac_gam_weights_local
        _x_local_centered = _aiac_gam_x_local - _x0_gam_local
        _design_gam_local = __import__("numpy").column_stack(
            [__import__("numpy").ones_like(_x_local_centered), _x_local_centered]
        )
        _xtwx_gam_local = (
            _design_gam_local.T
            @ (__import__("numpy").diag(_w_gam_local))
            @ _design_gam_local
        )
        _xtwy_gam_local = (
            _design_gam_local.T
            @ (__import__("numpy").diag(_w_gam_local))
            @ _aiac_gam_y_local
        )
        _beta_gam_local = (
            __import__("numpy").linalg.pinv(_xtwx_gam_local) @ _xtwy_gam_local
        )
        _aiac_gam_y_smooth_local.append(float(_beta_gam_local[0]))
        _aiac_gam_local_slope_local.append(float(_beta_gam_local[1]))

    _aiac_gam_curve_local = pd.DataFrame(
        {
            "network_influence_score": _aiac_gam_x_grid_local,
            "production_scale_impact_smooth": _aiac_gam_y_smooth_local,
            "local_slope": _aiac_gam_local_slope_local,
        }
    )
    _aiac_gam_curve_local["slope_change"] = _aiac_gam_curve_local[
        "local_slope"
    ].diff()
    _aiac_gam_curve_local["acceleration_proxy"] = (
        _aiac_gam_curve_local["local_slope"].diff().rolling(7, center=True).mean()
    )

    _aiac_gam_inner_curve_local = _aiac_gam_curve_local.iloc[12:-12].copy()
    if _aiac_gam_inner_curve_local["acceleration_proxy"].notna().any():
        _aiac_gam_turning_idx_local = int(
            _aiac_gam_inner_curve_local["acceleration_proxy"].idxmax()
        )
    else:
        _aiac_gam_turning_idx_local = int(
            _aiac_gam_curve_local["local_slope"].iloc[12:-12].idxmax()
        )
    _aiac_gam_turning_point_local = _aiac_gam_curve_local.loc[
        _aiac_gam_turning_idx_local
    ].to_dict()

    for _group_gam in [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]:
        _group_df_gam = _aiac_gam_plot_local[
            _aiac_gam_plot_local["rd_strength_tertile"] == _group_gam
        ]
        if not _group_df_gam.empty:
            _ax_aiac_gam.scatter(
                _group_df_gam["network_influence_score"],
                _group_df_gam["production_scale_impact_score"],
                s=_group_df_gam["bubble_size"],
                color=_aiac_gam_colors[_group_gam],
                alpha=0.68,
                edgecolors="white",
                linewidth=0.7,
                zorder=3,
            )

    _aiac_gam_top_labels = _aiac_gam_plot_local.sort_values(
        "bubble_size", ascending=False
    ).head(10)
    for _row_gam in _aiac_gam_top_labels.itertuples(index=False):
        _dx_gam = (
            0.045
            if _row_gam.network_influence_score
            <= float(_aiac_gam_plot_local["network_influence_score"].median())
            else -0.045
        )
        _dy_gam = (
            0.06
            if _row_gam.production_scale_impact_score
            <= float(
                _aiac_gam_plot_local["production_scale_impact_score"].median()
            )
            else -0.06
        )
        _ax_aiac_gam.text(
            _row_gam.network_influence_score + _dx_gam,
            _row_gam.production_scale_impact_score + _dy_gam,
            _row_gam.country_code,
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color=AF_PURPLE,
            ha="left" if _dx_gam > 0 else "right",
            va="bottom" if _dy_gam > 0 else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 0.28,
            },
            zorder=6,
        )

    _ax_aiac_gam.plot(
        _aiac_gam_curve_local["network_influence_score"],
        _aiac_gam_curve_local["production_scale_impact_smooth"],
        color=AF_CORAL,
        linewidth=2.9,
        zorder=4,
        label="Smoothed nonlinear fit",
    )

    _ax_aiac_gam.axvline(
        float(_aiac_gam_turning_point_local["network_influence_score"]),
        color=AF_CORAL,
        linestyle=(0, (4, 4)),
        linewidth=1.3,
        alpha=0.9,
        zorder=2,
    )
    _ax_aiac_gam.scatter(
        [float(_aiac_gam_turning_point_local["network_influence_score"])],
        [float(_aiac_gam_turning_point_local["production_scale_impact_smooth"])],
        s=70,
        color=AF_CORAL,
        edgecolors="white",
        linewidth=0.9,
        zorder=5,
    )
    _ax_aiac_gam.text(
        float(_aiac_gam_turning_point_local["network_influence_score"]) + 0.04,
        float(_aiac_gam_turning_point_local["production_scale_impact_smooth"])
        + 0.05,
        (
            f"candidate turning point\n"
            f"x = {float(_aiac_gam_turning_point_local['network_influence_score']):.2f}\n"
            f"local slope = {float(_aiac_gam_turning_point_local['local_slope']):.2f}"
        ),
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_CORAL,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CORAL,
            "linewidth": 0.6,
            "alpha": 0.92,
            "pad": 0.8,
        },
        zorder=6,
    )

    _ax_aiac_gam.text(
        0.015,
        0.985,
        (
            f"LOWESS-like fraction = {_aiac_gam_frac_local:.2f}\n"
            f"Max local slope = {float(_aiac_gam_turning_point_local['local_slope']):.2f}"
        ),
        transform=_ax_aiac_gam.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE,
        color=AF_PURPLE,
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.6,
            "alpha": 0.94,
            "pad": 0.85,
        },
        zorder=6,
    )

    _ax_aiac_gam.set_xlabel(
        "Network influence composite score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_gam.set_ylabel(
        "Production scale-impact score",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
        color="#1f2937",
    )
    _ax_aiac_gam.grid(
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color=AF_CYAN,
        alpha=0.9,
    )
    _ax_aiac_gam.set_axisbelow(True)
    _ax_aiac_gam.spines["top"].set_visible(False)
    _ax_aiac_gam.spines["right"].set_visible(False)
    _ax_aiac_gam.spines["left"].set_color("#374151")
    _ax_aiac_gam.spines["bottom"].set_color("#374151")
    _ax_aiac_gam.spines["left"].set_linewidth(0.8)
    _ax_aiac_gam.spines["bottom"].set_linewidth(0.8)
    _ax_aiac_gam.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_aiac_gam.legend(
        frameon=False, loc="lower right", fontsize=AF_ANNOTATION_FONT_SIZE
    )
    _fig_aiac_gam.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.13)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Extended Data Figures

    This section presents the Extended Data figures accompanying the main analysis. These figures provide additional stratification, robustness, and descriptive detail for the patterns reported in the main text.
    """)
    return


@app.cell(hide_code=True)
def _():
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    AF_PURPLE = "#0E2B72"
    AF_BLUE = "#3F66C3"
    AF_LABEL_FONT_SIZE = 18.4
    AF_ANNOTATION_FONT_SIZE = 14.1
    AF_CYAN = "#CDD6ED"
    AF_CORAL = "#F94E57"
    AF_EVENT_NEUTRAL = "#6B7280"
    AF_GUIDE_NEUTRAL = "#4B5563"

    AF_RD_PALETTE = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    AF_INCOME_PALETTE = {
        "High income": AF_PURPLE,
        "Upper-middle income": AF_BLUE,
        "Lower-middle income": AF_CYAN,
        "Low income": AF_CORAL,
    }


    def af_color_at(position, anchors=(AF_PURPLE, AF_BLUE, AF_CYAN)):
        position = float(min(max(position, 0), 1))
        if len(anchors) == 1:
            return anchors[0]
        scaled = position * (len(anchors) - 1)
        left = int(position * (len(anchors) - 1))
        right = min(left + 1, len(anchors) - 1)
        frac = scaled - left
        left_rgb = to_rgb(anchors[left])
        right_rgb = to_rgb(anchors[right])
        return tuple(
            (1 - frac) * left_channel + frac * right_channel
            for left_channel, right_channel in zip(left_rgb, right_rgb)
        )


    def af_gradient(n, anchors=(AF_PURPLE, AF_BLUE, AF_CYAN)):
        if n <= 1:
            return [af_color_at(0, anchors)]
        return [af_color_at(i / (n - 1), anchors) for i in range(n)]


    AF_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
        "af_sequential", [AF_PURPLE, AF_BLUE, AF_CYAN]
    )
    AF_FULL_CMAP = LinearSegmentedColormap.from_list(
        "af_full", [AF_PURPLE, AF_BLUE, AF_CYAN, AF_CORAL]
    )
    return (
        AF_ANNOTATION_FONT_SIZE,
        AF_BLUE,
        AF_CORAL,
        AF_CYAN,
        AF_EVENT_NEUTRAL,
        AF_GUIDE_NEUTRAL,
        AF_LABEL_FONT_SIZE,
        AF_PURPLE,
        AF_SEQUENTIAL_CMAP,
        af_color_at,
        to_rgb,
    )


@app.cell(hide_code=True)
def ext_fig_2_a(
    AF_BLUE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_PURPLE,
    pd,
    plt,
):
    _con_af_doc = __import__("duckdb").connect()
    _doc_type_sql = """
    WITH af_final AS (
      SELECT *
      FROM read_parquet('derived_tables_dedup/works.parquet')
      WHERE coalesce(is_alphafold_related, false)
        AND lower(coalesce(type, '')) NOT IN (
          'editorial', 'letter', 'paratext', 'erratum', 'peer-review', 'supplementary-materials'
        )
    ), typed AS (
      SELECT *,
        CASE
          WHEN lower(coalesce(type, '')) = 'review' THEN 'Review'
          WHEN lower(coalesce(type, '')) = 'article'
               AND (
                 coalesce(is_alphafold_db, false)
                 OR lower(coalesce(title, '')) LIKE '%database%'
                 OR lower(coalesce(title, '')) LIKE '%resource%'
                 OR lower(coalesce(title, '')) LIKE '%server%'
                 OR lower(coalesce(title, '')) LIKE '%web server%'
                 OR lower(coalesce(title, '')) LIKE '%benchmark%'
                 OR lower(coalesce(title, '')) LIKE '%method%'
                 OR af_usage_level IN ('database_or_resource_use', 'benchmark_or_method_discussion')
               ) THEN 'Methods/resource/database'
          WHEN lower(coalesce(type, '')) = 'article' THEN 'Article'
          ELSE 'Other'
        END AS category
      FROM af_final
    )
    SELECT category, COUNT(*) AS n
    FROM typed
    GROUP BY 1
    """
    _doc_type_comp = _con_af_doc.execute(_doc_type_sql).fetchdf()

    _doc_type_order = ["Article", "Methods/resource/database", "Review", "Other"]
    _doc_type_palette = {
        "Article": AF_PURPLE,
        "Methods/resource/database": AF_BLUE,
        "Review": AF_CYAN,
        "Other": AF_EVENT_NEUTRAL,
    }
    _doc_type_label_map = {
        "Article": "Article",
        "Methods/resource/database": "Methods/resource\ndatabase paper",
        "Review": "Review",
        "Other": "Other",
    }

    _doc_type_comp["category"] = pd.Categorical(
        _doc_type_comp["category"], categories=_doc_type_order, ordered=True
    )
    _doc_type_comp = _doc_type_comp.sort_values("category").reset_index(drop=True)
    _doc_type_total = int(_doc_type_comp["n"].sum())
    _doc_type_comp["share"] = _doc_type_comp["n"] / _doc_type_total
    _doc_type_comp["label"] = _doc_type_comp["category"].map(_doc_type_label_map)
    _doc_type_comp["color"] = _doc_type_comp["category"].map(_doc_type_palette)

    _fig_doc_type, _ax_doc_type = plt.subplots(figsize=(8.5, 5.3), dpi=320)
    _fig_doc_type.patch.set_facecolor("white")
    _ax_doc_type.set_facecolor("white")

    _y = list(range(len(_doc_type_comp)))
    _bar_height = 0.54
    _bars = _ax_doc_type.barh(
        _y,
        _doc_type_comp["n"],
        height=_bar_height,
        color=_doc_type_comp["color"],
        edgecolor="none",
        zorder=3,
    )

    _max_n = float(_doc_type_comp["n"].max())
    _ax_doc_type.set_xlim(0, _max_n * 1.22)
    _ax_doc_type.set_ylim(-0.82, len(_doc_type_comp) - 0.22)
    _ax_doc_type.invert_yaxis()

    _ax_doc_type.set_yticks(_y)
    _ax_doc_type.set_yticklabels(
        _doc_type_comp["label"], fontsize=12.0, color=AF_GUIDE_NEUTRAL
    )
    _ax_doc_type.tick_params(axis="y", length=0, pad=16)
    _ax_doc_type.tick_params(axis="x", length=0, labelbottom=False)

    for _spine in ["top", "right", "bottom", "left"]:
        _ax_doc_type.spines[_spine].set_visible(False)

    for _yi in _y:
        _ax_doc_type.hlines(
            _yi,
            0,
            _max_n * 1.16,
            color=AF_CYAN,
            linewidth=0.8,
            zorder=1,
        )

    for _bar, (_, _row) in zip(_bars, _doc_type_comp.iterrows()):
        _x = float(_row["n"])
        _share = float(_row["share"]) * 100
        _ax_doc_type.text(
            _x + _max_n * 0.018,
            _bar.get_y() + _bar.get_height() / 2,
            f"{int(_row['n']):,}  ({_share:.1f}%)",
            va="center",
            ha="left",
            fontsize=11.4,
            color=AF_GUIDE_NEUTRAL,
        )

    # _ax_doc_type.text(
    #     _max_n * 1.16,
    #     len(_doc_type_comp) + 0.02,
    #     "Methods/resource/database papers are article-type records with AFDB, database, resource, server, benchmark, method, or aligned usage-level signals. Other includes remaining eligible non-article, non-review records such as preprints, datasets, book chapters, and dissertations.",
    #     fontsize=8.8,
    #     color=AF_EVENT_NEUTRAL,
    #     ha="right",
    #     va="top",
    #     wrap=True,
    # )

    _fig_doc_type.subplots_adjust(left=0.29, right=0.96, top=0.87, bottom=0.18)
    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_2_b(
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_SEQUENTIAL_CMAP,
    discipline_af_base,
    plt,
):
    import pandas as _pd
    import numpy as _np

    _subfield_order = [
        "Structural Biology",
        "Molecular Biology",
        "Biophysics",
        "Genetics",
        "Molecular Medicine",
        "Biochemistry",
        "Cell Biology",
        "Biotechnology",
        "Cancer Research",
        "Developmental Biology",
        "Physiology",
        "Clinical Biochemistry",
        "Endocrinology",
        "Aging",
        "Immunology",
        "Microbiology",
        "Virology",
        "Applied Microbiology and Biotechnology",
        "Pharmacology",
        "Toxicology",
        "Pharmaceutical Science",
    ]

    _group_boundaries = [14, 18]
    _group_labels = [
        ("Biochemistry, Genetics and\nMolecular Biology", 6.5),
        ("Immunology and\nMicrobiology", 15.5),
        ("Pharmacology, Toxicology\nand Pharmaceutics", 19.5),
    ]

    _subfield_label_map = {
        "Applied Microbiology and Biotechnology": "Applied Microbiology\nand Biotechnology",
    }

    _heatmap_base = discipline_af_base.loc[
        discipline_af_base["is_alphafold_related"] == True,
        [
            "publication_date",
            "primary_field_display_name",
            "primary_subfield_display_name",
        ],
    ].copy()
    _heatmap_base["publication_year"] = _heatmap_base[
        "publication_date"
    ].dt.year.astype("Int64")
    _heatmap_base = _heatmap_base[
        _heatmap_base["primary_subfield_display_name"].isin(_subfield_order)
    ].copy()

    _year_counts = (
        _heatmap_base.groupby(
            ["primary_subfield_display_name", "publication_year"], as_index=False
        )
        .size()
        .rename(columns={"size": "n"})
    )

    _year_min = int(_year_counts["publication_year"].min())
    _year_max = int(_year_counts["publication_year"].max())
    _years = list(range(_year_min, _year_max + 1))

    _heatmap = (
        _year_counts.pivot(
            index="primary_subfield_display_name",
            columns="publication_year",
            values="n",
        )
        .reindex(index=_subfield_order, columns=_years)
        .fillna(0)
    )

    _log_heatmap = _np.log1p(_heatmap.to_numpy(dtype=float))
    _label_rows = [
        _subfield_label_map.get(_name, _name) for _name in _subfield_order
    ]

    _fig_subfield, _ax_subfield = plt.subplots(figsize=(11.2, 8.6), dpi=320)
    _fig_subfield.patch.set_facecolor("white")
    _ax_subfield.set_facecolor("white")

    _im = _ax_subfield.imshow(
        _log_heatmap,
        aspect="auto",
        cmap=AF_SEQUENTIAL_CMAP.reversed(),
        interpolation="nearest",
    )

    _ax_subfield.set_xticks(range(len(_years)))
    _ax_subfield.set_xticklabels(_years, fontsize=9.3, color=AF_GUIDE_NEUTRAL)
    _ax_subfield.set_yticks(range(len(_label_rows)))
    _ax_subfield.set_yticklabels(_label_rows, fontsize=9.5, color=AF_GUIDE_NEUTRAL)
    _ax_subfield.tick_params(axis="x", length=0, pad=8)
    _ax_subfield.tick_params(axis="y", length=0, pad=12)

    for _spine in ["top", "right", "bottom", "left"]:
        _ax_subfield.spines[_spine].set_visible(False)

    for _boundary in _group_boundaries:
        _ax_subfield.hlines(
            _boundary - 0.5,
            -0.5,
            len(_years) - 0.5,
            color=AF_BLUE,
            linewidth=1.15,
            zorder=5,
        )

    for _row in range(len(_subfield_order) + 1):
        _ax_subfield.hlines(
            _row - 0.5,
            -0.5,
            len(_years) - 0.5,
            color=AF_CYAN,
            linewidth=0.55,
            zorder=4,
        )

    for _col in range(len(_years) + 1):
        _ax_subfield.vlines(
            _col - 0.5,
            -0.5,
            len(_subfield_order) - 0.5,
            color=AF_CYAN,
            linewidth=0.45,
            zorder=4,
        )

    # _ax_subfield.text(
    #     -0.5,
    #     -2.55,
    #     "Subfield Diffusion by Year",
    #     fontsize=17.0,
    #     fontweight="bold",
    #     color=AF_PURPLE,
    #     ha="left",
    #     va="bottom",
    # )
    # _ax_subfield.text(
    #     -0.5,
    #     -1.65,
    #     "Three broad disciplines and 21 AlphaFold-relevant subfields",
    #     fontsize=10.2,
    #     color=AF_EVENT_NEUTRAL,
    #     ha="left",
    #     va="bottom",
    # )

    for _label, _y in _group_labels:
        _ax_subfield.text(
            -3.05,
            _y,
            _label,
            fontsize=10.0,
            fontweight="bold",
            color=AF_GUIDE_NEUTRAL,
            ha="right",
            va="center",
        )

    _cbar = _fig_subfield.colorbar(_im, ax=_ax_subfield, fraction=0.028, pad=0.02)
    _cbar.outline.set_visible(False)
    _cbar.ax.tick_params(length=0, labelsize=8.7, colors=AF_GUIDE_NEUTRAL)
    _cbar.set_label(
        "log(1 + annual AF paper count)",
        fontsize=9.0,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )

    _fig_subfield.subplots_adjust(left=0.34, right=0.92, top=0.88, bottom=0.10)
    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_4(
    AF_BLUE,
    AF_CYAN,
    AF_EVENT_NEUTRAL,
    AF_GUIDE_NEUTRAL,
    AF_PURPLE,
    country_first_af_adoption_rd,
    country_rd_strength_lookup,
    pd,
    plt,
    rd_strength_country_counts,
    sns,
):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    _rd_strength_plot = country_rd_strength_lookup.copy()
    _rd_strength_plot = _rd_strength_plot.dropna(
        subset=[
            "rd_strength_index",
            "rnd_gdp_pct_mean_2015_2018",
            "pre_af_life_science_fractional_output_2015_2018",
            "rd_strength_tertile",
        ]
    ).copy()
    _rd_strength_plot = _rd_strength_plot.sort_values(
        "rd_strength_index", ascending=False
    ).reset_index(drop=True)
    _rd_strength_plot["rank"] = np.arange(1, len(_rd_strength_plot) + 1)
    _rd_strength_plot["log10_pre_af_life_output"] = np.log10(
        _rd_strength_plot["pre_af_life_science_fractional_output_2015_2018"] + 1
    )

    _rd_group_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _rd_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _rd_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Medium",
        "Low R&D strength": "Low",
    }

    _rd_strength_plot["rd_strength_tertile"] = pd.Categorical(
        _rd_strength_plot["rd_strength_tertile"],
        categories=_rd_group_order,
        ordered=True,
    )

    _phase_gap_base = country_first_af_adoption_rd.copy()
    _phase_gap_base = _phase_gap_base.dropna(
        subset=["rd_strength_tertile", "entry_phase"]
    ).copy()
    _phase_gap_map = {
        "pre_af2": "Pre-AF2",
        "af2_phase": "AF2",
        "multimer_phase": "AF2",
        "afdb_phase": "AF2",
        "af3_phase": "AF3",
    }
    _phase_order = ["Pre-AF2", "AF2", "AF3"]
    _phase_group_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Medium",
        "Low R&D strength": "Low",
    }

    _phase_gap_base["phase3"] = (
        _phase_gap_base["entry_phase"].astype(str).map(_phase_gap_map)
    )
    _phase_gap_base = _phase_gap_base.dropna(subset=["phase3"]).copy()

    _phase_counts = _phase_gap_base.groupby(
        ["rd_strength_tertile", "phase3"], as_index=False
    ).agg(new_adopters=("country_code", "nunique"))

    _phase_grid = pd.MultiIndex.from_product(
        [_rd_group_order, _phase_order],
        names=["rd_strength_tertile", "phase3"],
    ).to_frame(index=False)

    _phase_counts = _phase_grid.merge(
        _phase_counts,
        on=["rd_strength_tertile", "phase3"],
        how="left",
    )
    _phase_counts["new_adopters"] = _phase_counts["new_adopters"].fillna(0)
    _phase_counts = _phase_counts.merge(
        rd_strength_country_counts,
        on="rd_strength_tertile",
        how="left",
    )
    _phase_counts["cumulative_share"] = (
        _phase_counts.sort_values(["rd_strength_tertile", "phase3"])
        .groupby("rd_strength_tertile")["new_adopters"]
        .cumsum()
        / _phase_counts.sort_values(["rd_strength_tertile", "phase3"])[
            "total_countries_in_group"
        ]
    )
    _phase_counts["phase_index"] = _phase_counts["phase3"].map(
        {"Pre-AF2": 0, "AF2": 1, "AF3": 2}
    )
    _phase_counts = _phase_counts.sort_values(
        ["rd_strength_tertile", "phase_index"]
    ).reset_index(drop=True)
    _phase_counts["cumulative_share"] = (
        _phase_counts.groupby("rd_strength_tertile")["new_adopters"].cumsum()
        / _phase_counts["total_countries_in_group"]
    )

    _fig_rd_global = plt.figure(figsize=(12.8, 9.0), dpi=320)
    _fig_rd_global.patch.set_facecolor("white")
    _rd_grid = GridSpec(
        2,
        2,
        figure=_fig_rd_global,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.26,
        hspace=0.34,
    )

    _ax_rank = _fig_rd_global.add_subplot(_rd_grid[0, 0])
    _ax_dist = _fig_rd_global.add_subplot(_rd_grid[0, 1])
    _ax_scatter = _fig_rd_global.add_subplot(_rd_grid[1, 0])
    _ax_phase_cum = _fig_rd_global.add_subplot(_rd_grid[1, 1])

    for _group in _rd_group_order:
        _group_df = _rd_strength_plot[
            _rd_strength_plot["rd_strength_tertile"] == _group
        ].copy()
        _ax_rank.plot(
            _group_df["rank"],
            _group_df["rd_strength_index"],
            color=_rd_palette[_group],
            linewidth=1.35,
            alpha=0.95,
            zorder=2,
        )
        _ax_rank.scatter(
            _group_df["rank"],
            _group_df["rd_strength_index"],
            s=13,
            color=_rd_palette[_group],
            alpha=0.95,
            zorder=3,
        )

    _ax_rank.axhline(
        0, color=AF_EVENT_NEUTRAL, linewidth=0.9, linestyle="--", zorder=1
    )
    _ax_rank.set_xlabel("Country rank", fontsize=10.2, color=AF_GUIDE_NEUTRAL)
    _ax_rank.set_ylabel(
        "Composite R&D strength index", fontsize=10.2, color=AF_GUIDE_NEUTRAL
    )
    _ax_rank.set_title(
        "a  Ranked distribution",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_PURPLE,
        pad=8,
    )

    for _group in _rd_group_order:
        _group_df = _rd_strength_plot[
            _rd_strength_plot["rd_strength_tertile"] == _group
        ]
        sns.histplot(
            _group_df["rd_strength_index"],
            bins=15,
            stat="density",
            element="step",
            fill=True,
            alpha=0.20,
            color=_rd_palette[_group],
            linewidth=0,
            ax=_ax_dist,
        )
        if len(_group_df) > 1:
            sns.kdeplot(
                _group_df["rd_strength_index"],
                color=_rd_palette[_group],
                linewidth=1.55,
                bw_adjust=1.05,
                ax=_ax_dist,
                clip=(
                    _rd_strength_plot["rd_strength_index"].min(),
                    _rd_strength_plot["rd_strength_index"].max(),
                ),
            )

    _ax_dist.axvline(
        0, color=AF_EVENT_NEUTRAL, linewidth=0.9, linestyle="--", zorder=1
    )
    _ax_dist.set_xlabel(
        "Composite R&D strength index", fontsize=10.2, color=AF_GUIDE_NEUTRAL
    )
    _ax_dist.set_ylabel("Density", fontsize=10.2, color=AF_GUIDE_NEUTRAL)
    _ax_dist.set_title(
        "b  Distribution",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_PURPLE,
        pad=8,
    )

    for _group in _rd_group_order:
        _group_df = _rd_strength_plot[
            _rd_strength_plot["rd_strength_tertile"] == _group
        ]
        _ax_scatter.scatter(
            _group_df["rnd_gdp_pct_mean_2015_2018"],
            _group_df["log10_pre_af_life_output"],
            s=34,
            color=_rd_palette[_group],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
            zorder=3,
        )

    _x_median = float(_rd_strength_plot["rnd_gdp_pct_mean_2015_2018"].median())
    _y_median = float(_rd_strength_plot["log10_pre_af_life_output"].median())
    _ax_scatter.axvline(
        _x_median, color=AF_EVENT_NEUTRAL, linewidth=0.9, linestyle="--", zorder=1
    )
    _ax_scatter.axhline(
        _y_median, color=AF_EVENT_NEUTRAL, linewidth=0.9, linestyle="--", zorder=1
    )

    for _label_country in [
        "United States of America",
        "China",
        "Germany",
        "United Kingdom",
        "Japan",
        "Israel",
        "South Korea",
    ]:
        _match = _rd_strength_plot[
            _rd_strength_plot["country_name_map"] == _label_country
        ]
        if _match.empty:
            continue
        _row = _match.iloc[0]
        _display_name = (
            "South Korea"
            if _label_country == "South Korea"
            else _label_country.replace(
                "United States of America", "United States"
            )
        )
        _ax_scatter.text(
            _row["rnd_gdp_pct_mean_2015_2018"] + 0.05,
            _row["log10_pre_af_life_output"] + 0.03,
            _display_name,
            fontsize=8.3,
            color=AF_GUIDE_NEUTRAL,
            ha="left",
            va="bottom",
            zorder=4,
        )

    _ax_scatter.set_xlabel(
        "R&D expenditure (% GDP), 2015-2018 mean",
        fontsize=10.2,
        color=AF_GUIDE_NEUTRAL,
    )
    _ax_scatter.set_ylabel(
        "log10(pre-AF life-science output + 1)",
        fontsize=10.2,
        color=AF_GUIDE_NEUTRAL,
    )
    _ax_scatter.set_title(
        "c  Structural components",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_PURPLE,
        pad=8,
    )

    _x = list(range(len(_phase_order)))
    for _group in _rd_group_order:
        _group_df = _phase_counts[
            _phase_counts["rd_strength_tertile"] == _group
        ].copy()
        _color = _rd_palette[_group]
        _ax_phase_cum.plot(
            _x,
            _group_df["cumulative_share"],
            color=_color,
            linewidth=2.2,
            marker="o",
            markersize=5.2,
            zorder=3,
        )

    _ax_phase_cum.set_xticks(_x)
    _ax_phase_cum.set_xticklabels(
        _phase_order, fontsize=10.0, color=AF_GUIDE_NEUTRAL
    )
    _ax_phase_cum.set_ylim(0, 1.05)
    _ax_phase_cum.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _ax_phase_cum.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9.3,
        color=AF_GUIDE_NEUTRAL,
    )
    _ax_phase_cum.set_xlim(-0.05, len(_phase_order) - 1 + 0.10)
    _ax_phase_cum.set_ylabel(
        "Cumulative share of adopters", fontsize=10.1, color=AF_GUIDE_NEUTRAL
    )
    _ax_phase_cum.set_title(
        "d  Cumulative adoption by phase end",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_PURPLE,
        pad=8,
    )

    _legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_rd_palette[_group],
            markeredgecolor="none",
            markersize=7.5,
            label=_rd_label_map[_group],
        )
        for _group in _rd_group_order
    ]
    _ax_scatter.legend(
        handles=_legend_handles,
        title="R&D group",
        frameon=False,
        fontsize=9.3,
        title_fontsize=9.4,
        loc="lower right",
    )

    for _ax in [_ax_rank, _ax_dist, _ax_scatter, _ax_phase_cum]:
        _ax.set_facecolor("white")
        _ax.grid(axis="y", color=AF_CYAN, linewidth=0.7)
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["left"].set_linewidth(0.8)
        _ax.spines["bottom"].set_linewidth(0.8)
        _ax.tick_params(axis="both", labelsize=9.4, colors=AF_GUIDE_NEUTRAL)

    _ax_rank.set_xlim(1, len(_rd_strength_plot))
    _ax_rank.margins(x=0.01)
    _ax_phase_cum.tick_params(axis="both", length=0)

    _fig_rd_global.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.09,
    )
    plt.gca()
    return Line2D, np


@app.cell(hide_code=True)
def ext_fig_5(AF_CYAN, AF_PURPLE, country_collab_edges, math, nx, pd, plt):
    _fig_f26_chord, _ax_f26_chord = plt.subplots(figsize=(14.5, 14.5), dpi=300)

    _f26_chord_edges = (
        country_collab_edges.copy()
        .sort_values("n_shared_works", ascending=False)
        .head(110)
    )
    _f26_chord_nodes = pd.Index(
        pd.concat(
            [
                _f26_chord_edges["source_country"],
                _f26_chord_edges["target_country"],
            ],
            ignore_index=True,
        ).unique()
    )
    _f26_chord_graph = nx.Graph()
    for _row_f26c in _f26_chord_edges.itertuples(index=False):
        _f26_chord_graph.add_edge(
            _row_f26c.source_country,
            _row_f26c.target_country,
            weight=_row_f26c.n_shared_works,
        )

    _f26_degree_weight = dict(_f26_chord_graph.degree(weight="weight"))
    _f26_node_order = sorted(
        _f26_chord_nodes,
        key=lambda _node_f26c: _f26_degree_weight.get(_node_f26c, 0),
        reverse=True,
    )
    _f26_n_nodes = len(_f26_node_order)
    _f26_angles = {
        _node_f26c: (math.pi / 2) - (2 * math.pi * _idx_f26c / _f26_n_nodes)
        for _idx_f26c, _node_f26c in enumerate(_f26_node_order)
    }
    _f26_radius = 1.0
    _f26_positions = {
        _node_f26c: (
            _f26_radius * math.cos(_f26_angles[_node_f26c]),
            _f26_radius * math.sin(_f26_angles[_node_f26c]),
        )
        for _node_f26c in _f26_node_order
    }


    def _f26_mix_colors(_color_a_f26c, _color_b_f26c, _t_f26c):
        _rgb_a_f26c = plt.matplotlib.colors.to_rgb(_color_a_f26c)
        _rgb_b_f26c = plt.matplotlib.colors.to_rgb(_color_b_f26c)
        return tuple(
            (1 - _t_f26c) * _channel_a_f26c + _t_f26c * _channel_b_f26c
            for _channel_a_f26c, _channel_b_f26c in zip(_rgb_a_f26c, _rgb_b_f26c)
        )


    _f26_color_lookup = {}
    _f26_rank_denominator = max(_f26_n_nodes - 1, 1)
    for _idx_f26c, _node_f26c in enumerate(_f26_node_order):
        _t_f26c = _idx_f26c / _f26_rank_denominator
        _f26_color_lookup[_node_f26c] = _f26_mix_colors(
            AF_PURPLE, AF_CYAN, _t_f26c
        )
    _f26_weights_chord = _f26_chord_edges["n_shared_works"]
    _f26_weight_min = float(_f26_weights_chord.min())
    _f26_weight_max = float(_f26_weights_chord.max())
    _f26_degree_values = list(_f26_degree_weight.values())
    _f26_q60 = (
        pd.Series(_f26_degree_values).quantile(0.6) if _f26_degree_values else 0
    )
    _f26_q80 = (
        pd.Series(_f26_degree_values).quantile(0.8) if _f26_degree_values else 0
    )

    for _edge_f26c in _f26_chord_edges.itertuples(index=False):
        _source_f26c = _edge_f26c.source_country
        _target_f26c = _edge_f26c.target_country
        _x0_f26c, _y0_f26c = _f26_positions[_source_f26c]
        _x1_f26c, _y1_f26c = _f26_positions[_target_f26c]
        _a0_f26c = _f26_angles[_source_f26c]
        _a1_f26c = _f26_angles[_target_f26c]
        _mid_angle_f26c = (_a0_f26c + _a1_f26c) / 2
        _angle_gap_f26c = abs(_a0_f26c - _a1_f26c)
        _ctrl_radius_f26c = 0.06 + 0.16 * (
            1 - min(_angle_gap_f26c, math.pi) / math.pi
        )
        _cx_f26c = _ctrl_radius_f26c * math.cos(_mid_angle_f26c)
        _cy_f26c = _ctrl_radius_f26c * math.sin(_mid_angle_f26c)
        _path_f26c = plt.matplotlib.path.Path(
            [(_x0_f26c, _y0_f26c), (_cx_f26c, _cy_f26c), (_x1_f26c, _y1_f26c)],
            [
                plt.matplotlib.path.Path.MOVETO,
                plt.matplotlib.path.Path.CURVE3,
                plt.matplotlib.path.Path.CURVE3,
            ],
        )
        _edge_scale_f26c = (_edge_f26c.n_shared_works - _f26_weight_min) / max(
            _f26_weight_max - _f26_weight_min, 1
        )
        _edge_color_f26c = _f26_color_lookup[_source_f26c]
        _patch_f26c = plt.matplotlib.patches.PathPatch(
            _path_f26c,
            facecolor="none",
            edgecolor=_edge_color_f26c,
            linewidth=1.6 + 8.8 * (_edge_scale_f26c**0.9),
            alpha=0.16 + 0.5 * (_edge_scale_f26c**0.85),
            capstyle="round",
            zorder=1,
        )
        _ax_f26_chord.add_patch(_patch_f26c)

    _f26_ring_outer = 1.075
    _f26_ring_inner = 0.955
    _f26_max_degree = max(_f26_degree_weight.values()) if _f26_degree_weight else 1
    for _node_f26c in _f26_node_order:
        _angle_f26c = _f26_angles[_node_f26c]
        _theta1_f26c = math.degrees(_angle_f26c - math.pi / _f26_n_nodes * 0.82)
        _theta2_f26c = math.degrees(_angle_f26c + math.pi / _f26_n_nodes * 0.82)
        _node_scale_f26c = _f26_degree_weight.get(_node_f26c, 0) / _f26_max_degree
        _wedge_f26c = plt.matplotlib.patches.Wedge(
            center=(0, 0),
            r=_f26_ring_outer,
            theta1=_theta1_f26c,
            theta2=_theta2_f26c,
            width=_f26_ring_outer - _f26_ring_inner + 0.07 * _node_scale_f26c,
            facecolor=_f26_color_lookup[_node_f26c],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.99,
            zorder=3,
        )
        _ax_f26_chord.add_patch(_wedge_f26c)

        _label_radius_f26c = 1.19
        _lx_f26c = _label_radius_f26c * math.cos(_angle_f26c)
        _ly_f26c = _label_radius_f26c * math.sin(_angle_f26c)
        _rotation_f26c = math.degrees(_angle_f26c)
        _ha_f26c = "left"
        if math.cos(_angle_f26c) < 0:
            _rotation_f26c += 180
            _ha_f26c = "right"
        _node_weight_f26c = _f26_degree_weight.get(_node_f26c, 0)
        _ax_f26_chord.text(
            _lx_f26c,
            _ly_f26c,
            _node_f26c,
            rotation=_rotation_f26c,
            rotation_mode="anchor",
            ha=_ha_f26c,
            va="center",
            fontsize=10.6 if _node_weight_f26c >= _f26_q60 else 8.8,
            fontweight="bold" if _node_weight_f26c >= _f26_q80 else "normal",
            color="#1f2937",
            zorder=4,
        )

    # _ax_f26_chord.add_patch(
    #     plt.Circle(
    #         (0, 0),
    #         0.67,
    #         facecolor="white",
    #         edgecolor="#e5e7eb",
    #         linewidth=1.0,
    #         zorder=2,
    #     )
    # )

    # _ax_f26_chord.text(
    #     0,
    #     0.08,
    #     "Figure 26A",
    #     ha="center",
    #     va="center",
    #     fontsize=17,
    #     fontweight="bold",
    #     color="#111827",
    # )
    # _ax_f26_chord.text(
    #     0,
    #     -0.02,
    #     "Chord-style map of\nAlphaFold country collaboration",
    #     ha="center",
    #     va="center",
    #     fontsize=12,
    #     color="#4b5563",
    #     linespacing=1.3,
    # )
    # _ax_f26_chord.text(
    #     0,
    #     -0.17,
    #     f"Top {_f26_n_nodes} countries, strongest {len(_f26_chord_edges)} ties",
    #     ha="center",
    #     va="center",
    #     fontsize=10,
    #     color="#6b7280",
    # )

    # _ax_f26_chord.set_title(
    #     "Nature-style chord visualization of the AlphaFold cross-country collaboration network",
    #     fontsize=18,
    #     fontweight="bold",
    #     loc="left",
    #     pad=26,
    # )
    # _ax_f26_chord.text(
    #     0,
    #     1.045,
    #     "Arc segments identify countries and curved ribbons encode repeated co-authorship ties. Ribbon thickness and opacity scale with the number of shared AlphaFold-related papers.",
    #     transform=_ax_f26_chord.transAxes,
    #     fontsize=10.5,
    #     color="#4b5563",
    #     va="bottom",
    # )
    _ax_f26_chord.set_xlim(-1.32, 1.32)
    _ax_f26_chord.set_ylim(-1.32, 1.32)
    _ax_f26_chord.set_aspect("equal")
    _ax_f26_chord.axis("off")
    _fig_f26_chord.patch.set_facecolor("white")
    _ax_f26_chord.set_facecolor("white")
    _fig_f26_chord.subplots_adjust(left=0.03, right=0.97, top=0.87, bottom=0.04)

    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_6(
    AF_CORAL,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_PURPLE,
    collab_work_base,
    pd,
    plt,
):
    import numpy as _np

    _team_size_plot = collab_work_base.dropna(subset=["team_size"]).copy()
    _team_size_plot = _team_size_plot[_team_size_plot["team_size"] > 0].copy()
    _team_size_plot["group"] = _team_size_plot["is_alphafold_related"].map(
        {True: "AF", False: "non-AF"}
    )

    _team_size_bins = [0.5, 1.5, 2.5, 4.5, 9.5, 19.5, float("inf")]
    _team_size_labels = ["1", "2", "3-4", "5-9", "10-19", "20+"]
    _team_size_order = ["AF", "non-AF"]
    _team_size_color_map = {
        "AF": AF_CORAL,
        "non-AF": AF_PURPLE,
    }

    _team_size_plot["team_size_bin"] = pd.cut(
        _team_size_plot["team_size"],
        bins=_team_size_bins,
        labels=_team_size_labels,
        right=True,
    )

    _team_size_bin_counts = (
        _team_size_plot.groupby(
            ["group", "team_size_bin"], as_index=False, observed=False
        )
        .size()
        .rename(columns={"size": "paper_count"})
    )
    _team_size_group_totals = (
        _team_size_plot.groupby("group", as_index=False)
        .size()
        .rename(columns={"size": "group_total"})
    )
    _team_size_bin_counts = _team_size_bin_counts.merge(
        _team_size_group_totals,
        on="group",
        how="left",
    )
    _team_size_bin_counts["share_pct"] = (
        100
        * _team_size_bin_counts["paper_count"]
        / _team_size_bin_counts["group_total"]
    )
    _team_size_bin_counts["team_size_bin"] = pd.Categorical(
        _team_size_bin_counts["team_size_bin"],
        categories=_team_size_labels,
        ordered=True,
    )
    _team_size_bin_counts = _team_size_bin_counts.sort_values(
        ["team_size_bin", "group"]
    ).reset_index(drop=True)

    _fig_team_size, (_ax_team_bins, _ax_team_ecdf) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.4),
        dpi=280,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    _fig_team_size.patch.set_facecolor("white")

    _x = _np.arange(len(_team_size_labels))
    _bar_width = 0.34

    for _offset, _group in [(-_bar_width / 2, "AF"), (_bar_width / 2, "non-AF")]:
        _group_df = _team_size_bin_counts[
            _team_size_bin_counts["group"] == _group
        ].copy()
        _group_df = (
            _group_df.set_index("team_size_bin")
            .reindex(_team_size_labels)
            .reset_index()
        )
        _bars = _ax_team_bins.bar(
            _x + _offset,
            _group_df["share_pct"],
            width=_bar_width,
            color=_team_size_color_map[_group],
            alpha=0.92 if _group == "AF" else 0.78,
            edgecolor="white",
            linewidth=0.8,
            label=_group,
            zorder=3,
        )
        for _bar, _share in zip(_bars, _group_df["share_pct"]):
            if float(_share) >= 5.5:
                _ax_team_bins.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    float(_share) + 0.8,
                    f"{float(_share):.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    color=AF_GUIDE_NEUTRAL,
                )

    for _group in _team_size_order:
        _ecdf_df = _team_size_plot[_team_size_plot["group"] == _group].copy()
        _values = _np.sort(_ecdf_df["team_size"].astype(float).to_numpy())
        _y = _np.arange(1, len(_values) + 1) / len(_values)
        _ax_team_ecdf.step(
            _values,
            _y,
            where="post",
            color=_team_size_color_map[_group],
            linewidth=2.1 if _group == "AF" else 1.9,
            alpha=0.98 if _group == "AF" else 0.88,
            label=_group,
            zorder=3,
        )

    _team_size_af_median = float(
        _team_size_plot.loc[_team_size_plot["group"] == "AF", "team_size"].median()
    )
    _team_size_nonaf_median = float(
        _team_size_plot.loc[
            _team_size_plot["group"] == "non-AF", "team_size"
        ].median()
    )
    _team_size_gap = _team_size_af_median - _team_size_nonaf_median

    _ax_team_bins.set_title(
        "a  Team-size bins",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_GUIDE_NEUTRAL,
        pad=8,
    )
    _ax_team_bins.set_ylabel("Share of papers (%)", fontsize=10.2, color="#111111")
    _ax_team_bins.set_xlabel("Authors per paper", fontsize=10.2, color="#111111")
    _ax_team_bins.set_xticks(_x)
    _ax_team_bins.set_xticklabels(_team_size_labels, fontsize=9.7, color="#111111")

    _ax_team_ecdf.set_title(
        "b  Empirical cumulative distribution",
        loc="left",
        fontsize=12.0,
        fontweight="bold",
        color=AF_GUIDE_NEUTRAL,
        pad=8,
    )
    _ax_team_ecdf.set_xlabel("Authors per paper", fontsize=10.2, color="#111111")
    _ax_team_ecdf.set_ylabel(
        "Cumulative share of papers", fontsize=10.2, color="#111111"
    )
    _ax_team_ecdf.set_ylim(0, 1.02)
    _ax_team_ecdf.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _ax_team_ecdf.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"], fontsize=9.5, color=AF_GUIDE_NEUTRAL
    )
    _ax_team_ecdf.set_xlim(
        1, min(40, int(_team_size_plot["team_size"].quantile(0.99)) + 2)
    )

    _ax_team_ecdf.axvline(
        _team_size_af_median,
        color=AF_CORAL,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.60,
        zorder=2,
    )
    _ax_team_ecdf.axvline(
        _team_size_nonaf_median,
        color=AF_PURPLE,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.72,
        zorder=2,
    )
    _ax_team_ecdf.text(
        0.985,
        0.08,
        f"Median gap: {_team_size_gap:.1f} authors\nAF median = {_team_size_af_median:.1f}; non-AF median = {_team_size_nonaf_median:.1f}",
        transform=_ax_team_ecdf.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": AF_CYAN,
            "linewidth": 0.5,
            "alpha": 0.92,
            "pad": 0.55,
        },
        zorder=5,
    )

    for _ax in [_ax_team_bins, _ax_team_ecdf]:
        _ax.set_facecolor("white")
        _ax.grid(
            axis="y",
            linestyle=(0, (2, 3)),
            linewidth=0.72,
            color=AF_CYAN,
            alpha=0.9,
        )
        _ax.grid(axis="x", visible=False)
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["left"].set_linewidth(0.8)
        _ax.spines["bottom"].set_linewidth(0.8)
        _ax.tick_params(axis="both", labelsize=9.5, colors=AF_GUIDE_NEUTRAL)

    _legend_handles = [
        plt.Line2D([0], [0], color=_team_size_color_map["AF"], lw=2.2, label="AF"),
        plt.Line2D(
            [0], [0], color=_team_size_color_map["non-AF"], lw=2.0, label="non-AF"
        ),
    ]
    _ax_team_ecdf.legend(
        handles=_legend_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=9.1,
        title="Paper group",
        title_fontsize=9.2,
    )

    # _fig_team_size.text(
    #     0.065,
    #     0.985,
    #     "Team-Size Distribution in AlphaFold-Related Research",
    #     ha="left",
    #     va="top",
    #     fontsize=16.0,
    #     fontweight="bold",
    #     color=AF_GUIDE_NEUTRAL,
    # )
    # _fig_team_size.text(
    #     0.065,
    #     0.947,
    #     "Binned composition and empirical distribution relative to non-AF papers",
    #     ha="left",
    #     va="top",
    #     fontsize=9.9,
    #     color="#5B5B5B",
    # )

    _fig_team_size.subplots_adjust(
        left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.22
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_7(
    AF_ANNOTATION_FONT_SIZE,
    AF_LABEL_FONT_SIZE,
    AF_SEQUENTIAL_CMAP,
    figure_3d_rd_pair_matrix,
    figure_3d_rd_pair_share,
    pd,
    plt,
    sns,
):
    _figure_rd_matrix_absolute = figure_3d_rd_pair_matrix.copy()
    _figure_rd_matrix_row_share = figure_3d_rd_pair_share.copy()
    _figure_rd_matrix_col_share = _figure_rd_matrix_absolute.div(
        _figure_rd_matrix_absolute.sum(axis=0).replace(0, pd.NA),
        axis=1,
    )

    _fig_rd_matrix_alt, _axes_rd_matrix_alt = plt.subplots(
        1,
        3,
        figsize=(17.2, 6.1),
        dpi=240,
    )
    _fig_rd_matrix_alt.patch.set_facecolor("white")

    _rd_tick_labels = ["High", "Middle", "Low"]
    _rd_label_font_size = AF_LABEL_FONT_SIZE * 0.82

    _rd_absolute_max = (
        float(_figure_rd_matrix_absolute.to_numpy().max())
        if _figure_rd_matrix_absolute.size
        else 0
    )
    _rd_share_max = max(
        float(_figure_rd_matrix_row_share.to_numpy().max())
        if _figure_rd_matrix_row_share.size
        else 0,
        float(_figure_rd_matrix_col_share.to_numpy().max())
        if _figure_rd_matrix_col_share.size
        else 0,
    )

    sns.heatmap(
        _figure_rd_matrix_absolute,
        annot=True,
        fmt=".0f",
        cmap=AF_SEQUENTIAL_CMAP.reversed(),
        vmin=0,
        vmax=_rd_absolute_max,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Shared AF collaboration ties (count)"},
        ax=_axes_rd_matrix_alt[0],
    )

    sns.heatmap(
        _figure_rd_matrix_row_share * 100,
        annot=True,
        fmt=".1f",
        cmap=AF_SEQUENTIAL_CMAP.reversed(),
        vmin=0,
        vmax=_rd_share_max * 100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Row-normalized collaboration share (%)"},
        ax=_axes_rd_matrix_alt[1],
    )

    sns.heatmap(
        _figure_rd_matrix_col_share * 100,
        annot=True,
        fmt=".1f",
        cmap=AF_SEQUENTIAL_CMAP.reversed(),
        vmin=0,
        vmax=_rd_share_max * 100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Column-normalized collaboration share (%)"},
        ax=_axes_rd_matrix_alt[2],
    )

    _panel_titles = [
        "a  Absolute count",
        "b  Row-normalized",
        "c  Column-normalized",
    ]

    for _ax, _title in zip(_axes_rd_matrix_alt, _panel_titles):
        _ax.set_title(
            _title,
            loc="left",
            fontsize=12.1,
            fontweight="bold",
            color="#111111",
            pad=8,
        )
        _ax.set_xlabel("Partner R&D strength group", fontsize=_rd_label_font_size)
        _ax.set_ylabel("Source R&D strength group", fontsize=_rd_label_font_size)
        _ax.set_xticklabels(_rd_tick_labels)
        _ax.set_yticklabels(_rd_tick_labels)
        plt.setp(_ax.get_xticklabels(), rotation=25, ha="right", color="#1f2937")
        plt.setp(_ax.get_yticklabels(), rotation=0, color="#1f2937")
        _ax.tick_params(
            axis="both", labelsize=AF_ANNOTATION_FONT_SIZE, colors="#1f2937"
        )
        _ax.set_facecolor("white")

    for _cbar_ax in _fig_rd_matrix_alt.axes[3:]:
        _cbar_ax.tick_params(labelsize=AF_ANNOTATION_FONT_SIZE, colors="#1f2937")
        _cbar_ax.yaxis.label.set_color("#1f2937")
        _cbar_ax.yaxis.label.set_size(AF_ANNOTATION_FONT_SIZE)

    # _fig_rd_matrix_alt.text(
    #     0.055,
    #     0.985,
    #     "Absolute-Count and Alternative-Normalization Views of the Collaboration Matrix",
    #     ha="left",
    #     va="top",
    #     fontsize=16.0,
    #     fontweight="bold",
    #     color="#111111",
    # )
    # _fig_rd_matrix_alt.text(
    #     0.055,
    #     0.947,
    #     "R&D-strength collaboration structure in AlphaFold-related research: scale, outward composition, and inward composition.",
    #     ha="left",
    #     va="top",
    #     fontsize=9.8,
    #     color="#5B5B5B",
    # )

    _fig_rd_matrix_alt.subplots_adjust(
        left=0.06, right=0.98, top=0.82, bottom=0.14, wspace=0.52
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_8(
    AF_ANNOTATION_FONT_SIZE,
    AF_CORAL,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    collab_work_base,
    discipline_af_base,
    pd,
    plt,
):
    _f3_intl_base = collab_work_base[
        ["work_id", "is_alphafold_related", "is_international_collab"]
    ].merge(
        discipline_af_base[
            ["work_id", "publication_date", "primary_subfield_display_name"]
        ],
        on="work_id",
        how="inner",
    )
    _f3_intl_base["publication_date"] = pd.to_datetime(
        _f3_intl_base["publication_date"]
    )
    _f3_intl_base["publication_year"] = _f3_intl_base[
        "publication_date"
    ].dt.year.astype("Int64")
    _f3_intl_base = _f3_intl_base.dropna(
        subset=["publication_year", "primary_subfield_display_name"]
    ).copy()
    _f3_intl_base = _f3_intl_base[
        _f3_intl_base["publication_year"].between(2019, 2025)
    ].copy()
    _f3_intl_base["research_type"] = _f3_intl_base["is_alphafold_related"].map(
        {True: "AF", False: "non-AF"}
    )

    _f3_intl_year = _f3_intl_base.groupby(
        ["publication_year", "research_type"],
        as_index=False,
    ).agg(
        total_papers=("work_id", "nunique"),
        international_papers=("is_international_collab", "sum"),
    )
    _f3_intl_year["international_collab_share_pct"] = (
        100 * _f3_intl_year["international_papers"] / _f3_intl_year["total_papers"]
    )
    _f3_intl_year = _f3_intl_year.sort_values(
        ["research_type", "publication_year"]
    ).reset_index(drop=True)

    _f3_intl_subfield = _f3_intl_base.groupby(
        ["primary_subfield_display_name", "research_type"],
        as_index=False,
    ).agg(
        total_papers=("work_id", "nunique"),
        international_papers=("is_international_collab", "sum"),
    )
    _f3_intl_subfield["international_collab_share_pct"] = (
        100
        * _f3_intl_subfield["international_papers"]
        / _f3_intl_subfield["total_papers"]
    )

    _f3_intl_subfield_wide = _f3_intl_subfield.pivot_table(
        index="primary_subfield_display_name",
        columns="research_type",
        values=["international_collab_share_pct", "total_papers"],
    )
    _f3_intl_subfield_wide.columns = [
        f"{_a}_{_b}" for _a, _b in _f3_intl_subfield_wide.columns
    ]
    _f3_intl_subfield_wide = _f3_intl_subfield_wide.reset_index().fillna(0)
    _f3_intl_subfield_wide = _f3_intl_subfield_wide.rename(
        columns={
            "international_collab_share_pct_non-AF": "international_collab_share_pct_non_AF",
            "total_papers_non-AF": "total_papers_non_AF",
        }
    )
    _f3_intl_subfield_wide["diff_pp"] = (
        _f3_intl_subfield_wide["international_collab_share_pct_AF"]
        - _f3_intl_subfield_wide["international_collab_share_pct_non_AF"]
    )
    _f3_intl_subfield_wide["abs_diff_pp"] = _f3_intl_subfield_wide["diff_pp"].abs()
    _f3_intl_subfield_wide["combined_total"] = (
        _f3_intl_subfield_wide["total_papers_AF"]
        + _f3_intl_subfield_wide["total_papers_non_AF"]
    )
    _f3_intl_subfield_top = (
        _f3_intl_subfield_wide.sort_values(
            ["abs_diff_pp", "combined_total"],
            ascending=[False, False],
        )
        .head(12)
        .copy()
    )
    _f3_intl_subfield_top = _f3_intl_subfield_top.sort_values(
        "diff_pp"
    ).reset_index(drop=True)

    _f3_intl_subfield_label_map = {
        "Applied Microbiology and Biotechnology": "Applied Microbiology\nand Biotechnology",
        "Clinical Biochemistry": "Clinical\nBiochemistry",
    }
    _f3_intl_subfield_top["label"] = _f3_intl_subfield_top[
        "primary_subfield_display_name"
    ].map(lambda _v: _f3_intl_subfield_label_map.get(_v, _v))

    _fig_f3_intl, (_ax_f3_intl_year, _ax_f3_intl_subfield) = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.4),
        dpi=280,
        gridspec_kw={"width_ratios": [1.15, 1.35]},
    )
    _fig_f3_intl.patch.set_facecolor("white")

    _f3_intl_color_map = {"AF": AF_CORAL, "non-AF": AF_PURPLE}

    for _group_f3_intl in ["AF", "non-AF"]:
        _group_df_f3_intl = _f3_intl_year[
            _f3_intl_year["research_type"] == _group_f3_intl
        ]
        _ax_f3_intl_year.plot(
            _group_df_f3_intl["publication_year"],
            _group_df_f3_intl["international_collab_share_pct"],
            color=_f3_intl_color_map[_group_f3_intl],
            linewidth=2.5,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=_group_f3_intl,
            zorder=3,
        )

    _ax_f3_intl_year.set_title(
        "a  By year",
        loc="left",
        fontsize=12.2,
        fontweight="bold",
        color="#111111",
        pad=8,
    )
    _ax_f3_intl_year.set_xlabel(
        "Publication year", fontsize=AF_LABEL_FONT_SIZE, labelpad=10
    )
    _ax_f3_intl_year.set_ylabel(
        "Internationally coauthored papers (%)",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
    )
    _ax_f3_intl_year.set_xticks(
        sorted(_f3_intl_year["publication_year"].dropna().unique().tolist())
    )
    _ax_f3_intl_year.grid(
        axis="y",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#D1D5DB",
        alpha=0.92,
    )
    _ax_f3_intl_year.grid(axis="x", visible=False)
    _ax_f3_intl_year.set_axisbelow(True)
    _ax_f3_intl_year.spines["top"].set_visible(False)
    _ax_f3_intl_year.spines["right"].set_visible(False)
    _ax_f3_intl_year.spines["left"].set_color("#374151")
    _ax_f3_intl_year.spines["bottom"].set_color("#374151")
    _ax_f3_intl_year.spines["left"].set_linewidth(0.8)
    _ax_f3_intl_year.spines["bottom"].set_linewidth(0.8)
    _ax_f3_intl_year.tick_params(axis="both", labelsize=10, colors="#1f2937")
    _ax_f3_intl_year.legend(
        frameon=False, loc="lower right", fontsize=AF_ANNOTATION_FONT_SIZE
    )

    _y_f3_intl = list(range(len(_f3_intl_subfield_top)))
    for _idx_f3_intl, _row_f3_intl in enumerate(
        _f3_intl_subfield_top.itertuples(index=False)
    ):
        _af_val_f3_intl = float(_row_f3_intl.international_collab_share_pct_AF)
        _nonaf_val_f3_intl = float(
            _row_f3_intl.international_collab_share_pct_non_AF
        )
        _ax_f3_intl_subfield.hlines(
            _idx_f3_intl,
            min(_af_val_f3_intl, _nonaf_val_f3_intl),
            max(_af_val_f3_intl, _nonaf_val_f3_intl),
            color=AF_GUIDE_NEUTRAL,
            linewidth=1.8,
            zorder=2,
        )
        _ax_f3_intl_subfield.scatter(
            _nonaf_val_f3_intl,
            _idx_f3_intl,
            s=56,
            color=AF_PURPLE,
            edgecolors="white",
            linewidth=0.7,
            zorder=3,
        )
        _ax_f3_intl_subfield.scatter(
            _af_val_f3_intl,
            _idx_f3_intl,
            s=56,
            color=AF_CORAL,
            edgecolors="white",
            linewidth=0.7,
            zorder=3,
        )
        _ax_f3_intl_subfield.text(
            max(_af_val_f3_intl, _nonaf_val_f3_intl) + 1.2,
            _idx_f3_intl,
            f"{_row_f3_intl.diff_pp:+.1f} pp",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#1f2937",
            zorder=4,
        )

    _ax_f3_intl_subfield.set_title(
        "b  By subfield",
        loc="left",
        fontsize=12.2,
        fontweight="bold",
        color="#111111",
        pad=8,
    )
    _ax_f3_intl_subfield.set_xlabel(
        "Internationally coauthored papers (%)",
        fontsize=AF_LABEL_FONT_SIZE,
        labelpad=10,
    )
    _ax_f3_intl_subfield.set_yticks(_y_f3_intl)
    _ax_f3_intl_subfield.set_yticklabels(_f3_intl_subfield_top["label"].tolist())
    _ax_f3_intl_subfield.grid(
        axis="x",
        linestyle=(0, (2, 3)),
        linewidth=0.72,
        color="#D1D5DB",
        alpha=0.92,
    )
    _ax_f3_intl_subfield.grid(axis="y", visible=False)
    _ax_f3_intl_subfield.set_axisbelow(True)
    _ax_f3_intl_subfield.spines["top"].set_visible(False)
    _ax_f3_intl_subfield.spines["right"].set_visible(False)
    _ax_f3_intl_subfield.spines["left"].set_visible(False)
    _ax_f3_intl_subfield.spines["bottom"].set_color("#374151")
    _ax_f3_intl_subfield.spines["bottom"].set_linewidth(0.8)
    _ax_f3_intl_subfield.tick_params(axis="x", labelsize=9.8, colors="#1f2937")
    _ax_f3_intl_subfield.tick_params(
        axis="y", labelsize=9.2, colors="#1f2937", length=0
    )

    _f3_intl_subfield_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=AF_CORAL,
            markeredgecolor="white",
            markersize=7.5,
            label="AF",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=AF_PURPLE,
            markeredgecolor="white",
            markersize=7.5,
            label="non-AF",
        ),
    ]
    _ax_f3_intl_subfield.legend(
        handles=_f3_intl_subfield_handles,
        frameon=False,
        loc="lower right",
        fontsize=AF_ANNOTATION_FONT_SIZE,
    )

    # _fig_f3_intl.text(
    #     0.06,
    #     0.985,
    #     "AlphaFold vs Non-AlphaFold International Co-Authorship",
    #     ha="left",
    #     va="top",
    #     fontsize=15.6,
    #     fontweight="bold",
    #     color="#111111",
    # )
    # _fig_f3_intl.text(
    #     0.06,
    #     0.947,
    #     "Temporal trends and the 12 subfields with the largest AF-versus-non-AF differences in international collaboration rates.",
    #     ha="left",
    #     va="top",
    #     fontsize=9.6,
    #     color="#5B5B5B",
    # )

    _fig_f3_intl.subplots_adjust(
        left=0.07, right=0.985, top=0.86, bottom=0.12, wspace=0.26
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def ext_fig_9(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    Line2D,
    figure_17_country_capacity,
    figure_17_group_boundaries,
    figure_17_group_summary,
    plt,
):
    fig_17_country_capacity, _fig17_axes = plt.subplots(
        3,
        1,
        figsize=(15.6, 9.8),
        dpi=280,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.12},
    )

    _fig17_plot = figure_17_country_capacity.copy()
    _fig17_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _fig17_metrics = [
        ("adoption_speed_z", "Adoption speed\n(z-score)"),
        (
            "production_scale_impact_score",
            "Production scale\nand impact\n(z-score composite)",
        ),
        ("network_influence_score", "Network influence\n(z-score composite)"),
    ]

    _fig17_ymins = []
    _fig17_ymaxs = []
    for _metric, _label in _fig17_metrics:
        _series = _fig17_plot[_metric].astype(float)
        _fig17_ymins.append(float(_series.min()))
        _fig17_ymaxs.append(float(_series.max()))
    _fig17_ymin = min(_fig17_ymins)
    _fig17_ymax = max(_fig17_ymaxs)
    _fig17_pad = (
        (_fig17_ymax - _fig17_ymin) * 0.08 if _fig17_ymax > _fig17_ymin else 0.5
    )

    for _ax, (_metric, _ylabel) in zip(_fig17_axes, _fig17_metrics):
        for _group in [
            "High R&D strength",
            "Middle R&D strength",
            "Low R&D strength",
        ]:
            _group_df = _fig17_plot[_fig17_plot["rd_strength_tertile"] == _group]
            if _group_df.empty:
                continue
            _ax.scatter(
                _group_df["country_position"],
                _group_df[_metric],
                s=28,
                color=_fig17_palette[_group],
                alpha=0.88,
                edgecolors="white",
                linewidth=0.45,
                zorder=3,
            )

        _ax.axhline(
            0,
            color=AF_GUIDE_NEUTRAL,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            zorder=1,
        )
        for _boundary in figure_17_group_boundaries:
            _ax.axvline(
                _boundary,
                color=AF_GUIDE_NEUTRAL,
                linewidth=0.8,
                linestyle=(0, (2, 3)),
                alpha=0.55,
                zorder=1,
            )

        _ax.set_ylabel(
            _ylabel,
            fontsize=AF_LABEL_FONT_SIZE * 0.72,
            color=AF_GUIDE_NEUTRAL,
            labelpad=10,
        )
        _ax.grid(
            axis="y",
            linestyle=(0, (2, 3)),
            linewidth=0.6,
            color=AF_GUIDE_NEUTRAL,
            alpha=0.25,
        )
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
        _ax.tick_params(axis="y", labelsize=10.5, colors=AF_GUIDE_NEUTRAL)
        _ax.tick_params(axis="x", colors=AF_GUIDE_NEUTRAL)
        _ax.set_ylim(_fig17_ymin - _fig17_pad, _fig17_ymax + _fig17_pad)

    for _row in figure_17_group_summary.itertuples(index=False):
        _fig17_axes[0].text(
            _row.mid,
            _fig17_ymax + _fig17_pad * 0.55,
            f"{_row.group_label}\n(n={_row.n_countries})",
            ha="center",
            va="bottom",
            fontsize=AF_ANNOTATION_FONT_SIZE * 0.82,
            color=AF_GUIDE_NEUTRAL,
        )

    # _fig17_axes[0].set_title(
    #     "Figure 17. Country-level AI4S capacity scores across three dimensions (n = 77 countries)",
    #     fontsize=AF_LABEL_FONT_SIZE * 0.92,
    #     color=AF_GUIDE_NEUTRAL,
    #     pad=20,
    # )
    _fig17_axes[-1].set_xlabel(
        "Countries ordered by R&D-strength group and within-group R&D-strength index",
        fontsize=AF_LABEL_FONT_SIZE * 0.76,
        color=AF_GUIDE_NEUTRAL,
        labelpad=12,
    )
    _fig17_axes[-1].set_xlim(0.5, len(_fig17_plot) + 0.5)
    _fig17_axes[-1].set_xticks([])

    _fig17_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_fig17_palette[_group],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7.5,
            label=_group,
        )
        for _group in [
            "High R&D strength",
            "Middle R&D strength",
            "Low R&D strength",
        ]
    ]
    _fig17_legend = _fig17_axes[0].legend(
        handles=_fig17_legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.82,
        title="R&D-strength group",
    )
    _fig17_legend.get_title().set_color(AF_GUIDE_NEUTRAL)
    for _text in _fig17_legend.get_texts():
        _text.set_color(AF_GUIDE_NEUTRAL)

    fig_17_country_capacity
    return


@app.cell(hide_code=True)
def ext_fig_10(
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    figure_18_capacity_density,
    plt,
    sns,
):
    fig_18_capacity_density, _fig18_axes = plt.subplots(
        1,
        3,
        figsize=(15.4, 5.2),
        dpi=280,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    _fig18_plot = figure_18_capacity_density.copy()
    _fig18_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _fig18_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    _fig18_short_labels = {
        "High R&D strength": "High R&D strength",
        "Middle R&D strength": "Middle R&D strength",
        "Low R&D strength": "Low R&D strength",
    }

    _fig18_x = _fig18_plot["production_scale_impact_score"].astype(float)
    _fig18_y = _fig18_plot["network_influence_score"].astype(float)
    _fig18_xpad = (
        (_fig18_x.max() - _fig18_x.min()) * 0.08
        if _fig18_x.max() > _fig18_x.min()
        else 0.5
    )
    _fig18_ypad = (
        (_fig18_y.max() - _fig18_y.min()) * 0.08
        if _fig18_y.max() > _fig18_y.min()
        else 0.5
    )
    _fig18_xlim = (
        float(_fig18_x.min() - _fig18_xpad),
        float(_fig18_x.max() + _fig18_xpad),
    )
    _fig18_ylim = (
        float(_fig18_y.min() - _fig18_ypad),
        float(_fig18_y.max() + _fig18_ypad),
    )

    for _ax, _group in zip(_fig18_axes, _fig18_order):
        _group_df = _fig18_plot[
            _fig18_plot["rd_strength_tertile"] == _group
        ].copy()
        _color = _fig18_palette[_group]
        _n = int(_group_df["country_code"].nunique())

        if len(_group_df) >= 3:
            sns.kdeplot(
                data=_group_df,
                x="production_scale_impact_score",
                y="network_influence_score",
                fill=True,
                levels=6,
                thresh=0.08,
                bw_adjust=0.95,
                cut=0,
                color=_color,
                alpha=0.56,
                ax=_ax,
                zorder=1,
            )
            sns.kdeplot(
                data=_group_df,
                x="production_scale_impact_score",
                y="network_influence_score",
                fill=False,
                levels=6,
                thresh=0.08,
                bw_adjust=0.95,
                cut=0,
                color=_color,
                linewidths=1.0,
                ax=_ax,
                zorder=2,
            )

        _ax.scatter(
            _group_df["production_scale_impact_score"],
            _group_df["network_influence_score"],
            s=34,
            color=_color,
            alpha=0.9,
            edgecolors="white",
            linewidth=0.55,
            zorder=3,
        )
        _ax.axhline(
            0,
            color=AF_GUIDE_NEUTRAL,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            alpha=0.65,
            zorder=0,
        )
        _ax.axvline(
            0,
            color=AF_GUIDE_NEUTRAL,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            alpha=0.65,
            zorder=0,
        )
        _ax.set_title(
            f"{_fig18_short_labels[_group]}\n(n={_n})",
            fontsize=AF_LABEL_FONT_SIZE * 0.72,
            color=AF_GUIDE_NEUTRAL,
            pad=10,
        )
        _ax.set_xlim(*_fig18_xlim)
        _ax.set_ylim(*_fig18_ylim)
        _ax.grid(
            linestyle=(0, (2, 3)),
            linewidth=0.55,
            color=AF_GUIDE_NEUTRAL,
            alpha=0.22,
        )
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color(AF_GUIDE_NEUTRAL)
        _ax.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
        _ax.tick_params(axis="both", labelsize=10.2, colors=AF_GUIDE_NEUTRAL)

    _fig18_axes[0].set_ylabel(
        "Network influence\n(z-score composite)",
        fontsize=AF_LABEL_FONT_SIZE * 0.72,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    for _ax in _fig18_axes[1:]:
        _ax.set_ylabel("")

    for _ax in _fig18_axes:
        _ax.set_xlabel(
            "Production scale and impact\n(z-score composite)",
            fontsize=AF_LABEL_FONT_SIZE * 0.7,
            color=AF_GUIDE_NEUTRAL,
            labelpad=10,
        )

    fig_18_capacity_density
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Supplementary Note Figures

    This section presents supplementary figures that support the accompanying methodological and interpretive notes. The figures extend the analytical narrative with additional diagnostics, alternative views, and supporting evidence.
    """)
    return


@app.cell(hide_code=True)
def supplementary_fig_1(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    country_collab_edges,
    country_rd_strength_lookup,
    nx,
    pd,
    plt,
    sns,
):
    _np_f3_alt = __import__("numpy")
    _stats_f3_alt = __import__("scipy.stats").stats

    _f3_alt_graph = nx.Graph()
    for _row_f3_alt in country_collab_edges.itertuples(index=False):
        _f3_alt_graph.add_edge(
            _row_f3_alt.source_country,
            _row_f3_alt.target_country,
            weight=_row_f3_alt.n_shared_works,
        )

    _f3_alt_weighted_degree = dict(_f3_alt_graph.degree(weight="weight"))
    _f3_alt_betweenness = nx.betweenness_centrality(
        _f3_alt_graph,
        weight="weight",
        normalized=True,
    )
    _f3_alt_eigenvector = nx.eigenvector_centrality_numpy(
        _f3_alt_graph,
        weight="weight",
    )
    _f3_alt_closeness = nx.closeness_centrality(_f3_alt_graph)

    _f3_alt_df = pd.DataFrame(
        {
            "country_code": list(_f3_alt_graph.nodes()),
            "weighted_degree": [
                _f3_alt_weighted_degree[_country_f3_alt]
                for _country_f3_alt in _f3_alt_graph.nodes()
            ],
            "betweenness_centrality": [
                _f3_alt_betweenness[_country_f3_alt]
                for _country_f3_alt in _f3_alt_graph.nodes()
            ],
            "eigenvector_centrality": [
                _f3_alt_eigenvector[_country_f3_alt]
                for _country_f3_alt in _f3_alt_graph.nodes()
            ],
            "closeness_centrality": [
                _f3_alt_closeness[_country_f3_alt]
                for _country_f3_alt in _f3_alt_graph.nodes()
            ],
        }
    )
    _f3_alt_df["country_code"] = (
        _f3_alt_df["country_code"].astype(str).str.strip().str.upper()
    )
    _f3_alt_df = (
        _f3_alt_df.merge(
            country_rd_strength_lookup[["country_code", "rd_strength_tertile"]],
            on="country_code",
            how="left",
        )
        .dropna(subset=["rd_strength_tertile"])
        .copy()
    )

    _f3_alt_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _f3_alt_label_map = {
        "High R&D strength": "High",
        "Middle R&D strength": "Middle",
        "Low R&D strength": "Low",
    }
    _f3_alt_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }

    _f3_alt_df = _f3_alt_df[
        _f3_alt_df["rd_strength_tertile"].isin(_f3_alt_order)
    ].copy()
    _f3_alt_df["rd_strength_tertile"] = pd.Categorical(
        _f3_alt_df["rd_strength_tertile"],
        categories=_f3_alt_order,
        ordered=True,
    )

    _f3_alt_eigen_floor = max(
        float(_f3_alt_df["eigenvector_centrality"].min()) * 0.5,
        1e-6,
    )
    _f3_alt_df["weighted_degree_log1p"] = _np_f3_alt.log10(
        1 + _f3_alt_df["weighted_degree"]
    )
    _f3_alt_df["eigenvector_centrality_log10"] = (
        _f3_alt_df["eigenvector_centrality"]
        .clip(lower=_f3_alt_eigen_floor)
        .map(lambda _v: __import__("math").log10(_v))
    )

    _f3_alt_metric_specs = [
        {
            "panel": "a",
            "column": "weighted_degree_log1p",
            "ylabel": "log10(1 + weighted degree)",
            "title": "Weighted degree",
        },
        {
            "panel": "b",
            "column": "betweenness_centrality",
            "ylabel": "Betweenness centrality",
            "title": "Betweenness centrality",
        },
        {
            "panel": "c",
            "column": "eigenvector_centrality_log10",
            "ylabel": "log10(eigenvector centrality)",
            "title": "Eigenvector centrality",
        },
        {
            "panel": "d",
            "column": "closeness_centrality",
            "ylabel": "Closeness centrality",
            "title": "Closeness centrality",
        },
    ]

    _fig_f3_alt, _axes_f3_alt = plt.subplots(2, 2, figsize=(13.6, 9.4), dpi=300)
    _fig_f3_alt.patch.set_facecolor("white")
    _axes_f3_alt = _axes_f3_alt.ravel()

    for _ax_f3_alt, _spec_f3_alt in zip(_axes_f3_alt, _f3_alt_metric_specs):
        _metric_col_f3_alt = _spec_f3_alt["column"]
        _plot_df_f3_alt = _f3_alt_df.dropna(subset=[_metric_col_f3_alt]).copy()

        sns.boxplot(
            data=_plot_df_f3_alt,
            x="rd_strength_tertile",
            y=_metric_col_f3_alt,
            order=_f3_alt_order,
            width=0.26,
            showcaps=True,
            showfliers=False,
            boxprops={
                "facecolor": "white",
                "alpha": 0.98,
                "edgecolor": AF_PURPLE,
                "linewidth": 1.0,
            },
            whiskerprops={"color": AF_PURPLE, "linewidth": 1.0},
            capprops={"color": AF_PURPLE, "linewidth": 1.0},
            medianprops={"color": AF_GUIDE_NEUTRAL, "linewidth": 1.8},
            ax=_ax_f3_alt,
        )

        for _box_patch_f3_alt, _group_name_f3_alt in zip(
            _ax_f3_alt.patches[: len(_f3_alt_order)], _f3_alt_order
        ):
            _box_patch_f3_alt.set_facecolor(
                plt.matplotlib.colors.to_rgba(
                    _f3_alt_palette[_group_name_f3_alt], 0.22
                )
            )

        sns.stripplot(
            data=_plot_df_f3_alt,
            x="rd_strength_tertile",
            y=_metric_col_f3_alt,
            order=_f3_alt_order,
            hue="rd_strength_tertile",
            palette=_f3_alt_palette,
            dodge=False,
            size=2.7,
            alpha=0.34,
            jitter=0.16,
            ax=_ax_f3_alt,
            legend=False,
        )

        _group_medians_f3_alt = (
            _plot_df_f3_alt.groupby("rd_strength_tertile", observed=False)[
                _metric_col_f3_alt
            ]
            .median()
            .reindex(_f3_alt_order)
        )
        _ymin_f3_alt = float(_plot_df_f3_alt[_metric_col_f3_alt].min())
        _ymax_f3_alt = float(_plot_df_f3_alt[_metric_col_f3_alt].max())
        _span_f3_alt = _ymax_f3_alt - _ymin_f3_alt
        _upper_pad_f3_alt = max(0.03, _span_f3_alt * 0.16)
        _lower_pad_f3_alt = max(0.02, _span_f3_alt * 0.08)

        for _idx_f3_alt, (_group_f3_alt, _median_f3_alt) in enumerate(
            _group_medians_f3_alt.items()
        ):
            if pd.notna(_median_f3_alt):
                _ax_f3_alt.text(
                    _idx_f3_alt,
                    float(_median_f3_alt) + max(0.01, _span_f3_alt * 0.035),
                    f"Median = {float(_median_f3_alt):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=AF_ANNOTATION_FONT_SIZE * 0.95,
                    color=AF_GUIDE_NEUTRAL,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.55,
                    },
                    zorder=5,
                )

        _samples_f3_alt = [
            _plot_df_f3_alt.loc[
                _plot_df_f3_alt["rd_strength_tertile"] == _group_f3_alt,
                _metric_col_f3_alt,
            ]
            .dropna()
            .to_numpy()
            for _group_f3_alt in _f3_alt_order
        ]
        _valid_samples_f3_alt = [
            _sample_f3_alt
            for _sample_f3_alt in _samples_f3_alt
            if len(_sample_f3_alt) > 0
        ]
        _kw_p_f3_alt = float(_stats_f3_alt.kruskal(*_valid_samples_f3_alt).pvalue)
        _ax_f3_alt.text(
            0.985,
            0.985,
            f"Kruskal-Wallis p = {_kw_p_f3_alt:.3g}",
            transform=_ax_f3_alt.transAxes,
            ha="right",
            va="top",
            fontsize=AF_ANNOTATION_FONT_SIZE,
            color="#1f2937",
            bbox={
                "facecolor": "white",
                "edgecolor": AF_CYAN,
                "linewidth": 0.5,
                "alpha": 0.9,
                "pad": 0.55,
            },
            zorder=6,
        )

        _ax_f3_alt.set_title(
            f"{_spec_f3_alt['panel']}  {_spec_f3_alt['title']}",
            loc="left",
            fontsize=12.1,
            fontweight="bold",
            color="#111111",
            pad=8,
        )
        _ax_f3_alt.set_xlabel(
            "R&D strength group",
            fontsize=AF_LABEL_FONT_SIZE * 0.9,
            color="#111111",
        )
        _ax_f3_alt.set_ylabel(
            _spec_f3_alt["ylabel"],
            fontsize=AF_LABEL_FONT_SIZE * 0.9,
            color="#111111",
        )
        _ax_f3_alt.set_xticks(range(len(_f3_alt_order)))
        _ax_f3_alt.set_xticklabels([_f3_alt_label_map[_g] for _g in _f3_alt_order])
        _ax_f3_alt.set_ylim(
            _ymin_f3_alt - _lower_pad_f3_alt, _ymax_f3_alt + _upper_pad_f3_alt
        )
        _ax_f3_alt.grid(
            axis="y",
            linestyle=(0, (2, 3)),
            linewidth=0.7,
            color="#D1D5DB",
            alpha=0.9,
        )
        _ax_f3_alt.grid(axis="x", visible=False)
        _ax_f3_alt.set_axisbelow(True)
        _ax_f3_alt.spines["top"].set_visible(False)
        _ax_f3_alt.spines["right"].set_visible(False)
        _ax_f3_alt.spines["left"].set_color("#374151")
        _ax_f3_alt.spines["bottom"].set_color("#374151")
        _ax_f3_alt.spines["left"].set_linewidth(0.8)
        _ax_f3_alt.spines["bottom"].set_linewidth(0.8)
        _ax_f3_alt.tick_params(
            axis="x", rotation=0, labelsize=10.0, colors="#1f2937"
        )
        _ax_f3_alt.tick_params(axis="y", labelsize=9.8, colors="#1f2937")

    # _fig_f3_alt.text(
    #     0.06,
    #     0.985,
    #     "Alternative Network Centrality Measures by R&D Group",
    #     ha="left",
    #     va="top",
    #     fontsize=16.0,
    #     fontweight="bold",
    #     color="#111111",
    # )
    # _fig_f3_alt.text(
    #     0.06,
    #     0.948,
    #     "Country-level distributions in the AlphaFold collaboration network using weighted degree, betweenness, eigenvector, and closeness centrality.",
    #     ha="left",
    #     va="top",
    #     fontsize=9.8,
    #     color="#5B5B5B",
    # )

    _fig_f3_alt.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.09, hspace=0.30, wspace=0.22
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def supplementary_fig_2(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    AF_SEQUENTIAL_CMAP,
    figure_19_decile_match_rate,
    figure_19_decile_matrix,
    figure_19_pca_results,
    figure_19_pca_summary,
    figure_19_scatter_fit,
    np,
    plt,
    sns,
):
    fig_19_pca_robustness, (_ax19a, _ax19b) = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.0),
        dpi=280,
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.22},
    )

    _fig19_plot = figure_19_pca_results.copy()
    _fig19_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }

    for _group in [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]:
        _group_df = _fig19_plot[_fig19_plot["rd_strength_tertile"] == _group]
        if _group_df.empty:
            continue
        _ax19a.scatter(
            _group_df["main_ai4s_capacity_score_z"],
            _group_df["pca_pc1_score_z"],
            s=34,
            color=_fig19_palette[_group],
            alpha=0.88,
            edgecolors="white",
            linewidth=0.55,
            label=_group,
            zorder=3,
        )

    _ax19a.plot(
        figure_19_scatter_fit["x"],
        figure_19_scatter_fit["y"],
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    _ax19a.axhline(
        0,
        color=AF_GUIDE_NEUTRAL,
        linewidth=0.9,
        linestyle=(0, (2, 3)),
        alpha=0.55,
        zorder=1,
    )
    _ax19a.axvline(
        0,
        color=AF_GUIDE_NEUTRAL,
        linewidth=0.9,
        linestyle=(0, (2, 3)),
        alpha=0.55,
        zorder=1,
    )
    _ax19a.grid(
        linestyle=(0, (2, 3)),
        linewidth=0.6,
        color=AF_GUIDE_NEUTRAL,
        alpha=0.22,
    )
    _ax19a.set_axisbelow(True)
    _ax19a.spines["top"].set_visible(False)
    _ax19a.spines["right"].set_visible(False)
    _ax19a.spines["left"].set_color(AF_GUIDE_NEUTRAL)
    _ax19a.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
    _ax19a.tick_params(axis="both", labelsize=10.2, colors=AF_GUIDE_NEUTRAL)
    _ax19a.set_xlabel(
        "Main AI4S capacity score\n(equal-weight composite, z)",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax19a.set_ylabel(
        "PCA-based alternative score\n(PC1, z)",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax19a.set_title(
        "a",
        fontsize=AF_LABEL_FONT_SIZE * 0.84,
        color=AF_GUIDE_NEUTRAL,
        pad=8,
        loc="left",
        fontweight="bold",
    )
    _ax19a.text(
        0.03,
        0.97,
        (
            f"Pearson r = {figure_19_pca_summary.loc[figure_19_pca_summary['metric'] == 'Pearson correlation: main vs PCA', 'value'].iloc[0]:.3f}\n"
            f"Spearman rho = {figure_19_pca_summary.loc[figure_19_pca_summary['metric'] == 'Spearman rank correlation: main vs PCA', 'value'].iloc[0]:.3f}\n"
            f"PC1 explained variance = {figure_19_pca_summary.loc[figure_19_pca_summary['metric'] == 'PC1 explained variance ratio', 'value'].iloc[0]:.1%}"
        ),
        transform=_ax19a.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.78,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 0.25,
        },
        zorder=4,
    )
    _ax19a.legend(
        frameon=False, loc="lower right", fontsize=AF_ANNOTATION_FONT_SIZE * 0.75
    )

    _fig19_heatmap = figure_19_decile_matrix.astype(float)
    _fig19_row_totals = _fig19_heatmap.sum(axis=1).replace(0, np.nan)
    _fig19_heatmap_share = _fig19_heatmap.div(_fig19_row_totals, axis=0).fillna(0)
    _fig19_annotations = _fig19_heatmap.copy().astype(str)
    for _row in _fig19_heatmap.index:
        for _col in _fig19_heatmap.columns:
            _count = int(_fig19_heatmap.loc[_row, _col])
            _share = float(_fig19_heatmap_share.loc[_row, _col])
            _fig19_annotations.loc[_row, _col] = f"{_count}\n{_share:.0%}"

    _fig19_vmax = max(0.5, float(_fig19_heatmap_share.to_numpy().max()))
    _fig19_cmap = AF_SEQUENTIAL_CMAP.reversed()
    sns.heatmap(
        _fig19_heatmap_share,
        ax=_ax19b,
        cmap=_fig19_cmap,
        vmin=0,
        vmax=_fig19_vmax,
        cbar=False,
        square=True,
        linewidths=0.7,
        linecolor="white",
        annot=_fig19_annotations,
        fmt="",
        annot_kws={
            "fontsize": AF_ANNOTATION_FONT_SIZE * 0.62,
            "color": AF_GUIDE_NEUTRAL,
        },
    )
    for _text, _share in zip(
        _ax19b.texts, _fig19_heatmap_share.to_numpy().ravel()
    ):
        _rgba = _fig19_cmap((_share - 0) / _fig19_vmax if _fig19_vmax else 0)
        _luminance = 0.2126 * _rgba[0] + 0.7152 * _rgba[1] + 0.0722 * _rgba[2]
        _text.set_color("#F9FAFB" if _luminance < 0.5 else AF_GUIDE_NEUTRAL)

    _ax19b.set_title(
        "b",
        fontsize=AF_LABEL_FONT_SIZE * 0.84,
        color=AF_GUIDE_NEUTRAL,
        pad=8,
        loc="left",
        fontweight="bold",
    )
    _ax19b.set_xlabel(
        "PCA-based capacity decile",
        fontsize=AF_LABEL_FONT_SIZE * 0.68,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax19b.set_ylabel(
        "Main capacity decile",
        fontsize=AF_LABEL_FONT_SIZE * 0.68,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax19b.tick_params(axis="both", labelsize=9.7, colors=AF_GUIDE_NEUTRAL)
    for _spine in _ax19b.spines.values():
        _spine.set_visible(False)
    _ax19b.text(
        0.02,
        -0.18,
        (
            f"Exact decile match rate = {figure_19_decile_match_rate:.1%}; "
            f"median abs. rank diff = {figure_19_pca_summary.loc[figure_19_pca_summary['metric'] == 'Median absolute rank difference', 'value'].iloc[0]:.1f}"
        ),
        transform=_ax19b.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.72,
        color=AF_GUIDE_NEUTRAL,
    )

    fig_19_pca_robustness
    return


@app.cell(hide_code=True)
def supplementary_fig_3_a(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    figure_20a_best,
    figure_20a_fit_grid,
    figure_20a_lowess_curve,
    figure_20a_sensitivity,
    figure_20a_threshold_base,
    plt,
):
    fig_20a_threshold_robustness, (_ax20a_left, _ax20a_right) = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.2),
        dpi=280,
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.22},
    )

    _20a_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    for _group in ["High R&D strength", "Middle R&D strength", "Low R&D strength"]:
        _group_df = figure_20a_threshold_base[
            figure_20a_threshold_base["rd_strength_tertile"] == _group
        ]
        if _group_df.empty:
            continue
        _ax20a_left.scatter(
            _group_df["network_influence_score"],
            _group_df["adoption_speed_z"],
            s=_group_df["bubble_size"] * 0.24,
            color=_20a_palette[_group],
            alpha=0.72,
            edgecolors="white",
            linewidth=0.55,
            zorder=3,
        )

    _ax20a_left.plot(
        figure_20a_fit_grid["x"],
        figure_20a_fit_grid["linear_y"],
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        label="Single linear model",
        zorder=4,
    )
    _ax20a_left.plot(
        figure_20a_fit_grid["x"],
        figure_20a_fit_grid["piecewise_y"],
        color=AF_CORAL,
        linewidth=2.0,
        label="Piecewise threshold model",
        zorder=5,
    )
    _ax20a_left.plot(
        figure_20a_lowess_curve["x"],
        figure_20a_lowess_curve["y_smooth"],
        color=AF_BLUE,
        linewidth=1.8,
        alpha=0.95,
        label="LOWESS-like smooth",
        zorder=4,
    )
    _ax20a_left.axvline(
        float(figure_20a_best["threshold_c"]),
        color=AF_CORAL,
        linewidth=1.1,
        linestyle=(0, (2, 3)),
        alpha=0.9,
        zorder=2,
    )
    _ax20a_left.text(
        float(figure_20a_best["threshold_c"]) + 0.03,
        float(figure_20a_threshold_base["adoption_speed_z"].min()) + 0.12,
        f"threshold = {float(figure_20a_best['threshold_c']):.2f}",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.72,
        color=AF_CORAL,
    )
    _ax20a_left.set_title(
        "(a) Adoption-speed outcome: linear vs threshold fit",
        fontsize=AF_LABEL_FONT_SIZE * 0.76,
        color=AF_GUIDE_NEUTRAL,
        pad=10,
    )
    _ax20a_left.set_xlabel(
        "Network influence score",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20a_left.set_ylabel(
        "Adoption speed (z-score)",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20a_left.grid(
        linestyle=(0, (2, 3)), linewidth=0.6, color=AF_GUIDE_NEUTRAL, alpha=0.22
    )
    _ax20a_left.set_axisbelow(True)
    _ax20a_left.spines["top"].set_visible(False)
    _ax20a_left.spines["right"].set_visible(False)
    _ax20a_left.spines["left"].set_color(AF_GUIDE_NEUTRAL)
    _ax20a_left.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
    _ax20a_left.tick_params(axis="both", labelsize=10.0, colors=AF_GUIDE_NEUTRAL)
    _ax20a_left.legend(
        frameon=False, loc="upper left", fontsize=AF_ANNOTATION_FONT_SIZE * 0.72
    )

    _ax20a_right.plot(
        figure_20a_sensitivity["threshold_c"],
        figure_20a_sensitivity["slope_post"],
        color=AF_CORAL,
        linewidth=2.0,
        zorder=3,
    )
    _ax20a_right.scatter(
        figure_20a_sensitivity["threshold_c"],
        figure_20a_sensitivity["slope_post"],
        s=18,
        color=AF_CORAL,
        edgecolors="white",
        linewidth=0.4,
        zorder=4,
    )
    _ax20a_right.axvline(
        float(figure_20a_best["threshold_c"]),
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        alpha=0.8,
        zorder=2,
    )
    _ax20a_right.axhline(
        float(figure_20a_best["slope_post"]),
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=1,
    )
    _ax20a_right.set_title(
        "(b) Adoption-speed post-threshold slope sensitivity",
        fontsize=AF_LABEL_FONT_SIZE * 0.76,
        color=AF_GUIDE_NEUTRAL,
        pad=10,
    )
    _ax20a_right.set_xlabel(
        "Alternative breakpoint choice",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20a_right.set_ylabel(
        "Estimated post-threshold slope",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20a_right.grid(
        linestyle=(0, (2, 3)), linewidth=0.6, color=AF_GUIDE_NEUTRAL, alpha=0.22
    )
    _ax20a_right.set_axisbelow(True)
    _ax20a_right.spines["top"].set_visible(False)
    _ax20a_right.spines["right"].set_visible(False)
    _ax20a_right.spines["left"].set_color(AF_GUIDE_NEUTRAL)
    _ax20a_right.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
    _ax20a_right.tick_params(axis="both", labelsize=10.0, colors=AF_GUIDE_NEUTRAL)
    _ax20a_right.text(
        0.02,
        0.97,
        (
            f"best threshold = {float(figure_20a_best['threshold_c']):.2f}\n"
            f"best post-threshold slope = {float(figure_20a_best['slope_post']):.2f}\n"
            f"range searched = +/-1 SD around threshold"
        ),
        transform=_ax20a_right.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.72,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 0.25,
        },
    )

    fig_20a_threshold_robustness
    return


@app.cell(hide_code=True)
def supplementary_fig_3_b(
    AF_ANNOTATION_FONT_SIZE,
    AF_BLUE,
    AF_CORAL,
    AF_CYAN,
    AF_GUIDE_NEUTRAL,
    AF_LABEL_FONT_SIZE,
    AF_PURPLE,
    figure_20b_best,
    figure_20b_fit_grid,
    figure_20b_lowess_curve,
    figure_20b_sensitivity,
    figure_20b_threshold_base,
    plt,
):
    fig_20b_threshold_robustness, (_ax20b_left, _ax20b_right) = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.2),
        dpi=280,
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.22},
    )

    _20b_palette = {
        "High R&D strength": AF_PURPLE,
        "Middle R&D strength": AF_BLUE,
        "Low R&D strength": AF_CYAN,
    }
    for _group in ["High R&D strength", "Middle R&D strength", "Low R&D strength"]:
        _group_df = figure_20b_threshold_base[
            figure_20b_threshold_base["rd_strength_tertile"] == _group
        ]
        if _group_df.empty:
            continue
        _ax20b_left.scatter(
            _group_df["network_influence_score"],
            _group_df["production_scale_impact_score"],
            s=_group_df["bubble_size"] * 0.24,
            color=_20b_palette[_group],
            alpha=0.72,
            edgecolors="white",
            linewidth=0.55,
            zorder=3,
        )

    _ax20b_left.plot(
        figure_20b_fit_grid["x"],
        figure_20b_fit_grid["linear_y"],
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        label="Single linear model",
        zorder=4,
    )
    _ax20b_left.plot(
        figure_20b_fit_grid["x"],
        figure_20b_fit_grid["piecewise_y"],
        color=AF_CORAL,
        linewidth=2.0,
        label="Piecewise threshold model",
        zorder=5,
    )
    _ax20b_left.plot(
        figure_20b_lowess_curve["x"],
        figure_20b_lowess_curve["y_smooth"],
        color=AF_BLUE,
        linewidth=1.8,
        alpha=0.95,
        label="LOWESS-like smooth",
        zorder=4,
    )
    _ax20b_left.axvline(
        float(figure_20b_best["threshold_c"]),
        color=AF_CORAL,
        linewidth=1.1,
        linestyle=(0, (2, 3)),
        alpha=0.9,
        zorder=2,
    )
    _ax20b_left.text(
        float(figure_20b_best["threshold_c"]) + 0.03,
        float(figure_20b_threshold_base["production_scale_impact_score"].min())
        + 0.12,
        f"threshold = {float(figure_20b_best['threshold_c']):.2f}",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.72,
        color=AF_CORAL,
    )
    _ax20b_left.set_title(
        "(a) Production outcome: linear vs threshold fit",
        fontsize=AF_LABEL_FONT_SIZE * 0.76,
        color=AF_GUIDE_NEUTRAL,
        pad=10,
    )
    _ax20b_left.set_xlabel(
        "Network influence score",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20b_left.set_ylabel(
        "Production scale and impact\n(z-score composite)",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20b_left.grid(
        linestyle=(0, (2, 3)), linewidth=0.6, color=AF_GUIDE_NEUTRAL, alpha=0.22
    )
    _ax20b_left.set_axisbelow(True)
    _ax20b_left.spines["top"].set_visible(False)
    _ax20b_left.spines["right"].set_visible(False)
    _ax20b_left.spines["left"].set_color(AF_GUIDE_NEUTRAL)
    _ax20b_left.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
    _ax20b_left.tick_params(axis="both", labelsize=10.0, colors=AF_GUIDE_NEUTRAL)
    _ax20b_left.legend(
        frameon=False, loc="upper left", fontsize=AF_ANNOTATION_FONT_SIZE * 0.72
    )

    _ax20b_right.plot(
        figure_20b_sensitivity["threshold_c"],
        figure_20b_sensitivity["slope_post"],
        color=AF_CORAL,
        linewidth=2.0,
        zorder=3,
    )
    _ax20b_right.scatter(
        figure_20b_sensitivity["threshold_c"],
        figure_20b_sensitivity["slope_post"],
        s=18,
        color=AF_CORAL,
        edgecolors="white",
        linewidth=0.4,
        zorder=4,
    )
    _ax20b_right.axvline(
        float(figure_20b_best["threshold_c"]),
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        alpha=0.8,
        zorder=2,
    )
    _ax20b_right.axhline(
        float(figure_20b_best["slope_post"]),
        color=AF_GUIDE_NEUTRAL,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=1,
    )
    _ax20b_right.set_title(
        "(b) Production post-threshold slope sensitivity",
        fontsize=AF_LABEL_FONT_SIZE * 0.76,
        color=AF_GUIDE_NEUTRAL,
        pad=10,
    )
    _ax20b_right.set_xlabel(
        "Alternative breakpoint choice",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20b_right.set_ylabel(
        "Estimated post-threshold slope",
        fontsize=AF_LABEL_FONT_SIZE * 0.7,
        color=AF_GUIDE_NEUTRAL,
        labelpad=10,
    )
    _ax20b_right.grid(
        linestyle=(0, (2, 3)), linewidth=0.6, color=AF_GUIDE_NEUTRAL, alpha=0.22
    )
    _ax20b_right.set_axisbelow(True)
    _ax20b_right.spines["top"].set_visible(False)
    _ax20b_right.spines["right"].set_visible(False)
    _ax20b_right.spines["left"].set_color(AF_GUIDE_NEUTRAL)
    _ax20b_right.spines["bottom"].set_color(AF_GUIDE_NEUTRAL)
    _ax20b_right.tick_params(axis="both", labelsize=10.0, colors=AF_GUIDE_NEUTRAL)
    _ax20b_right.text(
        0.02,
        0.97,
        (
            f"best threshold = {float(figure_20b_best['threshold_c']):.2f}\n"
            f"best post-threshold slope = {float(figure_20b_best['slope_post']):.2f}\n"
            f"range searched = +/-1 SD around threshold"
        ),
        transform=_ax20b_right.transAxes,
        ha="left",
        va="top",
        fontsize=AF_ANNOTATION_FONT_SIZE * 0.72,
        color=AF_GUIDE_NEUTRAL,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 0.25,
        },
    )

    fig_20b_threshold_robustness
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Extended Data Tables

    This section presents the Extended Data tables associated with the study. These tables summarize key quantitative results, country-level indicators, and operational definitions used in the analysis.
    """)
    return


@app.cell(hide_code=True)
def ext_table_1(discipline_af_base, mo, pd):
    table_1a_domain_counts = (
        discipline_af_base.assign(
            discipline=discipline_af_base["primary_field_display_name"],
            domain=discipline_af_base["primary_subfield_display_name"],
            research_type=discipline_af_base["is_alphafold_related"].map(
                {True: "AF works", False: "non-AF works"}
            ),
        )
        .dropna(subset=["discipline", "domain"])
        .groupby(["discipline", "domain", "research_type"], as_index=False)
        .agg(n_works=("work_id", "nunique"))
    )

    _table_1a_order = [
        "Biochemistry, Genetics and Molecular Biology",
        "Immunology and Microbiology",
        "Pharmacology, Toxicology and Pharmaceutics",
    ]
    _table_1a = table_1a_domain_counts.pivot_table(
        index=["discipline", "domain"],
        columns="research_type",
        values="n_works",
        fill_value=0,
    ).reset_index()
    _table_1a.columns.name = None
    for _col in ["AF works", "non-AF works"]:
        if _col not in _table_1a.columns:
            _table_1a[_col] = 0
    _table_1a["discipline"] = pd.Categorical(
        _table_1a["discipline"], categories=_table_1a_order, ordered=True
    )
    _table_1a = _table_1a.sort_values(
        ["discipline", "AF works", "domain"], ascending=[True, False, True]
    ).reset_index(drop=True)
    _table_1a["AF works"] = _table_1a["AF works"].astype(int)
    _table_1a["non-AF works"] = _table_1a["non-AF works"].astype(int)

    # Export-friendly version for pandas -> LaTeX
    export_table_1a = _table_1a.rename(
        columns={
            "discipline": "discipline",
            "domain": "domain",
            "AF works": "af_works",
            "non-AF works": "non_af_works",
        }
    ).copy()

    # Human-readable preview version
    preview_table_1a = _table_1a.rename(
        columns={"discipline": "Discipline", "domain": "Domain"}
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 1 | AlphaFold-related and non-AlphaFold publication counts by discipline and domain (2019-2025)"
            ),
            preview_table_1a,
        ]
    )
    return


@app.cell(hide_code=True)
def _(discipline_af_base, mo, pd):
    _table_1b_country_base = (
        discipline_af_base[
            [
                "work_id",
                "primary_field_display_name",
                "primary_subfield_display_name",
            ]
        ]
        .dropna(
            subset=["primary_field_display_name", "primary_subfield_display_name"]
        )
        .rename(
            columns={
                "primary_field_display_name": "discipline",
                "primary_subfield_display_name": "domain",
            }
        )
        .merge(
            pd.read_parquet("derived_tables_dedup/work_institutions.parquet")[
                ["work_id", "country_code"]
            ],
            on="work_id",
            how="inner",
        )
    )
    _table_1b_country_base["country_code"] = (
        _table_1b_country_base["country_code"].astype(str).str.strip().str.upper()
    )
    _table_1b_country_base = _table_1b_country_base[
        _table_1b_country_base["country_code"].notna()
        & (_table_1b_country_base["country_code"] != "")
        & (_table_1b_country_base["country_code"] != "NONE")
    ]
    _table_1b = _table_1b_country_base.groupby(
        ["discipline", "domain"], as_index=False
    ).agg(
        n_countries=("country_code", lambda _s: _s.nunique()),
        countries=("country_code", lambda _s: ", ".join(sorted(set(_s)))),
    )
    _table_1b_order = [
        "Biochemistry, Genetics and Molecular Biology",
        "Immunology and Microbiology",
        "Pharmacology, Toxicology and Pharmaceutics",
    ]
    _table_1b["discipline"] = pd.Categorical(
        _table_1b["discipline"], categories=_table_1b_order, ordered=True
    )
    _table_1b = _table_1b.sort_values(
        ["discipline", "n_countries", "domain"], ascending=[True, False, True]
    ).reset_index(drop=True)

    # Export-friendly version for pandas -> LaTeX
    export_table_1b = _table_1b.rename(
        columns={
            "discipline": "discipline",
            "domain": "domain",
            "n_countries": "n_countries",
            "countries": "country_list_iso2",
        }
    ).copy()

    # Human-readable preview version
    preview_table_1b = _table_1b.rename(
        columns={
            "discipline": "Discipline",
            "domain": "Domain",
            "n_countries": "N countries",
            "countries": "Country list (ISO2)",
        }
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 1B | Domain-level country coverage across the three disciplines"
            ),
            mo.md(
                "`export_table_1b` is the export-ready DataFrame for `to_latex()`."
            ),
            preview_table_1b,
        ]
    )
    return


@app.cell(hide_code=True)
def ext_table_2(country_rd_strength_lookup, mo, pd):
    _table_2 = country_rd_strength_lookup.copy()
    _table_2_cols = [
        "country_code",
        "country_name_map",
        "rnd_gdp_pct_mean_2015_2018",
        "pre_af_non_af_fractional_output_2015_2018",
        "log_pre_af_non_af_fractional_output_2015_2018",
        "z_rnd_gdp_pct_mean_2015_2018",
        "z_log_pre_af_non_af_fractional_output_2015_2018",
        "rd_strength_index",
        "rd_strength_tertile",
    ]
    _table_2 = _table_2[_table_2_cols].copy()
    _table_2.insert(0, "rd_rank", range(1, len(_table_2) + 1))
    for _col in [
        "rnd_gdp_pct_mean_2015_2018",
        "pre_af_non_af_fractional_output_2015_2018",
        "log_pre_af_non_af_fractional_output_2015_2018",
        "z_rnd_gdp_pct_mean_2015_2018",
        "z_log_pre_af_non_af_fractional_output_2015_2018",
        "rd_strength_index",
    ]:
        _table_2[_col] = _table_2[_col].astype(float).round(3)

    export_table_2 = _table_2.rename(
        columns={
            "country_code": "country_code",
            "country_name_map": "country_name",
            "rnd_gdp_pct_mean_2015_2018": "rnd_gdp_pct_mean_2015_2018",
            "pre_af_non_af_fractional_output_2015_2018": "pre_af_non_af_fractional_output_2015_2018",
            "log_pre_af_non_af_fractional_output_2015_2018": "log10_pre_af_non_af_output_plus_1",
            "z_rnd_gdp_pct_mean_2015_2018": "z_rnd_gdp_pct",
            "z_log_pre_af_non_af_fractional_output_2015_2018": "z_log_pre_af_output",
            "rd_strength_index": "rd_strength_index",
            "rd_strength_tertile": "rd_strength_tertile",
        }
    ).copy()

    preview_table_2 = export_table_2.rename(
        columns={
            "rd_rank": "R&D rank",
            "country_code": "Country code",
            "country_name": "Country name",
            "rnd_gdp_pct_mean_2015_2018": "R&D expenditure (% GDP), 2015-2018 mean",
            "pre_af_non_af_fractional_output_2015_2018": "Pre-AF non-AF fractional output, 2015-2018",
            "log10_pre_af_non_af_output_plus_1": "log10(pre-AF non-AF output + 1)",
            "z_rnd_gdp_pct": "z(R&D expenditure)",
            "z_log_pre_af_output": "z(log pre-AF output)",
            "rd_strength_index": "R&D strength index",
            "rd_strength_tertile": "R&D strength tertile",
        }
    ).copy()

    export_table_2_rule = pd.DataFrame(
        {
            "rule_component": [
                "input_1",
                "input_2",
                "transformation",
                "composite_index",
                "classification",
            ],
            "definition": [
                "Mean R&D expenditure as % of GDP, 2015-2018",
                "Pre-AF non-AF fractional publication output, 2015-2018",
                "log10(output + 1), then z-score each input",
                "Average of z(R&D expenditure) and z(log pre-AF output)",
                "Tertiles of the composite R&D strength index: Low / Middle / High",
            ],
        }
    )
    preview_table_2_rule = export_table_2_rule.rename(
        columns={"rule_component": "Rule component", "definition": "Definition"}
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 2 | Country rankings by pre-AlphaFold R&D strength (2015-2018 R&D expenditure and pre-AlphaFold life-science output)"
            ),
            mo.md("**Classification rule preview**"),
            preview_table_2_rule,
            mo.md("**Country ranking preview**"),
            preview_table_2,
        ]
    )
    return


@app.cell(hide_code=True)
def ext_table_3(mo, pd):
    export_table_3 = pd.DataFrame(
        [
            {
                "figure": "f3c2",
                "indicator": "International co-authorship rate",
                "operational_definition": "A paper is classified as internationally co-authored when author affiliations span at least two countries or regions. The figure reports the share of internationally co-authored papers within AlphaFold-related and non-AlphaFold life-science publications for each publication quarter.",
                "unit_or_scale": "% of papers",
                "primary_data_source": "collab_work_base + derived_tables_dedup/works.parquet",
                "notes": "Publication-level collaboration measure; AF and non-AF are compared over time. This corresponds to the publication-level international collaboration indicator described in M4.",
            },
            {
                "figure": "f3c_teamshare",
                "indicator": "Mean team size",
                "operational_definition": "Team size is defined as the total number of authors listed on a publication. The figure tracks the mean team size within AlphaFold-related and non-AlphaFold publications by publication year.",
                "unit_or_scale": "authors per paper",
                "primary_data_source": "collab_work_base + derived_tables_dedup/works.parquet",
                "notes": "Used to compare the organizational scale of AlphaFold-related research with the broader life-science baseline.",
            },
            {
                "figure": "f3c_teamshare",
                "indicator": "Large-team share",
                "operational_definition": "Publications are grouped by author count. The current figure highlights the share of papers in the 10-19 and 20+ author bins within AlphaFold-related and non-AlphaFold publications by year.",
                "unit_or_scale": "% of papers in a team-size bin",
                "primary_data_source": "collab_work_base + derived_tables_dedup/works.parquet",
                "notes": "Consistent with the broader team-size framework in M4; the present panel emphasizes the large-team bins most relevant to organizational complexity.",
            },
            {
                "figure": "f3d_rd",
                "indicator": "Group-level collaboration matrix",
                "operational_definition": "Countries are assigned to Low, Middle, and High R&D-strength groups. AlphaFold-related country-country co-authorship ties are aggregated into a weighted matrix by source and partner R&D group. Each cell records the row-normalized share of collaboration weight directed from a source group to a partner group.",
                "unit_or_scale": "% of collaboration weight within source group",
                "primary_data_source": "country_collab_edges + country_rd_strength_lookup",
                "notes": "Country nodes are linked by undirected weighted edges equal to the number of shared AlphaFold-related publications. This is the group-level collaboration matrix described in M4/M4b.",
            },
            {
                "figure": "f3d2_rd",
                "indicator": "Collaboration centrality",
                "operational_definition": "Country position in the AlphaFold international co-authorship network is measured using eigenvector centrality. The plotted value is log10-transformed eigenvector centrality.",
                "unit_or_scale": "log10(eigenvector centrality)",
                "primary_data_source": "figure_3b_centrality + country_rd_strength_lookup",
                "notes": "The underlying network is an undirected weighted country network in which edge weights equal the number of AlphaFold-related co-authored publications linking each pair of countries.",
            },
            {
                "figure": "f3d3_rd",
                "indicator": "First-author share",
                "operational_definition": "The share of total AlphaFold-related first-author country credit captured by each R&D-strength group, where first-author country credit is aggregated from affiliation-based first-author country assignments.",
                "unit_or_scale": "% of total AF first-author credit",
                "primary_data_source": "figure_3c2_rd_leadership_distribution",
                "notes": "This is a group-level authorship-share measure rather than a country-level ratio.",
            },
            {
                "figure": "f3d3_rd",
                "indicator": "Last-author share",
                "operational_definition": "The share of total AlphaFold-related last-author country credit captured by each R&D-strength group, where last-author country credit is aggregated from affiliation-based last-author country assignments.",
                "unit_or_scale": "% of total AF last-author credit",
                "primary_data_source": "figure_3c2_rd_leadership_distribution",
                "notes": "This is a group-level authorship-share measure rather than a country-level ratio.",
            },
            {
                "figure": "f3d4_rd",
                "indicator": "First-author ratio",
                "operational_definition": "For each country, the first-author ratio is defined as 100 x (fractional first-author AlphaFold output) / (fractional total AlphaFold output). First-author output is fractionalized across first-author countries only, whereas total AlphaFold output is fractionalized across all participating countries.",
                "unit_or_scale": "% ratio",
                "primary_data_source": "figure_3e_country_first_author_base -> figure_3c3_rd_country_metrics",
                "notes": "Because the numerator and denominator rely on different fractionalization logics, the ratio can exceed 100% and should be interpreted as a leadership ratio rather than a bounded share.",
            },
            {
                "figure": "f3d4_rd",
                "indicator": "Last-author ratio",
                "operational_definition": "For each country, the last-author ratio is defined as 100 x (fractional last-author AlphaFold output) / (fractional total AlphaFold output). Last-author output is fractionalized across last-author countries only, whereas total AlphaFold output is fractionalized across all participating countries.",
                "unit_or_scale": "% ratio",
                "primary_data_source": "figure_3e_country_last_author_base -> figure_3c3_rd_country_metrics",
                "notes": "Because the numerator and denominator rely on different fractionalization logics, the ratio can exceed 100% and should be interpreted as a leadership ratio rather than a bounded share.",
            },
        ]
    )

    preview_table_3 = export_table_3.rename(
        columns={
            "figure": "Figure",
            "indicator": "Indicator",
            "operational_definition": "Operational definition",
            "unit_or_scale": "Unit / scale",
            "primary_data_source": "Primary data source",
            "notes": "Notes",
        }
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 3 | Collaboration and authorship indicators and their operational definitions"
            ),
            preview_table_3,
        ]
    )
    return


@app.cell(hide_code=True)
def ext_table_4(country_af_nonaf_compare, country_af_output_named, mo, pd):
    _valid_iso_alpha2_codes = {
        "AD",
        "AE",
        "AF",
        "AG",
        "AI",
        "AL",
        "AM",
        "AO",
        "AQ",
        "AR",
        "AS",
        "AT",
        "AU",
        "AW",
        "AX",
        "AZ",
        "BA",
        "BB",
        "BD",
        "BE",
        "BF",
        "BG",
        "BH",
        "BI",
        "BJ",
        "BL",
        "BM",
        "BN",
        "BO",
        "BQ",
        "BR",
        "BS",
        "BT",
        "BV",
        "BW",
        "BY",
        "BZ",
        "CA",
        "CC",
        "CD",
        "CF",
        "CG",
        "CH",
        "CI",
        "CK",
        "CL",
        "CM",
        "CN",
        "CO",
        "CR",
        "CU",
        "CV",
        "CW",
        "CX",
        "CY",
        "CZ",
        "DE",
        "DJ",
        "DK",
        "DM",
        "DO",
        "DZ",
        "EC",
        "EE",
        "EG",
        "EH",
        "ER",
        "ES",
        "ET",
        "FI",
        "FJ",
        "FK",
        "FM",
        "FO",
        "FR",
        "GA",
        "GB",
        "GD",
        "GE",
        "GF",
        "GG",
        "GH",
        "GI",
        "GL",
        "GM",
        "GN",
        "GP",
        "GQ",
        "GR",
        "GS",
        "GT",
        "GU",
        "GW",
        "GY",
        "HK",
        "HM",
        "HN",
        "HR",
        "HT",
        "HU",
        "ID",
        "IE",
        "IL",
        "IM",
        "IN",
        "IO",
        "IQ",
        "IR",
        "IS",
        "IT",
        "JE",
        "JM",
        "JO",
        "JP",
        "KE",
        "KG",
        "KH",
        "KI",
        "KM",
        "KN",
        "KP",
        "KR",
        "KW",
        "KY",
        "KZ",
        "LA",
        "LB",
        "LC",
        "LI",
        "LK",
        "LR",
        "LS",
        "LT",
        "LU",
        "LV",
        "LY",
        "MA",
        "MC",
        "MD",
        "ME",
        "MF",
        "MG",
        "MH",
        "MK",
        "ML",
        "MM",
        "MN",
        "MO",
        "MP",
        "MQ",
        "MR",
        "MS",
        "MT",
        "MU",
        "MV",
        "MW",
        "MX",
        "MY",
        "MZ",
        "NA",
        "NC",
        "NE",
        "NF",
        "NG",
        "NI",
        "NL",
        "NO",
        "NP",
        "NR",
        "NU",
        "NZ",
        "OM",
        "PA",
        "PE",
        "PF",
        "PG",
        "PH",
        "PK",
        "PL",
        "PM",
        "PN",
        "PR",
        "PS",
        "PT",
        "PW",
        "PY",
        "QA",
        "RE",
        "RO",
        "RS",
        "RU",
        "RW",
        "SA",
        "SB",
        "SC",
        "SD",
        "SE",
        "SG",
        "SH",
        "SI",
        "SJ",
        "SK",
        "SL",
        "SM",
        "SN",
        "SO",
        "SR",
        "SS",
        "ST",
        "SV",
        "SX",
        "SY",
        "SZ",
        "TC",
        "TD",
        "TF",
        "TG",
        "TH",
        "TJ",
        "TK",
        "TL",
        "TM",
        "TN",
        "TO",
        "TR",
        "TT",
        "TV",
        "TW",
        "TZ",
        "UA",
        "UG",
        "UM",
        "US",
        "UY",
        "UZ",
        "VA",
        "VC",
        "VE",
        "VG",
        "VI",
        "VN",
        "VU",
        "WF",
        "WS",
        "YE",
        "YT",
        "ZA",
        "ZM",
        "ZW",
    }
    _table_4_base = country_af_output_named[
        ["country_code", "country_name", "af_fractional_count"]
    ].copy()

    if "country_af_nonaf_compare" in globals():
        _table_4_base = _table_4_base.merge(
            country_af_nonaf_compare[["country_code", "non_af_fractional_count"]],
            on="country_code",
            how="left",
        )
    else:
        _table_4_base["non_af_fractional_count"] = pd.NA

    _table_4_base["country_code"] = (
        _table_4_base["country_code"].astype(str).str.strip().str.upper()
    )
    _table_4_base = _table_4_base[
        _table_4_base["country_code"].isin(_valid_iso_alpha2_codes)
    ].copy()
    _table_4_base["country_name"] = _table_4_base["country_name"].fillna(
        _table_4_base["country_code"]
    )
    _table_4_base["af_fractional_count"] = _table_4_base[
        "af_fractional_count"
    ].astype(float)
    _table_4_base["non_af_fractional_count"] = pd.to_numeric(
        _table_4_base["non_af_fractional_count"], errors="coerce"
    )
    _table_4_base = _table_4_base[
        _table_4_base["af_fractional_count"].notna()
    ].copy()
    _table_4_base = _table_4_base.sort_values(
        ["af_fractional_count", "country_code"], ascending=[False, True]
    ).reset_index(drop=True)
    _table_4_base.insert(0, "af_rank", range(1, len(_table_4_base) + 1))

    export_table_4 = _table_4_base.rename(
        columns={
            "af_rank": "af_rank",
            "country_code": "country_code",
            "country_name": "country_name",
            "af_fractional_count": "af_fractional_output",
            "non_af_fractional_count": "non_af_fractional_output",
        }
    ).copy()

    for _col in ["af_fractional_output", "non_af_fractional_output"]:
        export_table_4[_col] = export_table_4[_col].round(3)

    preview_table_4 = export_table_4.rename(
        columns={
            "af_rank": "AF rank",
            "country_code": "Country code",
            "country_name": "Country name",
            "af_fractional_output": "AF fractional output",
            "non_af_fractional_output": "non-AF fractional output",
        }
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 4 | Country rankings by AlphaFold-related and non-AlphaFold life-science output"
            ),
            preview_table_4,
        ]
    )
    return


@app.cell(hide_code=True)
def ext_table_6(figure_19_pca_base, mo):
    _table_6_base = figure_19_pca_base[
        [
            "country_code",
            "country_name",
            "main_ai4s_capacity_score_z",
            "adoption_speed_z",
            "production_scale_impact_score",
            "network_influence_score",
            "rd_strength_tertile",
        ]
    ].copy()
    _table_6_base = _table_6_base.sort_values(
        ["main_ai4s_capacity_score_z", "country_code"],
        ascending=[False, True],
    ).reset_index(drop=True)
    _table_6_base.insert(0, "capacity_rank", range(1, len(_table_6_base) + 1))
    _table_6_base["rd_strength_tertile"] = _table_6_base[
        "rd_strength_tertile"
    ].map(
        {
            "High R&D strength": "High",
            "Middle R&D strength": "Middle",
            "Low R&D strength": "Low",
        }
    )

    export_table_6 = _table_6_base.rename(
        columns={
            "capacity_rank": "capacity_rank",
            "country_code": "country_code",
            "country_name": "country_name",
            "main_ai4s_capacity_score_z": "main_ai4s_capacity_score_z",
            "adoption_speed_z": "adoption_speed_z",
            "production_scale_impact_score": "production_scale_impact_score",
            "network_influence_score": "network_influence_score",
            "rd_strength_tertile": "rd_strength_group",
        }
    ).copy()

    for _col in [
        "main_ai4s_capacity_score_z",
        "adoption_speed_z",
        "production_scale_impact_score",
        "network_influence_score",
    ]:
        export_table_6[_col] = export_table_6[_col].astype(float).round(3)

    preview_table_6 = export_table_6.rename(
        columns={
            "capacity_rank": "Rank",
            "country_code": "Code",
            "country_name": "Country",
            "main_ai4s_capacity_score_z": "Composite AI4S capacity score (z-score)",
            "adoption_speed_z": "Adoption speed (z-score)",
            "production_scale_impact_score": "Production scale and impact (z-score)",
            "network_influence_score": "Network influence (z-score)",
            "rd_strength_group": "R&D-strength group",
        }
    ).copy()

    mo.vstack(
        [
            mo.md(
                "### Table 6 | Composite AI4S capacity scores and country rankings"
            ),
            preview_table_6,
        ]
    )
    return


@app.cell(hide_code=True)
def ext_table_7(
    figure_20a_piecewise_best,
    figure_20a_threshold_base,
    figure_20b_piecewise_best,
    figure_20b_threshold_base,
    mo,
    pd,
):
    export_table_7 = pd.DataFrame(
        [
            {
                "outcome": "Adoption speed",
                "threshold_network_influence": float(
                    figure_20a_piecewise_best["threshold_c"]
                ),
                "slope_pre": float(figure_20a_piecewise_best["slope_pre"]),
                "slope_change": float(figure_20a_piecewise_best["slope_change"]),
                "slope_post": float(figure_20a_piecewise_best["slope_post"]),
                "p_value_slope_change": float(
                    figure_20a_piecewise_best["p_value_slope_change"]
                ),
                "weighted_sse": float(figure_20a_piecewise_best["sse"]),
                "n_left": int(figure_20a_piecewise_best["n_left"]),
                "n_right": int(figure_20a_piecewise_best["n_right"]),
                "n_countries": int(len(figure_20a_threshold_base)),
            },
            {
                "outcome": "Production scale and impact",
                "threshold_network_influence": float(
                    figure_20b_piecewise_best["threshold_c"]
                ),
                "slope_pre": float(figure_20b_piecewise_best["slope_pre"]),
                "slope_change": float(figure_20b_piecewise_best["slope_change"]),
                "slope_post": float(figure_20b_piecewise_best["slope_post"]),
                "p_value_slope_change": float(
                    figure_20b_piecewise_best["p_value_slope_change"]
                ),
                "weighted_sse": float(figure_20b_piecewise_best["sse"]),
                "n_left": int(figure_20b_piecewise_best["n_left"]),
                "n_right": int(figure_20b_piecewise_best["n_right"]),
                "n_countries": int(len(figure_20b_threshold_base)),
            },
        ]
    )
    for _col in [
        "threshold_network_influence",
        "slope_pre",
        "slope_change",
        "slope_post",
        "weighted_sse",
    ]:
        export_table_7[_col] = export_table_7[_col].astype(float).round(3)
    preview_table_7 = export_table_7.copy()
    preview_table_7["p_value_slope_change"] = preview_table_7[
        "p_value_slope_change"
    ].map(
        lambda value: "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"
    )
    preview_table_7 = preview_table_7.rename(
        columns={
            "outcome": "Outcome",
            "threshold_network_influence": "Threshold (network influence score)",
            "slope_pre": "Pre-threshold slope",
            "slope_change": "Slope change",
            "slope_post": "Post-threshold slope",
            "p_value_slope_change": "P value for slope change",
            "weighted_sse": "Weighted SSE",
            "n_left": "N left of threshold",
            "n_right": "N right of threshold",
            "n_countries": "N countries",
        }
    ).copy()
    mo.vstack(
        [
            mo.md(
                "### Table 7 | Threshold (piecewise) regression results for AI4S outcomes"
            ),
            preview_table_7,
        ]
    )
    return


@app.cell(hide_code=True)
def _(Path, mo, pd, re):
    import matplotlib.figure as _mf

    _fig_export_dir = Path("outputs/figures/600dpi")
    _fig_export_dir.mkdir(parents=True, exist_ok=True)

    _seen_figure_ids = set()
    _export_rows = []
    _name_counts = {}

    for _name, _value in sorted(globals().items()):
        if not isinstance(_value, _mf.Figure):
            continue

        _figure_id = id(_value)
        if _figure_id in _seen_figure_ids:
            continue

        _base_name = re.sub(r"^_cell_[A-Za-z0-9]+_", "", _name)
        _base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", _base_name).strip("_")
        if not _base_name:
            _base_name = f"figure_{len(_seen_figure_ids) + 1:02d}"

        _name_counts[_base_name] = _name_counts.get(_base_name, 0) + 1
        _export_name = (
            _base_name
            if _name_counts[_base_name] == 1
            else f"{_base_name}_{_name_counts[_base_name]:02d}"
        )

        _export_path = _fig_export_dir / f"{_export_name}.png"
        _value.savefig(
            _export_path,
            dpi=600,
            bbox_inches="tight",
            facecolor=_value.get_facecolor(),
        )

        _seen_figure_ids.add(_figure_id)
        _export_rows.append(
            {
                "figure_name": _name,
                "export_name": _export_name,
                "path": str(_export_path),
            }
        )

    figure_export_registry = pd.DataFrame(_export_rows)
    mo.vstack(
        [
            mo.md("### Figure Export Registry (600 dpi PNG)"),
            mo.md(
                f"Generated **{len(figure_export_registry)}** figure exports in `{_fig_export_dir}`."
            ),
            figure_export_registry,
        ]
    )
    return


@app.cell(hide_code=True)
def _(duckdb):
    figure_3e_country_first_author_binary_base = duckdb.sql(
        """
        WITH af_works AS (
            SELECT work_id
            FROM read_parquet('derived_tables_dedup/works.parquet')
            WHERE work_id IS NOT NULL
              AND is_alphafold_related = TRUE
        ),
        work_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/work_institutions.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        first_author_country AS (
            SELECT DISTINCT
                work_id,
                UPPER(TRIM(country_code)) AS country_code
            FROM read_parquet('derived_tables_dedup/first_author_country_credit.parquet')
            WHERE work_id IS NOT NULL
              AND country_code IS NOT NULL
              AND TRIM(country_code) <> ''
        ),
        country_participation AS (
            SELECT
                wc.country_code,
                COUNT(DISTINCT wc.work_id) AS participated_af_papers
            FROM work_country AS wc
            INNER JOIN af_works AS aw
                ON wc.work_id = aw.work_id
            GROUP BY wc.country_code
        ),
        country_first_author_binary AS (
            SELECT
                wc.country_code,
                COUNT(DISTINCT wc.work_id) AS first_author_af_papers
            FROM work_country AS wc
            INNER JOIN af_works AS aw
                ON wc.work_id = aw.work_id
            INNER JOIN first_author_country AS fac
                ON wc.work_id = fac.work_id
               AND wc.country_code = fac.country_code
            GROUP BY wc.country_code
        )
        SELECT
            p.country_code,
            p.participated_af_papers,
            COALESCE(f.first_author_af_papers, 0) AS first_author_af_papers,
            100.0 * COALESCE(f.first_author_af_papers, 0) / NULLIF(p.participated_af_papers, 0) AS first_author_binary_share_pct
        FROM country_participation AS p
        LEFT JOIN country_first_author_binary AS f
            ON p.country_code = f.country_code
        """
    ).df()

    _ = figure_3e_country_first_author_binary_base
    return (figure_3e_country_first_author_binary_base,)


@app.cell(hide_code=True)
def _(
    country_rd_strength_lookup,
    figure_3b_centrality,
    figure_3e_country_first_author_base,
    figure_3e_country_last_author_base,
    math,
    pd,
):
    figure_3c3_rd_country_metrics = figure_3b_centrality[
        [
            "country_code",
            "eigenvector_centrality",
        ]
    ].merge(
        figure_3e_country_first_author_base[
            [
                "country_code",
                "first_author_share_pct",
                "af_output_fractional",
            ]
        ],
        on="country_code",
        how="inner",
    )
    figure_3c3_rd_country_metrics = figure_3c3_rd_country_metrics.merge(
        figure_3e_country_last_author_base[
            [
                "country_code",
                "last_author_share_pct",
            ]
        ],
        on="country_code",
        how="left",
    )
    figure_3c3_rd_country_metrics = figure_3c3_rd_country_metrics.merge(
        country_rd_strength_lookup[
            [
                "country_code",
                "rd_strength_tertile",
                "rd_strength_index",
            ]
        ],
        on="country_code",
        how="inner",
    )
    figure_3c3_rd_country_metrics["last_author_share_pct"] = (
        figure_3c3_rd_country_metrics["last_author_share_pct"].fillna(0)
    )
    figure_3c3_rd_country_metrics["log10_eigenvector_centrality"] = (
        figure_3c3_rd_country_metrics["eigenvector_centrality"]
        .clip(lower=1e-6)
        .map(lambda _v: math.log10(_v))
    )
    figure_3c3_rd_country_metrics["log_output_plus_1"] = (
        figure_3c3_rd_country_metrics["af_output_fractional"].map(
            lambda _v: math.log10(_v + 1)
        )
    )
    figure_3c3_rd_country_metrics["rd_strength_tertile"] = pd.Categorical(
        figure_3c3_rd_country_metrics["rd_strength_tertile"],
        categories=[
            "Low R&D strength",
            "Middle R&D strength",
            "High R&D strength",
        ],
        ordered=True,
    )
    figure_3c3_rd_country_metrics = figure_3c3_rd_country_metrics.dropna(
        subset=[
            "rd_strength_tertile",
            "log10_eigenvector_centrality",
            "first_author_share_pct",
            "last_author_share_pct",
            "log_output_plus_1",
        ]
    ).copy()
    figure_3c3_rd_country_metrics = figure_3c3_rd_country_metrics.sort_values(
        ["rd_strength_tertile", "rd_strength_index"],
        ascending=[True, False],
    ).reset_index(drop=True)

    figure_3c3_rd_country_metrics
    return (figure_3c3_rd_country_metrics,)


@app.cell(hide_code=True)
def _(aiac_country_rd_composite, pd):
    figure_17_country_capacity = aiac_country_rd_composite[
        [
            "country_code",
            "country_name",
            "rd_strength_tertile",
            "rd_strength_index",
            "adoption_speed_z",
            "production_scale_impact_score",
            "network_influence_score",
        ]
    ].copy()

    _figure_17_group_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    _figure_17_group_labels = {
        "High R&D strength": "High R&D strength",
        "Middle R&D strength": "Middle R&D strength",
        "Low R&D strength": "Low R&D strength",
    }

    figure_17_country_capacity["rd_strength_tertile"] = pd.Categorical(
        figure_17_country_capacity["rd_strength_tertile"],
        categories=_figure_17_group_order,
        ordered=True,
    )
    figure_17_country_capacity = figure_17_country_capacity.sort_values(
        ["rd_strength_tertile", "rd_strength_index", "network_influence_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    figure_17_country_capacity["country_position"] = range(
        1, len(figure_17_country_capacity) + 1
    )

    _figure_17_group_summary_rows = []
    _figure_17_group_boundaries = []
    _figure_17_last_position = 0
    for _group in _figure_17_group_order:
        _group_df = figure_17_country_capacity[
            figure_17_country_capacity["rd_strength_tertile"] == _group
        ].copy()
        if _group_df.empty:
            continue
        _start = int(_group_df["country_position"].min())
        _end = int(_group_df["country_position"].max())
        _mid = (_start + _end) / 2.0
        _figure_17_group_summary_rows.append(
            {
                "rd_strength_tertile": _group,
                "group_label": _figure_17_group_labels[_group],
                "start": _start,
                "end": _end,
                "mid": _mid,
                "n_countries": int(len(_group_df)),
            }
        )
        if _figure_17_last_position:
            _figure_17_group_boundaries.append(_start - 0.5)
        _figure_17_last_position = _end

    figure_17_group_summary = pd.DataFrame(_figure_17_group_summary_rows)
    figure_17_group_boundaries = _figure_17_group_boundaries

    figure_17_country_capacity
    return (
        figure_17_country_capacity,
        figure_17_group_boundaries,
        figure_17_group_summary,
    )


@app.cell(hide_code=True)
def _(aiac_country_rd_composite, pd):
    figure_18_capacity_density = aiac_country_rd_composite[
        [
            "country_code",
            "country_name",
            "rd_strength_tertile",
            "rd_strength_index",
            "production_scale_impact_score",
            "network_influence_score",
        ]
    ].copy()

    _figure_18_group_order = [
        "High R&D strength",
        "Middle R&D strength",
        "Low R&D strength",
    ]
    figure_18_capacity_density["rd_strength_tertile"] = pd.Categorical(
        figure_18_capacity_density["rd_strength_tertile"],
        categories=_figure_18_group_order,
        ordered=True,
    )
    figure_18_capacity_density = figure_18_capacity_density.sort_values(
        [
            "rd_strength_tertile",
            "rd_strength_index",
            "production_scale_impact_score",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    figure_18_group_counts = (
        figure_18_capacity_density.groupby(
            "rd_strength_tertile", as_index=False, observed=False
        )
        .agg(n_countries=("country_code", "nunique"))
        .dropna(subset=["rd_strength_tertile"])
    )

    figure_18_capacity_density
    return (figure_18_capacity_density,)


@app.cell(hide_code=True)
def _(aiac_country_rd_composite, pd):
    figure_19_pca_base = aiac_country_rd_composite[
        [
            "country_code",
            "country_name",
            "rd_strength_tertile",
            "rd_strength_index",
            "adoption_speed_z",
            "production_scale_impact_score",
            "network_influence_score",
        ]
    ].copy()


    def _figure_19_zscore(_series):
        _vals = _series.astype(float)
        _sigma = float(_vals.std(ddof=0))
        if not _sigma:
            return pd.Series(0.0, index=_series.index)
        return (_vals - float(_vals.mean())) / _sigma


    figure_19_pca_base["main_ai4s_capacity_score"] = (
        figure_19_pca_base["adoption_speed_z"]
        + figure_19_pca_base["production_scale_impact_score"]
        + figure_19_pca_base["network_influence_score"]
    ) / 3.0
    figure_19_pca_base["main_ai4s_capacity_score_z"] = _figure_19_zscore(
        figure_19_pca_base["main_ai4s_capacity_score"]
    )

    figure_19_pca_base
    return (figure_19_pca_base,)


@app.cell(hide_code=True)
def _(figure_19_pca_base, np, pd):
    def _figure_19_local_zscore(_series):
        _vals = _series.astype(float)
        _sigma = float(_vals.std(ddof=0))
        if not _sigma:
            return pd.Series(0.0, index=_series.index)
        return (_vals - float(_vals.mean())) / _sigma


    _figure_19_features = figure_19_pca_base[
        [
            "adoption_speed_z",
            "production_scale_impact_score",
            "network_influence_score",
        ]
    ].astype(float)
    _figure_19_matrix = _figure_19_features.to_numpy()
    _figure_19_centered = _figure_19_matrix - _figure_19_matrix.mean(
        axis=0, keepdims=True
    )
    _figure_19_cov = np.cov(_figure_19_centered, rowvar=False, ddof=0)
    _figure_19_evals, _figure_19_evecs = np.linalg.eigh(_figure_19_cov)
    _figure_19_order = np.argsort(_figure_19_evals)[::-1]
    _figure_19_evals = _figure_19_evals[_figure_19_order]
    _figure_19_evecs = _figure_19_evecs[:, _figure_19_order]
    _figure_19_pc1_loadings = _figure_19_evecs[:, 0]
    _figure_19_pc1_raw = _figure_19_centered @ _figure_19_pc1_loadings

    if (
        np.corrcoef(
            _figure_19_pc1_raw,
            figure_19_pca_base["main_ai4s_capacity_score_z"]
            .astype(float)
            .to_numpy(),
        )[0, 1]
        < 0
    ):
        _figure_19_pc1_raw = -_figure_19_pc1_raw
        _figure_19_pc1_loadings = -_figure_19_pc1_loadings

    figure_19_pca_results = figure_19_pca_base.copy()
    figure_19_pca_results["pca_pc1_score"] = _figure_19_pc1_raw
    figure_19_pca_results["pca_pc1_score_z"] = _figure_19_local_zscore(
        figure_19_pca_results["pca_pc1_score"]
    )
    figure_19_pca_results["main_score_rank"] = figure_19_pca_results[
        "main_ai4s_capacity_score_z"
    ].rank(ascending=False, method="first")
    figure_19_pca_results["pca_rank"] = figure_19_pca_results[
        "pca_pc1_score_z"
    ].rank(ascending=False, method="first")
    figure_19_pca_results["rank_diff_abs"] = (
        figure_19_pca_results["main_score_rank"]
        - figure_19_pca_results["pca_rank"]
    ).abs()
    figure_19_pca_results["main_decile"] = pd.qcut(
        figure_19_pca_results["main_ai4s_capacity_score_z"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    figure_19_pca_results["pca_decile"] = pd.qcut(
        figure_19_pca_results["pca_pc1_score_z"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )

    _figure_19_pearson = float(
        np.corrcoef(
            figure_19_pca_results["main_ai4s_capacity_score_z"].astype(float),
            figure_19_pca_results["pca_pc1_score_z"].astype(float),
        )[0, 1]
    )
    _figure_19_spearman = float(
        np.corrcoef(
            figure_19_pca_results["main_score_rank"].astype(float),
            figure_19_pca_results["pca_rank"].astype(float),
        )[0, 1]
    )
    _figure_19_explained_var = float(_figure_19_evals[0] / _figure_19_evals.sum())

    _figure_19_fit = np.polyfit(
        figure_19_pca_results["main_ai4s_capacity_score_z"].astype(float),
        figure_19_pca_results["pca_pc1_score_z"].astype(float),
        deg=1,
    )
    figure_19_scatter_fit = pd.DataFrame(
        {
            "x": np.linspace(
                float(figure_19_pca_results["main_ai4s_capacity_score_z"].min()),
                float(figure_19_pca_results["main_ai4s_capacity_score_z"].max()),
                200,
            )
        }
    )
    figure_19_scatter_fit["y"] = (
        _figure_19_fit[0] * figure_19_scatter_fit["x"] + _figure_19_fit[1]
    )

    figure_19_decile_comparison = figure_19_pca_results.groupby(
        ["main_decile", "pca_decile"], observed=False, as_index=False
    ).agg(n_countries=("country_code", "nunique"))
    _figure_19_decile_order = [f"D{i}" for i in range(1, 11)]
    figure_19_decile_matrix = (
        figure_19_decile_comparison.pivot(
            index="main_decile", columns="pca_decile", values="n_countries"
        )
        .reindex(index=_figure_19_decile_order, columns=_figure_19_decile_order)
        .fillna(0)
        .astype(int)
    )
    figure_19_decile_match_rate = float(
        (
            figure_19_pca_results["main_decile"]
            == figure_19_pca_results["pca_decile"]
        ).mean()
    )
    figure_19_pca_summary = pd.DataFrame(
        {
            "metric": [
                "PC1 explained variance ratio",
                "Pearson correlation: main vs PCA",
                "Spearman rank correlation: main vs PCA",
                "Exact decile match rate",
                "Median absolute rank difference",
                "Maximum absolute rank difference",
            ],
            "value": [
                _figure_19_explained_var,
                _figure_19_pearson,
                _figure_19_spearman,
                figure_19_decile_match_rate,
                float(figure_19_pca_results["rank_diff_abs"].median()),
                float(figure_19_pca_results["rank_diff_abs"].max()),
            ],
        }
    )
    figure_19_pca_loadings = pd.DataFrame(
        {
            "dimension": [
                "Adoption speed",
                "Production scale and impact",
                "Network influence",
            ],
            "pc1_loading": _figure_19_pc1_loadings,
        }
    )

    figure_19_pca_results
    return (
        figure_19_decile_match_rate,
        figure_19_decile_matrix,
        figure_19_pca_results,
        figure_19_pca_summary,
        figure_19_scatter_fit,
    )


@app.cell(hide_code=True)
def _(np, pd, stats):
    def figure20_fit_weighted_linear(x, y, w):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        w_matrix = np.diag(w)
        beta = np.linalg.pinv(design.T @ w_matrix @ design) @ (
            design.T @ w_matrix @ y
        )
        fit = design @ beta
        resid = y - fit
        sse = float((w * (resid**2)).sum())
        return {"beta": beta, "fit": fit, "resid": resid, "sse": sse}


    def figure20_fit_piecewise_at_threshold(x, y, w, c):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        term = np.maximum(x - float(c), 0)
        design = np.column_stack([np.ones_like(x), x, term])
        w_matrix = np.diag(w)
        beta = np.linalg.pinv(design.T @ w_matrix @ design) @ (
            design.T @ w_matrix @ y
        )
        fit = design @ beta
        resid = y - fit
        sse = float((w * (resid**2)).sum())
        df_resid = max(len(x) - design.shape[1], 1)
        sigma2 = float((w * (resid**2)).sum() / df_resid)
        cov = sigma2 * np.linalg.pinv(design.T @ w_matrix @ design)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        t_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        n_left = int((x <= float(c)).sum())
        n_right = int((x > float(c)).sum())
        return {
            "threshold_c": float(c),
            "beta": beta,
            "fit": fit,
            "resid": resid,
            "sse": sse,
            "p_value_slope_pre": float(p_value[1]),
            "p_value_slope_change": float(p_value[2]),
            "slope_pre": float(beta[1]),
            "slope_change": float(beta[2]),
            "slope_post": float(beta[1] + beta[2]),
            "n_left": n_left,
            "n_right": n_right,
        }


    def figure20_search_piecewise(x, y, w, candidates, min_side=8):
        rows = []
        for c in candidates:
            fit = figure20_fit_piecewise_at_threshold(x, y, w, c)
            if fit["n_left"] < int(min_side) or fit["n_right"] < int(min_side):
                continue
            rows.append(
                {
                    "threshold_c": float(fit["threshold_c"]),
                    "sse": float(fit["sse"]),
                    "slope_pre": float(fit["slope_pre"]),
                    "slope_change": float(fit["slope_change"]),
                    "slope_post": float(fit["slope_post"]),
                    "n_left": int(fit["n_left"]),
                    "n_right": int(fit["n_right"]),
                }
            )
        return pd.DataFrame(rows).sort_values("sse").reset_index(drop=True)


    def figure20_lowess_curve(x, y, w, frac=0.35, grid_n=220):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        w = w[order]
        x_grid = np.linspace(float(x.min()), float(x.max()), grid_n)
        k = max(8, int(np.ceil(frac * len(x))))
        smooth = []
        local_slope = []
        for x0 in x_grid:
            dist = np.abs(x - x0)
            bandwidth = np.partition(dist, k - 1)[k - 1]
            bandwidth = max(float(bandwidth), 1e-6)
            u = dist / bandwidth
            kernel = (1 - np.clip(u, 0, 1) ** 3) ** 3
            kernel[u >= 1] = 0
            w_local = kernel * w
            x_centered = x - x0
            design = np.column_stack([np.ones_like(x_centered), x_centered])
            beta = np.linalg.pinv(design.T @ np.diag(w_local) @ design) @ (
                design.T @ np.diag(w_local) @ y
            )
            smooth.append(float(beta[0]))
            local_slope.append(float(beta[1]))
        return pd.DataFrame(
            {"x": x_grid, "y_smooth": smooth, "local_slope": local_slope}
        )

    return (
        figure20_fit_piecewise_at_threshold,
        figure20_fit_weighted_linear,
        figure20_lowess_curve,
        figure20_search_piecewise,
    )


@app.cell(hide_code=True)
def _(
    aiac_country_rd_composite,
    figure20_fit_piecewise_at_threshold,
    figure20_fit_weighted_linear,
    figure20_lowess_curve,
    figure20_search_piecewise,
    np,
    pd,
):
    figure_20a_threshold_base = (
        aiac_country_rd_composite[
            [
                "country_code",
                "country_name",
                "rd_strength_tertile",
                "network_influence_score",
                "adoption_speed_z",
                "af_fractional_count",
                "bubble_size",
            ]
        ]
        .dropna(
            subset=[
                "network_influence_score",
                "adoption_speed_z",
                "af_fractional_count",
                "rd_strength_tertile",
            ]
        )
        .copy()
    )
    figure_20a_threshold_base["model_weight"] = (
        figure_20a_threshold_base["af_fractional_count"]
        .clip(lower=1)
        .map(lambda v: float(v) ** 0.5)
    )
    figure_20a_threshold_base = figure_20a_threshold_base.sort_values(
        "network_influence_score"
    ).reset_index(drop=True)

    _20a_x = (
        figure_20a_threshold_base["network_influence_score"]
        .astype(float)
        .to_numpy()
    )
    _20a_y = figure_20a_threshold_base["adoption_speed_z"].astype(float).to_numpy()
    _20a_w = figure_20a_threshold_base["model_weight"].astype(float).to_numpy()
    _20a_candidates = (
        pd.Series(_20a_x).quantile(np.linspace(0.20, 0.80, 25)).tolist()
    )
    figure_20a_piecewise_search = figure20_search_piecewise(
        _20a_x, _20a_y, _20a_w, _20a_candidates, min_side=8
    )
    figure_20a_best = figure_20a_piecewise_search.iloc[0].to_dict()
    figure_20a_linear = figure20_fit_weighted_linear(_20a_x, _20a_y, _20a_w)
    figure_20a_piecewise_best = figure20_fit_piecewise_at_threshold(
        _20a_x, _20a_y, _20a_w, figure_20a_best["threshold_c"]
    )
    figure_20a_lowess_curve = figure20_lowess_curve(
        _20a_x, _20a_y, _20a_w, frac=0.35, grid_n=220
    )

    figure_20a_fit_grid = pd.DataFrame(
        {"x": np.linspace(float(_20a_x.min()), float(_20a_x.max()), 240)}
    )
    figure_20a_fit_grid["linear_y"] = (
        figure_20a_linear["beta"][0]
        + figure_20a_linear["beta"][1] * figure_20a_fit_grid["x"]
    )
    figure_20a_fit_grid["piecewise_y"] = (
        figure_20a_piecewise_best["beta"][0]
        + figure_20a_piecewise_best["beta"][1] * figure_20a_fit_grid["x"]
        + figure_20a_piecewise_best["beta"][2]
        * np.maximum(
            figure_20a_fit_grid["x"] - float(figure_20a_best["threshold_c"]), 0
        )
    )

    _20a_x_sd = float(pd.Series(_20a_x).std(ddof=0))
    _20a_sens_min = max(
        float(_20a_x.min()) + 1e-6,
        float(figure_20a_best["threshold_c"]) - _20a_x_sd,
    )
    _20a_sens_max = min(
        float(_20a_x.max()) - 1e-6,
        float(figure_20a_best["threshold_c"]) + _20a_x_sd,
    )
    _20a_sens_candidates = np.linspace(_20a_sens_min, _20a_sens_max, 41)
    figure_20a_sensitivity = (
        figure20_search_piecewise(
            _20a_x, _20a_y, _20a_w, _20a_sens_candidates, min_side=8
        )
        .sort_values("threshold_c")
        .reset_index(drop=True)
    )
    figure_20a_sensitivity["delta_from_best"] = figure_20a_sensitivity[
        "threshold_c"
    ] - float(figure_20a_best["threshold_c"])
    return (
        figure_20a_best,
        figure_20a_fit_grid,
        figure_20a_lowess_curve,
        figure_20a_piecewise_best,
        figure_20a_sensitivity,
        figure_20a_threshold_base,
    )


@app.cell(hide_code=True)
def _(
    aiac_country_rd_composite,
    figure20_fit_piecewise_at_threshold,
    figure20_fit_weighted_linear,
    figure20_lowess_curve,
    figure20_search_piecewise,
    np,
    pd,
):
    figure_20b_threshold_base = (
        aiac_country_rd_composite[
            [
                "country_code",
                "country_name",
                "rd_strength_tertile",
                "network_influence_score",
                "production_scale_impact_score",
                "af_fractional_count",
                "bubble_size",
            ]
        ]
        .dropna(
            subset=[
                "network_influence_score",
                "production_scale_impact_score",
                "af_fractional_count",
                "rd_strength_tertile",
            ]
        )
        .copy()
    )
    figure_20b_threshold_base["model_weight"] = (
        figure_20b_threshold_base["af_fractional_count"]
        .clip(lower=1)
        .map(lambda v: float(v) ** 0.5)
    )
    figure_20b_threshold_base = figure_20b_threshold_base.sort_values(
        "network_influence_score"
    ).reset_index(drop=True)

    _20b_x = (
        figure_20b_threshold_base["network_influence_score"]
        .astype(float)
        .to_numpy()
    )
    _20b_y = (
        figure_20b_threshold_base["production_scale_impact_score"]
        .astype(float)
        .to_numpy()
    )
    _20b_w = figure_20b_threshold_base["model_weight"].astype(float).to_numpy()
    _20b_candidates = (
        pd.Series(_20b_x).quantile(np.linspace(0.20, 0.80, 25)).tolist()
    )
    figure_20b_piecewise_search = figure20_search_piecewise(
        _20b_x, _20b_y, _20b_w, _20b_candidates, min_side=8
    )
    figure_20b_best = figure_20b_piecewise_search.iloc[0].to_dict()
    figure_20b_linear = figure20_fit_weighted_linear(_20b_x, _20b_y, _20b_w)
    figure_20b_piecewise_best = figure20_fit_piecewise_at_threshold(
        _20b_x, _20b_y, _20b_w, figure_20b_best["threshold_c"]
    )
    figure_20b_lowess_curve = figure20_lowess_curve(
        _20b_x, _20b_y, _20b_w, frac=0.35, grid_n=220
    )

    figure_20b_fit_grid = pd.DataFrame(
        {"x": np.linspace(float(_20b_x.min()), float(_20b_x.max()), 240)}
    )
    figure_20b_fit_grid["linear_y"] = (
        figure_20b_linear["beta"][0]
        + figure_20b_linear["beta"][1] * figure_20b_fit_grid["x"]
    )
    figure_20b_fit_grid["piecewise_y"] = (
        figure_20b_piecewise_best["beta"][0]
        + figure_20b_piecewise_best["beta"][1] * figure_20b_fit_grid["x"]
        + figure_20b_piecewise_best["beta"][2]
        * np.maximum(
            figure_20b_fit_grid["x"] - float(figure_20b_best["threshold_c"]), 0
        )
    )

    _20b_x_sd = float(pd.Series(_20b_x).std(ddof=0))
    _20b_sens_min = max(
        float(_20b_x.min()) + 1e-6,
        float(figure_20b_best["threshold_c"]) - _20b_x_sd,
    )
    _20b_sens_max = min(
        float(_20b_x.max()) - 1e-6,
        float(figure_20b_best["threshold_c"]) + _20b_x_sd,
    )
    _20b_sens_candidates = np.linspace(_20b_sens_min, _20b_sens_max, 41)
    figure_20b_sensitivity = (
        figure20_search_piecewise(
            _20b_x, _20b_y, _20b_w, _20b_sens_candidates, min_side=8
        )
        .sort_values("threshold_c")
        .reset_index(drop=True)
    )
    figure_20b_sensitivity["delta_from_best"] = figure_20b_sensitivity[
        "threshold_c"
    ] - float(figure_20b_best["threshold_c"])
    return (
        figure_20b_best,
        figure_20b_fit_grid,
        figure_20b_lowess_curve,
        figure_20b_piecewise_best,
        figure_20b_sensitivity,
        figure_20b_threshold_base,
    )


if __name__ == "__main__":
    app.run()
