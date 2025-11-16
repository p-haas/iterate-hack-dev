# app/tools.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from .config import settings


def strip_code_fences(text: str) -> str:
    text = text.replace("```python", "")
    text = text.replace("```py", "")
    text = text.replace("```", "")
    return text.strip()


def _get_codegen_llm(temperature=0.0) -> ChatAnthropic:
    """
    LLM dédié à la génération de code.
    On utilise la même clé / modèle que pour le chat, mais avec un prompt système
    orienté "génération de scripts Python valides".
    """
    return ChatAnthropic(
        model=settings.claude_model,
        temperature=0.0,  # code = on veut du déterministe
        api_key=settings.anthropic_api_key,
    )


# ----------------------------------------------------------------------
# TOOL 1 : Générer un script d'ANALYSE d'erreurs
# ----------------------------------------------------------------------


def generate_error_analysis_script(
    column_names: List[str],
    metadata: str,
    file_type: Literal["excel", "csv"] = "excel",
    delimiter: str = ",",
) -> str:

    llm = _get_codegen_llm(temperature=0.0)

    # We protect the column names: inject them directly,
    # and explicitly instruct Claude NOT to invent any.
    columns_str = ", ".join(repr(c) for c in column_names)

    system_prompt = (
        "You are an expert data engineer. "
        "Your task is to generate robust, fully executable Python code. "
        "You MUST strictly follow the instructions and return ONLY Python code, "
        "without backticks, without explanations, without surrounding text."
    )

    user_prompt = f"""
Dataset context
-------------------
File type: {file_type}
EXACT column names (do NOT invent new ones, do NOT modify spelling):
[{columns_str}]

Dataset metadata / description:
{metadata}

Goal
--------
Generate a complete Python script (a single file) that:

1. Can be saved, for example as: detect_errors.py
2. Runs from the command line as:
   - python detect_errors.py path/to/dataset.ext

3. Contains:
   - an import block:
       import sys
       from typing import List
       import pandas as pd

   - a global variable EXPECTED_COLUMNS = [ ... ] with EXACTLY the provided column names

   - a function load_dataset(file_path: str) -> pd.DataFrame
       - if file_type = "excel":
             df = pd.read_excel(file_path)
         otherwise if "csv":
             df = pd.read_csv(file_path, delimiter={repr(delimiter)})


   - a function check_missing_values(df) that:
       * for each column in EXPECTED_COLUMNS, counts NaN values
       * if >0, adds an error message
       * messages MUST include example row indices AND a category prefix, e.g.:
           "[MISSING_VALUES] ERROR: Column 'age' has 12 missing values. Example rows: [0, 15, 42]."
       * returns the list of error messages

   - a function check_duplicates(df) that:
       * counts duplicate rows
       * if >0, adds an error message
       * messages MUST include example row indices AND a category prefix, e.g.:
           "[DUPLICATE_ROWS] ERROR: There are 5 duplicate rows. Example rows: [10, 11, 58]."

   - a function check_basic_types(df) that:
       * detects inconsistent column types
         (e.g., numeric columns stored as object with invalid values)
       * messages MUST include offending values, row indices AND a category prefix, e.g.:
           "[TYPE_INCONSISTENCY] WARNING: Column 'price' looks numeric but contains non-numeric values: ['N/A', 'unknown'] at rows [3, 77]."

   - a function check_value_ranges(df) that:
       * applies basic heuristics:
           - age must be 0 ≤ age ≤ 120
           - negative values in quantity-like columns → WARNING
       * MUST include example values + row indices AND a category prefix:
           "[VALUE_RANGE] ERROR: Column 'age' has values outside [0, 120]. Values: [999, -5] at rows [12, 98]."
           "[NEGATIVE_QUANTITY] WARNING: Column 'quantity' has negative values. Values: [-3, -10] at rows [5, 19]."

   - a function check_allowed_categories(df) that:
       * detects rare or suspicious categories in text columns
       * MUST include example row indices AND a category prefix:
           "[RARE_CATEGORY] WARNING: Column 'status' has rare category 'XYZ' (1 occurrence). Example row: [7]."


   - a function check_id_consistency(df) that:
        * checks that rows with the same ID have consistent values across important columns (e.g., 'name', 'description', 'price', 'category', etc.) 
        * when inconsistencies are found give examples of conflicting values along with their row indices, and a category prefix, for example:
           "[ID_INCONSISTENCY] ERROR: rows with product_code 'ABC123' have inconsistent 'name' values: ['Widget A' (row 3), 'Widget A - promo' (row 47)]."
           "[ID_INCONSISTENCY] ERROR: rows with id '42' have inconsistent 'price' values: [9.99 (row 10), 12.50 (row 25)]."

    - a function check_product_whitespace(df) that:
        * detects whitespace issues in product-related text fields (e.g., product name, product description)
        * specifically identifies:
            - leading spaces (e.g., "  Aspirin")
            - trailing spaces (e.g., "Aspirin  ")
            - multiple internal spaces (e.g., "Aspirin  Extra  Strength")
        * the function must:
            - scan all string columns that appear to represent product attributes
            - detect whitespace inconsistencies
        * messages MUST include example values, row indices AND a category prefix, e.g.:
            "[WHITESPACE] WARNING: Product name contains whitespace errors: '  Aspirin' at row 12 (leading spaces)."
            "[WHITESPACE] WARNING: Product description contains multiple internal spaces: 'Aspirin  Extra  Strength' at row 47."
            "[WHITESPACE] WARNING: Product name contains trailing spaces: 'Paracetamol  ' at row 88."


    - a function check_category_drifts(df) that:
       * detects “category drift” for products whose category/department changes over time.
       * for example, the product 'ExputexCoughSyrup200ml' appearing under different
         'Dept Fullname' values at different dates:
             - “OTC” before April 1
             - “OTC:Cold&Flu” after April 1
       * the function must:
           - identify products whose category varies across rows
           - group rows by product identifier (e.g., product_code, product_name)
           - check if 'Dept Fullname' changes over time or between transactions
       * messages MUST include examples with row indices, category values AND a category prefix, e.g.:
           "[CATEGORY_DRIFT] WARNING: Category drift detected for product 'ExputexCoughSyrup200ml': categories found ['OTC' (rows [12, 18]), 'OTC:Cold&Flu' (rows [33, 41])]."
       * if a date column exists (e.g., 'Sale Date'), the function should mention timing, if relevant.


   - a function check_near_duplicate_rows(df) that:
       * detects “near-duplicate” rows where all columns are identical
         EXCEPT for 'Sale Date', which differs by ±1 second.
       * the function must:
           - compare rows after temporarily rounding / normalizing datetime columns
           - identify pairs/groups of rows that match all columns except Sale Date
           - ensure the date-time difference is ≤ 1 second
       * messages MUST include example row indices, timestamps AND a category prefix, e.g.:
           "[NEAR_DUPLICATE] WARNING: Near-duplicate rows detected: rows [22, 23] differ only by Sale Date (2024-05-01 12:00:01 vs 2024-05-01 12:00:02)."
       * this helps detect accidental duplicate transactions or ingestion timing issues.



   - a function summarize_dataset(df) that:
       * prints useful summary stats:
           - number of rows and columns
           - for numeric columns: min, max, mean
           - for categorical columns: number of distinct categories
       * example outputs:
           "[SUMMARY] INFO: Column 'age' (numeric) - min=18, max=87, mean=45.2."
           "[SUMMARY] INFO: Column 'country' (categorical) - 5 distinct values."

   - a main() function that:
       * reads sys.argv (must receive exactly 2 arguments)
       * loads the dataset
       * calls ALL validation functions
       * prints a full text report, for example:
           DATASET ERROR REPORT
           ====================
           Shape: X rows x Y columns
           - [MISSING_VALUES] ERROR: Column 'age' has 12 missing values. Example rows: [0, 15, 42].
           - [NEGATIVE_QUANTITY] WARNING: Column 'quantity' has negative values. Example rows: [5, 19].
           - [DUPLICATE_ROWS] ERROR: There are 5 duplicate rows. Example rows: [10, 11, 58].
         etc.

5. Freedom to add additional functions
---------------------------------------
The LLM is allowed (and encouraged) to:
   - add more validation functions (check_date_consistency, check_outliers…)
   - add helper utilities
   - enrich the report with additional warnings/errors
   - ALWAYS include example values + row indices whenever applicable.
   - ALWAYS prefix each error/warning/info message with a CATEGORY label in square brackets, followed by the severity (ERROR/WARNING/INFO), e.g.:
       "[OUTLIER] WARNING: Column 'price' has extreme values [99999.0] at rows [101, 205]."

   All added functions must:
   - be clearly named (check_*/helper_*)
   - be called inside main()
   - return lists of error/warning/info strings with a category prefix.

6. The script must be syntactically VALID.
   - No pseudo-code.
   - No undefined variables.
   - No improperly closed f-strings.

Output format
----------------
RETURN ONLY the complete Python file code.
NO explanations, NO ```python or any formatting markers.

IMPORTANT — ABSOLUTE RULES:
- You MUST output ONLY raw Python code.
- NO backticks.
- NO Markdown.
- NO explanations.
- NO surrounding text.
If you output anything other than pure Python code, it will break the system.
"""

    ai_msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return strip_code_fences(ai_msg.content)


# ----------------------------------------------------------------------
# TOOL 2 : Générer un script de CORRECTION des erreurs
# ----------------------------------------------------------------------


def generate_error_correction_script(
    column_names: List[str],
    error_report: str,
    history_summary: Optional[str] = None,
) -> str:
    """
    Génère un script Python COMPLET et VALIDE qui corrige le dataset.

    Basé sur :
    - les noms de colonnes EXACTS (column_names)
    - un rapport d'erreurs (error_report) produit par le script d'analyse
    - éventuellement un résumé de l'historique de la conversation (history_summary)
      (contexte supplémentaire pour expliquer les décisions).

    Le script généré doit :
    - prendre DEUX arguments en ligne de commande :
        1) chemin du fichier d'entrée (dataset brut)
        2) chemin du fichier de sortie (dataset corrigé)
    - charger le dataset avec pandas
    - s'assurer du schéma (colonnes attendues)
    - appliquer des corrections cohérentes avec error_report :
        * ex : remplir les valeurs manquantes dans certaines colonnes
        * ex : supprimer les doublons
        * ex : caster certaines colonnes en int/float/str, etc.
    - sauvegarder le dataset corrigé dans le fichier de sortie.

    ⚠️ Cet outil ne lance PAS le script.
       Il renvoie uniquement le code Python sous forme de chaîne.
    """

    llm = _get_codegen_llm()
    columns_str = ", ".join(repr(c) for c in column_names)

    system_prompt = (
        "Tu es un expert data engineer. "
        "Tu génères des scripts Python robustes, prêts à l'emploi, sans texte autour. "
        "La sortie doit être UNIQUEMENT le code Python complet."
    )

    history_part = (
        f"\nHistorique / contexte supplémentaire :\n{history_summary}\n"
        if history_summary
        else ""
    )

    user_prompt = f"""
Noms de colonnes EXACTES :
[{columns_str}]

Rapport d'erreurs (résultats du script d'analyse) :
---------------------------------------------------
{error_report}
{history_part}

Objectif
--------
Générer un script Python complet (un seul fichier) qui corrige ces erreurs.

Contraintes / spécifications
----------------------------
1. Le script devra être sauvegardé par exemple en tant que : fix_errors.py

2. Il devra s'exécuter comme ceci :
   - python fix_errors.py raw_dataset.ext clean_dataset.ext

3. Le script doit contenir :
   - import sys
   - from typing import List
   - import pandas as pd

   - EXPECTED_COLUMNS = [ ... ] avec EXACTEMENT les noms de colonnes fournis

   - une fonction load_dataset(path: str) -> pd.DataFrame
       * si l'extension est .xlsx (insensible à la casse) -> pd.read_excel(path)
       * sinon -> pd.read_csv(path)

   - une fonction ensure_columns(df) qui :
       * compare df.columns à EXPECTED_COLUMNS
       * affiche des warnings si des colonnes manquent ou sont en trop
       * ne crash pas si le schéma n'est pas parfait

   - une fonction apply_fixes(df) qui :
       * se base sur le rapport d'erreurs fourni ci-dessus
       * applique des corrections simples et explicites, par exemple :
           - pour les colonnes mentionnées comme ayant des valeurs manquantes :
                * si la colonne semble numérique -> fillna(0)
                * sinon -> fillna('')
           - suppression des lignes dupliquées si le rapport mentionne des doublons
           - cast explicite de certaines colonnes si le rapport mentionne des problèmes de type
         (les règles exactes peuvent être simples, mais doivent être cohérentes avec le rapport)

       * logge ce qu'il fait avec des print(), par ex :
           print("Filled 23 missing values in column 'age' with 0")

   - une fonction main() qui :
       * lit sys.argv
       * attend 3 arguments (script + input_path + output_path)
       * charge le dataset brut
       * appelle ensure_columns(df)
       * appelle apply_fixes(df)
       * sauvegarde le df corrigé dans output_path :
           - to_excel si .xlsx
           - to_csv sinon
       * affiche un message de succès avec le chemin du fichier de sortie

4. Le script doit être syntaxiquement VALIDE :
   - pas de pseudo-code
   - pas de f-string cassée
   - toutes les variables utilisées doivent être définies

Format de sortie
----------------
RENVOIE UNIQUEMENT le code Python du fichier complet.
AUCUN texte explicatif, AUCUN ```python``` ou autre balise.
"""

    ai_msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return ai_msg.content


# Liste pratique de tools à enregistrer dans ton agent LangChain
def format_error_report_to_json(
    dataset_id: str,
    raw_report: str,
) -> Dict[str, Any]:
    """
    Clean raw script output and convert it into structured JSON matching IssueResponse.
    """
    llm = _get_codegen_llm(temperature=0.1)
    system_prompt = """You are a data quality analyst. Convert noisy text analysis reports into STRICT JSON.
The JSON MUST match this schema:
{
  "dataset_id": "string",
  "summary": "string",
  "completedAt": "ISO timestamp",
  "issues": [
    {
      "id": "string",
      "type": "missing_values|duplicates|outliers|whitespace|supplier_variations|category_drift|near_duplicates|other",
      "severity": "low|medium|high",
      "description": "Detailed description referencing concrete values",
      "affectedColumns": ["col1", "..."],
      "suggestedAction": "string",
      "category": "quick_fixes|smart_fixes",
      "affectedRows": 0,
      "temporalPattern": "optional string",
      "investigation": {
        "code": "Python code snippet or detector description",
        "success": true,
        "output": {
          "sample_rows": [1,2,3]
        }
      }
    }
  ]
}
Rules:
- Always include dataset_id from the user input.
- completedAt must be an ISO8601 timestamp (use current UTC time if missing).
- If the raw text lacks a field, make a best-effort guess but keep data faithful.
- NEVER include markdown, prose, or code fences. Return pure JSON only."""

    user_prompt = f"""DATASET_ID: {dataset_id}

RAW ANALYSIS LOG:
-----------------
{raw_report}

Return ONLY the JSON described above."""

    ai_msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    cleaned = strip_code_fences(ai_msg.content)
    return json.loads(cleaned)


TOOLS = [generate_error_analysis_script, generate_error_correction_script]
