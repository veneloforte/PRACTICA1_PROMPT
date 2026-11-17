from pathlib import Path
import os
import json
import re
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw")).resolve()
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed")).resolve()


REQUIRED_KEYS = [
    "name",
    "location",
    "main_characteristics",
    "type",
    "size",
    "capacity",
    "key_amenities",
    "proximity_highlights",
]


def _read_csv() -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / "listings1_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {csv_path}")
    return pd.read_csv(csv_path)


def _get_description_column(df: pd.DataFrame) -> str:
    if "description" in df.columns:
        return "description"

    normalized = {c.lower().strip(): c for c in df.columns}
    for candidate in ["description", "descriptions", "descripcion", "summary"]:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        "No se encontro una columna de descripcion. Columnas disponibles: "
        + ", ".join(df.columns.astype(str))
    )


def select_top_by_description_length(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if top_n < 10 or top_n > 20:
        raise ValueError("top_n debe estar entre 10 y 20.")

    description_col = _get_description_column(df)
    working_df = df.copy()
    working_df[description_col] = working_df[description_col].fillna("").astype(str)
    working_df["description_length"] = working_df[description_col].str.len()
    sorted_df = working_df.sort_values("description_length", ascending=False)
    return sorted_df.head(top_n)


def _build_prompt(description_text: str) -> str:
    schema_hint = {
        "name": "string",
        "location": "string",
        "main_characteristics": ["string"],
        "type": "string",
        "size": "string",
        "capacity": "string",
        "key_amenities": ["string"],
        "proximity_highlights": ["string"],
    }

    return (
        "Eres un extractor de informacion de alojamientos. "
        "Devuelve SOLO un JSON valido y estricto, sin markdown, sin comillas triples, "
        "sin texto adicional.\\n\\n"
        "Usa exactamente estas claves y no agregues otras: "
        f"{', '.join(REQUIRED_KEYS)}.\\n"
        "Si falta informacion, usa null o lista vacia segun corresponda.\\n"
        "Tipos esperados: name/location/type/size/capacity son string o null; "
        "main_characteristics/key_amenities/proximity_highlights son listas de strings.\\n\\n"
        f"Esquema de referencia: {json.dumps(schema_hint, ensure_ascii=True)}\\n\\n"
        f"Descripcion del alojamiento:\\n{description_text}"
    )


def _clean_llm_json(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)
    return text.strip()


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in REQUIRED_KEYS:
        normalized[key] = payload.get(key, None)

    list_keys = ["main_characteristics", "key_amenities", "proximity_highlights"]
    for key in list_keys:
        value = normalized[key]
        if value is None:
            normalized[key] = []
        elif not isinstance(value, list):
            normalized[key] = [str(value)]
        else:
            normalized[key] = [str(item) for item in value]

    str_keys = ["name", "location", "type", "size", "capacity"]
    for key in str_keys:
        value = normalized[key]
        if value is None:
            continue
        normalized[key] = str(value)

    return normalized


def _extract_with_groq(prompt: str) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Falta GROQ_API_KEY en variables de entorno.")

    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Responde siempre con JSON valido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content or "{}"


def _extract_with_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Falta GEMINI_API_KEY en variables de entorno.")

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return response.text or "{}"


def extract_entities_with_llm(description_text: str) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    prompt = _build_prompt(description_text)

    if provider == "groq":
        raw = _extract_with_groq(prompt)
    elif provider == "gemini":
        raw = _extract_with_gemini(prompt)
    else:
        raise ValueError("LLM_PROVIDER debe ser 'groq' o 'gemini'.")

    cleaned = _clean_llm_json(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"El modelo no devolvio JSON valido: {cleaned}") from exc

    if not isinstance(payload, dict):
        raise ValueError("La respuesta del LLM debe ser un objeto JSON.")

    return _validate_payload(payload)


def process_top_listings(top_n: int = 10) -> list[dict[str, Any]]:
    df = _read_csv()
    selected = select_top_by_description_length(df, top_n=top_n)
    description_col = _get_description_column(selected)

    results: list[dict[str, Any]] = []
    for idx, row in selected.iterrows():
        description_text = str(row.get(description_col, ""))
        extraction = extract_entities_with_llm(description_text)
        extraction["row_index"] = int(idx)
        extraction["description_length"] = int(row["description_length"])
        results.append(extraction)

    return results


def main() -> None:
    top_n = int(os.getenv("TOP_N", "10"))
    output_name = os.getenv("OUTPUT_FILE", "extracted_entities.json")
    output_path = PROCESSED_DATA_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entities = process_top_listings(top_n=top_n)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    print(f"Extraccion finalizada. Registros procesados: {len(entities)}")
    print(f"Resultado guardado en: {output_path}")


if __name__ == "__main__":
    main()
