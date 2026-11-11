"""Gemini calls: search with Google Search grounding, or analysis of already collected headlines."""
import json, os, re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from core.config import ROOT

class NoSearchQuota(Exception):
    """The API key has no quota for Google Search grounding."""

def get_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing: create the .env file from .env.example")
    return genai.Client(api_key=key)

def _generate(client, models, prompt, config):
    """Tries the models in order: moves to the next one if a model is overloaded (503), retired (404) or out of quota (429)."""
    last = None
    for model in models:
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as e:
            if getattr(e, "code", None) not in (404, 429, 503):
                raise
            last = e          # retired, overloaded or out of quota: try the next model
    raise last

def search_news(client, models, prompt):
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2)
    try:
        resp = _generate(client, models, prompt, cfg)
    except Exception as e:
        if getattr(e, "code", None) == 429:
            raise NoSearchQuota(str(e)[:200]) from e
        raise
    sources = []
    gm = resp.candidates[0].grounding_metadata if resp.candidates else None
    for ch in (gm.grounding_chunks or []) if gm else []:
        if ch.web:
            sources.append({"title": ch.web.title, "uri": ch.web.uri})
    return _parse_json(resp.text or ""), sources

ITEM_SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "index": {"type": "INTEGER"}, "title": {"type": "STRING"}, "summary": {"type": "STRING"},
        "tickers": {"type": "ARRAY", "items": {"type": "STRING"}},
        "asset_classes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "regions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "sentiment": {"type": "NUMBER"}, "impact": {"type": "INTEGER"},
        "horizon": {"type": "STRING", "enum": ["days", "weeks", "months"]}},
        "required": ["index", "title", "summary", "tickers", "asset_classes", "regions",
                     "sentiment", "impact", "horizon"]}}

def analyze_headlines(client, models, prompt):
    """Without search the output can be constrained to a JSON schema: no fragile parsing."""
    cfg = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json",
                                      response_schema=ITEM_SCHEMA)
    return _parse_json(_generate(client, models, prompt, cfg).text or "")

def _parse_json(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
