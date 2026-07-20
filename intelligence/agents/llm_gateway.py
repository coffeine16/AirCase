"""LLM gateway — one doorway for every agent.

Provider chain: Gemini (GEMINI_API_KEY) -> Groq (GROQ_API_KEY) -> None.
Callers ALWAYS have a deterministic rule-based fallback, so a missing key or a
rate limit degrades explanation quality, never correctness. Strict-JSON only:
we ask for JSON, strip code fences, parse hard, and return None on any doubt.
"""
import json
import os
import urllib.request

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-2.5-flash:generateContent?key={key}")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def _parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        start, end = t.index("{"), t.rindex("}") + 1
        return json.loads(t[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def complete_json(prompt: str) -> tuple[dict | None, str]:
    """Returns (parsed_json_or_None, provider_used)."""
    if key := os.environ.get("GEMINI_API_KEY"):
        try:
            out = _post(GEMINI_URL.format(key=key), {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            }, {})
            text = out["candidates"][0]["content"]["parts"][0]["text"]
            if parsed := _parse_json(text):
                return parsed, "gemini"
        except Exception as e:
            print(f"[llm] gemini failed: {type(e).__name__}: {e}")
    if key := os.environ.get("GROQ_API_KEY"):
        try:
            out = _post(GROQ_URL, {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }, {"Authorization": f"Bearer {key}"})
            if parsed := _parse_json(out["choices"][0]["message"]["content"]):
                return parsed, "groq"
        except Exception as e:
            print(f"[llm] groq failed: {type(e).__name__}: {e}")
    return None, "none"
