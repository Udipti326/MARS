import json


def extract_json_block(text: str):
    """
    Extract the first balanced JSON object from text.
    Handles nested braces safely.
    """

    start = text.find("{")

    if start == -1:
        return None

    depth = 0

    for i in range(start, len(text)):

        char = text[i]

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    return None


def safe_json_parse(data):
    """
    Robust parser:
    - handles dict
    - handles valid JSON string
    - extracts JSON from messy LLM output
    """

    # already parsed
    if isinstance(data, dict):
        return data

    if not data:
        return {}

    if not isinstance(data, str):
        data = str(data)

    data = data.strip()

    # -----------------------------------
    # 1. direct parse
    # -----------------------------------

    try:
        return json.loads(data)

    except Exception:
        pass

    # -----------------------------------
    # 2. extract balanced JSON block
    # -----------------------------------

    json_block = extract_json_block(data)

    if json_block:

        try:
            return json.loads(json_block)

        except Exception:
            pass

    # -----------------------------------
    # 3. fallback
    # -----------------------------------

    print("\n⚠️ FAILED TO PARSE JSON:\n")
    print(data)

    return {
        "raw_output": data,
        "error": "invalid_json"
    }