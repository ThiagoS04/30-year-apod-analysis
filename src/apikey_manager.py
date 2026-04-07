from pathlib import Path

KEY_FILE = Path("apikeys.txt")

DEFAULT_TEMPLATE = """# Add your API keys below, one per line.
# Format:
# WEBSITE_NAME=YOUR_API_KEY_HERE

NASA=your_nasa_api_key_here"""

def ensure_key_file() -> None:
    """
    Create apikeys.txt with instructions if it does not already exist.
    """
    if not KEY_FILE.exists():
        KEY_FILE.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
        raise FileNotFoundError(
            "apikeys.txt was not found, so it was created for you.\n"
            "Open apikeys.txt, add your API key(s), then run the program again."
        )

def load_keys() -> dict[str, str]:
    """
    Read apikeys.txt and return a dictionary of keys.
    Ignores blank lines and comments starting with #.
    """
    ensure_key_file()

    keys = {}

    with KEY_FILE.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid format in apikeys.txt on line {line_number}:\n"
                    f"{raw_line.strip()}\n"
                    "Expected format: WEBSITE_NAME=YOUR_API_KEY"
                )

            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()

            if not name:
                raise ValueError(
                    f"Missing website name in apikeys.txt on line {line_number}."
                )

            if not value:
                raise ValueError(
                    f"Missing API key value in apikeys.txt on line {line_number}."
                )

            keys[name] = value

    return keys

def get_api_key(website_name: str) -> str:
    """
    Return the API key for a given website name.

    Example:
        get_api_key("NASA")
    """
    keys = load_keys()

    if website_name not in keys:
        raise KeyError(
            f"No API key found for '{website_name}' in apikeys.txt.\n"
            f"Add a line like:\n{website_name}=YOUR_API_KEY_HERE"
        )

    value = keys[website_name]

    if value == "YOUR_API_KEY_HERE":
        raise ValueError(
            f"The key for '{website_name}' is still the placeholder value.\n"
            "Replace it with your real API key in apikeys.txt."
        )

    return value