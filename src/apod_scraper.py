### Astronomy Picture of the Day Scrapper
# This script scrapes the NASA Astronomy Picture of the Day (APOD) website using the NASA API
# The data is sorted by date and stored in a CSV file under data/
# If the CSV file already exists, it will be updated with new entries without duplicates
from pathlib import Path
from datetime import date, datetime, timedelta
import csv
import json
import hashlib
import os
import tempfile

import requests

from apikey_manager import get_api_key


API_URL = "https://api.nasa.gov/planetary/apod"
FIRST_APOD_DATE = date(1995, 6, 16)
REQUEST_TIMEOUT_SECONDS = 60
CHUNK_DAYS = 90

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "databases/raw/apod_data.csv"
METADATA_FILE = DATA_DIR / "metadata/apod_metadata.json"

TEMP_CSV_PREFIX = "apod_csv_"
TEMP_JSON_PREFIX = "apod_metadata_"

FIELDNAMES = [
    "date",
    "title",
    "media_type",
    "url",
    "hdurl",
    "thumbnail_url",
    "copyright",
    "service_version",
    "explanation",
]


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_stale_temp_files() -> None:
    """
    Remove stale temp files created by this script in the data folder.
    If a file is locked, leave it alone and continue.
    """
    patterns = [
        f"{TEMP_CSV_PREFIX}*.tmp",
        f"{TEMP_JSON_PREFIX}*.tmp",
    ]

    for pattern in patterns:
        for temp_file in DATA_DIR.glob(pattern):
            try:
                temp_file.unlink()
            except OSError:
                pass


def csv_exists_and_has_data() -> bool:
    return CSV_FILE.exists() and CSV_FILE.stat().st_size > 0


def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_metadata() -> dict | None:
    if not METADATA_FILE.exists():
        return None

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(csv_hash: str, latest_date: date, row_count: int) -> None:
    metadata = {
        "csv_hash": csv_hash,
        "latest_date": latest_date.isoformat(),
        "row_count": row_count,
    }
    atomic_write_json(METADATA_FILE, metadata)


def csv_matches_metadata() -> bool:
    metadata = load_metadata()

    if metadata is None:
        return False

    if not csv_exists_and_has_data():
        return False

    stored_hash = metadata.get("csv_hash")
    if not stored_hash:
        return False

    current_hash = compute_file_hash(CSV_FILE)
    return current_hash == stored_hash


def get_session() -> requests.Session:
    return requests.Session()


def raise_if_rate_limited(response: requests.Response) -> None:
    if response.status_code == 429:
        limit = response.headers.get("X-RateLimit-Limit", "unknown")
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        raise RuntimeError(
            "NASA API rate limit reached.\n"
            f"Rate limit: {limit}\n"
            f"Remaining requests: {remaining}\n"
            "Wait for the limit to reset before running the script again."
        )


def fetch_apod_range(
    session: requests.Session,
    start_date: date,
    end_date: date,
) -> list[dict]:
    api_key = get_api_key("NASA")

    params = {
        "api_key": api_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "thumbs": True,
    }

    try:
        response = session.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"Request timed out while fetching APOD data from {start_date} to {end_date}.\n"
            f"Current timeout: {REQUEST_TIMEOUT_SECONDS} seconds.\n"
            "Try reducing CHUNK_DAYS and/or increasing REQUEST_TIMEOUT_SECONDS."
        ) from exc

    raise_if_rate_limited(response)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        return [data]

    if not isinstance(data, list):
        raise ValueError("Unexpected API response format.")

    return data


def normalize_record(item: dict) -> dict:
    return {
        "date": item.get("date", ""),
        "title": item.get("title", ""),
        "media_type": item.get("media_type", ""),
        "url": item.get("url", ""),
        "hdurl": item.get("hdurl", ""),
        "thumbnail_url": item.get("thumbnail_url", ""),
        "copyright": item.get("copyright", ""),
        "service_version": item.get("service_version", ""),
        "explanation": item.get("explanation", ""),
    }


def load_existing_records() -> list[dict]:
    if not csv_exists_and_has_data():
        return []

    records = []

    with CSV_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = row.get("date", "").strip()
            if row_date:
                records.append(row)

    return records


def append_to_csv(records: list[dict]) -> None:
    if not records:
        return

    file_exists = CSV_FILE.exists()

    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        for record in records:
            writer.writerow(record)


def atomic_write_csv(file_path: Path, records: list[dict]) -> None:
    """
    Write the complete CSV to a temp file first, then replace the old file only
    after the temp file is fully written.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=file_path.parent,
        prefix=TEMP_CSV_PREFIX,
        suffix=".tmp",
    ) as tmp_file:
        writer = csv.DictWriter(tmp_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

        temp_path = Path(tmp_file.name)

    os.replace(temp_path, file_path)


def atomic_write_json(file_path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=file_path.parent,
        prefix=TEMP_JSON_PREFIX,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        temp_path = Path(tmp_file.name)

    os.replace(temp_path, file_path)


def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def chunk_date_ranges(
    start_date: date,
    end_date: date,
    chunk_days: int = CHUNK_DAYS,
) -> list[tuple[date, date]]:
    ranges = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return ranges


def find_missing_dates(
    existing_dates: set[date],
    start_date: date,
    end_date: date,
) -> list[date]:
    return [d for d in daterange(start_date, end_date) if d not in existing_dates]


def group_consecutive_dates(dates: list[date]) -> list[tuple[date, date]]:
    if not dates:
        return []

    dates = sorted(dates)
    ranges = []

    range_start = dates[0]
    range_end = dates[0]

    for current_date in dates[1:]:
        if current_date == range_end + timedelta(days=1):
            range_end = current_date
        else:
            ranges.append((range_start, range_end))
            range_start = current_date
            range_end = current_date

    ranges.append((range_start, range_end))
    return ranges


def deduplicate_and_sort_records(records: list[dict]) -> list[dict]:
    by_date = {}

    for record in records:
        record_date = record.get("date", "").strip()
        if record_date:
            by_date[record_date] = record

    return [by_date[d] for d in sorted(by_date.keys())]


def create_full_dataset(session: requests.Session) -> None:
    today = date.today()
    ranges = chunk_date_ranges(FIRST_APOD_DATE, today)

    print(f"Creating full dataset in {len(ranges)} chunk(s)...")

    total_written = 0
    latest_written_date = None

    # Start a brand new CSV for the full build
    if CSV_FILE.exists():
        CSV_FILE.unlink()

    for start_date, end_date in ranges:
        print(f"Fetching {start_date} to {end_date}...")
        items = fetch_apod_range(session, start_date, end_date)
        chunk_records = [normalize_record(item) for item in items]
        chunk_records = deduplicate_and_sort_records(chunk_records)

        if not chunk_records:
            continue

        append_to_csv(chunk_records)
        total_written += len(chunk_records)
        latest_written_date = parse_date(chunk_records[-1]["date"])

        print(f"Wrote {len(chunk_records)} record(s).")

    if total_written == 0 or latest_written_date is None:
        raise RuntimeError("No APOD records were written during full dataset creation.")

    csv_hash = compute_file_hash(CSV_FILE)
    write_metadata(csv_hash, latest_written_date, total_written)

    print(f"Created {CSV_FILE} with {total_written} record(s).")


def integrity_path(session: requests.Session) -> None:
    """
    Full integrity repair path:
    - scan the whole CSV
    - detect missing internal dates
    - fetch missing data in chunks
    - deduplicate and sort
    - atomically rewrite CSV
    - update metadata
    """
    today = date.today()
    existing_records = load_existing_records()
    existing_dates = {parse_date(record["date"]) for record in existing_records}

    missing_dates = find_missing_dates(existing_dates, FIRST_APOD_DATE, today)

    if not missing_dates:
        final_records = deduplicate_and_sort_records(existing_records)
        atomic_write_csv(CSV_FILE, final_records)
        csv_hash = compute_file_hash(CSV_FILE)
        latest_date = parse_date(final_records[-1]["date"])
        row_count = len(final_records)
        write_metadata(csv_hash, latest_date, row_count)

        print(
            "Integrity check finished. No missing dates found.\n"
            f"CSV rewritten into sorted canonical form.\n"
            f"Total records: {row_count}."
        )
        return

    missing_ranges = group_consecutive_dates(missing_dates)
    fetched_records = []

    print(
        f"Integrity path: found {len(missing_dates)} missing date(s) "
        f"across {len(missing_ranges)} range(s)."
    )

    for range_start, range_end in missing_ranges:
        chunk_ranges = chunk_date_ranges(range_start, range_end)

        for chunk_start, chunk_end in chunk_ranges:
            print(f"Repairing {chunk_start} to {chunk_end}...")
            items = fetch_apod_range(session, chunk_start, chunk_end)
            fetched_records.extend(normalize_record(item) for item in items)

    all_records = existing_records + fetched_records
    final_records = deduplicate_and_sort_records(all_records)

    atomic_write_csv(CSV_FILE, final_records)
    csv_hash = compute_file_hash(CSV_FILE)
    latest_date = parse_date(final_records[-1]["date"])
    row_count = len(final_records)
    write_metadata(csv_hash, latest_date, row_count)

    print(
        "Integrity repair complete.\n"
        f"Fetched {len(fetched_records)} record(s).\n"
        f"Saved {row_count} total record(s) to {CSV_FILE}."
    )


def normal_path(session: requests.Session) -> None:
    """
    Fast normal path:
    - trust metadata because hash matched
    - use metadata latest_date
    - fetch only trailing missing dates
    - append incrementally
    - update metadata once at the end
    """
    metadata = load_metadata()
    today = date.today()

    if metadata is None:
        print("Metadata missing. Switching to integrity path...")
        integrity_path(session)
        return

    latest_date_str = metadata.get("latest_date")
    row_count = metadata.get("row_count")

    if not latest_date_str or row_count is None:
        print("Metadata incomplete. Switching to integrity path...")
        integrity_path(session)
        return

    latest_date = parse_date(latest_date_str)
    current_row_count = int(row_count)

    if latest_date >= today:
        print("APOD dataset is already up to date.")
        return

    start_date = latest_date + timedelta(days=1)
    ranges = chunk_date_ranges(start_date, today)

    print(f"Normal path: appending {len(ranges)} chunk(s) from {start_date} to {today}...")

    total_appended = 0
    current_latest_date = latest_date

    for chunk_start, chunk_end in ranges:
        print(f"Fetching {chunk_start} to {chunk_end}...")
        items = fetch_apod_range(session, chunk_start, chunk_end)
        new_records = [normalize_record(item) for item in items]
        new_records = deduplicate_and_sort_records(new_records)

        if not new_records:
            continue

        append_to_csv(new_records)

        appended_count = len(new_records)
        total_appended += appended_count
        current_row_count += appended_count
        current_latest_date = parse_date(new_records[-1]["date"])

        print(f"Appended {appended_count} record(s).")

    if total_appended == 0:
        print("No new records returned.")
        return

    csv_hash = compute_file_hash(CSV_FILE)
    write_metadata(csv_hash, current_latest_date, current_row_count)

    print(
        "Normal update complete.\n"
        f"Appended {total_appended} record(s).\n"
        f"CSV now contains {current_row_count} record(s)."
    )


def update_apod_dataset() -> None:
    ensure_data_dir()
    cleanup_stale_temp_files()

    with get_session() as session:
        if not csv_exists_and_has_data():
            print("No dataset found. Creating from scratch...")
            create_full_dataset(session)
            return

        if not csv_matches_metadata():
            print("CSV differs from metadata. Running integrity path...")
            integrity_path(session)
            return

        print("CSV matches metadata. Running normal path...")
        normal_path(session)


def main() -> None:
    update_apod_dataset()


if __name__ == "__main__":
    main()