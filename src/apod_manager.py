### Astronomy Picture of the Day (APOD) Dataset Manager/Manipulator

import re
import pandas as pd
import hashlib
import json

from apod_scraper import update_apod_dataset
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime, UTC
from pathlib import Path


# Data paths for categorized APOD data set and metadata files
DATA_DIR = Path("data")
LABELED_CSV_PATH = DATA_DIR / "databases/cleaned/apod_labeled_data.csv"
LABELED_METADATA_PATH = DATA_DIR / "metadata/apod_labeled_metadata.json"
RAW_APOD_CSV_PATH = DATA_DIR / "databases/raw/apod_data.csv"

APOD_KEY_WORDS = {                  # APOD index categories, flattened; (top-level, sub-level) : [keywords]
    "Cosmos > Stars": [
        "Binary Stars",
        "Black Holes",
        "Globular Clusters",
        "Individual Stars",
        "Neutron Stars",
        "Nurseries",
        "Open Clusters",
        "Sun",
        "White Dwarfs"
    ],

    "Cosmos > Galaxies": [
        "Clusters of Galaxies",
        "Colliding Galaxies",
        "Elliptical Galaxies",
        "Local Group",
        "Milky Way",
        "Spiral Galaxies"
    ],

    "Cosmos > Nebulae": [
        "Dark Nebulae",
        "Emission Nebulae",
        "Planetary Nebulae",
        "Reflection Nebulae",
        "Supernova Remnants"
    ],

    "Cosmos > Miscellaneous": [
        "Quasars/Active Galactic Nuclei",
        "Dark Matter"
    ],

    "Solar System": [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Earth's Moon",
        "Mars",
        "Jupiter",
        "Jupiter's Moons",
        "Saturn",
        "Saturn's Moons",
        "Uranus",
        "Neptune",
        "Pluto",
        "Comets",
        "Asteroids"
    ],

    "Comets": [
        "Hyakutake",
        "Hale-Bopp",
        "Halley"
    ],

    "Space Technology": [
        "Rockets/Launch Vehicles",
        "Orbiting Observatories",
        "Space Stations",
        "Earth Observatories"
    ],

    "People": [
        "Scientists",
        "Astronauts"
    ],

    "Sky": [
        "Messier Objects",
        "Sky Views"
    ]
}



# Helper functions for hashing, metadata management, and text processing to support the main categorization function.
def _normalize_dataframe_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the DataFrame with reset index so hashes depend only
    on row content and row order, not the current index values.
    """
    return df.reset_index(drop=True).copy()


def _dataframe_sha256(df: pd.DataFrame) -> str:
    """
    Compute a SHA-256 hash of a DataFrame by converting it to a stable CSV string.
    """
    normalized_df = _normalize_dataframe_for_hash(df)
    csv_bytes = normalized_df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _file_sha256(path: Path) -> str:
    """
    Compute a SHA-256 hash of a file on disk.
    """
    hash_obj = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _load_metadata(path: Path) -> Optional[dict]:
    """
    Load metadata JSON if it exists, otherwise return None.
    """
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_metadata(path: Path, metadata: dict) -> None:
    """
    Write metadata dictionary as JSON.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def _get_latest_date_string(df: pd.DataFrame) -> Optional[str]:
    """
    Return the max date in YYYY-MM-DD string form, or None if DataFrame is empty.
    """
    if df.empty or "date" not in df.columns:
        return None

    return pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")



def _normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching and model preprocessing.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Lowercased text with repeated whitespace collapsed.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_document_text(df: pd.DataFrame) -> pd.Series:
    """
    Build the text that will be classified by combining title and explanation.

    Parameters
    ----------
    df : pd.DataFrame
        APOD DataFrame containing 'title' and 'explanation'.

    Returns
    -------
    pd.Series
        Combined text column.
    """
    title = df["title"].fillna("").astype(str)
    explanation = df["explanation"].fillna("").astype(str)
    return (title + " " + explanation).str.strip()



def _weak_label_from_keywords(
    text: str,
    keyword_map: Dict[str, List[str]]
) -> Optional[str]:
    """
    Assign one weak label based on direct keyword matches.

    Parameters
    ----------
    text : str
        Document text to inspect.
    keyword_map : Dict[str, List[str]]
        Flattened mapping from category label to keyword list.

    Returns
    -------
    Optional[str]
        The best matching category label as a string, or None if no keywords matched.
    """
    text = _normalize_text(text)

    best_label = None
    best_score = 0

    for label, keywords in keyword_map.items():
        score = 0

        for keyword in keywords:
            keyword_normalized = _normalize_text(keyword)
            if keyword_normalized and keyword_normalized in text:
                score += 1

        if score > best_score:
            best_score = score
            best_label = label

    return best_label



def categorize_apod_entries(
    df: pd.DataFrame,
    apod_keywords: Dict[str, List[str]],
    min_training_rows: int = 30,
    min_confidence: float = 0.25
) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
    """
    Categorize APOD rows using:
    1. weak labels from keyword matches
    2. TF-IDF + Multinomial Naive Bayes on title + explanation text

    Parameters
    ----------
    df : pd.DataFrame
        Uncleaned APOD DataFrame. It should contain 'title' and 'explanation'.
        If one of these columns is missing, it will be created as an empty string
        column before processing.

    apod_keywords : Dict[str, List[str]]
        Flattened APOD category dictionary where:
            key   = category label as a string
                    example: 'Cosmos > Stars' or 'People'
            value = list of keywords associated with that category

    min_training_rows : int, default=30
        Minimum number of weak-labeled rows required before training the
        TF-IDF + Naive Bayes classifier. If fewer rows are weak-labeled,
        the function returns keyword-only results and does not train a model.

    min_confidence : float, default=0.25
        Minimum predicted probability required to accept a model-generated
        label for rows that did not receive a direct keyword match.

    Returns
    -------
    categorized_df : pd.DataFrame
        Copy of the input DataFrame with the following added columns:

        - document_text :
            Combined text built from title + explanation

        - weak_label :
            Label assigned directly from keyword matching, if any

        - predicted_label :
            Label predicted by the trained classifier

        - predicted_confidence :
            Highest class probability returned by the classifier

        - final_label :
            Final chosen label:
                - weak_label if a direct keyword match exists
                - otherwise predicted_label if confidence >= min_confidence
                - otherwise None

        - label_source :
            Source of the final label:
                - 'keyword'
                - 'naive_bayes'
                - None

        - main_category :
            Top-level category extracted from final_label
            example: 'Cosmos' from 'Cosmos > Stars'

        - sub_category :
            Subcategory extracted from final_label
            example: 'Stars' from 'Cosmos > Stars'
            For labels without a subcategory, this will be None/NaN

    model : Optional[Pipeline]
        Trained sklearn Pipeline containing:
            - TfidfVectorizer
            - MultinomialNB

        Returns None if there are not enough weak-labeled rows to train.

    Notes
    -----
    This function uses weak supervision:
    initial labels are created from keyword matches, and those weak labels
    are then used to train a text classifier that can generalize to rows
    without exact keyword hits.
    """
    df = df.copy()

    if "title" not in df.columns:
        df["title"] = ""
    if "explanation" not in df.columns:
        df["explanation"] = ""

    df["document_text"] = _build_document_text(df)

    df["weak_label"] = df["document_text"].apply(
        lambda text: _weak_label_from_keywords(text, apod_keywords)
    )

    training_df = df[df["weak_label"].notna()].copy()

    if len(training_df) < min_training_rows:
        df["predicted_label"] = None
        df["predicted_confidence"] = None
        df["final_label"] = df["weak_label"]
        df["label_source"] = df["weak_label"].apply(
            lambda x: "keyword" if pd.notna(x) else None
        )

        split_columns = df["final_label"].str.split(" > ", n=1, expand=True)
        df["main_category"] = split_columns[0]
        df["sub_category"] = split_columns[1] if split_columns.shape[1] > 1 else None

        return df, None

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2
            )
        ),
        ("nb", MultinomialNB(alpha=0.5))
    ])

    X_train = training_df["document_text"]
    y_train = training_df["weak_label"].astype(str)

    model.fit(X_train, y_train)

    predicted_probabilities = model.predict_proba(df["document_text"])
    predicted_labels = model.classes_[predicted_probabilities.argmax(axis=1)]
    predicted_confidences = predicted_probabilities.max(axis=1)

    df["predicted_label"] = predicted_labels
    df["predicted_confidence"] = predicted_confidences

    def choose_final_label(row):
        if pd.notna(row["weak_label"]):
            return row["weak_label"], "keyword"
        if row["predicted_confidence"] >= min_confidence:
            return row["predicted_label"], "naive_bayes"
        return None, None

    final_pairs = df.apply(choose_final_label, axis=1)
    df["final_label"] = [pair[0] for pair in final_pairs]
    df["label_source"] = [pair[1] for pair in final_pairs]

    split_columns = df["final_label"].str.split(" > ", n=1, expand=True)
    df["main_category"] = split_columns[0]
    df["sub_category"] = split_columns[1] if split_columns.shape[1] > 1 else None

    return df, model



def update_labeled_apod_dataset(
    apod_df: pd.DataFrame,
    apod_keywords: Dict[str, List[str]],
    return_df: bool = False,
    min_training_rows: int = 30,
    min_confidence: float = 0.25
) -> pd.DataFrame | None:
    """
    Create or update the labeled APOD dataset stored under data/.

    Behavior
    --------
    1. If the labeled CSV or metadata does not exist:
       - build the full labeled dataset
       - save labeled CSV + metadata

    2. If the labeled CSV exists but does not match its metadata:
       - rebuild the full labeled dataset

    3. If the labeled CSV is intact and the source APOD dataset has only
       appended rows since the previous run:
       - re-run labeling on the current full source dataset
       - append only the newly added labeled rows to the labeled CSV
       - update metadata

    4. If the source APOD dataset appears historically changed/corrupted
       instead of just appended:
       - rebuild the full labeled dataset
    """
    apod_df = apod_df.sort_values("date").reset_index(drop=True).copy()

    # Case 1: missing outputs -> full rebuild
    if not LABELED_CSV_PATH.exists() or not LABELED_METADATA_PATH.exists():
        categorized_df, _ = categorize_apod_entries(
            apod_df,
            apod_keywords,
            min_training_rows=min_training_rows,
            min_confidence=min_confidence
        )
        _write_labeled_outputs(categorized_df, apod_df)
        return categorized_df

    metadata = _load_metadata(LABELED_METADATA_PATH)
    if metadata is None:
        categorized_df, _ = categorize_apod_entries(
            apod_df,
            apod_keywords,
            min_training_rows=min_training_rows,
            min_confidence=min_confidence
        )
        _write_labeled_outputs(categorized_df, apod_df)
        return categorized_df

    existing_labeled_df = pd.read_csv(LABELED_CSV_PATH).sort_values("date").reset_index(drop=True)

    # Case 2: labeled CSV integrity check
    current_labeled_file_hash = _file_sha256(LABELED_CSV_PATH)
    labeled_hash_matches = current_labeled_file_hash == metadata.get("categorized_hash")
    labeled_row_count_matches = len(existing_labeled_df) == metadata.get("categorized_row_count")
    labeled_latest_date_matches = _get_latest_date_string(existing_labeled_df) == metadata.get("categorized_latest_date")

    if not (labeled_hash_matches and labeled_row_count_matches and labeled_latest_date_matches):
        categorized_df, _ = categorize_apod_entries(
            apod_df,
            apod_keywords,
            min_training_rows=min_training_rows,
            min_confidence=min_confidence
        )
        _write_labeled_outputs(categorized_df, apod_df)
        return categorized_df

    # Check whether current source APOD dataset is append-only relative to previous source snapshot
    previous_source_row_count = metadata.get("source_row_count", 0)
    previous_source_hash = metadata.get("source_hash")
    previous_source_latest_date = metadata.get("source_latest_date")

    if previous_source_row_count > len(apod_df):
        # Source dataset somehow shrank -> rebuild
        categorized_df, _ = categorize_apod_entries(
            apod_df,
            apod_keywords,
            min_training_rows=min_training_rows,
            min_confidence=min_confidence
        )
        _write_labeled_outputs(categorized_df, apod_df)
        return categorized_df

    current_source_prefix = apod_df.iloc[:previous_source_row_count].copy()
    current_source_prefix_hash = _dataframe_sha256(current_source_prefix)

    source_is_append_only = (
        current_source_prefix_hash == previous_source_hash
    )

    current_source_latest_date = _get_latest_date_string(apod_df)

    # Already up to date
    if source_is_append_only and len(apod_df) == previous_source_row_count:
        return existing_labeled_df

    # Case 3: append only missing labeled rows
    if source_is_append_only and len(apod_df) > previous_source_row_count:
        full_categorized_df, _ = categorize_apod_entries(
            apod_df,
            apod_keywords,
            min_training_rows=min_training_rows,
            min_confidence=min_confidence
        )

        new_labeled_rows = full_categorized_df.iloc[previous_source_row_count:].copy()
        updated_labeled_df = pd.concat(
            [existing_labeled_df, new_labeled_rows],
            ignore_index=True
        ).sort_values("date").reset_index(drop=True)

        _write_labeled_outputs(updated_labeled_df, apod_df)
        return updated_labeled_df

    # Case 4: source changed historically -> rebuild full
    categorized_df, _ = categorize_apod_entries(
        apod_df,
        apod_keywords,
        min_training_rows=min_training_rows,
        min_confidence=min_confidence
    )
    _write_labeled_outputs(categorized_df, apod_df)
    return categorized_df if return_df else None



# Method to save metadata and CSV
def _write_labeled_outputs(categorized_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    """
    Write the categorized APOD CSV and matching metadata under the data folder.

    Metadata tracks:
    - hash of labeled CSV content
    - row count and latest date of labeled dataset
    - hash of full source APOD dataset
    - row count and latest date of source APOD dataset
    """
    LABELED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    categorized_df = categorized_df.sort_values("date").reset_index(drop=True)
    source_df = source_df.sort_values("date").reset_index(drop=True)

    categorized_df.to_csv(LABELED_CSV_PATH, index=False)

    metadata = {
        "categorized_hash": _file_sha256(LABELED_CSV_PATH),
        "categorized_row_count": len(categorized_df),
        "categorized_latest_date": _get_latest_date_string(categorized_df),
        "source_hash": _dataframe_sha256(source_df),
        "source_row_count": len(source_df),
        "source_latest_date": _get_latest_date_string(source_df),
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds")
    }

    _write_metadata(LABELED_METADATA_PATH, metadata)



## APOD Description:
#
#           Columns: ['date', 'title', 'media_type', 'url', 'hdurl', 'thumbnail_url', 'copyright', 'service_version', 'explanation']
#           Starts from latest date,     Every row is type Object,     copyright returns null for no copyright 
#           copyright column contains messy explanation data,
#           
##
def exploreDataset(df: pd.DataFrame) -> None:
    
    # 1. Basic shape
    print(f"Dataset Shape: \n{df.shape}\n")

    # 2. Column names
    print(f"Column Names:\n{df.columns.tolist()}\n")

    # 3. Data types + non-null counts + memory usage
    print(f"Data Info:\n{df.info()}\n")

    # 4. First few rows
    print(f"Head :\n{df[df.columns[:-1]].head()}\n")

    # 5. Last few rows
    print(f"Tail :\n{df[df.columns[:-1]].tail()}\n")

    # 6. Summary stats for numeric columns
    description_df = df.describe(include='all')
    print(f"Description:\n{description_df.transpose()}\n")

    # 8. Count missing values per column
    print(f"Missing Values:\n{df.isnull().sum()}\n")

    print("Rows with missing url and thumbnail_url:")
    print(apod_df.loc[apod_df['thumbnail_url'].isna() & 
                  apod_df['url'].isna(),
                  ['date', 'title', 'media_type', 'hdurl',  'copyright']])

    # Clean messy exlpanation data from copyright column then print
    apod_df['copyright'] = apod_df['copyright'].str.split('Explanation:').str[0].str.strip() 

    print("\nRows with copyright:")
    print(apod_df.loc[apod_df['copyright'].notna(),
                      ['date', 'title', 'media_type', 'url', 'hdurl', 'thumbnail_url', 'copyright']])



# Method for visualizer file to call in order to have up to date df before creating graphs
def categorizeDataset(df: pd.DataFrame, apod_keywords: Dict[str, List[str]]) -> None:
    
    update_apod_dataset()           # Always use latest version of APOD dataset when running the manager.

    # Get the APOD dataset as a pandas DataFrame
    apod_df = pd.read_csv(RAW_APOD_CSV_PATH)

    categorized_df = update_labeled_apod_dataset(apod_df, APOD_KEY_WORDS)



if __name__ == "__main__":

    update_apod_dataset()           # Always use latest version of APOD dataset when running the manager.

    # Get the APOD dataset as a pandas DataFrame
    apod_df = pd.read_csv(RAW_APOD_CSV_PATH)

    # Set pandas preferences for better display of the dataset when exploring it.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    #pd.set_option('display.width', None)

    # Remove 'explanation' column and save in separate database for easier exploration of the rest of the dataset.
    explanation_df = apod_df[['date', 'explanation']]
    apod_no_explanation_df = apod_df.drop(columns=['explanation'])

    # explores dataset
    #exploreDataset(apod_no_explanation_df)

    # Create or update the labeled APOD dataset under data/
    categorized_df = update_labeled_apod_dataset(apod_df, APOD_KEY_WORDS, return_df=True)

    print("\nLabeled APOD dataset ready.")
    print(categorized_df[["date", "title", "final_label", "main_category", "sub_category"]].tail().to_string(index=False))

