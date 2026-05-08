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
from urllib.parse import urljoin


# Data paths for categorized APOD data set and metadata files
DATA_DIR = Path("data")
LABELED_CSV_PATH = DATA_DIR / "databases/cleaned/apod_labeled_data.csv"
LABELED_METADATA_PATH = DATA_DIR / "metadata/apod_labeled_metadata.json"
RAW_APOD_CSV_PATH = DATA_DIR / "databases/raw/apod_data.csv"
APOD_INDEX_URL = "https://apod.nasa.gov/apod/lib/aptree.html"
APOD_BASE_URL = "https://apod.nasa.gov/apod/"
# APOD index categories, flattened; (top-level, sub-level) : [keywords]
APOD_KEY_WORDS = {
    "Cosmos > Stars": [
        "binary star",
        "binary stars",
        "stellar companion",
        "star cluster",
        "globular cluster",
        "open cluster",
        "young star cluster",
        "stellar nursery",
        "star forming region",
        "protostar",
        "newborn stars",
        "massive stars",
        "red giant",
        "blue giant",
        "white dwarf",
        "neutron star",
        "pulsar",
        "supernova progenitor",
        "stellar wind",
        "solar-type star",
        "variable star",
        "cepheid",
        "brown dwarf",
        "black hole binary",
        "accretion disk around a black hole",
        "event horizon",
        "x-ray binary"
    ],

    "Cosmos > Galaxies": [
        "spiral galaxy",
        "barred spiral galaxy",
        "elliptical galaxy",
        "irregular galaxy",
        "dwarf galaxy",
        "nearby galaxy",
        "galaxy cluster",
        "cluster of galaxies",
        "colliding galaxies",
        "interacting galaxies",
        "merging galaxies",
        "galactic halo",
        "galactic center",
        "milky way galaxy",
        "andromeda galaxy",
        "large magellanic cloud",
        "small magellanic cloud",
        "local group",
        "galaxy group",
        "active galaxy",
        "starburst galaxy",
        "edge-on galaxy",
        "face-on galaxy",
        "tidal tail",
        "galactic disk"
    ],

    "Cosmos > Nebulae": [
        "emission nebula",
        "reflection nebula",
        "dark nebula",
        "planetary nebula",
        "supernova remnant",
        "molecular cloud",
        "interstellar cloud",
        "dust cloud",
        "cosmic dust",
        "ionized gas",
        "hydrogen gas",
        "h ii region",
        "star-forming nebula",
        "nebular gas",
        "gas and dust",
        "horsehead nebula",
        "eagle nebula",
        "orion nebula",
        "carina nebula",
        "lagoon nebula",
        "trifid nebula",
        "crab nebula",
        "veil nebula",
        "rosette nebula"
    ],

    "Cosmos > Miscellaneous": [
        "dark matter",
        "dark energy",
        "gravitational lens",
        "gravitational lensing",
        "quasar",
        "active galactic nucleus",
        "active galactic nuclei",
        "agn",
        "cosmic microwave background",
        "cmb",
        "large-scale structure",
        "cosmic web",
        "early universe",
        "observable universe",
        "distant universe",
        "redshift",
        "high redshift",
        "gamma-ray burst",
        "grb",
        "fast radio burst",
        "frb",
        "cosmic ray",
        "gravitational wave",
        "intergalactic medium"
    ],

    "Solar System": [
        "solar flare",
        "solar prominence",
        "solar corona",
        "coronal mass ejection",
        "sunspot",
        "solar eclipse",
        "total solar eclipse",
        "partial solar eclipse",
        "annular eclipse",
        "aurora caused by solar activity",
        "mercury transit",
        "venus transit",
        "planet venus",
        "planet mars",
        "martian surface",
        "mars rover",
        "jupiter's atmosphere",
        "great red spot",
        "jupiter's moon",
        "galilean moon",
        "io",
        "europa",
        "ganymede",
        "callisto",
        "saturn's rings",
        "saturn's moon",
        "titan",
        "enceladus",
        "uranus",
        "neptune",
        "pluto",
        "kuiper belt",
        "oort cloud",
        "asteroid belt",
        "near-earth asteroid",
        "lunar eclipse",
        "earth's moon",
        "crescent moon",
        "full moon",
        "lunar surface",
        "planetary conjunction",
        "opposition of mars",
        "solar system object"
    ],

    "Comets": [
        "comet tail",
        "comet coma",
        "comet nucleus",
        "dust tail",
        "ion tail",
        "periodic comet",
        "long-period comet",
        "comet hyakutake",
        "comet hale-bopp",
        "halley's comet",
        "comet neowise",
        "comet leonard",
        "comet lovejoy",
        "comet mcnaught",
        "comet c/202",
        "comet 67p",
        "rosetta comet",
        "dirty snowball",
        "sungrazing comet"
    ],

    "Space Technology": [
        "space shuttle",
        "international space station",
        "iss",
        "hubble space telescope",
        "james webb space telescope",
        "jwst",
        "spitzer space telescope",
        "chandra x-ray observatory",
        "x-ray observatory",
        "orbiting observatory",
        "space telescope",
        "spacecraft",
        "space probe",
        "planetary probe",
        "mars rover",
        "curiosity rover",
        "perseverance rover",
        "opportunity rover",
        "sojourner rover",
        "cassini spacecraft",
        "voyager spacecraft",
        "new horizons spacecraft",
        "juno spacecraft",
        "osiris-rex",
        "artemis",
        "rocket launch",
        "launch vehicle",
        "falcon 9",
        "satellite",
        "space station",
        "earth observatory",
        "crew dragon"
    ],

    "Sky": [
        "night sky",
        "earth's sky",
        "dark sky",
        "twilight sky",
        "dawn sky",
        "evening sky",
        "morning sky",
        "star trails",
        "meteor shower",
        "aurora",
        "airglow",
        "zodiacal light",
        "light pillar",
        "noctilucent clouds",
        "constellation",
        "milky way over",
        "sky panorama",
        "all-sky view",
        "wide-field sky",
        "southern sky",
        "northern sky",
        "sky watchers",
        "skygazers",
        "moon and venus",
        "planetary alignment",
        "conjunction in the sky",
        "eclipse sky"
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
    Assign one weak label based on keyword/phrase matches.

    This version uses regex word boundaries so keywords match as full words/phrases,
    not as accidental substrings.

    Example:
    - "sun" will not match "sunlight"
    - "io" will not match inside "radiation"
    - "spiral galaxy" will match "spiral galaxy"
    """
    text = _normalize_text(text)

    best_label = None
    best_score = 0

    for label, keywords in keyword_map.items():
        score = 0

        for keyword in keywords:
            keyword_normalized = _normalize_text(keyword)

            if not keyword_normalized:
                continue

            pattern = r"\b" + re.escape(keyword_normalized) + r"\b"

            if re.search(pattern, text):
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
def exploreRawDataset(df: pd.DataFrame) -> None:
    
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
    print(df.loc[df['thumbnail_url'].isna() & 
                  df['url'].isna(),
                  ['date', 'title', 'media_type', 'hdurl',  'copyright']])

    # Clean messy exlpanation data from copyright column then print
    df['copyright'] = df['copyright'].str.split('Explanation:').str[0].str.strip() 

    print("\nRows with copyright:")
    print(df.loc[df['copyright'].notna(),
                      ['date', 'title', 'media_type', 'url', 'hdurl', 'thumbnail_url', 'copyright']])



# Method for visualizer file to call in order to have up to date df before creating graphs
def categorizeDataset(return_df: bool = False) -> pd.DataFrame | None:
    
    update_apod_dataset()           # Always use latest version of APOD dataset when running the manager.

    # Get the APOD dataset as a pandas DataFrame
    apod_df = pd.read_csv(RAW_APOD_CSV_PATH)

    return update_labeled_apod_dataset(apod_df, APOD_KEY_WORDS, return_df=return_df)



def seperateExplanation(df_to_split: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate the 'explanation' column from the main DataFrame for easier exploration.

    Parameters
    ----------
    df : pd.DataFrame
        The original APOD DataFrame containing an 'explanation' column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - df_no_explanation_df: The original DataFrame without the 'explanation' column.
        - explanation_df: A new DataFrame with only 'date' and 'explanation' columns.
    """

    explanation_df = df_to_split[['date', 'explanation']].copy()
    df_no_explanation_df = df_to_split.drop(columns=['explanation']).copy()
    
    return df_no_explanation_df, explanation_df



# Averages confidence of each label guess and prints
def print_confidence_distribution(
    df: pd.DataFrame,
    confidence_col: str = "predicted_confidence"
) -> None:
    if confidence_col not in df.columns:
        raise ValueError(f"Missing confidence column: {confidence_col}")

    # Confidence distribution
    if "predicted_confidence" in df.columns and "final_label" in df.columns:
        print("\nMean Predicted Confidence by Final Label:")
        print(df.groupby("final_label")["predicted_confidence"].mean().sort_values())
        print("\nOverall Mean of Label Means:")
        print(df.groupby("final_label")["predicted_confidence"].mean().mean())

        print("\nMedian Predicted Confidence by Final Label:")
        print(df.groupby("final_label")["predicted_confidence"].median().sort_values())
        print("\nOverall Mean of Label Medians:")
        print(df.groupby("final_label")["predicted_confidence"].median().mean())

        print("\nStandard Deviation of Predicted Confidence by Final Label:")
        print(df.groupby("final_label")["predicted_confidence"].std().sort_values())
        print("\nOverall Mean of Label Standard Deviations:")
        print(df.groupby("final_label")["predicted_confidence"].std().mean())



def remove_low_confidence_entries(
    labeled_df: pd.DataFrame,
    min_confidence: float = 0.50,
    confidence_col: str = "predicted_confidence",
    save: bool = False
) -> pd.DataFrame:
    """
    Remove APOD entries whose predicted confidence is below min_confidence.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Current labeled APOD dataset.

    min_confidence : float, default=0.50
        Minimum confidence required to keep a row.
        0.50 means 50%.

    confidence_col : str, default="predicted_confidence"
        Name of the confidence column.

    save : bool, default=False
        If True, overwrite the labeled CSV and update metadata.

    Returns
    -------
    pd.DataFrame
        Filtered labeled APOD DataFrame.
    """
    if confidence_col not in labeled_df.columns:
        raise ValueError(f"Missing confidence column: {confidence_col}")

    original_count = len(labeled_df)

    filtered_df = labeled_df[
        labeled_df[confidence_col].notna()
        & (labeled_df[confidence_col] >= min_confidence)
    ].copy()

    filtered_df = filtered_df.sort_values("date").reset_index(drop=True)

    removed_count = original_count - len(filtered_df)

    print(f"Original rows: {original_count}")
    print(f"Removed rows below {min_confidence:.0%} confidence: {removed_count}")
    print(f"Remaining rows: {len(filtered_df)}")

    if save:
        source_df = pd.read_csv(RAW_APOD_CSV_PATH)
        _write_labeled_outputs(filtered_df, source_df)
        print(f"Saved filtered dataset to {LABELED_CSV_PATH}")

    return filtered_df



# Method to explore the categorized dataset with all columns to understand the new label columns and confidence scores
# Check bottom of file for results
def exploreCategorizedDataset(df: pd.DataFrame) -> None:

    # Shape
    print(f"Dataset Shape: \n{df.shape}\n")

    # Column names
    print(f"Column Names:\n{df.columns.tolist()}\n")

    # Non null category counts
    print(f"Non-Null Values:\n{df.groupby('final_label').size()}\n")

    # Weakest confidence labels
    if "predicted_confidence" in df.columns and "final_label" in df.columns:
        low_confidence_df = df[df["predicted_confidence"] < 0.5][
            ["date", "title", "final_label", "predicted_label", "predicted_confidence"]
        ].sort_values("predicted_confidence")
        print("Low Confidence Predictions (<50% confidence):")
        print(low_confidence_df.head(10))

    # Confidence distribution
    print_confidence_distribution(df)

    # Low confidence label analysis
    print("\nLow Confidence Predictions Analysis:")
    labels_to_check = [
    "Cosmos > Miscellaneous",
    "Comets",
    ]

    for label in labels_to_check:
        label_df = categorized_df[categorized_df["final_label"] == label].copy()

        print(f"\n{'=' * 80}")
        print(f"{label}")
        print(f"{'=' * 80}")

        print("\nHighest 5 predicted confidences:")
        print(
            label_df
            .sort_values("predicted_confidence", ascending=False)
            [["date", "title", "final_label", "predicted_confidence", "label_source"]]
            .head(5)
            .to_string(index=False)
        )

        print("\nLowest 5 predicted confidences:")
        print(
            label_df
            .sort_values("predicted_confidence", ascending=True)
            [["date", "title", "final_label", "predicted_confidence", "label_source"]]
            .head(5)
            .to_string(index=False)
        )



if __name__ == "__main__":

    # Set pandas preferences for better display of the dataset when exploring it.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
#    pd.set_option('display.width', None)

    # Create or update the labeled APOD dataset under data/
    categorized_df = categorizeDataset(return_df=True)

    print("\nLabeled APOD dataset ready.")
    
    # Explore categorized dataset with all columns to understand the new label columns and confidence scores
    categorized_no_explanation_df = seperateExplanation(categorized_df)[0]
    exploreCategorizedDataset(categorized_no_explanation_df)
    
    #print_average_label_confidence(categorized_df)

    # explores dataset with no explanation column for easier reading
    #apod_no_explanation_df, explanation_df = seperateExplanation()
    #exploreDataset(apod_no_explanation_df)


"""                         Labeled APOD Dataset Exploration Results,       {} indicates thoughts
Dataset Shape: 
(11277, 16)

Column Names:
['date', 'title', 'media_type', 'url', 'hdurl', 'thumbnail_url', 'copyright', 'service_version', 'document_text', 'weak_label', 'predicted_label', 'predicted_confidence', 'final_label', 'label_source', 'main_category', 'sub_category']

Non-Null Values:
final_label
Comets                     274
Cosmos > Galaxies         1627
Cosmos > Miscellaneous     243
Cosmos > Nebulae          1141
Cosmos > Stars            1296
Sky                       1105
Solar System              2646
Space Technology          2944
dtype: int64

Low Confidence Predictions (<50% confidence):
             date                                   title       final_label  \
9922   2022-08-20                Stardust and Comet Tails            Comets   
7656   2016-06-05    Comet PanSTARRS and the Helix Nebula            Comets   
6740   2013-12-02         Comet Lovejoy Before Galaxy M63            Comets   
139    1995-11-05           Vela Satellites: The Watchers               NaN   
9671   2021-12-12    Comet Leonard Before Star Cluster M3            Comets   
10355  2023-10-27                  Encke and the Tadpoles    Cosmos > Stars   
373    1996-06-26      Happy Birthday Charles Messier: M1  Cosmos > Nebulae   
9620   2021-10-22                      A Comet and a Crab            Comets   
11275  2026-05-05                  Orion over Mount Teide  Cosmos > Nebulae   
686    1997-05-05  Sunset with Hale-Bopp at Keck\nCredit:    Cosmos > Stars   

        predicted_label  predicted_confidence  
9922   Cosmos > Nebulae              0.202837  
7656   Space Technology              0.232700  
6740             Comets              0.235541  
139        Solar System              0.245651  
9671       Solar System              0.245936  
10355      Solar System              0.259540  
373      Cosmos > Stars              0.260150  
9620     Cosmos > Stars              0.264377  
11275      Solar System              0.266328  
686        Solar System              0.269741  

Mean Predicted Confidence by Final Label:
final_label
Comets                    0.561154
Cosmos > Miscellaneous    0.626270
Sky                       0.773503
Space Technology          0.818386
Cosmos > Stars            0.843280
Solar System              0.846913
Cosmos > Nebulae          0.881178
Cosmos > Galaxies         0.889153
Name: predicted_confidence, dtype: float64

Overall Mean of Label Means:
0.7799796816755992

Median Predicted Confidence by Final Label:
final_label
Comets                    0.511577
Cosmos > Miscellaneous    0.596263
Sky                       0.840698
Space Technology          0.872822
Solar System              0.931366
Cosmos > Stars            0.938444
Cosmos > Nebulae          0.975096
Cosmos > Galaxies         0.981934
Name: predicted_confidence, dtype: float64

Overall Mean of Label Medians:
0.8310249819052439

Standard Deviation of Predicted Confidence by Final Label:
final_label
Space Technology          0.170029
Cosmos > Galaxies         0.170866
Solar System              0.174044
Cosmos > Nebulae          0.178382
Cosmos > Stars            0.183222
Cosmos > Miscellaneous    0.184515
Comets                    0.197845
Sky                       0.203706
Name: predicted_confidence, dtype: float64

Overall Mean of Label Standard Deviations:
0.18282603161342062

Low Confidence Predictions Analysis:

================================================================================
Cosmos > Miscellaneous
================================================================================

Highest 5 predicted confidences:
      date                                 title            final_label  predicted_confidence label_source
2000-10-10 The Einstein Cross Gravitational Lens Cosmos > Miscellaneous              0.981899      keyword      {Cosmos>Miscellaneous}
2007-03-11 The Einstein Cross Gravitational Lens Cosmos > Miscellaneous              0.981899      keyword      {Cosmos>Miscellaneous, same as previous}
2005-03-27 The Einstein Cross Gravitational Lens Cosmos > Miscellaneous              0.981899      keyword      {Cosmos>Miscellaneous, same as first}   
2021-10-17 The Einstein Cross Gravitational Lens Cosmos > Miscellaneous              0.981899      keyword      {Cosmos>Miscellaneous, same as 2nd}
2010-02-07 The Einstein Cross Gravitational Lens Cosmos > Miscellaneous              0.981899      keyword      {Cosmos>Miscellaneous, same as first and third}

Lowest 5 predicted confidences:
      date                                    title            final_label  predicted_confidence label_source
1997-06-14                       The Early Universe Cosmos > Miscellaneous              0.306183      keyword       {Cosmos>Miscellaneous}
1996-02-16                       The Early Universe Cosmos > Miscellaneous              0.307814      keyword       {Cosmos>Miscellaneous, same as previous}
2005-12-20              Star Trails Above Mauna Kea Cosmos > Miscellaneous              0.315136      keyword       {Sky}
2000-12-02 SN 1006: Pieces of the Cosmic Ray Puzzle Cosmos > Miscellaneous              0.315140      keyword       {Cosmos>Miscellaneous}
1996-10-16 SN 1006: Pieces of the Cosmic Ray Puzzle Cosmos > Miscellaneous              0.322310      keyword       {Cosmos>Miscellaneous, same as previous}

================================================================================
Comets
================================================================================

Highest 5 predicted confidences:
      date                                title final_label  predicted_confidence label_source
2008-03-02 Comet Hale-Bopp Over Val Parola Pass      Comets              0.997507      keyword      {Comets}
2012-12-23 Comet Hale-Bopp Over Val Parola Pass      Comets              0.996851      keyword      {Comets, same as previous}
2005-09-04 Comet Hale-Bopp Over Val Parola Pass      Comets              0.996647      keyword      {Comets, same as previous}
2001-05-27 Comet Hale-Bopp Over Val Parola Pass      Comets              0.996647      keyword      {Comets, same as previous}
2004-03-14 Comet Hale-Bopp Over Val Parola Pass      Comets              0.996647      keyword      {Comets, same as previous}

Lowest 5 predicted confidences:
      date                                title final_label  predicted_confidence label_source
2022-08-20             Stardust and Comet Tails      Comets              0.202837      keyword      {Comets}
2016-06-05 Comet PanSTARRS and the Helix Nebula      Comets              0.232700      keyword      {Comets}
2013-12-02      Comet Lovejoy Before Galaxy M63      Comets              0.235541      keyword      {Comets}
2021-12-12 Comet Leonard Before Star Cluster M3      Comets              0.245936      keyword      {Comets}
2021-10-22                   A Comet and a Crab      Comets              0.264377      keyword      {Comets/Cosmos>Nebulae}


{

Low confidence accuracy: 9/10, without repeat: 7/8
High confidence accuracy: 10/10, without repeat: 2/2

Total confidence: 19/20, without repeat 9/10

}
"""