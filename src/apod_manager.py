### Astronomy Picture of the Day (APOD) Dataset Manager/Manipulator

import re
import pandas as pd

from apod_scraper import update_apod_dataset
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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



def _normalize_text(text: str) -> str:
    """
    Lowercase and remove extra whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text



def _build_document_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + explanation into one text field.
    Uses explanation if present, otherwise just title.
    """
    title = df["title"].fillna("").astype(str)
    explanation = df["explanation"].fillna("").astype(str)
    return (title + " " + explanation).str.strip()



def _weak_label_from_keywords(text: str, flat_keywords: Dict[str, List[str]]) -> str | None:
    """
    Assigns the single best label based on keyword matches.
    Returns None if no keywords matched.

    Strategy:
    - count how many keywords from each class appear
    - choose the class with the highest count
    """
    text = _normalize_text(text)

    best_label = None
    best_score = 0

    for label, keywords in flat_keywords.items():
        score = 0
        for kw in keywords:
            kw_norm = _normalize_text(kw)
            if kw_norm and kw_norm in text:
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

    # Build text column
    df["document_text"] = _build_document_text(df)

    # Keyword-based weak labeling
    df["weak_label"] = df["document_text"].apply(
        lambda txt: _weak_label_from_keywords(txt, apod_keywords)
    )

    training_df = df[df["weak_label"].notna()].copy()

    # If not enough rows to train, return keyword-only result
    if len(training_df) < min_training_rows:
        df["predicted_label"] = None
        df["predicted_confidence"] = None
        df["final_label"] = df["weak_label"]
        df["label_source"] = df["weak_label"].apply(
            lambda x: "keyword" if pd.notna(x) else None
        )

        # Optional split into main/sub columns
        split_cols = df["final_label"].str.split(" > ", n=1, expand=True)
        df["main_category"] = split_cols[0]
        df["sub_category"] = split_cols[1] if split_cols.shape[1] > 1 else None

        return df, None

    # Train TF-IDF + Naive Bayes
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

    # Predict for all rows
    pred_probs = model.predict_proba(df["document_text"])
    pred_labels = model.classes_[pred_probs.argmax(axis=1)]
    pred_confidences = pred_probs.max(axis=1)

    df["predicted_label"] = pred_labels
    df["predicted_confidence"] = pred_confidences

    # Choose final label
    def choose_final_label(row):
        if pd.notna(row["weak_label"]):
            return row["weak_label"], "keyword"
        if row["predicted_confidence"] >= min_confidence:
            return row["predicted_label"], "naive_bayes"
        return None, None

    final_pairs = df.apply(choose_final_label, axis=1)
    df["final_label"] = [pair[0] for pair in final_pairs]
    df["label_source"] = [pair[1] for pair in final_pairs]

    # Split label into main/sub columns
    split_cols = df["final_label"].str.split(" > ", n=1, expand=True)
    df["main_category"] = split_cols[0]
    df["sub_category"] = split_cols[1] if split_cols.shape[1] > 1 else None

    return df, model



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



if __name__ == "__main__":

    update_apod_dataset()           # Always use latest version of APOD dataset when running the manager.

    # Get the APOD dataset as a pandas DataFrame
    apod_df = pd.read_csv("data/apod_data.csv")

    # Set pandas preferences for better display of the dataset when exploring it.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    #pd.set_option('display.width', None)

    # Remove 'explanation' column and save in separate database for easier exploration of the rest of the dataset.
    explanation_df = apod_df[['date', 'explanation']]
    apod_no_explanation_df = apod_df.drop(columns=['explanation'])

    exploreDataset(apod_no_explanation_df)

    categorized_df = categorize_apod_entries(apod_df, APOD_KEY_WORDS)[0]
    print("\nCategorized DataFrame with weak labels and predictions:")
    print(categorized_df.head())

    target_dates = [
    "1997-01-15",
    "2002-09-08",
    "2004-02-24"
    ]

    print(categorized_df.loc[categorized_df["date"].isin(target_dates), 
                             ["date", "title", "weak_label", "predicted_label", "predicted_confidence", 
                              "final_label", "label_source", "main_category", "sub_category"]])
