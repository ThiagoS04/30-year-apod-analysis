import pandas as pd
import matplotlib.pyplot as plt

from apod_manager import categorizeDataset, remove_low_confidence_entries

def plot_category_counts_by_period(
    labeled_df: pd.DataFrame,
    period_years: int = 5,
    category_col: str = "final_label",
    date_col: str = "date"
) -> None:
    """
    Plot a heatmap showing how many APOD entries each category has
    over custom year periods.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Current labeled APOD dataset.

    period_years : int, default=5
        Number of years per time period.
        Example: 5 creates 5-year periods, 10 creates 10-year periods.

    category_col : str, default="final_label"
        Column containing APOD category labels.

    date_col : str, default="date"
        Column containing APOD dates.
    """
    import matplotlib.pyplot as plt

    if period_years <= 0:
        raise ValueError("period_years must be greater than 0.")

    if date_col not in labeled_df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    if category_col not in labeled_df.columns:
        raise ValueError(f"Missing category column: {category_col}")

    df = labeled_df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()]
    df = df[df[category_col].notna()]

    df["year"] = df[date_col].dt.year

    df["period_start"] = (df["year"] // period_years) * period_years
    df["period_end"] = df["period_start"] + period_years - 1

    df["period"] = (
        df["period_start"].astype(str)
        + "-"
        + df["period_end"].astype(str)
    )

    category_period_counts = (
        df.groupby(["period", category_col])
        .size()
        .unstack(fill_value=0)
    )

    plt.figure(figsize=(14, 7))
    plt.imshow(category_period_counts.T, aspect="auto")

    plt.xticks(
        ticks=range(len(category_period_counts.index)),
        labels=category_period_counts.index,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        ticks=range(len(category_period_counts.columns)),
        labels=category_period_counts.columns
    )

    plt.xlabel(f"{period_years}-Year Period")
    plt.ylabel("Category")
    plt.title(f"APOD Category Counts by {period_years}-Year Period")

    plt.colorbar(label="Number of Entries")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    labeled_df = categorizeDataset(True)                             # Get the labeled dataset with predicted confidences
    confident_df = remove_low_confidence_entries(labeled_df, .5)     # Remove entries with <50% confidence

    plot_category_counts_by_period(confident_df, period_years=1)              # Plot category counts of confident df by 1-year periods