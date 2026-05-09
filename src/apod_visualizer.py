import math

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math


from apod_manager import categorizeDataset, print_confidence_distribution, remove_low_confidence_entries

OUTPUT_FOLDER = "data/vis/"

def normalize_df(
        df: pd.DataFrame,
        date_col: str = "date",
        category_col: str = "final_label"
        ) -> pd.DataFrame:

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    if category_col not in df.columns:
        raise ValueError(f"Missing category column: {category_col}")

    normal_df = df.copy()

    normal_df[date_col] = pd.to_datetime(normal_df[date_col], errors="coerce")
    normal_df = normal_df[normal_df[date_col].notna()]
    normal_df = normal_df[normal_df[category_col].notna()]
    return normal_df


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

    if period_years <= 0:
        raise ValueError("period_years must be greater than 0.")

    df = normalize_df(labeled_df)

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

    # Add count labels inside each heatmap box
    for row_index, category in enumerate(category_period_counts.columns):
        for col_index, period in enumerate(category_period_counts.index):
            count = category_period_counts.loc[period, category]

            plt.text(
                col_index,
                row_index,
                str(count),
                ha="center",
                va="center"
            )

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


def save_yearly_category_pie_chart_pages(
    labeled_df: pd.DataFrame,
    output_folder: str = f"{OUTPUT_FOLDER}/apod_yearly_pie_chart_pages",
    charts_per_page: int = 9,
    category_col: str = "final_label",
    date_col: str = "date",
    min_pct_label: float = 3.0
) -> None:
    """
    Save yearly APOD category pie charts as PNG image pages.

    FIXED:
    - Each category has a stable color
    - Slice names are removed from the pie charts
    - Legend is used instead
    """

    if charts_per_page <= 0:
        raise ValueError("charts_per_page must be greater than 0.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = normalize_df(labeled_df)

    df = df[df[category_col].notna()]

    df["year"] = df[date_col].dt.year

    years = sorted(df["year"].unique())
    all_categories = sorted(df[category_col].unique())

    # Stable color mapping
    cmap = plt.get_cmap("tab10")
    category_colors = {
        category: cmap(i % cmap.N)
        for i, category in enumerate(all_categories)
    }

    cols = math.ceil(math.sqrt(charts_per_page))
    rows = math.ceil(charts_per_page / cols)

    def autopct_format(pct):
        if pct >= min_pct_label:
            return f"{pct:.1f}%"
        return ""

    page_number = 1

    for page_start in range(0, len(years), charts_per_page):
        page_years = years[page_start:page_start + charts_per_page]

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 5, rows * 5)
        )

        if charts_per_page == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax_index, ax in enumerate(axes):
            if ax_index >= len(page_years):
                ax.axis("off")
                continue

            year = page_years[ax_index]
            year_df = df[df["year"] == year]

            category_counts = (
                year_df[category_col]
                .value_counts()
                .reindex(all_categories, fill_value=0)
            )

            category_counts = category_counts[category_counts > 0]

            pie_colors = [category_colors[cat] for cat in category_counts.index]

            ax.pie(
                category_counts.values,
                colors=pie_colors,
                autopct=autopct_format,
                startangle=90,
                textprops={"fontsize": 8}
            )

            ax.set_title(f"{year}", fontsize=12)
            ax.axis("equal")

        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=category_colors[category],
                markeredgecolor=category_colors[category],
                markersize=8,
                label=category
            )
            for category in all_categories
        ]

        fig.legend(
            handles=legend_handles,
            title="Category",
            loc="center right",
            bbox_to_anchor=(1.05, 0.5)
        )

        fig.suptitle(
            "APOD Category Percentages by Year",
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])

        output_path = output_folder / f"apod_pie_charts_page_{page_number}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {output_path}")

        page_number += 1


def save_apod_distribution_line_graph(
    labeled_df: pd.DataFrame,
    output_path: str = "data/vis/apod_distribution_over_time.png",
    category_col: str = "final_label",
    date_col: str = "date",
    normalize: bool = True,
    rolling_window: int = 3,
    start_year: int = 1996
) -> None:
    """
    Save a line graph showing APOD category distribution over time.

    All labels beginning with 'Cosmos >' are combined into one category:
    'Cosmology'.

    A centered rolling average is applied to smooth year-to-year noise.
    """

    if rolling_window <= 0:
        raise ValueError("rolling_window must be greater than 0.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = normalize_df(
        labeled_df,
        date_col=date_col,
        category_col=category_col
    )

    df["year"] = df[date_col].dt.year

    # Start at 1996 because 1995 is incomplete
    df = df[df["year"] >= start_year]

    df["plot_category"] = df[category_col].apply(
        lambda label: "Cosmology" if str(label).startswith("Cosmos >") else label
    )

    yearly_counts = (
        df.groupby(["year", "plot_category"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if normalize:
        yearly_values = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100
        y_label = "Percentage of APOD Posts"
        title = f"APOD Category Distribution Over Time ({rolling_window}-Year Rolling Average)"
    else:
        yearly_values = yearly_counts
        y_label = "Number of APOD Posts"
        title = f"APOD Category Counts Over Time ({rolling_window}-Year Rolling Average)"

    # 3-year centered rolling average
    yearly_values = yearly_values.rolling(
        window=rolling_window,
        center=True,
        min_periods=1
    ).mean()

    plt.figure(figsize=(14, 7))

    for category in yearly_values.columns:
        plt.plot(
            yearly_values.index,
            yearly_values[category],
            marker="o",
            linewidth=2,
            label=category
        )

    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title="Category")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved line graph to {output_path}")



if __name__ == "__main__":

    labeled_df = categorizeDataset(True)                             # Get the labeled dataset with predicted confidences
    confident_df = remove_low_confidence_entries(labeled_df, .5)     # Remove entries with <50% confidence
    print_confidence_distribution(confident_df)                    # Print average label confidences for new "confident" df


    plot_category_counts_by_period(confident_df, period_years=1)              # Plot category counts of confident df by 1-year periods

    save_yearly_category_pie_chart_pages(confident_df)           # Save yearly category pie charts of confident df to PDF

    save_apod_distribution_line_graph(confident_df)              # Save line graph of category distribution over time for confident df