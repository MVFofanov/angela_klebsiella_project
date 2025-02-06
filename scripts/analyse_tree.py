from collections import Counter
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from ete3 import Tree, TreeStyle, NodeStyle, TextFace

# Set environment variable for non-interactive backend
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# Step 1: Merge metadata sheets
def merge_metadata(file_path: str, output_tsv: str) -> pd.DataFrame:
    """Merge 'Continent' and 'Country' sheets, replace missing values, and format strings."""
    df_continent = pd.read_excel(file_path, sheet_name="Continent")
    df_country = pd.read_excel(file_path, sheet_name="Country")

    # Merge on "Tree node ID"
    merged_df = pd.merge(df_country, df_continent, on="Tree node ID", how="outer")

    # Convert "Tree node ID" to string and strip spaces
    merged_df["Tree node ID"] = merged_df["Tree node ID"].astype(str).str.strip()

    # Replace missing or empty country/continent values with "unknown"
    merged_df["Country"] = (
        merged_df["Country"]
            .fillna("unknown")
            .replace("", "unknown")
            .str.replace(" ", "_", regex=False)  # Replace spaces with underscores
    )

    merged_df["Continent"] = (
        merged_df["Continent"]
            .fillna("unknown")
            .replace("", "unknown")
            .str.replace(" ", "_", regex=False)  # Replace spaces with underscores
    )

    # Save to TSV
    merged_df.to_csv(output_tsv, sep="\t", index=False)
    return merged_df


# Step 2: Annotate the tree
def annotate_tree(tree_file: str, metadata_df: pd.DataFrame) -> Tree:
    """Load tree and annotate it with metadata (Country, Continent)."""
    logging.info("🔍 Starting tree annotation...")

    # Load tree
    tree = Tree(tree_file, format=1)
    logging.info(f"🌳 Tree loaded from {tree_file} with {len(tree.get_leaves())} leaves.")

    # Convert metadata keys to strings and strip spaces
    metadata_dict = metadata_df.set_index("Tree node ID").to_dict("index")

    logging.info(f"Annotation dictionary {metadata_dict=}.")

    for node in tree.traverse():
        if node.name:  # Skip unnamed internal nodes
            node.name = str(node.name).strip()  # Ensure node names are strings and remove extra spaces

    # Annotate each node
    for node in tree.traverse():
        logging.info(f"Name of the node: {node.name=}.")
        if node.name in metadata_dict:
            logging.info(f"Name of the node found in the annotation table: {node.name=}.")
            metadata = metadata_dict[node.name]
            node.add_features(
                country=metadata["Country"],
                continent=metadata["Continent"],
                label=f'{node.name} | {metadata["Country"]} | {metadata["Continent"]}'
            )
            logging.debug(f"✅ Annotated node: {node.name} | Country: {metadata['Country']} | "
                          f"Continent: {metadata['Continent']}")
        else:
            logging.warning(f"⚠️ Node {node.name} has no metadata in the table!")

    logging.info("✅ Tree annotation completed.")
    return tree


# Step 3: Find Kenyan samples and their closest sister clades
def find_kenya_sister_clades(tree: Tree) -> Tuple[
    List[Tuple[str, int, int, str, Dict[str, int], int, str, Dict[str, int], str, str, int, float]], set, set]:
    """Find closest non-Kenyan sister clades for Kenyan samples and extract country and continent distribution."""
    results = []
    unique_countries = set()
    unique_continents = set()

    logging.info("🔍 Starting to find Kenyan samples and their sister clades.")

    for leaf in tree.iter_leaves():
        if hasattr(leaf, "country") and leaf.country == "Kenya":
            node = leaf.up  # Start at the parent node
            logging.debug(
                f"🟢 Found Kenyan sample: {leaf.name} | Country: Kenya | "
                f"Continent: {getattr(leaf, 'continent', 'unknown')}")

            sister_clade = None
            while node is not None:
                sisters = node.get_sisters()
                logging.debug(f"🔄 Checking sisters of {node.name}: {[s.name for s in sisters]}")

                for sister in sisters:
                    sister_leaves = sister.get_leaves()
                    sister_countries = {n.country for n in sister_leaves if hasattr(n, "country")}
                    sister_continents = {n.continent for n in sister_leaves if hasattr(n, "continent")}

                    # Find a sister clade that contains at least ONE non-Kenyan sample
                    non_kenyan_countries = sister_countries - {"Kenya"}
                    if non_kenyan_countries:
                        sister_clade = sister
                        logging.info(
                            f"✅ Found a valid sister clade for {leaf.name}: Contains {len(sister_leaves)} "
                            f"samples from {non_kenyan_countries} and continents {sister_continents}")
                        break

                if sister_clade:
                    break  # Stop searching once we find a valid non-Kenyan sister clade

                logging.debug(f"⏫ Moving up to parent node: {node.name if node else 'None'}")
                node = node.up  # Move further up in the tree

            if sister_clade:
                sister_leaves = sister_clade.get_leaves()
                num_sister_members = len(sister_leaves)
                sister_countries = [n.country for n in sister_leaves if hasattr(n, "country")]
                sister_continents = [n.continent for n in sister_leaves if hasattr(n, "continent")]

                country_counts = dict(Counter(sister_countries))
                continent_counts = dict(Counter(sister_continents))

                unique_countries.update(sister_countries)
                unique_continents.update(sister_continents)

                sister_countries_uniq = sorted(set(sister_countries))
                sister_continents_uniq = sorted(set(sister_continents))

                # Sister Clade Members (sorted node names)
                sister_clade_members = ", ".join(sorted(n.name for n in sister_leaves if n.name))

                # Sister Clade Members with known countries
                sister_clade_members_known = ", ".join(
                    sorted(n.name for n in sister_leaves if n.name and hasattr(n, "country") and hasattr(n, "continent"))
                )
                sister_clade_members_known_size = len(
                    [n for n in sister_leaves if hasattr(n, "country") and hasattr(n, "continent")]
                )
                sister_clade_members_known_ratio = round(sister_clade_members_known_size / num_sister_members, 2)

                results.append((
                    leaf.name, num_sister_members,
                    len(sister_continents_uniq), ", ".join(sister_continents_uniq), continent_counts,
                    len(sister_countries_uniq), ", ".join(sister_countries_uniq), country_counts,
                    sister_clade_members, sister_clade_members_known,
                    sister_clade_members_known_size, sister_clade_members_known_ratio
                ))

    logging.info(f"📝 Total Kenyan samples processed: {len(results)}")
    logging.info("✅ Completed Kenyan sister clade analysis.")

    return results, unique_countries, unique_continents


# Step 4: Save the Kenya sister clade results
def save_kenya_sister_clades(
        results: List[Tuple[str, int, int, str, Dict[str, int], int, str, Dict[str, int], str, str, int, float]],
        unique_countries: set, unique_continents: set, output_file: str):
    """Save the Kenya sister clade data in the specified format."""
    columns = [
        "Kenya Sample", "Sister Clade Size",
        "Number of continents", "Sister Clade Continents", "Continent Counts",
        "Number of countries", "Sister Clade Countries", "Country Counts",
        "Sister Clade Members", "Sister Clade Members with known countries",
        "Sister Clade Members with known countries size", "Sister Clade Members with known countries ratio"
    ]
    columns.extend(sorted(unique_countries))  # Add unique country columns
    columns.extend(sorted(unique_continents))  # Add unique continent columns

    data = []
    for entry in results:
        row = {col: 0 for col in columns}  # Initialize row
        (row["Kenya Sample"], row["Sister Clade Size"],
         row["Number of continents"], row["Sister Clade Continents"], row["Continent Counts"],
         row["Number of countries"], row["Sister Clade Countries"], country_counts,
         row["Sister Clade Members"], row["Sister Clade Members with known countries"],
         row["Sister Clade Members with known countries size"], row["Sister Clade Members with known countries ratio"]) = entry

        # Sort country dictionary in descending order
        country_counts = {k: v for k, v in sorted(country_counts.items(), key=lambda item: item[1], reverse=True)}
        row["Country Counts"] = ', '.join([f'{k}: {v}' for k, v in country_counts.items()])

        # Fill specific country count columns
        for country, count in country_counts.items():
            row[country] = count

        # Fill specific continent count columns
        continent_counts = {k: v for k, v in
                            sorted(row["Continent Counts"].items(), key=lambda item: item[1], reverse=True)}
        row["Continent Counts"] = ', '.join([f'{k}: {v}' for k, v in continent_counts.items()])

        for continent, count in continent_counts.items():
            row[continent] = count

        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values(by="Sister Clade Size", ascending=False)
    df.to_csv(output_file, sep="\t", index=False)


def save_annotated_tree(tree: Tree, output_dir: str):
    """Save tree as SVG and PDF with continent and country annotations."""
    tree_style = TreeStyle()
    tree_style.show_leaf_name = False  # Do not display node.name directly

    # Customize node appearance
    for node in tree.traverse():
        if node.is_leaf():
            if hasattr(node, "label"):  # Use label if it exists
                node.add_face(TextFace(node.label, fsize=10), column=0, position="branch-right")
            else:
                node.add_face(TextFace(node.name, fsize=10), column=0, position="branch-right")

            # Set style
            nstyle = NodeStyle()
            nstyle["size"] = 5
            nstyle["fgcolor"] = "black"
            node.set_style(nstyle)

    # Save the tree
    svg_file = f"{output_dir}/annotated_tree.svg"
    pdf_file = f"{output_dir}/annotated_tree.pdf"
    tree.render(svg_file, tree_style=tree_style, dpi=300)
    tree.render(pdf_file, tree_style=tree_style, dpi=300)

    print(f"✅ Annotated tree saved as:\n - {svg_file}\n - {pdf_file}")


def plot_scatter_countries_vs_clade_size(data_file: str, output_file: str):
    """Creates and saves a scatterplot of Number of Countries vs Log10 Sister Clade Size."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_file, sep="\t")

    # Check if required columns exist
    if "Number of countries" not in df.columns or "Sister Clade Size" not in df.columns:
        raise ValueError("Missing required columns in the dataset!")

    # Log-transform the Sister Clade Size column
    df["Log10 Sister Clade Size"] = np.log10(df["Sister Clade Size"])  # Avoid log(0) issues

    # Create scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Number of countries", y="Log10 Sister Clade Size")

    # Set labels and title
    plt.xlabel("Number of Countries")
    plt.ylabel("Log10 Number of members in a sister clade")
    plt.title("Scatterplot of Number of Countries vs. Number of members in a sister clade")

    # Save the figure
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Scatterplot saved to: {output_file}")


def plot_scatter_continents_vs_clade_size(data_file: str, output_file: str):
    """Creates and saves a scatterplot of Number of Continents vs Log10 Sister Clade Size."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_file, sep="\t")

    # Check if required columns exist
    if "Number of continents" not in df.columns or "Sister Clade Size" not in df.columns:
        raise ValueError("Missing required columns in the dataset!")

    # Log-transform the Sister Clade Size column
    df["Log10 Sister Clade Size"] = np.log10(df["Sister Clade Size"] + 1)  # Avoid log(0) issues

    # Create scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Number of continents", y="Log10 Sister Clade Size")

    # Set labels and title
    plt.xlabel("Number of Continents")
    plt.ylabel("Log10 Number of members in a sister clade")
    plt.title("Scatterplot of Number of Continents vs. Number of members in a sister clade")

    # Save the figure
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Scatterplot saved to: {output_file}")


def plot_boxplot_continents_vs_clade_size(data_file: str, output_file: str):
    """Creates and saves a boxplot of Number of Continents vs Log10 Sister Clade Size,
       with individual data points displayed as a swarmplot overlay."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_file, sep="\t")

    # Check if required columns exist
    if "Number of continents" not in df.columns or "Sister Clade Size" not in df.columns:
        raise ValueError("Missing required columns in the dataset!")

    # Log-transform the Sister Clade Size column
    df["Log10 Sister Clade Size"] = np.log10(df["Sister Clade Size"] + 1)  # Avoid log(0) issues

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot boxplot
    sns.boxplot(data=df, x="Number of continents", y="Log10 Sister Clade Size", showfliers=False, width=0.6,
                boxprops={'facecolor': 'lightgray'})

    # Overlay with jittered scatterplot (swarm)
    sns.stripplot(data=df, x="Number of continents", y="Log10 Sister Clade Size", color="black", jitter=0.2, alpha=0.7,
                  size=5)

    # Set labels and title
    plt.xlabel("Number of Continents")
    plt.ylabel("Log10 Number of Members in a Sister Clade")
    plt.title("Boxplot of Number of Continents vs. Number of Members in a Sister Clade")

    # Save the figure
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Boxplot with individual points saved to: {output_file}")


def extract_kenya_sister_clades_countries(input_file: str, metadata_df: pd.DataFrame, output_file: str):
    """Extract rows where Sister Clade Size <= 1000, count occurrences of countries,
       track Kenya Samples, and add continent information."""

    # Load the dataset
    df = pd.read_csv(input_file, sep="\t")

    # Filter rows where 'Sister Clade Size' <= 1000
    filtered_df = df[df["Sister Clade Size"] <= 1000]

    # Create a dictionary mapping countries to continents from metadata
    country_to_continent = metadata_df.set_index("Country")["Continent"].to_dict()

    # Dictionary to count occurrences of each country
    country_counts = Counter()
    country_kenya_samples = {}  # Store Kenya Samples per country

    for _, row in filtered_df.iterrows():
        kenya_sample = row["Kenya Sample"]
        country_count_str = row["Country Counts"]

        if pd.notna(country_count_str):  # Ensure the value is not NaN
            country_entries = country_count_str.split(", ")
            for entry in country_entries:
                country, count = entry.rsplit(": ", 1)  # Split country and count
                country_counts[country] += int(count)  # Accumulate the count

                # Track Kenya Samples for each country
                if country not in country_kenya_samples:
                    country_kenya_samples[country] = set()
                country_kenya_samples[country].add(kenya_sample)

    # Convert to DataFrame
    country_data = []
    for country, total_members in country_counts.items():
        continent = country_to_continent.get(country, "unknown")  # Assign continent or Unknown
        kenya_samples = sorted(country_kenya_samples[country])  # Sort for consistency
        country_data.append([
            country,
            continent,
            total_members,
            len(kenya_samples),  # Number of Kenya Samples
            ", ".join(kenya_samples)  # Names of Kenya Samples
        ])

    # Create a DataFrame
    country_df = pd.DataFrame(
        country_data,
        columns=["Country", "Continent", "Total Members", "Number of Kenya Samples", "Kenya Sample Names"]
    )

    # Sort by total members in descending order
    country_df = country_df.sort_values(by="Total Members", ascending=False)

    # Save to file
    country_df.to_csv(output_file, sep="\t", index=False)

    print(f"✅ Extracted Kenya sister clade country counts saved to: {output_file}")


def plot_scatter_kenya_samples_vs_total_members(data_file: str, output_file: str):
    """Creates and saves a scatterplot of Number of Kenya Samples vs Log10 Total Members,
       colored by Continent."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_file, sep="\t")

    # Check if required columns exist
    required_columns = {"Number of Kenya Samples", "Total Members", "Continent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in the dataset: {required_columns - set(df.columns)}")

    # Log-transform the 'Total Members' column
    df["Log10 Total Members"] = np.log10(df["Total Members"] + 1)  # Avoid log(0) issues

    # Define a color palette for continents
    continent_palette = {
        "Asia": "#E74C3C",  # Red
        "Europe": "#3498DB",  # Blue
        "N_America": "#2ECC71",  # Green
        "S_America": "#F1C40F",  # Yellow
        "No_Match": "black",  # Black
        "Africa": "#8B4513",  # Dark Brown
        "Oceania": "#E67E22"  # Orange
    }

    # Create scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Number of Kenya Samples",
        y="Log10 Total Members",
        hue="Continent",
        palette=continent_palette,
        edgecolor="black"
    )

    # Set labels and title
    plt.xlabel("Number of Kenya Samples")
    plt.ylabel("Log10 Total Members from all the countries in the sister clades")
    plt.title("Scatterplot of Number of Kenya Samples vs. Total Country Members for all the countries")

    # Save the figure
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Scatterplot saved to: {output_file}")


def plot_clustered_heatmap_kenya_sister_clades(data_file: str, metadata_file: str, output_file: str):
    """Creates a clustered heatmap of Kenya samples (rows) vs country counts (columns),
       for 'Sister Clade Size' <= 1000. Uses log normalization, hierarchical clustering,
       and continent annotation as colored boxes aligned below the heatmap, with country labels below them."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_file, sep="\t")
    df.columns = df.columns.str.strip()  # Strip whitespace in column names

    # Ensure "Kenya Sample" exists
    if "Kenya Sample" not in df.columns:
        raise KeyError("Column 'Kenya Sample' is missing in the dataset!")

    # Load metadata for country-to-continent mapping
    metadata_df = pd.read_csv(metadata_file, sep="\t")  # Ensure 'Country' and 'Continent' columns exist
    country_to_continent = dict(zip(metadata_df["Country"], metadata_df["Continent"]))

    # Define continent color palette
    continent_palette = {
        "Asia": "#E74C3C",  # Red
        "Europe": "#3498DB",  # Blue
        "N_America": "#2ECC71",  # Green
        "S_America": "#F1C40F",  # Yellow
        "No_Match": "black",  # Black
        "Africa": "#8B4513",  # Dark Brown
        "Oceania": "#E67E22"  # Orange
    }

    # Filter rows where 'Sister Clade Size' <= 1000
    df_filtered = df[df["Sister Clade Size"] <= 1000].copy()

    # Identify metadata columns
    metadata_columns = {
        "Kenya Sample", "Sister Clade Size", "Number of continents",
        "Sister Clade Continents", "Continent Counts", "Number of countries",
        "Sister Clade Countries", "Country Counts", "Sister Clade Members",
        "Sister Clade Members with known countries",
        "Sister Clade Members with known countries size",
        "Sister Clade Members with known countries ratio"
    }

    continents = {"Africa", "Asia", "Europe", "N_America", "No_Match", "Oceania", "S_America"}
    columns_to_exclude = metadata_columns | continents
    country_columns = [col for col in df_filtered.columns if col not in columns_to_exclude]

    # Ensure at least some country columns exist
    if not country_columns:
        raise ValueError("No valid country columns found in the dataset!")

    # Subset only the required columns
    heatmap_data = df_filtered.set_index("Kenya Sample")[country_columns]
    heatmap_data = heatmap_data.fillna(0)  # Convert NaN to 0
    heatmap_data = np.log10(heatmap_data + 1)  # Log10 Normalization

    # Remove rows & columns with all zeros
    heatmap_data = heatmap_data.loc[(heatmap_data != 0).any(axis=1), (heatmap_data != 0).any(axis=0)]

    # Extract continent information for country columns
    country_continents = [country_to_continent.get(col, "No_Match") for col in heatmap_data.columns]
    continent_colors = [continent_palette.get(continent, "black") for continent in country_continents]

    # Plot heatmap with hierarchical clustering
    plt.figure(figsize=(20, 18))  # Increased figure size
    g = sns.clustermap(
        heatmap_data,
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="gray",
        standard_scale=1,  # Normalize across columns
        figsize=(20, 18),
        method="ward",  # Clustering method
        metric="euclidean",  # Distance metric
        col_cluster=True,  # Cluster columns (countries)
        row_cluster=True  # Cluster rows (Kenya samples)
    )

    # Extract correct clustered order of rows (Kenya Samples)
    row_order = g.dendrogram_row.reordered_ind
    col_order = g.dendrogram_col.reordered_ind  # Correct order of columns (countries)

    # ✅ Reorder heatmap_data based on clustering results
    heatmap_data = heatmap_data.iloc[row_order, :]
    reordered_country_names = heatmap_data.columns[col_order]
    reordered_colors = [continent_colors[i] for i in col_order]  # Reorder colors to match clustering

    # ✅ Move continent annotation BELOW the heatmap
    ax_continent = g.fig.add_axes([g.ax_heatmap.get_position().x0,  # Match heatmap width
                                   g.ax_heatmap.get_position().y0 - 0.06,  # Move lower
                                   g.ax_heatmap.get_position().width,  # Match heatmap width
                                   0.02])  # Small height

    ax_continent.set_xticks(np.arange(len(col_order)) + 0.5)
    ax_continent.set_xticklabels([])  # Hide labels
    ax_continent.set_yticks([])
    ax_continent.set_frame_on(False)

    # ✅ Draw continent annotation boxes
    for x, color in enumerate(reordered_colors):
        ax_continent.add_patch(Rectangle((x, 0), 1, 1, color=color, transform=ax_continent.transData, clip_on=False))

    # ✅ Move country labels BELOW continent annotation
    ax_countries = g.fig.add_axes([g.ax_heatmap.get_position().x0,  # Match heatmap width
                                    g.ax_heatmap.get_position().y0 - 0.10,  # Move even lower
                                    g.ax_heatmap.get_position().width,
                                    0.02])  # Small height for labels

    ax_countries.set_xticks(np.arange(len(col_order)) + 0.5)
    ax_countries.set_xticklabels(reordered_country_names, rotation=90, fontsize=8)
    ax_countries.set_yticks([])
    ax_countries.set_frame_on(False)

    # ✅ Add a legend for continents
    legend_patches = [Patch(facecolor=color, label=continent) for continent, color in continent_palette.items()]
    g.ax_heatmap.legend(handles=legend_patches, title="Continents", bbox_to_anchor=(1.05, 1), loc="upper left")

    # ✅ Save the figure
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Clustered Heatmap with Continent Annotations saved to: {output_file}")


# Run the pipeline
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    # Paths
    wd = "/mnt/c/crassvirales/angela_klebsiella_project"
    data_dir = f"{wd}/data"
    results_dir = f"{wd}/results"
    figures_dir = f'{results_dir}/figures'

    metadata_file = f"{data_dir}/Klebsiella_tree_metadata.xlsx"
    tree_file = f"{data_dir}/matrix.tab.tree.phangorn.nwk"
    output_metadata_tsv = f"{data_dir}/Klebsiella_tree_metadata.tsv"

    output_kenya_clades = f"{results_dir}/kenya_sister_clades.tsv"
    output_kenya_sister_clades_countries = f"{results_dir}/kenya_sister_clades_countries.tsv"

    output_scatterplot_countries_vs_clade_size = f'{figures_dir}/scatterplot_countries_vs_clade_size.png'
    output_scatterplot_continents_vs_clade_size = f'{figures_dir}/scatterplot_continents_vs_clade_size.png'
    output_boxplot_continents_vs_clade_size = f'{figures_dir}/boxplot_continents_vs_clade_size.png'

    output_scatterplot_kenya_samples_vs_total_members = f"{figures_dir}/scatterplot_number_of_kenya_samples_vs_total_country_members_with_continents.png"
    output_heatmap = f"{figures_dir}/heatmap_kenya_sample_vs_countries.png"

    # Configure logging
    logging.basicConfig(
        filename=f"{results_dir}/analyse_tree.log",
        filemode="w",  # Overwrite log file each run
        level=logging.DEBUG,  # Change to logging.INFO for less verbosity
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    metadata_df = merge_metadata(metadata_file, output_metadata_tsv)
    tree = annotate_tree(tree_file, metadata_df)

    # Finding non-Kenyan sister clades
    kenya_results, unique_countries, unique_continents = find_kenya_sister_clades(tree)
    save_kenya_sister_clades(kenya_results, unique_countries, unique_continents, output_kenya_clades)

    # Save high-quality tree visualization
    save_annotated_tree(tree, "results")

    print(f"✅ Metadata saved to {output_metadata_tsv}")
    print(f"✅ Kenya sister clades saved to {output_kenya_clades}")

    plot_scatter_countries_vs_clade_size(output_kenya_clades, output_scatterplot_countries_vs_clade_size)
    plot_scatter_continents_vs_clade_size(output_kenya_clades, output_scatterplot_continents_vs_clade_size)
    plot_boxplot_continents_vs_clade_size(output_kenya_clades, output_boxplot_continents_vs_clade_size)

    # Extract country counts for small clades (<=1000 members)
    extract_kenya_sister_clades_countries(output_kenya_clades, metadata_df, output_kenya_sister_clades_countries)

    plot_scatter_kenya_samples_vs_total_members(output_kenya_sister_clades_countries,
                                                output_scatterplot_kenya_samples_vs_total_members)

    plot_clustered_heatmap_kenya_sister_clades(output_kenya_clades, output_metadata_tsv, output_heatmap)
