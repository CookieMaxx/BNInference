#data loading and gene selection functions

import numpy as np
import pandas as pd
from typing import List

#loading expression data from CSV file and prepare for analysis
def load_and_prepare_data(csv_file: str) -> pd.DataFrame:

    print(f"Loading expression data from {csv_file}...")

    #load data
    data = pd.read_csv(csv_file, index_col=0)  #first column (locus tags) as index

    print(f"Loaded data shape: {data.shape}")
    print(f"Genes (rows): {data.shape[0]}")
    print(f"Samples (columns): {data.shape[1]}")
    print(f"Sample names: {list(data.columns)}")
    print(f"First few gene names: {list(data.index[:5])}")

    #transpose so genes are columns and samples are rows (required for sklearn)
    expression_df = data.T

    print(f"After transpose - Samples (rows): {expression_df.shape[0]}, Genes (columns): {expression_df.shape[1]}")

    #checking for any missing values
    missing_values = expression_df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values. Filling with 0.")
        expression_df = expression_df.fillna(0)

    #checking for negative values (counts should be non-negative)
    negative_values = (expression_df < 0).sum().sum()
    if negative_values > 0:
        print(f"Warning: Found {negative_values} negative values. Setting to 0.")
        expression_df = expression_df.clip(lower=0)

    return expression_df

#select subset of genes for analysis based on specified method
def select_genes_for_analysis(expression_df: pd.DataFrame,
                              n_genes: int = 100,
                              selection_method: str = 'variance') -> List[str]:

    print(f"Selecting {n_genes} genes using method: {selection_method}")

    total_genes = len(expression_df.columns)
    if n_genes >= total_genes:
        print(f"Requested {n_genes} genes >= total {total_genes} genes. Using all genes.")
        return list(expression_df.columns)

    if selection_method == 'variance':
        #genes with highest variance (most variable)
        gene_variance = expression_df.var()
        selected_genes = gene_variance.nlargest(n_genes).index.tolist()
        print(
            f"Selected genes with variance range: {gene_variance[selected_genes].min():.2f} - {gene_variance[selected_genes].max():.2f}")

    elif selection_method == 'mean':
        #genes with highest mean expression
        gene_means = expression_df.mean()
        selected_genes = gene_means.nlargest(n_genes).index.tolist()
        print(
            f"Selected genes with mean expression range: {gene_means[selected_genes].min():.2f} - {gene_means[selected_genes].max():.2f}")

    elif selection_method == 'random':
        #random selection
        np.random.seed(42)
        selected_genes = np.random.choice(expression_df.columns, size=n_genes, replace=False).tolist()
        print(f"Randomly selected {len(selected_genes)} genes")

    elif selection_method == 'first':
        #first n genes
        selected_genes = list(expression_df.columns[:n_genes])
        print(f"Selected first {len(selected_genes)} genes")

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    print(f"Selected genes: {selected_genes[:5]}..." + (
        f" and {len(selected_genes) - 5} more" if len(selected_genes) > 5 else ""))
    return selected_genes