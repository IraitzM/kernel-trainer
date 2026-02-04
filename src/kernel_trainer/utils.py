"""
Utilities
"""
import os
import pandas as pd
import plotly.express as px

from kernel_trainer.config import logger

def visualize_3d(data):
    """
    Create a 3D scatter plot of an ellipsoid dataset.

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        Dataset containing columns ``x0``, ``x1``, ``x2`` and a target column
        ``y`` which is used to colour the points.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly figure object representing the 3D scatter plot.
    """

    # Scatter plot of generated points
    fig = px.scatter_3d(
        data, x="x0", y="x1", z="x2", color="y", color_continuous_scale="rdbu"
    )

    return fig

def collect_results(path: str, metric:str, summarize:bool = True) -> pd.DataFrame:
    """
    Collect and aggregate results from multiple CSV files organized by dataset categories.
    
    This function reads CSV files from a specified directory, extracts metric values for
    different kernel methods across dataset categories, and returns a consolidated DataFrame.
    
    Parameters
    ----------
    path : str
        Directory path containing the results CSV files. Files should be named to include
        category identifiers (e.g., '1a', '1b', '1c', etc.).
    metric : str
        The column name in the CSV files to extract as the metric of interest
        (e.g., 'accuracy', 'f1_score', 'auc').
    summarize : bool, optional
        If True, only process files containing "compact" in their filename.
        If False, process all matching files. Default is True.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'Dataset': Category identifier (1a, 1b, 1c, 2a, 2b, 2c, 3a, 3b, 3c)
        - 'linear': Metric value for linear kernel method
        - 'poly': Metric value for polynomial kernel method
        - 'rbf': Metric value for RBF kernel method
        - 'z': Metric value for Z kernel method
        - 'zy': Metric value for ZY kernel method
        - 'zz-full': Metric value for ZZ-full kernel method
        - 'best': Metric value for best PennyLane method
        
        Missing or invalid values ('--' or NaN) are converted to None.
    
    Notes
    -----
    - Files not found in the specified path trigger a warning but do not halt execution.
    - Metric values are converted to float; non-numeric values are treated as None.
    - The function processes files from nine dataset categories: 1a-1c, 2a-2c, 3a-3c.
    - Seven kernel methods are extracted from each file: linear, poly, rbf, Z, ZY, ZZ-full,
        and best (pennylane).
    
    Raises
    ------
    FileNotFoundError
        If the specified path directory does not exist.
    """
    # Define the categories
    categories = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c']

    # Define the methods we want to extract
    methods = ['linear', 'poly', 'rbf', 'Z', 'ZY','ZZ-full', 'best (pennylane)']

    # Initialize a list to store rows
    data_rows = []

    # Process each category
    for category in categories:
        files = [x for x in os.listdir(path) if category in x]
        for filename in files:
            if summarize and "compact" not in filename:
                continue

            # Check if file exists
            if not os.path.exists(f"{path}/{filename}"):
                logger.warning(f"{filename} not found, skipping...")
                continue

            # Read the CSV file
            df = pd.read_csv(f"{path}/{filename}")

            # Create a row for this category
            row = {'Dataset': category}

            # Extract values for each method
            for method in methods:
                # Find the row for this method
                method_row = df[df['Method'] == method]

                if not method_row.empty:
                    value = method_row[metric].values[0]
                    # Handle '--' values
                    if value == '--' or pd.isna(value):
                        row[method] = None
                    else:
                        row[method] = float(value)
                else:
                    row[method] = None

            data_rows.append(row)

    # Create the final DataFrame
    result_df = pd.DataFrame(data_rows)

    # Rename columns to match your desired output
    column_mapping = {
        'Dataset': 'Dataset',
        'linear': 'linear',
        'poly': 'poly',
        'rbf': 'rbf',
        'Z': 'z',
        'ZY': 'zy',
        'ZZ-full': 'zz-full',
        'best (pennylane)': 'best'
    }
    result_df = result_df.rename(columns=column_mapping)

    # Reorder columns
    if result_df.empty:
        return pd.DataFrame(columns=['Dataset', 'linear', 'poly', 'rbf', 'z', 'zy', 'zz-full', 'best'])

    # else
    return result_df[['Dataset', 'linear', 'poly', 'rbf', 'z', 'zy', 'zz-full', 'best']]
