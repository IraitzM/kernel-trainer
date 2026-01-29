"""
Utilities
"""

import plotly.express as px


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
