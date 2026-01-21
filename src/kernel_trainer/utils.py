"""
Utilities
"""
import plotly.express as px


def visualize_3d(data):
    """
    Visualize the 3D distribution of points in an ellipsoid.

    Args:
        points
    """

    # Scatter plot of generated points
    fig = px.scatter_3d(
        data, x="x0", y="x1", z="x2", color="y", color_continuous_scale="rdbu"
    )

    return fig
