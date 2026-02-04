import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Union, Literal

def save_matplotlib_figure(fig, format: Literal['png', 'pdf', 'svg', 'tiff'] = 'png', dpi: int = 300) -> io.BytesIO:
    """
    Saves a Matplotlib figure to a BytesIO object with high quality.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

def save_plotly_figure(fig: go.Figure, format: Literal['png', 'pdf', 'svg', 'html'] = 'png', 
                       width: int = 1200, height: int = 800, scale: float = 3.0) -> io.BytesIO:
    """
    Saves a Plotly figure to a BytesIO object. 
    Note: Requires kaleido for static image export, usually. 
    If kaleido is not installed, might need to fallback or warn.
    HTML export is always safe.
    """
    buf = io.BytesIO()
    if format == 'html':
        fig.write_html(buf)
    else:
        # For static image export from plotly
        # Use high scale for better DPI equivalent
        img_bytes = fig.to_image(format=format, width=width, height=height, scale=scale)
        buf.write(img_bytes)
        
    buf.seek(0)
    return buf
