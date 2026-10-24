import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

def compute_spectrogram_or_cwt(lead_signal: np.ndarray, fs: int = 100, nperseg: int = 64, noverlap: int = 48):
    """
    Computes time-frequency energy distribution of an ECG lead.
    Returns: time array, frequency array, power spectral density (dB).
    """
    f, t, Sxx = signal.spectrogram(
        lead_signal,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='magnitude'
    )
    # Convert to log-scale magnitude (dB)
    sxx_db = 10 * np.log10(Sxx + 1e-8)
    return t, f, sxx_db

def plot_time_frequency_scalogram(lead_signal: np.ndarray, lead_name: str, fs: int = 100):
    """
    Generates an interactive Time-Frequency Scalogram using Plotly.
    """
    t, f, sxx_db = compute_spectrogram_or_cwt(lead_signal, fs=fs)
    
    fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=f,
            z=sxx_db,
            colorscale='Viridis',
            colorbar=dict(title='Power (dB)')
        )
    )
    fig.update_layout(
        title=f"Time-Frequency Energy Scalogram — Lead {lead_name}",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_standard_12lead_ecg(ecg: np.ndarray, lead_names: list, fs: int = 100):
    """
    Renders the classic 12-lead ECG paper recording (4 columns x 3 rows).
    ecg: shape (1000, 12)
    """
    cols = 4
    rows = 3
    time_axis = np.linspace(0, 2.5, int(fs * 2.5))  # 2.5s segment per lead column
    
    lead_matrix = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[lead for row in lead_matrix for lead in row],
        vertical_spacing=0.08,
        horizontal_spacing=0.02
    )
    
    for r_idx, row in enumerate(lead_matrix):
        for c_idx, lead in enumerate(row):
            lead_idx = lead_names.index(lead)
            segment_start = int(c_idx * 2.5 * fs)
            segment_end = segment_start + len(time_axis)
            lead_trace = ecg[segment_start:segment_end, lead_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=lead_trace,
                    mode='lines',
                    line=dict(color='#111111', width=1.3),
                    showlegend=False
                ),
                row=r_idx + 1,
                col=c_idx + 1
            )
            
    fig.update_layout(
        plot_bgcolor="#FFF5F5",  # Clinical pink grid background
        paper_bgcolor="#FFFFFF",
        height=520,
        title_text="Standard 12-Lead Clinical Paper Tracing (25 mm/s, 10 mm/mV Calibration)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add subtle gridlines simulating standard ECG calibration blocks
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#FFD1D1", dtick=0.2)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#FFD1D1", dtick=0.5)
    
    return fig