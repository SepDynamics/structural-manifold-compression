import gradio as gr
import plotly.graph_objects as go
from wrappers import run_market_manifold, run_trace_manifold


def plot_bridge(market_json, trace_json):
    fig = go.Figure()

    # 1. Market Data (Left Axis) - Entropy
    if market_json and "windows" in market_json:
        market_windows = market_json["windows"]
        # Extract timeseries (using index as proxy for time if timestamp unavailable)
        x_m = [w.get("index", i) for i, w in enumerate(market_windows)]
        y_m_entropy = [w["metrics"]["entropy"] for w in market_windows]

        fig.add_trace(
            go.Scatter(
                x=x_m,
                y=y_m_entropy,
                name="Market Entropy (EUR/USD)",
                line=dict(color="blue"),
                yaxis="y1",
            )
        )

        # Highlight collapses
        collapse_x = [
            w.get("index", i)
            for i, w in enumerate(market_windows)
            if w["state"] == "collapsed"
        ]
        collapse_y = [
            w["metrics"]["entropy"] for w in market_windows if w["state"] == "collapsed"
        ]
        if collapse_x:
            fig.add_trace(
                go.Scatter(
                    x=collapse_x,
                    y=collapse_y,
                    mode="markers",
                    marker=dict(color="red", size=8, symbol="x"),
                    name="Market Collapse",
                    yaxis="y1",
                )
            )

    # 2. Trace Data (Right Axis) - Entropy
    if trace_json and "windows" in trace_json:
        trace_windows = trace_json["windows"]
        x_t = [w.get("index", i) for i, w in enumerate(trace_windows)]
        y_t_entropy = [w["metrics"]["entropy"] for w in trace_windows]

        fig.add_trace(
            go.Scatter(
                x=x_t,
                y=y_t_entropy,
                name="System Trace Entropy",
                line=dict(color="orange"),
                yaxis="y2",  # Secondary axis
            )
        )

        # Highlight collapses
        t_collapse_x = [
            w.get("index", i)
            for i, w in enumerate(trace_windows)
            if w["state"] == "collapsed"
        ]
        t_collapse_y = [
            w["metrics"]["entropy"] for w in trace_windows if w["state"] == "collapsed"
        ]
        if t_collapse_x:
            fig.add_trace(
                go.Scatter(
                    x=t_collapse_x,
                    y=t_collapse_y,
                    mode="markers",
                    marker=dict(color="darkorange", size=8, symbol="circle-open"),
                    name="Trace Collapse",
                    yaxis="y2",
                )
            )

    # Layout
    fig.update_layout(
        title="The Bridge: Universal Structural Entropy",
        xaxis=dict(title="Window Index (Time)"),
        yaxis=dict(
            title="Market Entropy",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Trace Entropy",
            titlefont=dict(color="orange"),
            tickfont=dict(color="orange"),
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
    )

    return fig


def process_market(file_obj):
    if file_obj is None:
        return None, "Upload a CSV file."

    try:
        # Create a temp file path because wrapper expects path
        # Gradio v4 passes file path directly usually, but handle both object and path
        if isinstance(file_obj, str):
            path = file_obj
        else:
            path = file_obj.name

        result = run_market_manifold(path)
        return result, f"Processed {len(result.get('windows', []))} market windows."
    except Exception as e:
        return None, f"Error: {str(e)}"


def process_trace(text, window_size):
    if not text:
        return None, "Enter text to analyze."

    try:
        result = run_trace_manifold(text, int(window_size))
        return result, f"Processed {len(result.get('windows', []))} trace windows."
    except Exception as e:
        return None, f"Error: {str(e)}"


def update_viz(market_data, trace_data):
    return plot_bridge(market_data, trace_data)


with gr.Blocks(title="The Bridge: Structural Entropy Demo") as demo:
    gr.Markdown("# The Bridge: Structural Entropy Across Domains")
    gr.Markdown(
        "Visualizing the isomorphism between **Financial Market Instability** and **AI/System Trace Decoherence**."
    )

    with gr.Row():
        with gr.Column():
            gr.Header("Left Realm: Financial Markets")
            market_file = gr.File(
                label="Upload OHLCV CSV (Time, Open, High, Low, Close, Volume)"
            )
            market_btn = gr.Button("Analyze Market Structure")
            market_status = gr.Textbox(label="Status")
            market_json = gr.JSON(label="Manifold Output", visible=False)

        with gr.Column():
            gr.Header("Right Realm: Cognitive/System Traces")
            trace_input = gr.Textbox(
                label="System Logs / AI Thoughts",
                lines=10,
                placeholder="Paste execution logs or chain-of-thought here...",
            )
            window_size = gr.Number(value=256, label="Window Size (bits)")
            trace_btn = gr.Button("Analyze Trace Structure")
            trace_status = gr.Textbox(label="Status")
            trace_json = gr.JSON(label="Manifold Output", visible=False)

    gr.Header("The Bridge Visualization")
    plot = gr.Plot()

    # Event wiring
    market_btn.click(
        process_market, inputs=[market_file], outputs=[market_json, market_status]
    )
    trace_btn.click(
        process_trace,
        inputs=[trace_input, window_size],
        outputs=[trace_json, trace_status],
    )

    # Update plot when either changes (chained)
    market_json.change(update_viz, inputs=[market_json, trace_json], outputs=[plot])
    trace_json.change(update_viz, inputs=[market_json, trace_json], outputs=[plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
