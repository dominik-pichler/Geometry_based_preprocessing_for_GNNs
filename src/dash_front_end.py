import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loader_GNN import get_data
from training import train_gnn
from inference import infer_gnn
import json
import io
import sys

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("GNN Training and Inference Dashboard"),

    html.Div([
        html.Label("Seed:"),
        dcc.Input(id="seed-input", type="number", value=42),
    ]),

    html.Div([
        html.Label("Mode:"),
        dcc.RadioItems(
            id='mode-selection',
            options=[
                {'label': 'Training', 'value': 'training'},
                {'label': 'Inference', 'value': 'inference'}
            ],
            value='training'
        ),
    ]),

    html.Button("Run", id="run-button"),

    html.Div(id="output-area"),

    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="loading-output")
    )
])


@app.callback(
    [Output("output-area", "children"),
     Output("loading-output", "children")],
    [Input("run-button", "n_clicks")],
    [State("seed-input", "value"),
     State("mode-selection", "value")]
)
def run_gnn(n_clicks, seed, mode):
    if n_clicks is None:
        return "", ""

    # Capture print outputs
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    # Setup logging to capture logs
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)

    try:
        # Load config
        with open('data_config.json', 'r') as config_file:
            data_config = json.load(config_file)

        # Setup
        logger_setup()
        set_seed(seed)

        # Create a mock args object
        class Args:
            def __init__(self):
                self.seed = seed
                self.inference = (mode == 'inference')

        args = Args()

        # Get data
        logging.info("Retrieving data")
        t1 = time.perf_counter()
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
        t2 = time.perf_counter()
        logging.info(f"Retrieved data in {t2 - t1:.2f}s")

        if not args.inference:
            # Training
            logging.info(f"Running Training")
            train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
        else:
            # Inference
            logging.info(f"Running Inference")
            infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    # Restore stdout
    sys.stdout = old_stdout

    # Get captured output
    output = buffer.getvalue()
    log_output = log_capture_string.getvalue()

    return html.Div([
        html.H3("Output:"),
        html.Pre(output),
        html.H3("Logs:"),
        html.Pre(log_output)
    ]), ""


if __name__ == "__main__":
    app.run_server(debug=True)