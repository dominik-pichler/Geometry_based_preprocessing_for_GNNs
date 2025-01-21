import pandas as pd
import torch
from pydantic import BaseModel, ValidationError
from typing import Literal
from src.models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from app_util import extract_param
import json
import torch.nn as nn


###### STATIC VARIABLES ######

file_path_model_configs = 'src/model_settings.json'


class DataFrameSchema(BaseModel):
    Timestamp: str
    From_Bank: int
    From_Account: str
    To_Bank: int
    To_Account: str
    Amount_Received: float
    Receiving_Currency: str
    Amount_Paid: float
    Payment_Currency: str
    Payment_Format: str


def check_and_validate_uploadData(df_uploadData:pd.DataFrame) -> tuple[bool,str]:
    '''
    Function utilizing pydantic to check if the uploaded data comes in the right shape and style!

    Parameters
    ----------
    df_uploadData : Uploaded tabular (hopefully) data from the user that should be scored.

    Returns: True if uploadData is valid, False otherwise.
    -------
    '''
 # Expected column names and number of columns
    expected_columns = [
            "Timestamp", "From_Bank", "From_Account", "To_Bank", "To_Account",
            "Amount_Received", "Receiving_Currency", "Amount_Paid",
            "Payment_Currency", "Payment_Format"
    ]

    # Check number of columns
    if len(df_uploadData.columns) != len(expected_columns):
        return False,"Number of columns does not match!"

        # Check column names
    if list(df_uploadData.columns) != expected_columns:
        print("Column names do not match!")
        return False,"Column names do not match!"

        # Validate each row using Pydantic schema
    for index, row in df_uploadData.iterrows():
        try:
            # Convert row to dictionary and validate using Pydantic model
            DataFrameSchema(
                Timestamp=row["Timestamp"],
                From_Bank=row["From_Bank"],
                From_Account=row["From_Account"],
                To_Bank=row["To_Bank"],
                To_Account=row["To_Account"],
                Amount_Received=row["Amount_Received"],
                Receiving_Currency=row["Receiving_Currency"],
                Amount_Paid=row["Amount_Paid"],
                Payment_Currency=row["Payment_Currency"],
                Payment_Format=row["Payment_Format"]
            )

        except ValidationError as e:
            return False, f"Validation error in row {index}: {e}"

        return True, ""

def extract_param(param_name, params):
    return params.get(param_name)

def get_settings() -> dict:

    with open(file_path_model_configs, "r") as file:
        model_config = json.load(file)
    # Generate configurations for each model
    configs = {}
    for model_name, model_data in model_config.items():
        params = model_data["params"]
        configs[f"Config_{model_name}"] = {
            # Extract required parameters
            "lr": extract_param("lr", params),
            "n_hidden": extract_param("n_hidden", params),
            "n_gnn_layers": extract_param("n_gnn_layers", params),
            "loss": extract_param("loss", params),  # Assuming loss is always 'ce'
            "w_ce1": extract_param("w_ce1", params),
            "w_ce2": extract_param("w_ce2", params),
            "dropout": extract_param("dropout", params),
            "final_dropout": extract_param("final_dropout", params),
            # Include n_heads only if the model is 'gat', otherwise set to None
            "n_heads": extract_param("n_heads", params) if model_name == 'gat' else None
        }
    return configs
def get_model(selected_option:str):
    """
    Dynamically maps the selected option to the correct model class and instantiates it.

    Args:
        selected_option (str): The name of the model to instantiate (e.g., "GINe", "PNA").
        **kwargs: Additional arguments required for the model initialization.

    Returns:
        torch.nn.Module: An instance of the selected model class.
    """
    n_feats = 1  #TODO: should be centralized or ideally infered from the inferencedataset
    e_dim = 4    #TODO: should be centralized or ideally infered from the inferencedataset
    configs = get_settings()

    if selected_option == "Graph Attention Network with edge features":
        config_gat = configs["Config_gat"]
        model = GATe(
            num_features=n_feats, num_gnn_layers=config_gat['n_gnn_layers'], n_classes=2,
            n_hidden=round(config_gat['n_hidden']), n_heads=round(config_gat['n_heads']),
            edge_updates=False, edge_dim=e_dim,
            dropout=config_gat['dropout'], final_dropout=config_gat['final_dropout']
        )
    elif selected_option == "Graph Isomorphism Network with node features":
        config_gin = configs["Config_gin"]

        model = GINe(
            num_features=n_feats, num_gnn_layers=config_gin['n_gnn_layers'], n_classes=2,
            n_hidden=round(config_gin['n_hidden']), residual=False, edge_updates=True, edge_dim=e_dim,
            dropout=config_gin['n_hidden'], final_dropout=config_gin['final_dropout']
        )

    elif selected_option == "Relational Graph Convolutional Network":
        config_rgcn = configs["Config_rgcn"]

        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config_rgcn['n_gnn_layers']),
            n_classes=2, n_hidden=round(config_rgcn['n_hidden']),
            edge_update=False, dropout=config_rgcn['dropout'], final_dropout=config_rgcn['final_dropout'], n_bases=None  # (maybe)
        )
    else:
        raise ValueError(f"Selected option {selected_option} is not supported.")

    return model


def infer_on_new_data(node_tensor,edge_tensor, edge_features,selected_option):
    # Example data
    num_nodes = 10
    num_features = 6  # Node feature dimension
    edge_dim = 5  # Edge feature dimension
    n_classes = 2  # Number of output classes

    # Node features: Shape (N, num_features)
    x = torch.rand(num_nodes, num_features)
    # torch.Size([100, 10])

    # Edge index: Shape (2, E) torch.Size([2, 3])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Example edges

    # Edge attributes: Shape (E, edge_dim) torch.Size([3, 5])
    edge_attr = torch.rand(edge_index.shape[1], edge_dim)

    # Initialize model with matching dimensions
    model = GATe(
        num_features=num_features,
        num_gnn_layers=2,
        n_classes=n_classes,
        n_hidden=64,
        n_heads=4,
        edge_updates=True,
        edge_dim=edge_dim,
        dropout=0.009,
        final_dropout=0.1
    )

    # Run forward pass
    with torch.no_grad():
        predictions = model(x, edge_index, edge_attr)

    threshold = 0.5
    binary_predictions = (predictions >= threshold).int()
    return binary_predictions  # Output shape depends on task (e.g., edge classification)
