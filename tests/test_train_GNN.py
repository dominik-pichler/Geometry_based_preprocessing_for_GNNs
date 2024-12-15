import pytest
from torch_geometric.data import Data
import torch
from src.models import GINe
from src.training import train_homo, get_model


def test_get_model():
    # Mock input data
    sample_batch = Data(x=torch.rand((10, 16)), edge_attr=torch.rand((20, 5)))
    config = {
        "n_gnn_layers": 2,
        "n_hidden": 32,
        "dropout": 0.5,
        "final_dropout": 0.3,
    }
    args = {"model": "gin", "emlps": False}

    # Call get_model function
    model = get_model(sample_batch, config, args)

    # Assertions
    assert isinstance(model, GINe), "Model is not an instance of GINe"
    assert model.num_features == 16, "Model feature size mismatch"


def test_train_homo():
    from torch.utils.data import DataLoader

    # Mock data and loader
    data = Data(
        x=torch.rand((100, 16)),
        edge_index=torch.randint(0, 100, (2, 300)),
        edge_attr=torch.rand((300, 5)),
        y=torch.randint(0, 2, (100,))
    )
    loader = DataLoader([data], batch_size=10)

    # Mock model and optimizer
    model = GINe(num_features=16, num_gnn_layers=2, n_classes=2, n_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Call train_homo function (basic test to ensure no errors)
    try:
        train_homo(
            loader,
            loader,
            loader,
            torch.arange(100),
            torch.arange(100),
            torch.arange(100),
            model,
            optimizer,
            loss_fn,
            args={"tqdm": False},
            config={"epochs": 1},
            device="cpu",
            val_data=None,
            te_data=None,
            data_config=None,
        )
    except Exception as e:
        pytest.fail(f"train_homo raised an exception: {e}")
