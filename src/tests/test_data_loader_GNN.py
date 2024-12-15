import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from src.data_loader_GNN import get_data
from datetime import datetime
@pytest.fixture
def mock_args():
    class Args:
        GBPre = False
        ports = False
        tds = False
        model = 'some_model'
        reverse_mp = False

    return Args()


@patch('py2neo.Graph')
def test_get_data(mock_graph, mock_args):
    # Mock the database query results
    mock_result = [
        {'from_id': '1', 'to_id': '2', 'time_of_transaction': datetime(1970, 1, 1), 'amount_paid': 500.0,
         'currency_paid': 'USD', 'payment_format': 'online', 'is_laundering': 0},
        {'from_id': '2', 'to_id': '3', 'time_of_transaction': datetime(1970, 1, 1), 'amount_paid': 300.0,
         'currency_paid': 'EUR', 'payment_format': 'offline', 'is_laundering': 1}
    ]

    # Setup mock graph.run() to return a mock result set
    mock_graph.return_value.run.return_value = [MagicMock(**record) for record in mock_result]

    # Call the function under test
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(mock_args)

    # Check if the returned data objects have expected properties
    assert isinstance(tr_data.x, torch.Tensor)
    assert isinstance(tr_data.y, torch.Tensor)
    assert isinstance(tr_data.edge_index, torch.Tensor)
    assert isinstance(tr_data.edge_attr, torch.Tensor)

    # Check if indices are tensors and match expected shapes or conditions
    assert isinstance(tr_inds, torch.Tensor)
    assert isinstance(val_inds, torch.Tensor)
    assert isinstance(te_inds, torch.Tensor)

if __name__ == "__main__":
    pytest.main()