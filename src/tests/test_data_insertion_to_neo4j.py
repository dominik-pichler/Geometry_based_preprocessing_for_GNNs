import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data_insertion_to_neo4j import insert_and_connect_data



mock_data = {
    'Timestamp': pd.date_range(start='1/1/2020', periods=5),
    'From Bank': ['BankA', 'BankB', 'BankC', 'BankD', 'BankE'],
    'To Bank': ['BankF', 'BankG', 'BankH', 'BankI', 'BankJ'],
    'Account': [123, 456, 789, 101, 112],
    'Amount Received': [1000, 1500, 2000, 2500, 3000],
    'Receiving Currency': ['USD', 'EUR', 'GBP', 'JPY', 'AUD'],
    'Amount Paid': [950, 1450, 1950, 2450, 2950],
    'Payment Currency': ['USD', 'EUR', 'GBP', 'JPY', 'AUD'],
    'Payment Format': ['Cash', 'Card', 'Transfer', 'Check', 'Online'],
    'Is Laundering': [0, 0, 1, 0, 1]
}


# Test for insert_and_connect_data function
def test_insert_and_connect_data():
    mock_tx = MagicMock()
    row_data = {
        'id': f"{mock_data['Timestamp'][0]}_{mock_data['From Bank'][0]}_{mock_data['Account'][0]}_{mock_data['To Bank'][0]}_{mock_data['Account'][0]}",
        'Timestamp': mock_data['Timestamp'][0],
        'Amount_Received': mock_data['Amount Received'][0],
        'Receiving_Currency': mock_data['Receiving Currency'][0],
        'Amount_Paid': mock_data['Amount Paid'][0],
        'Payment_Currency': mock_data['Payment Currency'][0],
        'Payment_Format': mock_data['Payment Format'][0],
        'Is_Laundering': mock_data['Is Laundering'][0],
        'From_Bank': mock_data['From Bank'][0],
        'To_Bank': mock_data['To Bank'][0],
        'From_Account': f"{mock_data['From Bank'][0]}_{mock_data['Account'][0]}",
        'To_Account': f"{mock_data['To Bank'][0]}_{mock_data['Account'][0]}"
    }

    insert_and_connect_data(mock_tx, row_data)

    # Check if the query was run with the correct parameters
    mock_tx.run.assert_called_once()
    args, kwargs = mock_tx.run.call_args
    assert kwargs == row_data

# Run the tests
if __name__ == "__main__":
    pytest.main()
