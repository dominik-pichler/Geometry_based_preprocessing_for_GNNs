import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Assuming the functions are imported from the module
# from your_module import load_demo_data, isNaN, insert_and_connect_data

# Mock data for testing
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

# Test for load_demo_data function
def test_load_demo_data():
    with patch('pandas.read_csv') as mock_read_csv:
        # Mock the DataFrame returned by read_csv
        mock_df = pd.DataFrame(mock_data)
        mock_read_csv.return_value = mock_df

        # Call the function
        df = load_demo_data()

        # Check if the DataFrame is shuffled and has expected columns
        assert len(df) == len(mock_df)
        assert set(df.columns) == set(mock_df.columns)

        # Check if approximately 50% of the timestamps are modified (for local_test=True)
        modified_timestamps = df['Timestamp'] > mock_df['Timestamp']
        assert modified_timestamps.sum() == int(len(mock_df) * 0.5)

        # Check if approximately 30% of 'Is Laundering' is set to 1
        assert df['Is Laundering'].sum() == int(len(mock_df) * 0.3)

# Test for isNaN function
def test_isNaN():
    assert isNaN(np.nan) is True
    assert isNaN(1) is False
    assert isNaN('string') is False

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
