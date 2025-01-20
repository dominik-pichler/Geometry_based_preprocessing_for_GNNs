import pandas as pd
import numpy as np
import torch
import logging
from sklearn.preprocessing import LabelEncoder

# Configure the basic settings for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

# Create a logger
logger = logging.getLogger(__name__)

def preprocess_uploaded_data(df_uploadData: pd.DataFrame) -> torch.tensor:

    # Querying the DataFrame
    df_edges = df_uploadData.assign(
        from_id=df_uploadData["From_Account"],
        to_id=df_uploadData["To_Account"],
        time_of_transaction=df_uploadData["Timestamp"],
        amount_paid=df_uploadData["Amount_Paid"],
        currency_paid=df_uploadData["Payment_Currency"],
        payment_format=df_uploadData["Payment_Format"],
        is_laundering=None  # Add a column with NULL values (None in Python)
    )[[
        "from_id",
        "to_id",
        "time_of_transaction",
        "amount_paid",
        "currency_paid",
        "payment_format",
        "is_laundering"
    ]]

    df_edges = df_edges.rename(columns={
        'amount_paid': 'Amount_Received',
        'payment_format': 'Payment_Format',
        'is_laundering': 'Is_Laundering',
        'currency_paid': 'Received_Currency',
        'time_of_transaction': 'Timestamp'
    })

    df_edges['Timestamp'] = pd.to_datetime(df_edges['Timestamp'], format='%d.%m.%y %H:%M')
    # Datatype Conversion
    df_edges['Timestamp'] = ( df_edges['Timestamp'] - df_edges['Timestamp'].min())  # Normailization to turn dates into floats
    df_edges = df_edges.sort_values(by='Timestamp', ascending=True)

    df_edges['from_id'] = df_edges['from_id'].astype(str)
    df_edges['to_id'] = df_edges['to_id'].astype(str)
    le = LabelEncoder();
    df_edges['from_id'] = le.fit_transform(df_edges['from_id']);
    df_edges['to_id'] = le.fit_transform(df_edges['to_id'])
    df_edges['from_id'] = pd.to_numeric(df_edges['from_id'], errors='coerce')
    df_edges['to_id'] = pd.to_numeric(df_edges['to_id'], errors='coerce')

    # Specifying relevant features for the following
    edge_features = ['Timestamp', 'Amount_Received', 'Received_Currency', 'Payment_Format']
    node_features = ['Feature']

    max_n_id = df_edges.loc[:, ['from_id','to_id']].to_numpy().max() + 1  # Determining the number of edges for future index determination
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(
        max_n_id)})  # 2 x number of edges matrix with default features of 1.0. Needed for #TODO

    df_edges['Timestamp'] = df_edges['Timestamp'].dt.total_seconds()
    timestamps = torch.tensor(df_edges['Timestamp'].to_numpy(), dtype=torch.int64)
    logging.info(f"Timestamp-Tensor: {timestamps}***")


    logging.info(f"Number of nodes (holdings doing transactions) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    # encoder to transform categorical variables to numerical ones (in order to be able to process via numpy.
    label_encoder = LabelEncoder()
    df_edges['Payment_Format'] = label_encoder.fit_transform(df_edges['Payment_Format'])
    df_edges['Received_Currency'] = label_encoder.fit_transform(df_edges['Received_Currency'])
    df_edges['Amount_Received'] = df_edges['Amount_Received'].astype(float)
    df_edges.to_csv("TestEdges.csv")
    node_tensor = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_tensor =  edge_index = torch.tensor(df_edges[['from_id', 'to_id']].to_numpy().T, dtype=torch.long)
    edge_features = torch.tensor(df_edges[['Timestamp', 'Amount_Received', 'Payment_Format']].to_numpy(),
                                 dtype=torch.float)
    return node_tensor,edge_tensor, edge_features

