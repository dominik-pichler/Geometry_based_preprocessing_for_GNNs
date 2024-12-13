import random
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import numpy as np
import logging
import argparse
from util import logger_setup

logger_setup()

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--rows_to_insert', type=int, default=5000,
                    help='Number of rows to insert (default: 5000)')
parser.add_argument('--local_test', action='store_true',
                    help='Flag to indicate if this is a local test')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"Rows to insert: {args.rows_to_insert}")
print(f"Local test: {args.local_test}")


rows_to_insert = 5000  # Change this value to insert more or fewer rows
local_test = True

def load_demo_data():
    df = pd.read_csv('../data/LI-Small_Trans.csv', sep=',', nrows=args.rows_to_insert)

    if args.local_test:
        # Due to the sheer enourmous size of the dataset I had to take a subset and create a variable distribution that makes sense.
        # Randomly modify about 50% of the "Timestamp" column
        # Assuming "Timestamp" is a column in your DataFrame
        # Convert 'Timestamp' column to datetime, coercing errors to NaT
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        logging.info(f"Converted Timestamp {df['Timestamp']}")

        # df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
        # Ensure that there are no NaT values before proceeding
        if df['Timestamp'].isna().any():
            print("Warning: Some timestamps could not be converted and are set to NaT.")

        # Check if 'Timestamp' exists in columns and modify a portion of the rows
        if 'Timestamp' in df.columns:
            # Calculate number of rows to modify (50% of the DataFrame)
            num_rows_to_modify = int(len(df) * 0.5)

            # Randomly select indices to modify
            indices_to_modify = np.random.choice(df.index, size=num_rows_to_modify, replace=False)


            # Modify selected timestamps by adding a random number of days (1 to 10 days)
            df.loc[indices_to_modify, 'Timestamp'] += pd.to_timedelta(np.random.randint(1, 11, size=num_rows_to_modify),
                                                                      unit='d')
            logging.info(f"Converted Timestamp {df['Timestamp']}")

        # Set about 30% of all entries in the "is laundering" column to 1
        if 'Is Laundering' in df.columns:
            # Calculate number of rows to set to 1
            num_rows_to_set = int(len(df) * 0.3)

            # Randomly select indices to set to 1
            indices_to_set = np.random.choice(df.index, size=num_rows_to_set, replace=False)

            # Set selected entries to 1
            df.loc[indices_to_set, 'Is Laundering'] = 1

        df.to_csv('../data/Artificially_Generated_Local_Training_Data.csv', sep=',', index=False)
        # Shuffle the DataFrame rows
    return df.sample(frac=1).reset_index(drop=True)


def isNaN(num):
    return num != num


# Connect to Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"
driver = GraphDatabase.driver(uri, auth=(user, password))


def insert_and_connect_data(tx, row_data):
    query = (
        """

        // Delete everything in the database
        MATCH (n)
        DETACH DELETE n;
        
    //Create Transaction node
       // CREATE (t:Transaction {
         //   id: $id,  // Unique identifier for each transaction
            // timestamp: $Timestamp,
            // amount_received: $Amount_Received,
            // receiving_currency: $Receiving_Currency,
            // amount_paid: $Amount_Paid,
            // payment_currency: $Payment_Currency,
            // payment_format: $Payment_Format, //
            // is_laundering: $Is_Laundering
        //})
        
        // Create or merge Bank nodes and Account nodes
        MERGE (fromBank:Bank {id: $From_Bank})
        MERGE (toBank:Bank {id: $To_Bank})
        MERGE (fromAccount:Account {id: $From_Account})
        MERGE (toAccount:Account {id: $To_Account})
        >>
        // Create relationships with additional context
        MERGE (fromBank)-[:BANK_OWNS_ACCOUNT]->(fromAccount)
        MERGE (toBank)-[:BANK_OWNS_ACCOUNT]->(toAccount)
                
        CREATE (fromAccount)-[:TRANSFERRED_TO {
            amount_paid: $Amount_Paid, 
            currency_paid: $Payment_Currency, 
            time_of_transaction: $timestamp,
            payment_format: $Payment_Format,
            is_laundering: $Is_Laundering
        }]->(toAccount)        
        """

    )

    tx.run(query, **row_data)


df = load_demo_data()
print("loaded data!") 
# Create unique constraints
with driver.session() as session:
    session.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE")
    session.run("CREATE CONSTRAINT bank_id IF NOT EXISTS FOR (b:Bank) REQUIRE b.id IS UNIQUE")
    session.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")

# Specify the number of rows to insert

# Insert data into Neo4j
with driver.session() as session:
    for index, row in tqdm(df.iterrows()):
        try:
            if index >= args.rows_to_insert:
                break  # Stop after inserting the specified number of rows

            # Create a unique ID for the transaction
            transaction_id = f"{row['Timestamp']}_{row['From Bank']}_{row['Account']}_{row['To Bank']}_{row['Account']}"

            # Prepare row data
            row_data = row.to_dict()
            row_data['id'] = transaction_id
            row_data['From_Bank'] = row['From Bank']
            row_data['To_Bank'] = row['To Bank']
            row_data['timestamp'] = row['Timestamp']
            row_data['From_Account'] = f"{row['From Bank']}_{row['Account']}"
            row_data['To_Account'] = f"{row['To Bank']}_{row['Account']}"
            row_data['Amount_Received'] = row['Amount Received']
            row_data['Receiving_Currency'] = row['Receiving Currency']
            row_data['Amount_Paid'] = row['Amount Paid']
            row_data['Payment_Currency'] = row['Payment Currency']
            row_data['Payment_Format'] = row['Payment Format']
            row_data['Is_Laundering'] = row['Is Laundering']

            session.write_transaction(insert_and_connect_data, row_data)
        except Exception as e:
            print(e)
print(f"Data insertion complete. Inserted {min(args.rows_to_insert, len(df))} rows.")

# Close the driver connection
driver.close()
