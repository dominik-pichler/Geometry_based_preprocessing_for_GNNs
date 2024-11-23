import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm



rows_to_insert = 1000  # Change this value to insert more or fewer rows


def load_demo_data():
    return pd.read_csv('../data/LI-Small_Trans.csv', sep=',', nrows = rows_to_insert)


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
            if index >= rows_to_insert:
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
print(f"Data insertion complete. Inserted {min(rows_to_insert, len(df))} rows.")

# Close the driver connection
driver.close()
