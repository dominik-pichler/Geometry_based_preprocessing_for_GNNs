from neo4j import GraphDatabase

class Neo4jStatisticsReport:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def generate_report(self):
        with self.driver.session() as session:
            result = session.run("""
                WITH 
                    (MATCH (n) RETURN count(n) AS total_nodes) AS nodes,
                    (MATCH (t:Transaction) RETURN count(t) AS total_transactions) AS transactions,
                    (MATCH (b:Bank) RETURN count(b) AS total_banks) AS banks,
                    (MATCH (a:Account) RETURN count(a) AS total_accounts) AS accounts,
                    (MATCH (t:Transaction) RETURN sum(t.amount_received) AS total_amount_received) AS amount_received,
                    (MATCH (t:Transaction) RETURN sum(t.amount_paid) AS total_amount_paid) AS amount_paid,
                    (MATCH (t:Transaction) RETURN avg(t.amount_received) AS average_amount_received) AS avg_received,
                    (MATCH (t:Transaction) RETURN avg(t.amount_paid) AS average_amount_paid) AS avg_paid,
                    (MATCH (t:Transaction) RETURN count(distinct t.receiving_currency) AS unique_receiving_currencies) AS unique_receiving,
                    (MATCH (t:Transaction) RETURN count(distinct t.payment_currency) AS unique_payment_currencies) AS unique_payment,
                    (MATCH ()-[r:BANK_OWNS_ACCOUNT]->() RETURN count(r) AS total_bank_account_relationships) AS bank_account_relationships,
                    (MATCH ()-[r:TRANSFERRED_TO]->() RETURN count(r) AS total_transfer_relationships) AS transfer_relationships,
                    (MATCH (t:Transaction) RETURN min(t.timestamp) AS earliest_transaction_time) AS earliest_time,
                    (MATCH (t:Transaction) RETURN max(t.timestamp) AS latest_transaction_time) AS latest_time,
                    (MATCH (t:Transaction) WHERE t.is_laundering = true RETURN count(t) AS suspected_laundering_transactions) AS laundering_count

                RETURN 
                  '<html><body>' +
                  '<h1>Graph Descriptive Statistics Report</h1>' +
                  '<ul>' +
                  '<li>Total Nodes: ' + toString(nodes.total_nodes) + '</li>' +
                  '<li>Total Transactions: ' + toString(transactions.total_transactions) + '</li>' +
                  '<li>Total Banks: ' + toString(banks.total_banks) + '</li>' +
                  '<li>Total Accounts: ' + toString(accounts.total_accounts) + '</li>' +
                  '<li>Total Amount Received: ' + toString(amount_received.total_amount_received) + '</li>' +
                  '<li>Total Amount Paid: ' + toString(amount_paid.total_amount_paid) + '</li>' +
                  '<li>Average Amount Received per Transaction: ' + toString(avg_received.average_amount_received) + '</li>' +
                  '<li>Average Amount Paid per Transaction: ' + toString(avg_paid.average_amount_paid) + '</li>' +
                  '<li>Unique Receiving Currencies: ' + toString(unique_receiving.unique_receiving_currencies) + '</li>' +
                  '<li>Unique Payment Currencies: ' + toString(unique_payment.unique_payment_currencies) + '</li>' +
                  '<li>Total Bank-Account Relationships: ' + toString(bank_account_relationships.total_bank_account_relationships) + '</li>' +
                  '<li>Total Transfer Relationships: ' + toString(transfer_relationships.total_transfer_relationships) + '</li>' +
                  '<li>Earliest Transaction Time: ' + toString(earliest_time.earliest_transaction_time) + '</li>' +
                  '<li>Latest Transaction Time: ' + toString(latest_time.latest_transaction_time) + '</li>' +
                  '<li>Suspected Money Laundering Transactions: ' + toString(laundering_count.suspected_laundering_transactions) + '</li>' +
                  '</ul></body></html>' AS html_report
            """)

            # Fetch the HTML report from the result
            for record in result:
                html_report = record["html_report"]

            # Save the report as an HTML file
            with open("graph_statistics_report.html", "w") as file:
                file.write(html_report)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "your_password"

    report_generator = Neo4jStatisticsReport(uri, user, password)
    report_generator.generate_report()
    report_generator.close()

    print("Report generated successfully as graph_statistics_report.html")