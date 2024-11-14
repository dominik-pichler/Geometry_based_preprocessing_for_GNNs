```
       .---.
      /_____\
      ( '.' )     STOP BEING A FRAUD!
       \_-_/_
    .-"`'V'//-.
   / ,   |// , \
  / /|Ll //Ll|\ \
 / / |__//   | \_\
 \ \/---|[]==| / /
  \/\__/ |   \/\/
   |/_   | Ll_\|
     |`^"""^`|
     |   |   |
     |   |   |
     |   |   |
     |   |   |
     L___l___J
      |_ | _|
     (___|__)
```


# Setup
1. Setup the DB Infrastructure via `docker-compose up -d`.
2. Install all needed packages via `poetry install`.
3. Download the data from [IBM - Syntetic Transaction Data for Anti-Money-Laundry Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data) and store it in the `/data` dir.
4. Specify the data path in `src/data_insertion_to_neo4j.py` and run it via `python data_insertion_to_neo4j.py`.


# 1. Introduction
In this project I am investigating **Graph Structure-based Fraud Detection** in Financial Transaction Networks with the help of **Graph Neural Networks**.


Initially Inspired by the big issue of Value-Added-Tax Fraud, where in 2021 alone approx. 15 Mrd.€have been stolen by criminals applying Value-Added-Tax Fraud-techniques in the EU according to [*Ott (2024)*](https://epub.jku.at/obvulihs/download/pdf/10500928).
Due to lack of publically available data, I had to switch to something similar: Money Laundry Patterns in Transaction Networks. As I chose a geometry based approach, it might still be useful for VAT Fraud as well.


Therefore, I try to **bring my own method** to detect money laundering in the geometrical structures present in the IBM Transactions for Anti Money Laundering (AML) while orienting myself on the following three papers:
- [The geometry of suspicious money laundering activities in financial networks](https://perfilesycapacidades.javeriana.edu.co/en/publications/the-geometry-of-suspicious-money-laundering-activities-in-financi)
- [Provably Powerful Graph Neural Networks for Directed Multigraph](https://arxiv.org/pdf/2306.11586)
- [Smurf-based Anti-money Laundering in Time Evolving Transction Networks](https://www.researchgate.net/publication/354487674_Smurf-Based_Anti-money_Laundering_in_Time-Evolving_Transaction_Networks)

For this purpose, I am using the following dataset:
-  [IBM - Syntetic Transaction Data for Anti-Money-Laundry Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data): 



This data looks the following and contains 180 Mio. of those transactions (rows), with a high and low Laundering-Transaction Share.



# 2. Dataset

| Timestamp       | From Bank | Account    | To Bank | Account    | Amount Received | Receiving Currency | Amount Paid | Payment Currency | Payment Format | Is Laundering |
|-----------------|-----------|------------|---------|------------|-----------------|--------------------|-------------|------------------|----------------|---------------|
| 01.09.22 0:08   | 11        | 8000ECA90  | 11      | 8000ECA90  | 3195403         | US Dollar          | 3195403     | US Dollar        | Reinvestment   | 0             |
| 01.09.22 0:21   | 3402      | 80021DAD0  | 3402    | 80021DAD0  | 1858.96         | US Dollar          | 1858.96     | US Dollar        | Reinvestment   | 0             |
| 01.09.22 0:00   | 11        | 8000ECA90  | 1120    | 8006AA910  | 592571          | US Dollar          | 592571      | US Dollar        | Cheque         | 0             |
| 01.09.22 0:16   | 3814      | 8006AD080  | 3814    | 8006AD080  | 12.32           | US Dollar          | 12.32       | US Dollar        | Reinvestment   | 0             |
| 01.09.22 0:00   | 20        | 8006AD530  | 20      | 8006AD530  | 2941.56         | US Dollar          | 2941.56     | US Dollar        | Reinvestment   | 0             |
| 01.09.22 0:24   | 12        | 8006ADD30  | 12      | 8006ADD30  | 6473.62         | US Dollar          | 6473.62     | US Dollar        | Reinvestment   | 0             |
| 01.09.22 0:17   | 11        | 800059120  | 1217    | 8006AD4E0  | 60562           | US Dollar          | 60562       | US Dollar        | ACH            | 0             |



and includes the following columns:


| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Timestamp          | The date and time when the transaction occurred.                            |
| From Bank          | The identifier of the bank from which the funds were sent.                  |
| Account            | The account number from which the funds were sent.                          |
| To Bank            | The identifier of the bank to which the funds were sent.                    |
| Account            | The account number to which the funds were sent.                            |
| Amount Received    | The total amount of money received in the transaction.                      |
| Receiving Currency | The currency in which the amount was received.                              |
| Amount Paid        | The total amount of money paid in the transaction.                          |
| Payment Currency   | The currency in which the payment was made.                                 |
| Payment Format     | The method or format used for the payment (e.g., Reinvestment, Cheque, ACH).|
| Is Laundering      | Indicates whether the transaction is suspected of money laundering (0 = No).|


A more in-depth dataset analysis can be found [here](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data:)

# 3. Project plan:
## 3.1 Dataset Collection and Preprocessing
- **Research and Identify Suitable Datasets**: 10 hours
- **Data Cleaning and Preprocessing**: 20 hours
- **Feature Engineering**: 10 hours

## 3.2 Designing and Building the Graph Neural Network
- **Literature Review on Graph Neural Networks**: 10 hours
- **Network Architecture Design**: 15 hours
- **Implementation of the Network**: 20 hours

## 3.3 Training and Fine-Tuning the Network
- **Initial Model Training**: 20 hous
- **Hyperparameter Tuning**: 15 hours
- **Validation and Testing**: 15 hours

## 3.4 Building an Application to Present Results
- **Design User Interface**: 10 hours
- **Develop Application Backend**: 15 hours
- **Integrate Model with Application**: 10 hours

## 3.5 Writing the Final Report
- **Drafting the Report Structure**: 5 hours
- **Writing and Editing Content**: 15 hours
- **Creating Visualizations and Appendices**: 5 hours

## 3.6 Preparing the Presentation
- **Design Presentation Slides**: 5 hours
- **Rehearse Presentation Delivery**: 5 hours


# Implementation
The core concept here is a neurosymbolic approach, combining symbolic AI (logics, rules) and graph neural networks. 
