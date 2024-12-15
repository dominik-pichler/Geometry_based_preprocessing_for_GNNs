#!/bin/bash

# Setting up the training data
python data_insertion_to_neo4j.py --rows_to_insert=100000 --local_test

# Baseline - Without geometric preprocessing
python main.py --model gin --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "no_GBPre_GIN"
python main.py --model gat --batch_size 4096 --n_epochs 200  --num_neighs 150 150 --save_model --unique_name "no_GBPre_GAT"
python main.py --model rgcn --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "no_GBPre_RCGN"
python main.py --model pna --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "no_GBPre_PNA"

# Testing Suit - with geometric preprocessing
python main.py --model gin  --GBPre --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "GBPre_GIN"
python main.py --model gat  --GBPre --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "GBPre_GAT"
python main.py --model rgcn --GBPre --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "GBPre_RCGN"
python main.py --model pna  --GBPre --batch_size 4096 --n_epochs 200 --num_neighs 150 150 --save_model --unique_name "GBPre_PNA"