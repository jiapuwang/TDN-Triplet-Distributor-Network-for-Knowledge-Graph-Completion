# TDN-Triplet-Distributor-Network-for-Knowledge-Graph-Completion

Train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: python run.py
```

Knowledge Graph Embedding

```
optional arguments:
  --dataset {FB15K,WN,Kinship,UMLS}
                        Knowledge Graph dataset
  --model {InceptE,ComplEx,RotatE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
```
