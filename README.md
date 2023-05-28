# Ad auction bidding strategy

Consider the case of participating in `N ≫ 1` online ad auctions with a limited bidding 
budget. The task is to create such a bidding strategy that you can win some of them, and 
that the placed ads generate at least `N_C` clicks. This should be done by spending as 
little money as possible.

A detailed description of the problem and a possible solution are provided 
in [this document](pdf/bidding_strategy.pdf).  

To execute the optimization algorithm run either:
```shell script
python main.py --name exponpow
```
or use the [Jupyter notebook](Example.ipynb).


Optimal bids for a sample problem (a detailed exlpanation 
is given in [this document](pdf/bidding_strategy.pdf))
<img width="600" alt="teaser" src="./pdf/fig/biding_strategy.png">