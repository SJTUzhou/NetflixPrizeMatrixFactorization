**Netflix Prize**

Implementation of Matrix Factorization from the article 

https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

Use algorithm 'stochastic gradient descent' and 'alternating least square' to train parameter matrices

**const.py**: Specify the path and filename constants. You can change to your local file path.

**prepare_data.py**: Run it first to get the reformed dataset and other auxiliary files.

**ALS_extra_data.py** Run it to get the extra needed data for ALS method

**matrix_stoc_grad_desc.py**: Run it to train and evaluate the matrices by stochastic gradient descent

Command: **python3 matrix_stoc_grad_desc.py start** for the first time of training

Command: **python3 matrix_stoc_grad_desc.py continue** for continuing training

**ALS_matrix.py**: Run it to train and evaluate the matrices by alternating least square

Command: **python3 ALS_matrix.py.py start** for the first time of training

Command: **python3 ALS_matrix.py.py continue** for continuing training
