**Netflix Prize**

Implementation of Matrix Factorization from the article 

https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

Use algorithm 'stochastic gradient descent' and 'alternating least square' to train parameter matrices,

Use multi-processing in python by default

**const.py**: Specify the path and filename constants. You can change to your local file path.

**prepare_data.py**: Run it first to get the dataset in format of NumPy and other auxiliary files. It will divide train set and test set randomly.

**ALS_extra_data.py**: Run it AFTER **prepare_data.py** to get the extra needed data in format of NumPy for ALS method

**matrix_stoc_grad_desc.py**: Run it to train and evaluate the matrices by stochastic gradient descent

**ALS_matrix.py**: Run it to train and evaluate the matrices by alternating least squares

