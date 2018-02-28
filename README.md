# IDR
*Isometric. For arbitrary metric.*

TensorFlow implementation of algorithm of non-linear dimension reduction. The main goal of the algorithm is to make an isometric mapping from high-dimensional space with arbitrary distance measure to low-dimensional space for given points. The distance measure defined in high-dimensional space can be arbitrary, not necessarily l2 metric or even the metric in mathematical sense. In fact, IDR needs only the matrix M, where M[i, j] is the distance measure between points we want to embed.

The reason of creating such a framework is that sometimes we need to look on data in high-dimensional space with given distance measure. We can look only on 2-dimensional picture, so we may apply usual algorithms like PCA or tSNE, but they change choosen distance measure unpredictably.

file_name | what is there 
------------ | ------------- |
idr.py | IDR class implementation. |
idr_example_3d.ipynb | example of IDR usage  |
utils.py | some useful functions |
