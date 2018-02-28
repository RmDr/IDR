import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) # filter warnings of tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from collections import defaultdict
from sklearn.metrics import mean_absolute_error as mae

OUTPUT_LINE_LENGTH = 72

def draw_nearest_farthest_points(points, distance_matrix, n_neighbors=30, n_tries=1, neighbors_size=None, points_size=1):
    '''
    Draws n_tries plots, for each plot choose random point from points with n_neighbors nearest and n_neighbors 
    farthest points in distance of distance_matrix. This can be applied to measure the quality
    of the dimension reduction algorithm and visualize structure of high-dimensional data.
    
    points: points to draw.
    n_neighbors: int in [0, len(points)). Must be less than the number of all points.
    n_tries: int. Number of plots.
    neighbors_size: size of points which are neighbors of the chosen random point
    '''
    
    if n_neighbors > len(points):
        raise ValueError('''n_neighbors must be less than number of 
all points. {} > {}.'''.format(n_neighbors, len(points)))
    
    for attempt_i in range(n_tries):
        point_index = np.random.randint(len(points))
        plt.scatter(*points.T, c='black', s=points_size)
        nearest_indices = np.argsort(distance_matrix[point_index])[1: n_neighbors + 1]
        farthest_indices = np.argsort(distance_matrix[point_index])[-n_neighbors: ]
        plt.scatter(*points[nearest_indices].T, c='red', s=neighbors_size)
        plt.scatter(*points[farthest_indices].T, c='blue', s=neighbors_size)
        point = points[point_index]
        plt.scatter(point[0], point[1], c='green', s=neighbors_size)
        plt.show()

class IDR(object):
    def __init__(self, n_components=2, learning_rate=100, n_steps=1000, supervision_bound=None, 
                 random_state=None, show_progress_bar=True):
        '''
        Implementation of algorithm of non-linear dimension reduction on tensorflow. 
        The main goal of the algorithm is to make an isometric mapping from high-dimensional 
        space with arbitrary distance measure to low-dimensional space for given points. 
        The distance measure defined in high-dimensional space can be arbitrary, 
        not necessarily l2 metric or even the metric in mathematical sense. In fact, 
        IDR needs only the matrix M, where M[i, j] is the distance measure between points we 
        want to embed.
        
        n_components: int. The dimension of the low-dimensional space.
        learning_rate: float. Learning rate for gradient descent.
        n_steps: int. Number of steps for gradient descent.
        random_state: int. Random seed to use for random initialization of points. If None, nothing will be set.
        show_progress_bar: bool. Wheter to show the progress bar of learning process.
        
        '''
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.supervision_bound = supervision_bound
        self.random_state = random_state
        self.show_progress_bar = show_progress_bar
    
    def fit(self, true_distances, init_points=None):
        '''
        Fits algorithm.
        
        true_distances: np.array of shape (n_points, n_points). true_distances[i, j] is the distance measure
        between i-th and j-th points.
        init_points: None or np.array of shape (n_points, n_components). The initial state for the optimization
        algorithm. If None initial state is chosen randomly from standart normal distribution.
        '''
        
        self.true_distances = true_distances
        self.n_points = true_distances.shape[0]
        with tf.Session() as sess:
            true_distances_tf = tf.placeholder(dtype=tf.float32)
            
            if init_points is None:
                rnd = np.random.RandomState(self.random_state)
                points = tf.Variable(tf.constant(rnd.uniform(size=(self.n_points, self.n_components)), dtype=tf.float32))
            else:
                points = tf.Variable(tf.constant(init_points))
            
            lengths = tf.reduce_sum(points * points, 1)
            lengths = tf.reshape(lengths, [-1, 1])
            pred_distances_tf = lengths - 2 * tf.matmul(points, tf.transpose(points)) + tf.transpose(lengths)
            pred_distances_tf = tf.sqrt(tf.maximum(pred_distances_tf, 1e-30))

            loss = tf.square(true_distances_tf - pred_distances_tf)
            if self.supervision_bound is not None: 
                loss = (1 - true_distances_tf) * loss
            loss = tf.reduce_mean(loss)
            
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            make_train_iteration = optimizer.minimize(loss)

            sess.run(tf.global_variables_initializer())
            self.losses = defaultdict(list)
            
            for i in range(self.n_steps):                
                feed_dict = {true_distances_tf: true_distances}
                sess.run(make_train_iteration, feed_dict=feed_dict)
                loss_mse = sess.run(loss, feed_dict=feed_dict)
                self.losses['mse'].append(loss_mse)
                pred_distances = sess.run(pred_distances_tf, feed_dict=feed_dict)
                loss_mae = mae(self.true_distances, pred_distances)
                self.losses['mae'].append(loss_mae)
             
                if self.show_progress_bar:
                    output = "\rprogress: {}/{}. losses: mse={:.4f}; mae={:.4f}.".format(i + 1, self.n_steps, loss_mse, loss_mae)
                    output = output.ljust(OUTPUT_LINE_LENGTH, ' ')
                    print(output, end='')
            self.points = sess.run(points)
    
    def draw_points(self):
        '''
        Draws points if self.n_components equals 2.       
        '''
        
        if self.n_components != 2:
            raise ValueError('''Can not draw {}-dimensional points. \
Works only when n_components equals two.'''.format(n_components))
        plt.scatter(*self.points.T, c='green')
        plt.show()

    def draw_nearest_farthest_points(self, n_neighbors=30, n_tries=1):
        '''
        Draws n_tries plots, for each plot choose random point with n_neightbours nearest and n_neighbors 
        farthest points in distance measure of high-dimensional space. This can be applied to measure the quality
        of the algorithm and visualize structure of high-dimensional data.        
        
        n_neighbors: int in [0, true_distances.shape[0]). Must be less than the number of all points.
        n_tries: int. Number of plots       
        '''
        
        if self.n_components != 2:
            raise ValueError('''Can not draw {}-dimensional points. \
Works only when n_components equals two.'''.format(n_components))
        draw_nearest_farthest_points(self.points, self.true_distances, n_tries=n_tries, 
                            n_neighbors=n_neighbors, neighbors_size=None, points_size=1)
        
    def draw_losses(self):
        '''
        Draw MSE and MAE losses.
        '''
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['mse'])
        plt.ylim(0, 0.5)
        plt.xlabel('step_number')
        plt.ylabel('MSE value')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.losses['mae'])
        plt.ylim(0, 0.5)
        plt.xlabel('step_number')
        plt.ylabel('MAE value')
        plt.show()