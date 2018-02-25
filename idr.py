import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) # filter warnings of tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from itertools import product
from collections import defaultdict
from sklearn.metrics import mean_absolute_error as mae

class IDR(object):
    def __init__(self, n_components=2, learning_rate=100, n_steps=1000, show_progress_bar=True):
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
        show_progress_bar: bool. Wheter to show the tqdm progress bar of learning process.
        
        '''
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_steps = n_steps
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
                points = tf.Variable(tf.random_uniform((self.n_points, self.n_components)))
            else:
                points = tf.Variable(tf.constant(init_points))
            
            lengths = tf.reduce_sum(points * points, 1)
            lengths = tf.reshape(lengths, [-1, 1])
            pred_distances_tf = lengths - 2 * tf.matmul(points, tf.transpose(points)) + tf.transpose(lengths)
            eps_matrix_tf = tf.constant(1e-30, dtype=tf.float32) * tf.ones_like(pred_distances_tf)
            pred_distances_tf = tf.sqrt(tf.maximum(pred_distances_tf, eps_matrix_tf))
            
            loss_mse = tf.losses.mean_squared_error(true_distances_tf, pred_distances_tf)
            
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            make_train_iteration = optimizer.minimize(loss_mse)

            sess.run(tf.global_variables_initializer())
            
            self.losses = defaultdict(list)
            if self.show_progress_bar:
                iteration_range = tqdm.tqdm(range(self.n_steps))
            else:
                iteration_range = range(self.n_steps)
            for i in iteration_range:
                sess.run(make_train_iteration, feed_dict={true_distances_tf: true_distances})
                self.losses['mse'].append(sess.run(loss_mse, feed_dict={true_distances_tf: true_distances}))
                pred_distances = sess.run(pred_distances_tf, feed_dict={true_distances_tf: true_distances})
                self.losses['mae'].append(mae(self.true_distances, pred_distances))
            self.points = sess.run(points)
    
    def draw_points(self, n_neighbours=30, n_tries=1, plot_type='neighbours'):
        '''
        Draws n_tries plots, for each plot choose random point with n_neightbours nearest and n_neighbours 
        farthest points in distance measure of high-dimensional space. This can be applied to measure the quality
        of the algorithm and visualize structure of high-dimensional data.        
        
        n_neighbours: int in [0, true_distances.shape[0]). Must be less than the number of all points.
        n_tries: int. Number of plots
        plot_type: str in ['neighbours', 'heatmap']. Two different types of plots.        
        '''
        
        if self.n_components != 2:
            raise ValueError('''Can not draw {}-dimensional points. 
Works only when n_components equals two.'''.format(n_components))
        if n_neighbours > self.n_points:
            raise ValueError('''n_neighbours must be less than number of 
all points. {} > {}.'''.format(n_neighbours, self.n_points))
        
        for attempt_i in range(n_tries):
            point_index = np.random.randint(self.n_points)
            point = self.points[point_index]
            if plot_type == 'neighbours':
                plt.scatter(*self.points.T, s=1)
                nearest_indices = np.argsort(self.true_distances[point_index])[1: n_neighbours + 1]
                farthest_indices = np.argsort(self.true_distances[point_index])[-n_neighbours: ]
                for index in nearest_indices:
                    plt.scatter(self.points[index][0], self.points[index][1], c='red')
                for index in farthest_indices:
                    plt.scatter(self.points[index][0], self.points[index][1], c='blue')
            elif plot_type == 'heatmap':
                r = 1 - self.true_distances[point_index]
                g = np.zeros(self.n_points)
                b = self.true_distances[point_index]
                plt.scatter(*self.points.T, c=list(zip(r, g, b)), s=1)
            plt.scatter(point[0], point[1], c='green')
            plt.show()
        
    def draw_losses(self):
        '''
        draw mse and mae losses.
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