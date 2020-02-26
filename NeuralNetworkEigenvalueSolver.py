##################################################
# Neural Network Eigenvalue Solver
# Julie Butler Hartley
# Version 2.0
# Created: January 16, 2020
# Last Modified: February 26, 2020
#
# Finds an eigenvalue of either a given matrix or a randomly generated matrix 
# with a neural network.  See the paper located at the below link as a 
# reference:
# https://www.sciencedirect.com/science/article/pii/S0898122104901101
#
# Algorithm given by the paper converges to an eigenvalue, but not the 
# smallest or largest as claimed. It converges to all of them equally.
#
# A change to the loss function allows for either the smallest or the largest
# eigenvalue to be found everytime.
#
# Notes: relu does not seem to be as effective as sigmoid, changing x0 to 
# non-randomized does not help convergence to the same eigenvalue
# Code has been changed from orginal to include a loss function that ensures the
# smallest postive eigenvalue
##################################################

##############################
# IMPORTS
##############################
# Third-Party Imports
# For machine learning, making sure to import version 1 of Tensorflow, not 
# version 2
import tensorflow as tf
# For matrix manipulation and finding eigenvalues
import numpy as np

##############################
# F(X)
##############################
def f_x(x, A):
    """
        Inputs:
            x (a Tensorflow tensor): the trial eigenvector (i.e. the output
                of the neural network)
            A (a 2D Numpy array): the matrix to find eigenvalues and 
                eigenvectors of
        Returns:
            f (a Tensorflow tensor): the result of the function f(x(t)), 
                defined in the paper referenced above.  When x(t) is 
                converged, f(x(t)) = x(t)
        Returns the value of f(x) at a given value of x.
    """
    xTxA = (tf.tensordot(tf.transpose(x), x, axes=1)*A)
    # (1- xTAx)*I
    xTAxI = (1- tf.tensordot(tf.transpose(x), tf.tensordot(A, x, axes=1), axes=1))*np.eye(matrix_size)
    # (xTx*A - (1- xTAx)*I)*x
    f = tf.tensordot((xTxA + xTAxI), x, axes=1)

    return f

##############################
# NN EIGENVALUE
##############################
def NN_Eigenvalue(matrix_size, A, max_iterations, nn_structure, eigen_guess, 
                    eigen_lr, delta_threshold):
    """
        Inputs:
            matrix_size (an int): the dimension of the matrix
            A (a 2D Numpy array): A square, symmetric matrix to find 
                an eigenvector and eigenvalue of.
            max_iterations (an int): the maximum number of training iterations 
                to be used by the neural network
            nn_structure (a list): the number of neurons in each layer of the
                neural network
            eigen_guess (an int): to find the lowest eigenvalue, a number smaller
                than the predicted eigenvalue.  To find the largest eigenvalue,
                a number larger than the predicted eigenvalue.
            eigen_lr (a float): the learning rate for the portion of the loss
                function that controls which eigenvalue is found.  Set to 0.0
                to find a random eigenvalue.
            delta_threshold (a float): the minimum value desired between two
                sequentially calculated eigenvalues
    """
    # Defining the 6x6 identity matrix
    I = np.identity(matrix_size)
    
    # Create a vector of random numbers and then normalize it
    # This is the beginning trial solution eigenvector
    x0 = np.random.rand(matrix_size)
    x0 = x0/np.sqrt(np.sum(x0*x0))
    # Reshape the trial eigenvector into the format for Tensorflow
    x0 = np.reshape(x0, (1, matrix_size))

    # Convert the above matrix and vector into tensors that can be used by
    # Tensorflow
    I_tf = tf.convert_to_tensor(I)
    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)

    # Set up the neural network with the specified architecture
    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(nn_structure)

        # x0 is the input to the neural network
        previous_layer = x0_tf

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, nn_structure[l],activation=tf.nn.relu )
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, matrix_size)

    ## Execute the Tensorflow session
    with tf.Session() as sess:  
      
        # Define the loss function
        with tf.name_scope('loss'):
            # trial eigenvector is the output of the neural network
            x_trial = tf.transpose(dnn_output) 
            # f(x)
            f_trial = tf.transpose(f_x(x_trial, A))
            # eigenvalue calculated using the trial eigenvector using the 
            # Rayleigh quotient formula
            eigenvalue_trial = tf.transpose(x_trial)@A@x_trial/(tf.transpose(x_trial)@x_trial)
            
            x_trial = tf.transpose(x_trial) 

            # Define the loss function
            loss = tf.losses.mean_squared_error(f_trial, x_trial) + \
                 eigen_lr*tf.losses.mean_squared_error([[eigen_guess]], eigenvalue_trial)
                                                                                                            
        # Define the training algorithm and loss function
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)

        # Initialize the Tensorflow variables
        init = tf.global_variables_initializer()
        init.run()

        # Define for calculating the change between successively calculated
        # eigenvalues
        old_eig = 0

        for i in range(max_iterations):
            sess.run(training_op)
            # Compute the eigenvalue using the Rayleigh quotient
            eigenvalue = (x_trial.eval() @ (A @ x_trial.eval().T)
                        /(x_trial.eval() @ x_trial.eval().T))[0,0]
            eigenvector = x_trial.eval()

            # Calculate the change between the currently calculated eigenvalue
            # and the previous one
            delta = np.abs(eigenvalue-old_eig)
            old_eig = eigenvalue
            
            # Print useful information every 100 steps
            if i % 100 == 0:
                l = loss.eval()
                print("Step:", i, "/",max_iterations, "loss: ", l, "Eigenvalue:" , eigenvalue)
                # Kill the loop if the loss becomes zero, i.e. traning is 
                # complete
                if delta < delta_threshold:
                    break

    # Return the converged eigenvalue and eigenvector
    return eigenvalue, eigenvector

def pairing_model_4p4h (g, delta):
    """
        Inputs:
            g (a float): the interaction strength
            delta (a float): the spacing between energy levels
        Returns:
            A (a 2D Numpy array): the Hamiltoian for the 4 particle, 4 hole
                pairing model
        Calculates the Hamiltonian for the 4 particle 4 hole pairing model for
        the no broken pairs case.  For more information see Chapter 10 of
        LNP 936.
    """
    A = np.array(
        [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
        [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
        [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ], 
        [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ], 
        [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ], 
        [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]])
    return A


# Defining variables
matrix_size = 6 # Size of the matrix
max_iterations = 5000 # Maximum number of iterations
nn_structure = [100,100] # Number of hidden neurons in each layer
eigen_guess =  70 # Guess for the eigenvalue (see the header of NN_Eigenvalue)
eigen_lr = 0.01 # Eigenvalue learnign rate (see the header of NN_Eigenvalue)
delta_threshold = 1e-16 # Kill condition

# Create the matrix to be used
A = pairing_model_4p4h (0.5, 1.0)

# Find the eigenvalues and the eigenvectors using Numpy to compare to the 
numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)

# Reset the Tensorflow graph, causes an error if this is not here
# Since the above cells are not re-ran every time this one is, they are not 
# reset.  This line is needed to reset the Tensorflow computational graph to
# avoid variable redefinition errors. 
tf.reset_default_graph()

# Calcualte the estimate of the eigenvalue and the eigenvector
eigenvalue, eigenvector = NN_Eigenvalue(matrix_size, A, max_iterations, nn_structure, eigen_guess, 
                    eigen_lr, delta_threshold)

## Compare with the analytical solution
print("\n Numpy Eigenvalues: \n", numpy_eigenvalues)
print("\n Final Numerical Eigenvalue \n", eigenvalue)
diff = np.min(abs(numpy_eigenvalues - eigenvalue))
print("\n")
print('Absolute difference between Numerical Eigenvalue and TensorFlow DNN = ',diff)


