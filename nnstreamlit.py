import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *






np.random.seed(1)


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Simple Neural Networks Web App")
    st.sidebar.title("Simple Neural Networks Web App")
 
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    st.markdown(" Problem Statement: You are given a dataset ('data.h5') containing:")
    message="- a training set of " + (train_x_flatten.size()) " images labelled as cat (1) or non-cat (0)"
    st.markdown(message)
    st.markdown("- a test set of m_test images labelled as cat and non-cat")
    st.markdown("- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).")
    
    
    def display(index):
        plt.rcParams['figure.figsize'] = (4.0, 4.0) # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.imshow(train_x_orig[index])
        st.write("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
        st.pyplot()
        
        
    def predict1(X, y, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
    
        # Forward propagation
        probas, caches = L_model_forward(X, parameters)

    
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
    
        Accuracy=np.sum((p == y)/m)
        st.write(Accuracy)
        
        return p 
    
    
    def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
    
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
    
        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        parameters = initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###
    
        # Loop (gradient descent)
        my_bar = st.progress(0)
        placeholder=st.empty()
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(X, parameters)
            ### END CODE HERE ###
        
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(AL, Y)
            ### END CODE HERE ###
    
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
 
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                st.markdown("Cost after iteration %i: %f" %(i, cost))
            if i % 100 == 0:
                costs.append(cost)
            time.sleep(0)
            if (i/(num_iterations/100)) in range(0,100):
                placeholder.text(str(i/(num_iterations/100)))
                my_bar.progress(int(i/(num_iterations/100)))
             
            
            
        my_bar.progress(100)
        placeholder.text(str("Completed"))
            
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        st.pyplot()
    
        return parameters
    
   
    
    
    
    
       
    def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations 
    
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """
        
        np.random.seed(1)
        grads = {}
        costs = []                              # to keep track of the cost
        m = X.shape[1]                           # number of examples
        (n_x, n_h, n_y) = layers_dims
    
        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = initialize_parameters(layers_dims[0],layers_dims[1],layers_dims[2])
        ### END CODE HERE ###
    
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
    
        # Loop (gradient descent)
        my_bar = st.progress(0)
        placeholder=st.empty()
        for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2,cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        
            A1,cache1 = linear_activation_forward(X, W1, b1, "relu")
            A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        ### END CODE HERE ###
        
        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(A2, Y)
        ### END CODE HERE ###
        
        # Initializing backward propagation
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
        
        # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                st.markdown("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0:
                costs.append(cost)
            time.sleep(0)
            if (i/(num_iterations/100)) in range(0,100):
                
                placeholder.text(str(i/(num_iterations/100)))
                my_bar.progress(int(i/(num_iterations/100)))
             
            
            
        my_bar.progress(100)
        placeholder.text(str("Completed"))
        # plot the cost

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        st.pyplot()
    
        return parameters
        
    st.sidebar.markdown("Let's get more familiar with the dataset")
    option=st.sidebar.checkbox("Explore the Data Set ?",False,key="option")
    
    if option:
        st.sidebar.markdown("The following will show you an image in the dataset")
        index=st.sidebar.number_input("Feel free to change the index multiple times to see other images",1,208,step=1,key="index")
        display(index)
        
    
    choice1=st.sidebar.selectbox("Choose which Neural Network model you wish to use",("2-Layer","4-Layer"),key="choice1")
    if choice1=="2-Layer":
        num_iterations=st.sidebar.slider("Number of Iterations for Gradient Descent: ",1500,2500,key="n_h")
        cost=st.sidebar.checkbox("Do you wish to see cost of the model after every 100 iterations ?",False,key="cost")
        ### CONSTANTS DEFINING THE MODEL ####
        n_x = 12288     # num_px * num_px * 3
        n_h = 7
        n_y = 1
        layers_dims = (n_x, n_h, n_y)
        if st.sidebar.button("Click to Train", key='train'):
            st.subheader("2-Layer Neural Net in Action")
            st.markdown("PS:- Neural Net is working hard in the background,it may take some time") 
            parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = num_iterations, print_cost=cost)
            st.subheader("Training Completed")
            
            
            st.markdown("Accuracy on Train set is: ")
            predictions_train=predict1(train_x, train_y, parameters)
            
            st.markdown("Accuracy on Test set is: ")
            predictions_test = predict1(test_x, test_y, parameters)
            
           
            
            
            
            
            
            
            
            
            
                
        
           
            
            
            
            
    if choice1=="4-Layer":
        num_iterations=st.sidebar.slider("Number of Iterations for Gradient Descent: ",1500,2500,key="n_h")
        cost=st.sidebar.checkbox("Do you wish to see cost of the model after every 100 iterations ?",False,key="cost")
        layers_dims = [12288, 20, 7, 5, 1]
        if st.sidebar.button("Click to Train", key='train'):
            st.subheader("4-Layer Neural Net in Action")
            st.markdown("PS:- Neural Net is working hard in the background,it may take some time")
            parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = num_iterations, print_cost = cost)
            st.subheader("Training Completed")
            
            st.markdown("Accuracy on Train set is: ")
            pred_train = predict1(train_x, train_y, parameters)
            
            st.markdown("Accuracy on Test set is: ")
            pred_test = predict1(test_x, test_y, parameters)
            
            
    
            
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
