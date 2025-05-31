import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import pandas as pd
from sklearn.model_selection import KFold
from A2helpers import generateData,polyKernel,linearKernel,gaussKernel 
from sklearn.metrics import accuracy_score

#--------------------------Q1a----------------------------------------------------------------------------------------------------
def bionimail_deviance_loss(p, X, y, lamb):
    n, d = X.shape
    w = p[:d]
    w0 = p[d]
    logits = X @ w + w0
    logistic_loss = np.sum(np.logaddexp(0, -y * logits))  # log(1 + exp(-y * (Xw + w0)))
    L2_regularization = (lamb / 2) * np.sum(w**2)
    
    return logistic_loss + L2_regularization


def minBinDev(X,y,lamb):
    n,d = X.shape
    p = np.zeros(d+1)
    result = minimize(bionimail_deviance_loss, p, args=(X, y, lamb), method='BFGS', options={'disp': False})    
    w = result.x[:d]
    w0 = result.x[d]
    
    return w,w0

#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q1b----------------------------------------------------------------------------------------------------

def minHinge(X, y, lamb, stablizer=1e-5):
    n, d = X.shape
    y = y.flatten() 

    # Quadratic matrix P
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = lamb * np.eye(d)
    P += stablizer * np.eye(d + 1 + n)
    P = matrix(P)

    # Linear term q
    q = np.hstack([np.zeros(d + 1), np.ones(n)])
    q = matrix(q)

    G = np.zeros((2 * n, d + 1 + n))
    delta_y = np.diag(y)
    G[:n, d + 1:] = -np.eye(n)  # First set: ξ >= 0 constraints (slack variables)
    G[n:, :d] = -(delta_y @ X)  # Use matrix multiplication instead of broadcasting
    G[n:, d] = -delta_y @ np.ones(n)  # The intercept w0 part
    G[n:, d + 1:] = -np.eye(n)  # Slack variables
    G = matrix(G)


    # Inequality constraints bounds h
    h = np.hstack([np.zeros(n), -np.ones(n)])
    h = matrix(h)

    # Solve QP problem
    result = solvers.qp(P, q, G, h, options={'show_progress': False})

    sol_result = np.array(result['x']).flatten()
    w = sol_result[:d]
    w0 = sol_result[d]

    return w,w0


#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q1c----------------------------------------------------------------------------------------------------

def classify(Xtest, w, w0):
    ycap = np.sign(Xtest@w + w0)
    return ycap


#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q1d----------------------------------------------------------------------------------------------------

def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.0]
    gen_model_list = [1, 2, 3]
    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                # Binomial deviance logistic regression
                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)  # Use classify function for training predictions
                ytest_pred = classify(Xtest, w, w0)    # Use classify function for test predictions
                train_acc_bindev[i, j, r] = np.mean(ytrain == ytrain_pred)
                test_acc_bindev[i, j, r] = np.mean(ytest == ytest_pred)

                # Hinge loss model
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)  # Use classify function for training predictions
                ytest_pred = classify(Xtest, w, w0)    # Use classify function for test predictions
                train_acc_hinge[i, j, r] = np.mean(ytrain == ytrain_pred)
                test_acc_hinge[i, j, r] = np.mean(ytest == ytest_pred)

    # Compute the average accuracies over the 100 runs
    avg_train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    avg_test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    # Combine the results into 4x6 matrices (4 lambdas, 3 models for BinDev, 3 models for Hinge)
    train_acc = np.hstack([avg_train_acc_bindev, avg_train_acc_hinge])
    test_acc = np.hstack([avg_test_acc_bindev, avg_test_acc_hinge])


    return train_acc, test_acc


# train_acc , test_acc = synExperimentsRegularize()
# print("Training acc 1: ")
# print(train_acc)
# print()
# print("Test acc 1: ")
# print(test_acc)
# print()

#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------------------Q2a---------------------------------------------------------------------------------------------

def adjBinDev(X,y,lamb,kernel_func):
    K = kernel_func(X,X)
    n,d = X.shape
    alpha = np.zeros(n+1)

    def loss(mater):
        alpha = mater[:-1]
        alpha_0 = mater[-1]
        linear_term = K @ alpha + alpha_0
        dev_loss = np.sum(np.log(1 + np.exp(-y*linear_term)))
        re_loss = (lamb/2) * (alpha.T @ K @ alpha)
        return dev_loss + re_loss
    
    solution = minimize(loss,alpha,method='L-BFGS-B')
    alpha_star = solution.x[:-1]
    alpha_0_star = solution.x[-1]
    return alpha_star,alpha_0_star
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    K = kernel_func(X, X)  # Kernel matrix
    y = y.flatten()

    # Quadratic matrix P
    P = np.zeros((2 * n + 1, 2 * n + 1))
    P[:n, :n] = lamb * K  # Regularization on kernel matrix
    P += stabilizer * np.eye(2 * n + 1)  # Stabilizer
    P = matrix(P)

    q = np.zeros(2 * n + 1)
    q[n+1:] = np.ones(n) 
    q = matrix(q)
    delta_y = np.diag(y)
    G = np.zeros((2 * n, 2 * n + 1))
    G[:n, n+1:] = -np.eye(n)  

    G[n:, :n] = -(delta_y @ K)
    G[n:, n] = -delta_y @ np.ones(n)  
    G[n:, n + 1:] = -np.eye(n)
    G = matrix(G)

    # Inequality bounds h
    h = np.zeros(2 * n)
    h[n:] = -1  # Margin constraint
    h = matrix(h)

    # Solve QP problem
    solution = solvers.qp(P, q, G, h, options={'show_progress': False})
    alpha_alpha_0 = np.array(solution['x']).flatten()
    alpha = alpha_alpha_0[:n]
    alpha_0 = alpha_alpha_0[n]

    return alpha, alpha_0
# def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
#     n, d = X.shape
#     K = kernel_func(X, X)  # Kernel matrix
#     y = y.flatten()

#     # Quadratic matrix P
#     P = np.zeros((2 * n + 1, 2 * n + 1))
#     P[:n, :n] = lamb * K  # Regularization on kernel matrix
#     P += stabilizer * np.eye(2 * n + 1)  # Stabilizer
#     P = matrix(P)

#     # Linear term q
#     q = np.zeros(2 * n + 1)
#     q[n:] = 1  # Slack variables part
#     q = matrix(q)

#     # Inequality matrix G
#     delta_y = np.diag(y)
#     G = np.zeros((2 * n, 2 * n + 1))

#     # Adjust this section: include an extra column for the slack variables
#     G[:n, n+1:] = -np.eye(n)  # Slack variables ξ >= 0, extra column for intercept

#     G[n:, :n] = -(delta_y @ K)  # Margin constraints
#     G[n:, n] = -y  # Intercept term
#     G[n:, n + 1:] = -np.eye(n)
#     G = matrix(G)

#     # Inequality bounds h
#     h = np.zeros(2 * n)
#     h[n:] = -1  # Margin constraint
#     h = matrix(h)

#     # Solve QP problem
#     solution = solvers.qp(P, q, G, h, options={'show_progress': False})
#     alpha_alpha_0 = np.array(solution['x']).flatten()
#     alpha = alpha_alpha_0[:n]
#     alpha_0 = alpha_alpha_0[n]

#     return alpha, alpha_0

def adjClassify(X_test,a,a0,X,kernel_func):
    K_test = kernel_func(X_test,X)
    decision = np.dot(K_test,a) + a0
    yhat = np.sign(decision)
    return yhat


def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    
    # Define the list of kernel functions
    kernel_list = [
        linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)
    ]
    
    # List of generative models
    gen_model_list = [1, 2, 3]

    # Storage for accuracies: kernel_list x gen_model_list x n_runs
    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                # Generate training and testing data
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                
                # Binary Deviance model
                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel)
                
                # Predict on training data and calculate accuracy
                yhat_train = adjClassify(Xtrain, a, a0, Xtrain, kernel)
                train_acc_bindev[i, j, r] = np.mean(yhat_train == ytrain)
                
                # Predict on test data and calculate accuracy
                yhat_test = adjClassify(Xtest, a, a0, Xtrain, kernel)
                test_acc_bindev[i, j, r] = np.mean(yhat_test == ytest)
                
                # Hinge Loss model
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                
                # Predict on training data and calculate accuracy
                yhat_train = adjClassify(Xtrain, a, a0, Xtrain, kernel)
                train_acc_hinge[i, j, r] = np.mean(yhat_train == ytrain)
                
                # Predict on test data and calculate accuracy
                yhat_test = adjClassify(Xtest, a, a0, Xtrain, kernel)
                test_acc_hinge[i, j, r] = np.mean(yhat_test == ytest)

    # Compute average accuracies across runs
    avg_train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    avg_test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    # Combine the accuracies into 5x6 matrices for Binary Deviance and Hinge Loss
    train_acc = np.hstack([avg_train_acc_bindev, avg_train_acc_hinge])
    test_acc = np.hstack([avg_test_acc_bindev, avg_test_acc_hinge])

    return train_acc, test_acc

# train_acc , test_acc = synExperimentsRegularize()
# print("Training acc 2: ")
# print(train_acc)
# print()
# print("Test acc 2: ")
# print(test_acc)
# print()


#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q3a----------------------------------------------------------------------------------------------------

def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    
    # Compute the kernel matrix K
    K = kernel_func(X, X)  # K should be n × n matrix

    # Create Δ(y) matrix
    delta_y = np.diag(y.flatten())
    
    # Formulate the quadratic program components
    P = delta_y @ K @ delta_y / (lamb)  # P is n × n
    P += stabilizer * np.eye(n)  # Add stabilizer for numerical stability
    q = -np.ones((n, 1))  # q is a vector of -1s (n × 1)
    
    # Convert to cvxopt matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    
    # Constraints: 0 ≤ α ≤ 1
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))  # G is 2n × n
    h = matrix(np.hstack([np.zeros(n), np.ones(n)]))  # h is 2n × 1
    
    # Equality constraint: α^T y = 0
    A = matrix(y.reshape(1, -1))  # A is 1 × n
    b_eq = matrix(0.0)  # b is 0

    # Solve the quadratic program
    solvers.options['show_progress'] = False
    solution = solvers.qp(P_cvx, q_cvx, G, h, A, b_eq)
    
    # Extract α from the solution
    alpha = np.array(solution['x']).flatten()  # n × 1 vector
    
    # Calculate b using an αi where 0 < αi < 1
    mask = (alpha > 0) & (alpha < 1)
    if np.any(mask):
        i = np.argmin(np.abs(alpha[mask] - 0.5))  # Find closest αi to 0.5
        idx = np.where(mask)[0][i]  # Get the index in the original array
        b = y[idx] - (1 / lamb) * (K[idx, :] @ (delta_y @ alpha))
    else:
        b = None  # No suitable αi found for calculating intercept b

    return alpha, b
 
#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q3b----------------------------------------------------------------------------------------------------
   
def  dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    # Check if 'a' is None
    y = y.flatten()
    
    # Compute the kernel matrix between Xtest and X
    K = kernel_func(Xtest, X)  # K will be an m × n matrix
    
    # Calculate predictions
    delta_y = y * a.flatten()  # n × 1 vector
    predictions = (1 / lamb) * (K @ delta_y) + b  # m × 1 vector
    
    # Apply sign function to get class predictions
    yhat = np.sign(predictions).reshape(-1, 1)  # m × 1 vector

    return yhat

#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------Q3c----------------------------------------------------------------------------------------------------

def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.0  # Normalize pixel values
    y = train_data[:, 0][:, None]   # Class labels
    y[y == 4] = -1                   # Convert class 4 to -1
    y[y == 9] = 1                    # Convert class 9 to +1

    # Prepare cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Ensure reproducibility with random_state
    cv_acc = np.zeros((k, len(lamb_list), len(kernel_list)))  # Accuracy storage

    # Iterate through hyperparameter combinations
    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                train_index, val_index = list(kf.split(X))[l]  # Extract train and validation indices for fold l
                Xtrain, Xval = X[train_index], X[val_index]  # Training and validation data
                ytrain, yval = y[train_index], y[val_index]  # Training and validation labels

                # Train the model using dualHinge
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)

                # Predict using dualClassify
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)

                # Calculate validation accuracy
                cv_acc[l, i, j] = np.mean(yhat == yval)  # Compare predictions to true labels

    # Compute the average accuracies over k folds
    avg_cv_acc = np.mean(cv_acc, axis=0)

    # Identify the best lamb and kernel function based on average accuracy
    best_index = np.unravel_index(np.argmax(avg_cv_acc), avg_cv_acc.shape)
    best_lamb = lamb_list[best_index[0]]
    best_kernel = kernel_list[best_index[1]]

    return avg_cv_acc, best_lamb, best_kernel
