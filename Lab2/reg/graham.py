import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    x_range = np.linspace(-1, 1, 200)
    y_range = np.linspace(-1, 1, 200)
    mesh_x, mesh_y = np.meshgrid(x_range, y_range)

    mu_a = np.array([0, 0])
    cov_a = np.diag([beta, beta])

    stacked_mesh = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=-1)
    prior = util.density_Gaussian(mu_a, cov_a, stacked_mesh).reshape(mesh_x.shape)

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_xlabel(r'$a_0$')
    ax.set_ylabel(r'$a_1$')
    ax.set_title(r'Prior distribution $p(\mathbf{a})$')

    plt.plot(-0.1, -0.5, 'k+', markersize=14, label=r'True $\mathbf{a}$')
    plt.contourf(mesh_x, mesh_y, prior, 100, cmap=cm.viridis)

    ax.legend()
    plt.savefig('graham/prior.pdf')
    #plt.show()
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from training set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    n = len(x)
    X = np.ones((n, 2))
    X[:, 1] = x[:, 0] # Pad with a column of ones

    x_range = np.linspace(-1, 1, 200)
    y_range = np.linspace(-1, 1, 200)
    mesh_x, mesh_y = np.meshgrid(x_range, y_range)
    stacked_mesh = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=-1)

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_xlabel(r'$a_0$')
    ax.set_ylabel(r'$a_1$')

    if n == 1:
        title = r'$p(\mathbf{a}|x_1, z_1)$'
    elif n == 5:
        title = r'$p(\mathbf{a}|x_1, z_1, \dots, x_5, z_5)$'
    elif n == 100:
        title = r'$p(\mathbf{a}|x_1, z_1, \dots, x_{100}, z_{100})$'

    plt.plot(-0.1, -0.5, 'k+', markersize=14, label=r'True $\mathbf{a}$')

    Cov = sigma2 * np.linalg.inv(sigma2/beta * np.eye(2) + X.T @ X)
    mu = (1/sigma2) * Cov @ X.T @ z

    # print("Cov", Cov)
    # print("mu", mu)

    prior = util.density_Gaussian(mu.flatten(), Cov, stacked_mesh).reshape(mesh_x.shape)
    plt.contourf(mesh_x, mesh_y, prior, 200, cmap=cm.viridis)
    ax.set_title('Posterior Distribution ' + title)
    ax.legend()
    plt.savefig('graham/posterior{}.pdf'.format(n))
    #plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """

    n = len(x)
    num_train = len(x_train)
    X = np.ones((n, 2))
    X[:, 1] = x

    # Predicted target mean
    preds = X @ mu

    # Predicted target variance
    var = np.diag(X @ Cov @ X.T) + sigma2
    stdev = np.sqrt(var)
    print("stdev", stdev)

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1)
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    ax.set_xlabel(r'Input')
    ax.set_ylabel(r'Target')
    ax.set_title('Predictions with {} training samples'.format(num_train))

    plt.errorbar(x, preds, yerr=stdev, fmt='k', capsize=3, label='Predictions')
    plt.scatter(x_train.squeeze(), z_train.squeeze(), c='r', label='Training samples')

    ax.grid()
    ax.legend()
    plt.savefig('graham/predict{}.pdf'.format(num_train))
    # plt.show()

    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1

    # prior distribution p(a)
    priorDistribution(beta)
    
    # number of training samples used to compute posterior
    for ns in [1, 5, 100]:
    
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)

        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
