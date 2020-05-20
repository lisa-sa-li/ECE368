import numpy as np
import matplotlib.pyplot as plt
import util

def getCoordinates():
    # Create the meshgrid and get coordinates
    xx = np.linspace(-1., 1., 100)
    yy = np.linspace(-1., 1., 100)
    X, Y = np.meshgrid(xx, yy)

    x = X.flatten()
    y = Y.flatten()

    coordinates = np.zeros([len(x), 2])
    coordinates[:, 0] = x
    coordinates[:, 1] = y

    return X, Y, coordinates

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    # Get the axis and coordinates
    X, Y, coordinates = getCoordinates()

    # Get the density of the Gaussian distribution
    mean = [0, 0]
    cov = [[beta, 0], [0, beta]]
    density = util.density_Gaussian(mean, cov, coordinates).reshape(X.shape)

    # Plot
    plt.clf()
    plt.contour(X, Y, density, 100)
    plt.plot(-0.1, -0.5, marker='o', markersize=10, color='blue', label="True a")
    plt.title('Prior distribution of a')
    plt.xlabel("a_0")
    plt.ylabel("a_1")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.legend()
    plt.savefig('prior.pdf')
    return 

def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    # Create the meshgrid and get coordinates
    n = x.shape[0]
   
    # Get the axis and coordinates
    X, Y, coordinates = getCoordinates()

    # Add [1] to the beginning of the x's 
    x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

    # Calculate Cov and mu and density
    Cov = sigma2 * np.linalg.inv(sigma2/beta * np.eye(x.shape[1]) + np.transpose(x) @ x)
    mu = (1/sigma2 * Cov @ np.transpose(x) @ z).reshape(-1)
    density = util.density_Gaussian(mu, Cov, coordinates).reshape(X.shape)

    # Plot
    plt.clf()
    plt.title('Prior distribution of a with n=' + str(n))
    plt.contour(X, Y, density, 100)
    plt.plot(-0.1, -0.5, marker='o', markersize=10, color='blue', label='True a')
    plt.xlabel("a_0")
    plt.ylabel("a_1")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.legend()
    plt.savefig("posterior" + str(n) + '.pdf')
   
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
    # Add [1] to the beginning of the x's 
    n = len(x_train)
    x_new = np.append(np.ones([1, len(x)]), np.array([x]), axis=0)
    
    # Calculate predictions and standard deviation
    pred = np.transpose(mu) @ x_new
    var = np.transpose(x_new) @ Cov @ x_new + sigma2 * np.eye(x_new.shape[1])
    std_dev = np.sqrt(np.diag(var))

    # Plot
    plt.clf()
    plt.title('Predictions with n=' + str(n))
    plt.errorbar(x, pred, yerr=std_dev)
    plt.scatter(x_train, z_train, color="r", label="Training")
    plt.xlabel("Input")
    plt.ylabel("Target")
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    plt.legend()
    plt.savefig("predict" + str(n) + '.pdf')
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    for ns in [1, 5, 100]: 
    
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        
        # prior distribution p(a)
        priorDistribution(beta)
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
