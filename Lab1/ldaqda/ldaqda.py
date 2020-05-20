import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    # Extract the male and female data
    N_male = (y == 1).sum()
    N_female = (y == 2).sum()
    x_male = np.array([x[i] for i in range(len(y)) if y[i] == 1])
    x_female = np.array([x[i] for i in range(len(y)) if y[i] == 2])

    # Calculate means
    mu_male = np.mean(x_male, axis=0) 
    mu_female = np.mean(x_female, axis=0)
    mu = np.mean(x, axis=0)

    # Calculate covariances
    cov = np.matmul(np.transpose(x - mu), (x - mu)) / len(y)
    cov_male = np.matmul(np.transpose(x_male - mu_male), (x_male - mu_male)) / N_male
    cov_female = np.matmul(np.transpose(x_female - mu_female), (x_female - mu_female)) / N_female

    plot_contours(mu_male, mu_female, cov, cov_male, cov_female, x_male, x_female)

    return (mu_male,mu_female,cov,cov_male,cov_female)

def plot_contours(mu_male, mu_female, cov, cov_male, cov_female, x_male, x_female):

    # Create the meshgrid
    xx = np.linspace(50., 80., 100)
    yy = np.linspace(80., 280., 100)
    X, Y = np.meshgrid(xx, yy)

    LDA_male, LDA_female, QDA_male, QDA_female = [], [], [], []

    # Extract coordinates to find the Gaussian density at that point
    for i in range(len(X)):
        xy = []
        for j in range(len(X[0])):
            xy.append([X[i][j], Y[i][j]])

        xy = np.array(xy)

        LDA_male.append(util.density_Gaussian(mu_male, cov, xy))
        LDA_female.append(util.density_Gaussian(mu_female, cov, xy))
        QDA_male.append(util.density_Gaussian(mu_male, cov_male, xy))
        QDA_female.append(util.density_Gaussian(mu_female, cov_female, xy))

    LDA_male, LDA_female = np.array(LDA_male), np.array(LDA_female)
    QDA_male, QDA_female = np.array(QDA_male), np.array(QDA_female)

    # Find boundary points
    LDA_boundary = LDA_male - LDA_female
    QDA_boundary = QDA_male - QDA_female

    # Plot everything 
    plt.figure(1)
    plt.contour(X, Y, LDA_male)
    plt.contour(X, Y, LDA_female)
    plt.contour(X, Y, LDA_boundary, [0])
    plt.scatter(x_male[:, 0], x_male[:, 1], color="b")
    plt.scatter(x_female[:, 0], x_female[:, 1], color="r")
    plt.title('Height vs Weight by Gender')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.xlim((50, 80))
    plt.ylim((80, 280))
    plt.savefig('lda.pdf')

    plt.figure(2)
    plt.contour(X, Y, QDA_male)
    plt.contour(X, Y, QDA_female)
    plt.contour(X, Y, QDA_boundary, [0])
    plt.scatter(x_male[:, 0], x_male[:, 1], color="b")
    plt.scatter(x_female[:, 0], x_female[:, 1], color="r")
    plt.title('Height vs Weight by Gender')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.xlim((50, 80))
    plt.ylim((80, 280))
    plt.savefig('qda.pdf')
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
 
    # LDA
    LDA_male = util.density_Gaussian(mu_male, cov, x)
    LDA_female = util.density_Gaussian(mu_female, cov, x)
 
    LDA_pred = (LDA_female > LDA_male)*1 + 1
    LDA_num_correct = (y == LDA_pred).sum()
    mis_lda = 1 - LDA_num_correct/len(y)
 
    # QDA
    QDA_male = util.density_Gaussian(mu_male, cov_male, x)
    QDA_female = util.density_Gaussian(mu_female, cov_female, x)
 
    QDA_pred = (QDA_female > QDA_male)*1 + 1
    QDA_num_correct = (y == QDA_pred).sum()
    mis_qda = 1 - QDA_num_correct/len(y)

    print("mis_lda: %.3f | mis_qda: %.3f" % (mis_lda, mis_qda))
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
