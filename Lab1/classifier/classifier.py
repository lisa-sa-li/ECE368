import os.path
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import util
    
def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    spam = file_lists_by_category[0]
    ham = file_lists_by_category[1]

    # Dictionary containing every word and its frequency between the list of emails
    s_words = util.get_word_freq(spam)
    h_words = util.get_word_freq(ham)

    # Total number of words
    N_spam = sum(s_words.values())
    N_ham = sum(h_words.values())

    # Create the vocabulary
    W = set(list(s_words.keys()) + list(h_words.keys()))
    D = len(W)

    p_d = util.Counter()
    q_d = util.Counter()

    # Apply Laplace Smoothing
    for w in W:
        p_d[w] = (s_words[w] + 1)/(N_spam + D)
        q_d[w] = (h_words[w] + 1)/(N_ham + D)
    
    probabilities_by_category = (p_d, q_d)
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    freq = util.get_word_freq([filename])
    p_d = probabilities_by_category[0]
    q_d = probabilities_by_category[1]

    # log(pi) and log(1 - pi)
    spam = np.log(prior_by_category[0])
    ham = np.log(prior_by_category[1])

    # Applying MAP rule
    for w in freq:
        if w in p_d: # Must check for this condition otherwise log(0) will error
            spam += freq[w] * np.log(p_d[w])
        if w in q_d:
            ham += freq[w] * np.log(q_d[w])

    # Choose the larger value of the two
    email = "spam" if spam > ham else "ham"
    classify_result = (email, [spam, ham])

    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance\

    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                probabilities_by_category,
                                                priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    print("-------")

    # Modify the decision rule such that Type 1 and Type 2 errors can be traded off
    type_1, type_2 = [], []
    c = -1000

    # Iterate through 10 different values of c
    while c < 1001:
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            label = "spam" if log_posterior[0] > (log_posterior[1] + c) else "ham"
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0],totals[0],correct[1],totals[1]))
        type_1.append(totals[0]-correct[0])
        type_2.append(totals[1]-correct[1])

        c += 100

    # Plot the trade-off curve
    plt.figure()
    plt.plot(type_1, type_2)
    plt.title('Type 1 vs Type 2 Errors')
    plt.xlabel("Type 1")
    plt.ylabel("Type 2")
    plt.savefig('nbc.pdf')
    
   
   

 