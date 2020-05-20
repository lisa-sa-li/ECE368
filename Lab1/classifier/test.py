
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
	
import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
 
from collections import defaultdict
 
 
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
    ### TODO: Write your code here
 
    ALPHA = 1
 
    spam = file_lists_by_category[0]
    ham = file_lists_by_category[1]
 
    N_spam = len(spam)
    N_ham = len(ham)
 
    def constant_factory(v):
        return lambda: v
 
    ft_spam = constant_factory(ALPHA / (N_spam + 2 * ALPHA))
    ft_ham = constant_factory(ALPHA / (N_ham + 2 * ALPHA))
 
    spam_dict = defaultdict(ft_spam)
    ham_dict = defaultdict(ft_ham)
    t_spam_dict = {}
 
    for file in spam:
        t_spam_dict = {}
        with open(file, "r", encoding='ISO-8859-1') as f:
            line = f.readlines()
            for l in line:
                word = l.split(" ")
                for w in word:
                    t_spam_dict[w] = 1
 
        for w in t_spam_dict:
            spam_dict[w] += 1
 
    for file in ham:
        t_ham_dict = {}
        with open(file, "r", encoding='ISO-8859-1') as f:
            line = f.readlines()
            for l in line:
                word = l.split(" ")
                for w in word:
                    t_ham_dict[w] = 1
 
        for w in t_ham_dict:
            ham_dict[w] += 1
 
    for w in spam_dict:
        spam_dict[w] -= ALPHA / (N_spam + 2 * ALPHA)  # DEFAULT VALUE
        spam_dict[w] = (spam_dict[w] + ALPHA) / (N_spam + 2 * ALPHA)
 
    for w in ham_dict:
        ham_dict[w] -= ALPHA / (N_ham + 2 * ALPHA)
        ham_dict[w] = (ham_dict[w] + ALPHA) / (N_ham + 2 * ALPHA)
 
    probabilities_by_category = [spam_dict, ham_dict]
    return probabilities_by_category
 
 
def classify_new_email(filename, probabilities_by_category, prior_by_category):
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
    ### TODO: Write your code here
    EPS = 1e-12
 
    is_in = defaultdict(int)
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        line = f.readlines()
        for l in line:
            word = l.split(' ')
            for w in word:
                is_in[w] = 1
 
    is_true = np.log(prior_by_category[0])
    is_false = np.log(prior_by_category[1])
    for w in is_in:
        is_true += np.log(probabilities_by_category[0][w])
        is_false += np.log(probabilities_by_category[1][w])
 
    minv = np.amin((is_true, is_false))
    normalize = minv + np.log(np.exp(is_true - minv) + np.exp(is_false - minv))
    is_true -= normalize
    is_false -= normalize
 
    if is_true >= is_false:
        res = 'spam'
    else:
        res = 'ham'
 
    classify_result = (res, [is_true, is_false])
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
    performance_measures = np.zeros([2, 2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'
 
    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label, log_posterior = classify_new_email(filename,
                                                  probabilities_by_category,
                                                  priors_by_category)
 
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1
 
    template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0], totals[0], correct[1], totals[1]))
 
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    ### basic type_cost = [[1,0],[0,1]] matrix
    ### want to adjust the former for type 1, latter for type 2
    ### refer to bishop ch.1.5 for more information on decision theory
    type_1 = []
    type_2 = []
 
    # Classify emails from testing set and measure the performance
    for i in range(100):
        performance_measures = np.zeros([2, 2])
        ratio = 10 * i - 700
        # all we need is the ratio of the cost assoc. with type 1 and type 2 errors
        # here the ratio is the log of (cost_of type 2/cost of type 1)
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category)
 
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = 1 if (log_posterior[0]) < (log_posterior[1] + ratio) else 0
            performance_measures[int(true_index), int(guessed_index)] += 1
 
            # print(performance_measures)
 
        totals = np.sum(performance_measures, 1)
 
        type_1.append(performance_measures[0][1] / totals[0])
        type_2.append(performance_measures[1][0] / totals[1])
 
    plt.plot(np.arange(100) * 10 - 700, type_1, label='Type 1 Errors')
    plt.plot(np.arange(100) * 10 - 700, type_2, label='Type 2 Errors')
    plt.legend()
    plt.xlabel('Type 2 vs Type 1 Weight Logratio')
    plt.ylabel('Frequency of Error')
    plt.savefig('Classification_Errors.png')
