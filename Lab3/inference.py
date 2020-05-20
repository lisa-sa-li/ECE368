import numpy as np
import graphics
import rover
from rover import Distribution
from time import time

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: 
        a function that takes a hidden state and returns a Distribution for the next state
    observation_model: 
        a function that takes a hidden state and returns a Distribution for the observation from that hidden state
    observations: 
        a list of observations, one per hidden state (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    start = time()
    # Compute the forward messages
    print("starting forward...")
    for n in range(1, num_time_steps):
        forward_distr = Distribution()
        prev_msg = forward_messages[n-1]
        obs = observations[n]

        for state in prev_msg:
            # Get Distribution for the next hidden state at time = n
            next_states = transition_model(state)

            for s in next_states:
                observe = 1 if obs is None else observation_model(s)[obs]
                prev_val = next_states[s] * prev_msg[state] * observe
                # Only include states with non-zero probability 
                if prev_val > 0.0:
                    forward_distr[s] += prev_val

        forward_distr.renormalize()
        forward_messages[n] = forward_distr
    
    print("starting backwards...")
    for i in range(num_time_steps):
        n = num_time_steps - 1 - i # goes backwards
        back_distr = Distribution()
        for state in all_possible_hidden_states:
            if n == num_time_steps - 1:
                next_sum = 1
            else:
                # Get the message and observation before it
                next_msg = backward_messages[n+1]
                next_obs = observations[n+1]
                trans_model_state = transition_model(state)
                to_sum = []

                for s in next_msg:
                    observe = 1 if next_obs is None else observation_model(s)[next_obs]
                    to_sum.append(next_msg[s] * trans_model_state[s] * observe)
                next_sum = np.sum(to_sum)

            # Only include states with non-zero probability 
            if next_sum > 0.0:
                back_distr[state] = next_sum
        back_distr.renormalize()
        backward_messages[n] = back_distr

   
    # Compute the marginals 
    print("starting marg...")
    for n in range(num_time_steps):
        marg_distr = Distribution()
        forw_msg = forward_messages[n]
        back_msg = backward_messages[n]

        for state in forw_msg:
            marg_distr[state] = forw_msg[state] * back_msg[state]

        marg_distr.renormalize()
        marginals[n] = marg_distr

    print("This took %.2fs to run" % (time()- start))
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)
    w = [None] * num_time_steps
    backtrack = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    # Initialization
    init_distr = Distribution()
    for state in prior_distribution:
        if prior_distribution[state] != 0.0 and observation_model(state)[observations[0]]!= 0.0: # Make sure np.log doesn't error out
            init_distr[state] = np.log(prior_distribution[state]) + np.log(observation_model(state)[observations[0]]) 
    w[0] = init_distr

    for n in range(1, num_time_steps):
        w_n = w[n - 1]
        obs = observations[n]
        w_distr = Distribution()
        backtrack_dic = Distribution()
        for s in all_possible_hidden_states:
            max_val = -np.inf
            max_state = None

            for state in w_n:
                next_states = transition_model(state)
                if s in next_states:
                    probability = next_states[s]
                    curr_val = -np.inf if probability == 0 else np.log(probability) + w_n[state]

                    # If there's a new max, keep track of its value and state
                    if curr_val > max_val:
                        max_val = curr_val
                        max_state = state
                        
                    observe = 1 if obs is None else observation_model(s)[obs]
                    if observe > 0:
                        w_distr[s] = np.log(observe) + max_val
                        backtrack_dic[s] = max_state

        w[n] = w_distr
        backtrack[n] = backtrack_dic
                    
    # Backtrack
    estimated_hidden_states[-1] = max(w[-1], key=lambda key: w[-1][key]) # argmax at the last move
    for n in range(num_time_steps-2, -1, -1):
        prev_state = estimated_hidden_states[n+1]
        estimated_hidden_states[n] = backtrack[n+1][prev_state]

    return estimated_hidden_states


if __name__ == '__main__':
   
    # enable_graphics = True
    enable_graphics = False
    
    missing_observations = True
    # missing_observations = False
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    # Question 1 & 2
    timestep = 30 if missing_observations else (num_time_steps - 1)

    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    # Question 3
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # Question 4
    marginal_sequence = [max(marginals[i], key=lambda key: marginals[i][key]) for i in range(len(hidden_states))]
    fb_err = 1 - np.sum([hidden_states[i] == marginal_sequence[i] for i in range(len(hidden_states))]) / 100
    v_err = 1 - np.sum([hidden_states[i] == estimated_states[i] for i in range(len(hidden_states))]) / 100
    print("\nForward-back error: %.3f | Viterbi error: %.3f" % (fb_err, v_err))

    # Question 5
    for i in range(len(marginal_sequence)-1):
        if marginal_sequence[i+1] not in rover.transition_model(marginal_sequence[i]).keys():
            print("\nz_%d: %s -> z_%d: %s is invalid\n" % (i, marginal_sequence[i], i+1, marginal_sequence[i+1]))
            break

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
