import math
import numpy as np
import graphics
import rover
import itertools as it

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

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

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
    beta_init = rover.Distribution()
    for i, hid_state in enumerate(all_possible_hidden_states):
        beta_init[hid_state] = 1
    backward_messages[-1] = beta_init

    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    test = transition_model((0,0,'stay'))

    for iter in range(num_time_steps-1):
        next_states = rover.Distribution()
        for hs_id, hid_state in enumerate(forward_messages[iter]):
            next_hid_states = transition_model(hid_state)
            for nhs_id, nhs in enumerate(next_hid_states):
                #testy = observation_model(nhs)[observations[iter]]
                if observations[iter+1] is not None:
                    obs_val = observation_model(nhs)[observations[iter + 1]]
                else:
                    obs_val = 1
                next_states[nhs] += obs_val * next_hid_states[nhs] * forward_messages[iter][hid_state]

        #normalize and assign
        next_states.renormalize()
        forward_messages[iter+1] = next_states

    # TODO: Compute the backward messages
    for iter in reversed(range(1, num_time_steps)):
        prior_states = rover.Distribution()
        for hs_id, hid_state in enumerate(all_possible_hidden_states):
            # if (hid_state[0] == 10) and (hid_state[1] == 0):
            #     bleh = 3
            #     pass
            next_hid_states = transition_model(hid_state)
            for nhs_id, nhs in enumerate(next_hid_states):

                # testy = observation_model(nhs)[observations[iter]]
                # testy2 = next_hid_states[nhs]
                # testy3 = backward_messages[iter][nhs]
                # testy4 = testy * testy2 * testy3
                if observations[iter] is not None:
                    obs_val = observation_model(nhs)[observations[iter]]
                else:
                    obs_val = 1
                prior_states[hid_state] += obs_val * next_hid_states[nhs] * backward_messages[iter][nhs]
                # if observation_model(nhs)[observations[iter]] * next_hid_states[nhs] * backward_messages[iter][nhs] > 0:
                #     pass


        # normalize and assign
        a = {}
        for i, key in enumerate(prior_states):
            if key[0] >= 9:
                a[key] = prior_states[key]

        prior_states.renormalize()
        backward_messages[iter-1] = prior_states

        a = {}
        for i, key in enumerate(prior_states):
            if key[0] >= 9:
                a[key] = prior_states[key]

    # TODO: Compute the marginals
    for iter in range(num_time_steps):
        marginals[iter] = rover.Distribution()
        for ahs_id, a_hid_state in enumerate(forward_messages[iter]):
             marginals[iter][a_hid_state] = forward_messages[iter][a_hid_state] * backward_messages[iter][a_hid_state]
        marginals[iter].renormalize()

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

    # TODO: Write your code here
    num_time_steps = len(observations)
    messages = [None] * num_time_steps
    links = [None] * num_time_steps

    for idx, z0 in enumerate(prior_distribution):
        prior_distribution[z0] = real_log(prior_distribution[z0]) +  real_log(observation_model(z0)[observations[0]])
    messages[0] = prior_distribution

    #Now inductive case
    for znp1_idx in range(1, num_time_steps):
        np1_links = rover.Distribution()
        np1_dist = rover.Distribution()
        for i, state in enumerate(all_possible_hidden_states):
            np1_dist[state] = -np.inf

        wn = messages[znp1_idx - 1]
        obs = observations[znp1_idx]

        #find the max
        for i, zn in enumerate(wn):
            next_states = transition_model(zn)
            for i, znp1 in enumerate(next_states):
                val = real_log(next_states[znp1]) + wn[zn]
                if val > np1_dist[znp1]:
                    np1_dist[znp1] = val
                    np1_links[znp1] = zn

        for i, znp1 in enumerate(np1_dist):
            if observations[znp1_idx] is None:
                obs_val = 1
            else:
                obs_val = observation_model(znp1)[obs]
            np1_dist[znp1] = real_log(obs_val) + np1_dist[znp1]

        messages[znp1_idx] = np1_dist
        links[znp1_idx] = np1_links

    #compute best path
    new_top = find_max(messages[-1])
    estimated_hidden_states = []
    estimated_hidden_states.append(new_top)
    for i in reversed(range(1, num_time_steps)):
        estimated_hidden_states.append(links[i][new_top])
        new_top = links[i][new_top]

    return estimated_hidden_states

def find_max(messages):
    val = None
    top = -np.inf
    for i, znp1 in enumerate(messages):
        if messages[znp1] > top:
            val = znp1
    return val

def real_log(input):
    if input == 0:
        return -np.inf
    else:
        return np.log(input)

def compute_error_prob(est_hid_states, hid_states):
    count = 0
    for state in est_hid_states:
        if est_hid_states[state] == hid_states[state]:
            count = count + 1
    return 1 - (count/100)

if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
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



    timestep = 30#num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    # print("error = " + str(compute_error_prob(marginals, hidden_states)))

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")

    for time_step in range(9, -1, -1):  ##NEW CODE
        print(estimated_states[time_step])

    fb_err = 1 - np.sum([hidden_states[i] == max(marginals[i], key=lambda key: marginals[i][key]) for i in range(len(hidden_states))]) / 100
    v_err = 1 - np.sum([hidden_states[i] == estimated_states[i] for i in range(len(hidden_states))]) / 100
    print("Forward-back error: %.3f | Viterbi error: %.3f" % (fb_err, v_err))


    # print("error = " + str(compute_error_prob(estimated_states, hidden_states)))

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

