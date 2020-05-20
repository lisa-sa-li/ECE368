import numpy as np
import graphics
import rover
from rover import Distribution

class WDict(dict):
    def __missing__(self, key):
        return -np.inf

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
        - P(Z0)
    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
        - Transition probability P(Zn | Z(n-1))
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
        - emission probability P(Xn | Zn)
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
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    print('Computing forwards messages')
    for n, obs in enumerate(observations):
        if n%10 == 0:
            print('{}%'.format(n))
        fwd_n = Distribution()
        for state in all_possible_hidden_states:
            if n == 0: # Initial case for Z_0
                sum_prev_fwd = prior_distribution[state]
            else:
                # Sum is over all previous states, list comprehension makes that easy
                # Recall that transition_model(s)(k) = P(Zn=k | Zn-1=s)
                # Don't need to sum over all_possible_hidden_states here, only needs to sum over states in
                # forward_messages[n-1]
                # sum_prev_fwd = np.sum([forward_messages[n - 1][prev_state] *
                #                        transition_model(prev_state)[state] for prev_state in
                #                        all_possible_hidden_states])
                sum_prev_fwd = np.sum([forward_messages[n - 1][prev_state] *
                                       transition_model(prev_state)[state] for prev_state in
                                       forward_messages[n-1]])

            # For missing observations, P(X|Z) = 1
            if sum_prev_fwd > 0:
                if obs is None:
                    fwd_n[state] = sum_prev_fwd
                else:
                    state_obs = observation_model(state)[obs]
                    if state_obs > 0:
                        fwd_n[state] = state_obs * sum_prev_fwd

        # Renormalize to prevent underflow / overflow
        fwd_n.renormalize()

        forward_messages[n] = fwd_n

    # Compute the backward messages

    print('Computing backwards messages')
    # for i, obs in enumerate(reversed(observations)): # Backwards messages start at the end and go to beginning
    for i in range(num_time_steps):
        n = num_time_steps-i-1  # Time step we are currently computing
        if i%10 == 0:
            print('{}%'.format(i))
        back_n = Distribution()
        for state in all_possible_hidden_states:
            if n == num_time_steps-1:  # Initial case for Z_(N-1)
                sum_next_back = 1
            else:
                state_transition = transition_model(state)
                next_obs = observations[n+1]

                if next_obs is None:
                    sum_next_back = np.sum([backward_messages[n + 1][next_state] *
                                            state_transition[next_state] for next_state in backward_messages[n+1]])
                else:
                    sum_next_back = np.sum([backward_messages[n + 1][next_state] *
                                            observation_model(next_state)[next_obs] *
                                            state_transition[next_state] for next_state in backward_messages[n+1]])

            if sum_next_back > 0:
                back_n[state] = sum_next_back

        back_n.renormalize()

        backward_messages[n] = back_n

    print('Done.')

    # Compute the marginals

    print('Computing marginals')
    for n in range(num_time_steps):
        marginal_n = Distribution()
        for state in all_possible_hidden_states:
            marginal_n[state] = forward_messages[n][state]*backward_messages[n][state]

        if sum(marginal_n.values()) > 0:
            marginal_n.renormalize()
        marginals[n] = marginal_n
            
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
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states
        - P(Z0)
    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
        - Transition probability P(Zn | Z(n-1))
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
        - emission probability P(Xn | Zn)
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)

    # Preallocate arrays
    estimated_hidden_states = [None] * num_time_steps
    # w = [None] * num_time_steps
    # backtrack = [None] * num_time_steps
    w = [WDict() for _ in range(num_time_steps)]
    backtrack = [dict() for _ in range(num_time_steps)]

    for n, obs in enumerate(observations):
        if n % 10 == 0:
            print('{}%'.format(n))

        for state in all_possible_hidden_states:
            # Compute the most likely state we came from
            max_log_state_prob = -np.inf
            max_log_state = None  # Corresponding state

            if n == 0:  # Base case
                if prior_distribution[state] != 0:  # Necessary because of the log
                    max_log_state_prob = np.log(prior_distribution[state])
                else:
                    max_log_state_prob = -np.inf
            else:
                # for prev_state in all_possible_hidden_states:
                for prev_state in w[n-1]:
                    transition_prob = transition_model(prev_state)[state]
                    if transition_prob != 0:
                        log_state_prob = np.log(transition_prob) + w[n-1][prev_state]
                    else:
                        log_state_prob = -np.inf

                    if log_state_prob > max_log_state_prob:
                        max_log_state_prob = log_state_prob
                        max_log_state = prev_state

            emission_prob = observation_model(state)[obs] if obs is not None else 1
            if emission_prob != 0:
                # Will default to -np.inf
                w[n][state] = np.log(emission_prob) + max_log_state_prob
                backtrack[n][state] = max_log_state

    # Backtracking
    end_state = max(w[-1], key=lambda k: w[-1][k])
    estimated_hidden_states[-1] = end_state

    for n in range(num_time_steps-2, -1, -1):
        prev = estimated_hidden_states[n+1]
        estimated_hidden_states[n] = backtrack[n+1][prev]

    return estimated_hidden_states

def get_sequence_from_marginals(_marginals):
    _sequence = []
    for m in _marginals:
        _sequence.append(m.get_mode())
    return _sequence


def error_probabilities(_marginals, _map_sequence, _hidden_states):

    marginal_correct = 0
    map_correct = 0

    for i in range(len(_hidden_states)):
        if _marginals[i] == _hidden_states[i]:
            marginal_correct += 1
        if _map_sequence[i] == _hidden_states[i]:
            map_correct += 1

    return 1-marginal_correct/100, 1-map_correct/100


def check_valid_sequence(_marginals, transition_model):
    _sequence = None
    for i in range(1, len(_marginals)):
        if _marginals[i] not in transition_model(_marginals[i-1]).keys():
            _sequence = [_marginals[i-1], _marginals[i]]
            break

    return _sequence, (i-1, i)


if __name__ == '__main__':

    enable_graphics = False
    parts = [1, 2, 3, 4]

    if 1 in parts:
        # Part 1: Run on file with no missing data
        # load data
        hidden_states, observations = rover.load_data('test.txt')
        num_time_steps = len(hidden_states)

        all_possible_hidden_states = rover.get_all_hidden_states()
        all_possible_observed_states = rover.get_all_observed_states()
        prior_distribution = rover.initial_distribution()

        print('Running forward-backward...')
        marginals = forward_backward(all_possible_hidden_states,
                                     all_possible_observed_states,
                                     prior_distribution,
                                     rover.transition_model,
                                     rover.observation_model,
                                     observations)
        print('\n')

        timestep = num_time_steps - 1
        print("Most likely parts of marginal at time %d:" % (timestep))
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
        print('\n')

    if 2 in parts:
        # Part 2: Run on file with missing data
        print('== Loading missing data ==')

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



        # timestep = num_time_steps - 1
        timestep = 30
        print("Most likely parts of marginal at time %d:" % (timestep))
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
        print('\n')

    if 3 in parts:
        print('Running Viterbi...')
        estimated_states = Viterbi(all_possible_hidden_states,
                                   all_possible_observed_states,
                                   prior_distribution,
                                   rover.transition_model,
                                   rover.observation_model,
                                   observations)
        print('\n')

        print("Last 10 hidden states in the MAP estimate:")
        for time_step in range(num_time_steps - 10, num_time_steps):
            print(estimated_states[time_step])

    if 4 in parts:
        marginal_sequence = get_sequence_from_marginals(marginals)

        marginal_error, map_error = error_probabilities(marginal_sequence, estimated_states, hidden_states)
        print('Marginal Error: {} | MAP Error: {}'.format(marginal_error, map_error))


         # Question 4
        marginal_sequence = [max(marginals[i], key=lambda key: marginals[i][key]) for i in range(len(hidden_states))]
        fb_err = 1 - np.sum([hidden_states[i] == marginal_sequence[i] for i in range(len(hidden_states))]) / 100
        v_err = 1 - np.sum([hidden_states[i] == estimated_states[i] for i in range(len(hidden_states))]) / 100
        print("\nForward-back error: %.3f | Viterbi error: %.3f" % (fb_err, v_err))


        # Check for a valid sequence
        # sequence, idx = check_valid_sequence(marginal_sequence, rover.transition_model)
        # if sequence is None:
        #     print('No illegal state transition')
        # else:
        #     print('Illegal state transition:')
        #     print('{} -> {}'.format(*sequence))
        #     print('From time:')
        #     print('{} -> {}'.format(*idx))

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

