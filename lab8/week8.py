import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """

        initial_emission = seq[0]
        v_initial_list = [self.pi[i] * self.B[i][self.emissions_dict[initial_emission]] for i in range(len(self.pi))]
        path_dict = {idx:[] for idx, value in enumerate(v_initial_list)}
        vt_dict = {}

        for emission in seq[1:]:

            current_emission_idx = self.emissions_dict[emission]
            new_path_dict = {key:value[:] for (key, value) in path_dict.items()}
            new_v_list = []

            for state in self.states_dict:
                current_state_idx = self.states_dict[state]
                current_emission_probability = self.B[current_state_idx][current_emission_idx]
                vt_dict = {idx:(value * self.A[idx][current_state_idx] * current_emission_probability) for idx, value in enumerate(v_initial_list)}
                max_probability_state_idx = max(vt_dict, key=lambda x: vt_dict[x])
                new_v_list.append(vt_dict[max_probability_state_idx])
                new_path_dict[current_state_idx] = path_dict[max_probability_state_idx] + [max_probability_state_idx]

            v_initial_list = new_v_list
            path_dict = {key:value[:] for (key, value) in new_path_dict.items()}

        vt_dict = {idx:value for idx, value in enumerate(v_initial_list)}
        state_lookup_dict = {value:key for (key, value) in self.states_dict.items()}

        max_probability_state_idx = max(vt_dict, key=lambda x: vt_dict[x])
        final_path_list = path_dict[max_probability_state_idx]
        final_path_list.append(max_probability_state_idx)

        final_state_seq = [state_lookup_dict[state_index] for state_index in final_path_list]
        return final_state_seq
