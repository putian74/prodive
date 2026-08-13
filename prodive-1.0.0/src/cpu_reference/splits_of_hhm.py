import numpy as np  
import tools as tools
import os
import pfam_selfhmm_read_hhm
import json

np.set_printoptions(linewidth=500) 
# Define the transition-matrix coordinate map.
def p_mapping(row_A):
    mapping_ = []
    for i in range(row_A):
        if i > 0 and i < row_A - 1:
            row_B = 2 + (i-1) * 3
            col_B = 5 + (i-1) * 3
            new_mapping = [[row_B, col_B],
                        [row_B, col_B - 1],
                        [row_B, col_B + 1],
                        [row_B + 2, col_B],
                        [row_B + 2, col_B - 1],
                        [row_B + 1, col_B],
                        [row_B + 1, col_B + 1]]
            mapping_.append(new_mapping)
    return mapping_ 


def split_emission(start_1,end_1, am, ai): 

    M_emssion_ = am[start_1:end_1-1]  # Length is end_1 - start_1 - 1.
    # print(M_emssion_)
    I_emission = np.power(2, -ai/1000)
    # print(I_emission)
    #print(sum(I_emission))
    I_emission = (np.append(I_emission, 0)).tolist()
    #I_emission = I_emission.tolist()
    D_emssion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    #D_emssion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    M_emssion = (np.power(2, -M_emssion_/1000)).tolist()

    #print(M_emssion_fin)
    # print(I_emission)
    # print(D_emssion)
    emssion_matrix = []
    emssion_matrix.append(I_emission)
    for i in range(end_1 - start_1 - 1):
           row = M_emssion[i] + [0.0]
           #row = M_emssion[i]
           emssion_matrix.append(row)
           emssion_matrix.append(D_emssion)
           emssion_matrix.append(I_emission)
    # emssion_matrix.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return emssion_matrix
#start, end, adress_transition, adress_emission_m, adress_emission_i, adress_fasta_number, save_adress_transition, save_adress_start, save_adress_emission, save_fasta_number
def split_transition_and_emission(start_1, end_1, at, am, ai):
    # The selected match states span M(start_1 + 1) through M(end_1 - 1).
    # start_1 = np.array(start_1)
    # end_1 = np.array(end_1)
    at = np.array(at)
    am = np.array(am)
    ai = np.array(ai)
    A_ = at[start_1:end_1]
    # A_ = np.array([  
    #     [0.00997, 5.00874, 5.73109, 0.61958, 0.77255, 0.71866, 0.66827],  
    #     [0.01699, 5.00874, 4.58831, 0.61958, 0.77255, 0.68016, 0.70631],  
    #     [0.00996, 5.0103, 5.73265, 0.61958, 0.77255, 0.61678, 0.77584],  
    #     [0.00974, 5.03238, 5.75472, 0.61958, 0.77255, 0.51817, 0.90537],  
    #     [0.90465, 0.89203, 1.68472, 0.97678, 0.47244, 0.53346, 0.88327]  
    #print(A_)
    filter_matrix = np.power(2, -A_/1000)
    #print(filter_matrix)
    len_filter_matrix = len(filter_matrix)-1
    #print(len_filter_matrix)
    start_matrix_1 = np.zeros((1, (end_1 - start_1 - 1) * 3 + 1))
    ## I0 M1 D1 I1
    #print(initial_probability_list[start_1])
    start_matrix_1[0,0] =  0
    start_matrix_1[0,1] =  1
    start_matrix_1[0,2] =  0

    # Normalize the initial match/delete transition probabilities.
    total = filter_matrix[0,4] + filter_matrix[0,3]

    if total > 0:
        if filter_matrix[0,3] == 0:
            filter_matrix[0,3] = 0.01
            total = filter_matrix[0,4] + filter_matrix[0,3]
        filter_matrix[0] = [
            filter_matrix[0,4] / total,
            filter_matrix[0,3] / total,
            0, 0, 0, 0, 0
        ]
    else:
        # Use an all-zero row when the normalization denominator is zero.
        filter_matrix[0] = [0, 0, 0, 0, 0, 0, 0]
    #filter_matrix[0] = [filter_matrix[0,4]/(filter_matrix[0,4]+filter_matrix[0,3]), filter_matrix[0,3]/(filter_matrix[0,4]+filter_matrix[0,3]), 0, 0, 0, 0, 0]
    trans_matrix_1 = np.zeros((1 + 3 * (len_filter_matrix), 1 + 3 * (len_filter_matrix)))  

    mapping = p_mapping(len(filter_matrix))

    # mapping = [  
    #     [(2, 5), (2, 4), (2, 6), (4, 5), (4, 4), (3, 5), (3, 6)],  
    #     [(5, 8), (5, 7), (5, 9), (7, 8), (7, 7), (6, 8), (6, 9)],  
    #     [(8, 11), (8, 10), (8, 12), (10, 11), (10, 10), (9, 11), (9, 12)],  
    #     [(11, 14), (11, 13), (11, 15), (13, 14), (13, 13), (12, 14), (12, 15)],  
    #     [(14, 17), (14, 16), (14, 18), (16, 17), (16, 16), (15, 17), (15, 18)]  
    # ]  

    for i in range(filter_matrix.shape[0] - 2):  
        for j in range(filter_matrix.shape[1]):  
                row, col = mapping[i][j]
                trans_matrix_1[row - 1, col - 1] = filter_matrix[i + 1, j]  

    for i in range(2):
        trans_matrix_1[0, i] = filter_matrix[0, i]
    #trans_matrix_1[2 + 3 * (len_filter_matrix) - 4, 2 + 3 * (len_filter_matrix) - 1] = filter_matrix[len_filter_matrix][0]
    trans_matrix_1[2 + 3 * (len_filter_matrix) - 4, 2 + 3 * (len_filter_matrix) - 2] = filter_matrix[len_filter_matrix][1]
    #trans_matrix_1[2 + 3 * (len_filter_matrix) - 2, 2 + 3 * (len_filter_matrix) - 1] = filter_matrix[len_filter_matrix][2]
    trans_matrix_1[2 + 3 * (len_filter_matrix) - 2, 2 + 3 * (len_filter_matrix) - 2] = filter_matrix[len_filter_matrix][4]
    #trans_matrix_1[2 + 3 * (len_filter_matrix) - 3, 2 + 3 * (len_filter_matrix) - 1] = filter_matrix[len_filter_matrix][4]

    emssion_matrix_1 =  split_emission(start_1, end_1, am, ai)
    start_matrix_1 = np.array(start_matrix_1)
    trans_matrix_1 = np.array(trans_matrix_1)
    emssion_matrix_1 = np.array(emssion_matrix_1)

    return start_matrix_1,trans_matrix_1,emssion_matrix_1


# profile_hmm1_start_probs = []
# profile_hmm1_trans_probs = []
# profile_hmm1_emit_probs = []

# hmm_file_path_1 = "/path/to/PF00028.hhm"
# fragment = 3
# start_hmm = 0
# end_hmm = 1

# # print(hmm_initial_probability_list)
# transition_probability_data,emission_probability_m_data,emission_probability_i_data = pfam_selfhmm_read_hhm.hmm_read(hmm_file_path_1)
# # print(transition_probability_data)
# # print(emission_probability_m_data)
# # print(emission_probability_i_data)
# # print("------------------------------------------")
# for start_1 in range(start_hmm, end_hmm):  # Iterate over profile-HMM windows.
#     end_1= start_1 + fragment+1

#     profile_hmm1_start_probss,profile_hmm1_trans_probss,profile_hmm1_emit_probss = split_transition_and_emission(start_1, end_1,
#                                                                                                                 transition_probability_data,
#                                                                                                                 emission_probability_m_data,
#                                                                                                                 emission_probability_i_data,

#                                                                                                                 ) 

#     profile_hmm1_start_probs.append(profile_hmm1_start_probss)
#     profile_hmm1_trans_probs.append(profile_hmm1_trans_probss)
#     # print(profile_hmm1_trans_probss)#I0 M1 D1 I1 M2 D2 I2 M3 D3 I3 
#     # print("------------------------------------------")
# #     # print("------------------------------------------")
#     profile_hmm1_emit_probs.append(profile_hmm1_emit_probss)

# print(profile_hmm1_start_probs)
# print("------------------------------------------")
# print(profile_hmm1_trans_probs)
# print("------------------------------------------")
# print(profile_hmm1_emit_probs)
# print("------------------------------------------")
