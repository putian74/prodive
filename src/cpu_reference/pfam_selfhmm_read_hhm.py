import math
import numpy as np
import pandas as pd
import json
import tools as tools

# Example input: /path/to/profile.hhm
def hmm_read(adress):
    null_line = int(tools.get_null_number(adress))
    with open(f'{adress}', 'r') as file:
        lines = file.readlines()
    transition_probability_data = []
    split_line = lines[null_line+2].split()  # Split transition-probability row.
    series = pd.Series(split_line) 
    # Read all transition probabilities from the HHM file.
    transition_probability_data.append(pd.to_numeric(series, errors='coerce').fillna(np.inf).tolist()[:-3])
    # transition_probability_data = [lines[i].split() for i in range(27, len(lines), 3)]
    for row in range(null_line+4, len(lines), 3):  
        split_line = lines[row].split()
        series = pd.Series(split_line)
        cleaned_series = pd.to_numeric(series, errors='coerce').fillna(np.inf)[:-3]
        transition_probability_data.append(cleaned_series.tolist())
        
    split_line = lines[null_line-1].split()
    series = pd.Series(split_line[1:])       # Skip the leading NULL label.
    emission_probability_i_data = pd.to_numeric(series, errors='coerce').fillna(np.inf).tolist()
    emission_probability_m_data = []  # Match-state emission probabilities.

    #fasta_number = []
    #dict_match_col = {}
    for row in range(null_line+3, len(lines), 3): 
            value = lines[row].split()
            values = value[2:-1]
            if values != []:
                #dict_match_col[int(value[0])] = [int(value[-5])] 
                #fasta_number.append(int(value[-5]))
                series = pd.Series(values)
                cleaned_series = pd.to_numeric(series, errors='coerce').fillna(np.inf)
                emission_probability_m_data.append(cleaned_series.tolist())
    #print(transition_probability_data)   
    # print(emission_probability_m_data)
    # print(emission_probability_i_data)

    # print(start_transition_probability_data)
    return  transition_probability_data,emission_probability_m_data, emission_probability_i_data#fasta_number
# print(emission_probability_m_data)
# hmm_read("/path/to/PF00001.hhm")

