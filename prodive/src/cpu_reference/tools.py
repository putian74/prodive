import os


def get_hmm_length(hmm_file_path):
    with open(f'{hmm_file_path}', 'r') as file:
        # Read the file line by line.
        for line in file:
            # Remove leading and trailing whitespace.
            line = line.strip()
            # Locate the HHM length record.
            if line.startswith('LENG'):
                parts = line.split()
                if len(parts) >= 2:
                    value = parts[1]
                    return value


def get_hmm_number(hmm_file_path):
    with open(f'{hmm_file_path}', 'r') as file:
        # Read line by line while tracking the one-based line number.
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line.startswith('HMM  '):
                return line_number
    
def mkdir_hmm(folder_path,fragment):
    os.makedirs(folder_path+f'/{fragment}', exist_ok=True)

def get_null_number(hmm_file_path):
    with open(f'{hmm_file_path}', 'r') as file:
        # Read line by line while tracking the one-based line number.
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            # Locate the HHM null-model record.
            if line.startswith('NULL   '):
                return line_number
