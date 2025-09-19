"""
Computes parameter matches of initial axion field value and axion field velocity that give the correct dark matter abundance for a range of axion decay constants. Also computes the maximum field value reached.
Collects the results in a dictionary with fAGeV values as keys and a list of tuples (theta1, vheta1, theta_max, error) as values.
The results are saved to a JSON file named 'parameter_matches.json' (converts NumPy floats to native Python floats for JSON compatibility).
"""

# Modules
import numpy as np
import json
from multiprocessing import Pool
from kin_mis_utils import evolution_kinetic_mis, find_params

# Input
vheta1_range_list = [(10, 11, 1.0)] # Define vheta1 ranges and steps
fAGeV = 1e10 # Axion decay constant in GeV

def worker(vheta1_range):
    tautab = np.linspace(1, 5, 1000)  # Normalized conformal time, SHOULD START AT 1!
    matches = find_params(fAGeV, tautab, vheta1_range, target = 0.12, tol = 0.005)
    result = []

    for theta1, vheta1, err in matches:
        try:
            evolution = evolution_kinetic_mis(fAGeV, theta1, vheta1, tautab)
            theta_max = np.max(evolution["thetatab"])
        except Exception:
            theta_max = None # If failed, mark as None
        
        result.append((
            float(theta1),
            float(vheta1),
            float(theta_max) if theta_max is not None else None,
            float(err)
        ))

    return result

num_cpus = len(vheta1_range_list)

if __name__ == '__main__':
    with Pool(processes = num_cpus) as pool:
        results = pool.map(worker, vheta1_range_list)

     # Flatten the list of lists into a single list of matches
    all_matches = [match for sublist in results for match in sublist]

    # Create a single-key dictionary
    results_dict = {str(fAGeV): all_matches}

    with open(f'parameter_matches_{fAGeV:.0e}.json', 'w') as f:
        json.dump(results_dict, f)