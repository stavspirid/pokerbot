"""
This script generates synthetic datasets of noisy card images for training and testing a card recognition model. 
It uses the functions `generate_data_set` and `dataset_to_npz` from the data_sets module to create image datasets, 
split them into training and testing sets, and store them in compressed NPZ files for later use.

Dependencies:
- data_sets: This script imports `generate_data_set` and `dataset_to_npz` from the `data_sets` module.
- The generated datasets are saved in directories 't_dataset' and 't_dataset_testing'.

"""

from data_sets import generate_data_set,dataset_to_npz

def main():
    #Generate and save training data
    generate_data_set(15000, 't_dataset')
    dataset_to_npz('t_dataset','dataset_train_4_15000_08.npz')

    #Generate and save testing data
    generate_data_set(1000, 't_dataset_testing')
    dataset_to_npz('t_dataset_testing','dataset_test_4_1000_08.npz')

    
if __name__ == "__main__":
    main()