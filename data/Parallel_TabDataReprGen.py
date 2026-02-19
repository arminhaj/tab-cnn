from TabDataReprGen import main
from multiprocessing import Pool
import sys

# number of files to process overall
num_filenames = 360
modes = ["c","m","cm","s"]

filename_indices = list(range(num_filenames)) * 4
mode_list = ( 
        [modes[0]] * num_filenames 
    +   [modes[1]] * num_filenames 
    +   [modes[2]] * num_filenames 
    +   [modes[3]] * num_filenames
    )


if __name__ == "__main__":
    # number of processes will run simultaneously
    pool = Pool(11)
    results = pool.map(main, zip(filename_indices, mode_list))
