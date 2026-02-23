from TabDataReprGen import main, TabDataReprGen
from multiprocessing import Pool

modes = ["c","m","cm","s"]

if __name__ == "__main__":
    num_filenames = len(TabDataReprGen().get_filenames())
    filename_indices = list(range(num_filenames)) * len(modes)
    mode_list = (
            [modes[0]] * num_filenames
        +   [modes[1]] * num_filenames
        +   [modes[2]] * num_filenames
        +   [modes[3]] * num_filenames
        )
    args = zip(filename_indices, mode_list)
    with Pool(11) as pool:
        pool.map(main, args)
