from TabDataReprGen import main, TabDataReprGen
from multiprocessing import Pool

modes = ["c","m","cm","s"]

if __name__ == "__main__":
    gen = TabDataReprGen()
    filenames = gen.get_filenames()
    num_filenames = len(filenames)
    if num_filenames == 0:
        raise FileNotFoundError(
            f"No .jams annotation files found in {gen.path_anno}. "
            "Expected GuitarSet annotations at data/GuitarSet/annotation."
        )
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
