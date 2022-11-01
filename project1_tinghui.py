import os
import numpy as np
from mpi4py import MPI
import pickle
import time


# Mapper
def mapper(file_list, tmp_dir, rank):
    '''This mapper function collect data into a map (dictionary)
    '''

    counts = {}
    for file_name in file_list:
        f = open(file_name, 'r')
        for number in f:
            number = int(number)
            if not number in counts:
                counts[number] = 1
            else:
                counts[number] += 1
        f.close()

    with open(os.path.join(tmp_dir, 'tmp_{}.pkl'.format(str(rank))), 'wb') as tmp_file:
        pickle.dump(counts, tmp_file, pickle.HIGHEST_PROTOCOL)


# Reducer
def reducer(tmp_dir, n_per_node, k, rank):
    '''This reducer function merge all counts together
    '''
    total = np.zeros(n_per_node)
    for curr_file in os.listdir(tmp_dir):
        with open(os.path.join(tmp_dir, curr_file), 'rb') as read_file:
            curr_map = pickle.load(read_file)

        for i in range(n_per_node):
            target = i + rank * n_per_node
            total[i] += curr_map[target]

    index = np.argsort(total)[::-1][:k]
    count = total[index]
    index += rank * n_per_node
    # print("Rank {}: {}; {} \n".format(rank, idx, count))
    return index, count


if __name__ == '__main__':

    time_0 = time.time()
    comm = MPI.COMM_WORLD

    ## Set up path and parameter
    # folder that contains all the txt files
    data_dir = '/gpfs/projects/AMS598/Projects2022/project1'

    # folder of current job
    curr_dir = '/gpfs/projects/AMS598/class2022/tinghwu/project1'

    # the top k integers with the highest frequencies
    k = 5

    tmp_dir = os.path.join(curr_dir, 'tmp')
    if comm.rank == 0:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    
    comm.Barrier()
    time_1 = time.time()

    ## Set up processed files and numbers for each node
    # Processed files
    total_list = os.listdir(data_dir)
    total_list = [os.path.join(data_dir, name) for name in total_list if name[-4:] == '.txt']
    f_per_node = int(len(total_list) / comm.size)
    # TODO fix if f_per_node not divided exactly
    file_list = total_list[comm.rank*f_per_node:(comm.rank+1)*f_per_node]
    # print("{}: {}".format(comm.rank, file_list))

    # Processed numbers
    n_per_node = int(100 / comm.size)
    # TODO fix if n_per_node not divided exactly

    ## Run the mapper
    mapper(file_list, tmp_dir, comm.rank)
    time_2 = time.time()
    comm.Barrier()


    ## Run the reducer
    idx, count = reducer(tmp_dir, n_per_node, k, comm.rank)
    time_3 = time.time()
    comm.Barrier()

    ## Gather the results from all nodes to root node
    all_count = comm.gather(count)
    all_idx = comm.gather(idx)

    if comm.rank == 0:
        all_count = np.concatenate(all_count)
        all_idx = np.concatenate(all_idx)
        index = np.argsort(all_count)[::-1][:k]
        print("The top {} numbers with highest frequencies are:".format(k))
        print("Integer / Frequency")
        for i in index:
            print("{} / {}".format(all_idx[i], int(all_count[i])))

    time_4 = time.time()
    t_setup = time_1 - time_0
    t_mapper = time_2 - time_1
    t_reducer = time_3 - time_2
    t_finish = time_4 - time_3
    print("Node {}: {}, {}, {}, {}".format(comm.rank, t_setup, t_mapper, t_reducer, t_finish))


