import math
import random
import os
import re
import numpy as np
import CPS_tree
import ourServer

# range_query folder
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

np.random.seed(1)
random.seed(1)

def read_data(data_path, Domain):
    print('reading data and partitioning users...')
    dataset = np.loadtxt(data_path, np.int32)
    users = dataset.shape[0]
    np.random.shuffle(dataset)
    space_true_value = np.zeros(Domain, dtype=int)
    for data_per_user in dataset:
        space_true_value[data_per_user] += 1
    return [dataset, space_true_value, users] 

def convert_read_data_to_load_format(data_path, domain_size):
    dataset, space_true_value, users_num = read_data(data_path, domain_size)
    selected_data = dataset
    domain_sizes = domain_size
    attributes = ["dimension_1"]
    
    return selected_data, domain_sizes, attributes, space_true_value, users_num

def load_query_data(query_path, query_time):
    query_list = []
    with open(query_path, 'r') as inf_query:
        for _ in range(query_time):
            str_line = inf_query.readline()
            matches = re.findall(r'\d+', str_line)
            if matches:
                query_list.append(list(map(int, matches[:2])))  
    
    return query_list


def run_fed_dim1_now():

    domain = pow(2, 10)
    num_parties = 3
    alpha = 0.3
    error_list = []
    for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]: 
        MSE_list = []
        Loop = 10
        num_queries = 1000

        query_wise_MSE_accumulator = np.zeros((num_queries, Loop))
        for loop in range(0, Loop):
            MSE_loop_list = []
            queries = load_query_data(query_path, query_time = num_queries)
            cps_tree_list = []
            user_num_list = []
            space_true_value_party_list = []
            for idx in range(num_parties):

                data, domain_size, _, space_true_value_list, users_num = convert_read_data_to_load_format(data_set_path, domain)
                user_num_list.append(users_num)
                cps_tree = CPS_tree.CPS_tree(data, domain_size, epsilon = epsilon, alpha = alpha, search_granularity=2 ** 7)
                cps_tree.phase_1()
                cps_tree_list.append(cps_tree)
                
                space_true_value_party_list.append(space_true_value_list)
            
            server = ourServer.ourServer(party_list=cps_tree_list, domain_size=domain_size, epsilon=epsilon)
            server._initialize_overall_CPS_tree()
            
            print('Phase I done.')
            
            for idx in range(num_parties):
                current_cps_tree = server.party_list[idx]
                current_cps_tree.calibrate_CPS_tree_segments(server)
                current_cps_tree.phase_2(server)
                print('Phase II done.')
            server.phase_3()
            print('Phase III done.')
            for q_idx, a_query in enumerate(queries):
                fed_ans = 0
                fed_true_fre = 0
                for idx in range(num_parties):
                    fed_true_fre += space_true_value_party_list[idx][a_query[0]:a_query[1] + 1].sum() 
                fed_ans = server.query(a_query)
                fed_true_fre /= sum(user_num_list)
                var_MSE = math.pow(fed_true_fre - fed_ans, 2)
                MSE_loop_list.append(var_MSE)
                query_wise_MSE_accumulator[q_idx, loop] = var_MSE
            MSE_loop_list = np.array(MSE_loop_list)
            loop_mean_MSE = np.mean(MSE_loop_list)
            loop_std_MSE = np.std(MSE_loop_list)
            
            MSE_list.append(np.mean(MSE_loop_list))
        epsilon_mean_MSE = np.mean(MSE_list)
        epsilon_std_MSE = np.std(MSE_list)
        error_list.append((epsilon, epsilon_mean_MSE))
        print(f"Current epsilon: {epsilon}, mean MSE: {epsilon_mean_MSE}, std MSE: {epsilon_std_MSE}")
        print(MSE_list)
        
    print(error_list)


if __name__ == '__main__':

    print(PROJECT_PATH)
    
    desired_directory = PROJECT_PATH + '/ours'
    data_set_path = desired_directory 
    run_fed_dim1_now()
    
