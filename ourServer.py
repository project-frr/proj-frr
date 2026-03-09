import queue
import random
import numpy as np
from frequency_oracle import *
from utils import *
from collections import deque


np.random.seed(1)
random.seed(1)


class ourServer(object):
    def __init__(self, party_list, domain_size, epsilon):
        self.num_party = len(party_list)
        assert self.num_party > 0
        self.party_list = party_list
        self.domain_size = int(domain_size)
        self.epsilon = epsilon
    
    def _merge_distribution_phase_1(self):
        self.overall_ns_frequency_hist_phase_1 = np.zeros(self.domain_size)
        self.overall_user_num_phase_1 = 0
        self.data_list_phase_1 = []
        for party in self.party_list:
            self.overall_ns_frequency_hist_phase_1 += (party.ns_frequency_hist_phase_1 * party.underlying_users[1])
            self.overall_user_num_phase_1 += party.underlying_users[1]
            self.data_list_phase_1.append(party.data[party.underlying_users[0]:party.underlying_users[1]])  
        self.overall_sw_transform = party.sw_transform
        self.overall_sw_n = party.sw_n
        self.overall_sw_max_iteration = party.sw_max_iteration
        self.overall_sw_loglikelihood_threshold = party.sw_loglikelihood_threshold
        self.overall_ns_frequency_hist_phase_1 /= self.overall_user_num_phase_1
        fo = Frequency_oracle(data=[], frequency_oracle_name = 'SW', epsilon=self.epsilon, domain_size=self.domain_size)
        overall_sw_hist_smooth = fo._EMS(n=self.overall_sw_n, ns_hist=self.overall_ns_frequency_hist_phase_1, transform=self.overall_sw_transform, \
            max_iteration=self.overall_sw_max_iteration, loglikelihood_threshold=self.overall_sw_loglikelihood_threshold)
        overall_sw_hist_woS = fo._EM(n=self.overall_sw_n, ns_hist=self.overall_ns_frequency_hist_phase_1, transform=self.overall_sw_transform, \
            max_iteration=self.overall_sw_max_iteration, loglikelihood_threshold=self.overall_sw_loglikelihood_threshold)
        self.overall_sw_hist_smooth = overall_sw_hist_smooth
        self.overall_sw_hist_woS = overall_sw_hist_woS
        self.flatten_data_phase_1 = np.concatenate([np.array(data).flatten() for data in self.data_list_phase_1])

    
    def _initialize_overall_CPS_tree(self):
        self._merge_distribution_phase_1()
        domain_item_set = set(list(np.arange(self.domain_size, step=1, dtype=int)))
        self.approximated_segments, self.approximated_intervals, self.coefficients, self.segment_to_interval_map = \
            self._CPS_fitting(domain_item_set=domain_item_set, data_list_alpha_frac=self.flatten_data_phase_1, epsilon=self.epsilon, plot=True)

    
    def phase_3(self):
        self.add_freq_to_overall_tree()
        self._post_weighted_average_and_freq_consistency()
    
    def add_freq_to_overall_tree(self):
        self.overall_CPS_tree = self.party_list[0]
        for node_idx, node_overall in enumerate(self.overall_CPS_tree.tree.all_nodes()):
            if not node_overall.is_root():
                this_node_count = node_overall.data.frequency * node_overall.user_num
                this_node_count_all = node_overall.user_num
                for current_CPS_tree in self.party_list[1:]:
                    node_current = current_CPS_tree.tree.all_nodes()[node_idx]
                    this_node_count += (node_current.data.frequency * node_current.user_num)
                    this_node_count_all += node_current.user_num
                node_overall.data.frequency = this_node_count / this_node_count_all
                node_overall.data.error = self.get_one_theoretical_square_error(q = 1.0 / (math.exp(self.epsilon) + 1), p = 0.5, \
                    user_number = this_node_count_all, frequency_oracle_name = 'ORR')
    
    def get_one_theoretical_square_error(self, q, p, user_number, frequency_oracle_name, frequency = True, b = None):
        if frequency:
            var = q * (1 - q) / (user_number * (p - q) ** 2)
        else:
            var = user_number * q * (1 - q) / (p - q) ** 2
        if frequency_oracle_name == 'SW':
            var /= (2 * b) **2
        return var
    
    def _post_weighted_average_and_freq_consistency(self):
        self._weighted_average(self.overall_CPS_tree.tree, consider_underlying_distribution=True)
        self._frequency_consistency(self.overall_CPS_tree.tree)
    
    def _weighted_average(self, pripl_tree, consider_underlying_distribution = True):
        '''
        dfs: postorder traversal, implemented through tow stacks
        '''
        # global parameter
        nodes_num = len(pripl_tree.all_nodes()) - 1
        self.error_vector = np.zeros(nodes_num)
        # root node
        root = pripl_tree.get_node(pripl_tree.root)
        root.data.error_coef = np.zeros(nodes_num)
        # trasversal of all nodes
        node_index = 0
        for node in self._dfs_postorder_traveral(pripl_tree):
            node.data.initial_error_coef(node_index, nodes_num)
            if node.is_leaf():
                if consider_underlying_distribution:
                    self._update_frequency_in_wa(pripl_tree, node, is_leaf = True)
                self.error_vector[node_index] = node.data.error
            else:
                self.error_vector[node_index] = node.data.error
                alpha_1, alpha_2 = self._update_frequency_in_wa(pripl_tree, node, is_leaf = False)
                update_error_coef = alpha_1 * node.data.error_coef
                for child in pripl_tree.children(node.identifier):
                    update_error_coef += alpha_2 * child.data.error_coef
                node.data.error_coef = update_error_coef.copy()
                node.data.error = np.dot(update_error_coef, self.error_vector)
            node_index += 1
    
    def _frequency_consistency(self, pripl_tree, non_negative = True):
        '''
        bfs: implemented through queue
        '''
        for node in self._bfs(pripl_tree, return_root = True):
            if not node.is_leaf():
                parent_f = node.data.frequency
                children = pripl_tree.children(node.identifier)
                children_f = np.zeros(len(children))
                error_coef_matrix = np.zeros((len(children), len(pripl_tree.all_nodes())-1))
                for i in range(len(children)):
                    children_f[i] = children[i].data.frequency
                    error_coef_matrix[i] = children[i].data.error_coef
                update_children_f = self._update_frequency_in_fc(children_f, parent_f, non_negative)
                C_plus = np.where(update_children_f > 0)[0]
                coef = np.zeros(len(children))
                if len(C_plus) > 0:
                    coef[C_plus] = - 1 / len(C_plus)
                for i in range(len(children)):
                    children[i].data.frequency = update_children_f[i]
                    self.update_error = True
                    if self.update_error:
                        if i in C_plus: # f_c in C+, then we update its variance
                            coef_w = coef.copy()
                            coef_w[i] += 1
                            children[i].data.error_coef = (1 / len(C_plus)) * node.data.error_coef + np.dot(coef_w, error_coef_matrix)
                            children[i].data.error = np.dot(np.square(children[i].data.error_coef), self.error_vector)
    
    def _update_frequency_in_fc(self, children_f:np.array, parent_f, non_negative = True):
        # we can not compute this directly
        if parent_f == 0:
            C = np.zeros(np.shape(children_f))
        else:
            C = children_f.copy()
            C = C - (C.sum() - parent_f) / np.size(C)
            if non_negative:
                while (C < 0).any():
                    C[C < 0] = 0
                    mask = (C > 0) 
                    C[mask] += (parent_f - C.sum()) / np.sum(mask)
        return C
    
    def _bfs(self, CPS_tree, return_root = False):
        travl_queue = queue.Queue()
        root = CPS_tree.get_node(CPS_tree.root)
        travl_queue.put(root)
        while not travl_queue.empty():
            node = travl_queue.get()
            if not node.is_leaf():
                for child in CPS_tree.children(node.identifier):
                    travl_queue.put(child)
            if return_root or not node.is_root():
                yield node
    
    def _dfs_postorder_traveral(self, pripl_tree):
        # we do not return root, which does not store any estimate value
        stack_travl = deque()
        stack_visit = deque()
        root = pripl_tree.get_node(pripl_tree.root)
        for child in pripl_tree.children(root.identifier):
            stack_travl.append(child)
        while len(stack_travl) != 0:
            node = stack_travl[-1]
            if node.is_leaf():
                yield stack_travl.pop()
            else:
                if len(stack_visit) > 0 and node is stack_visit[-1]:
                    yield stack_travl.pop()
                    stack_visit.pop()
                else:
                    stack_visit.append(node)
                    for child in pripl_tree.children(node.identifier):
                        stack_travl.append(child)

    def _update_frequency_in_wa(self, CPS_tree, node, is_leaf):
        x_1 = node.data.frequency
        var_1 = node.data.error
        x_2 = 0
        var_2 = 0
        if is_leaf:
            x_2 = self.overall_sw_hist_smooth[node.data.interval_left: node.data.interval_right+1].sum()
            var_2 = np.square(x_2)
        else:
            for child in CPS_tree.children(node.identifier):
                x_2 += child.data.frequency
                var_2 += child.data.error
        alpha = var_2 / (var_1 + var_2)
        x = alpha * x_1 + (1 - alpha) * x_2
        node.data.frequency = x
        node.data.error = (var_1 * var_2-1) / (var_1 + var_2-2)
        return alpha, 1 - alpha
    
    def _CPS_fitting(self, domain_item_set, data_list_alpha_frac, epsilon, plot=False):
        _, int_to_item_dict = map_data(data_list_alpha_frac, domain_item_set)
        hist_input = self.overall_sw_hist_smooth
        check_points, intervals, intervals_middle_idx_dict, first_derivatives, first_derivatives_left, first_derivatives_right\
            = get_checkpoints(histogram = hist_input, int_to_item_dict = int_to_item_dict)
        approximated_segments, approximated_intervals, coefficients, segment_to_interval_mapping = \
            get_cubic_hermite_spline_approximated_intervals(check_points = check_points, intervals = intervals, \
            intervals_middle_idx_dict = intervals_middle_idx_dict, first_derivatives = first_derivatives, histogram = hist_input, \
                first_derivatives_left = first_derivatives_left, first_derivatives_right = first_derivatives_right, epsilon=epsilon, \
                    histogram_sw_woS=self.overall_sw_hist_woS)

        return approximated_segments, approximated_intervals, coefficients, segment_to_interval_mapping
    
    
    def query(self, query_range, server=None, CPS_tree = None):
        if CPS_tree is None:
            CPS_tree = self.overall_CPS_tree.tree
        ans = 0
        query_queue = queue.Queue()
        root = CPS_tree.get_node(CPS_tree.root)
        query_queue.put(root)
        while not query_queue.empty():
            node = query_queue.get()
            intersect_s = self.__intersection([node.data.interval_left, node.data.interval_right], query_range)
            sub_ans = 0
            if intersect_s > 0:
                sub_ans = node.data.frequency
                ans += sub_ans
            elif intersect_s < 0:
                if node.is_leaf():
                    l = max(query_range[0], node.data.interval_left)
                    r = min(query_range[1], node.data.interval_right)
                    d = node.data.interval_right - node.data.interval_left + 1
                    sub_ans = self._CPS_range_query(l, r, node.data.interval_left, node.data.interval_right, \
                        node.data.this_node_coefficients, node.data.frequency, server)
                    ans += sub_ans
                else:
                    for child in CPS_tree.children(node.identifier):
                        query_queue.put(child)
        ans = min(1, ans)
            
        return ans
    
    def __intersection(self, node_range, query_range):
        if node_range[0] > query_range[1] or node_range[1] < query_range[0]:
            # no overlap
            return 0
        elif node_range[0] >= query_range[0] and node_range[1] <= query_range[1]:
            # complete overlap
            return 1
        else:
            # partial overlap
            return -1
    
    def _CPS_range_query(self, intersect_query_l, intersect_query_r, segment_start, segment_end, this_node_coefficients, count, server=None):
        '''
        Args:
            [l,r]: the bound of range query in the segment.
            segment_start: i.e. ceil(s_{i}). he start point of the segment.
            d: the domain size of this segment
            slope: the refined slope of this segment
            count: the sum of counts or frequencies of items in this segment
        '''
        d = segment_end - segment_start + 1
        full_intergral = cubic_integral_(this_node_coefficients, segment_start, segment_end)
        intersect_intergral = cubic_integral_(this_node_coefficients, intersect_query_l, intersect_query_r, inter=True)
        ratio = intersect_intergral / max(full_intergral, 1e-10)
        if full_intergral == 0 or ratio > 1 or ratio < 0:
            ratio = (intersect_query_r - intersect_query_l) / (segment_end - segment_start)
        res = ratio * count 
        
        return res
