import random
import numpy as np
from frequency_oracle import *
from treelib import Tree, Node
import queue, os
from collections import deque
from dim1_exp import PROJECT_PATH
from utils import *

np.random.seed(1)
random.seed(1)

class Node_CPS(object):
    def __init__(self, this_node_interval_left, this_node_interval_right, domain_size, frequency = None, this_node_coefficients = None, error = None):
        self.interval_left = this_node_interval_left
        self.interval_right = this_node_interval_right
        self.set_data(frequency, this_node_coefficients, error, domain_size, comp_partial_query_weight = True)
        self.left_user_ratio = 0
        self.error_coef = None
    
    def set_data(self, frequency = None, this_node_coefficients = None, error = None, domain_size = None, comp_partial_query_weight = False):
        if frequency is not None:
            self.frequency = frequency
        if this_node_coefficients is not None:
            self.this_node_original_coefficients = this_node_coefficients
            self.this_node_coefficients = this_node_coefficients
        if error is not None:
            self.error = error
        if comp_partial_query_weight:
            self.partial_query_weight = 2 * (self.interval_left + 1) * (domain_size - self.interval_right) / (domain_size * (domain_size - 1))
    
    def set_allocated_users(self, total_num, left_num, proportion):
        '''
        as we allocate users according indexes from small to large, given left num and the porpotion we will take,\
        we can refer the users' ids for the node
        '''
        self.user_num = round(left_num * proportion)
        self.left_user_num = left_num - self.user_num
        index_l = total_num - left_num
        index_r =index_l + self.user_num - 1
        self.allocated_users = [index_l, index_r]
    
    def reset_this_node_coefficients(self, coefficients):
        self.this_node_coefficients = coefficients
    
    def initial_error_coef(self, nodes_id, nodes_number):
        self.error_coef = np.zeros(nodes_number)
        self.error_coef[nodes_id] = 1
    

class CPS_tree(object):
    def __init__(self, data, domain_size, epsilon, alpha, search_granularity):
        '''
        Args:
            data(np.array): a 2-d ndarray with only 1-d data
            domain_size(int): the domain size of data
            epsilon(float): privacy budget
            error_metric(str): "max_absolute" or "square"
        '''
        assert np.ndim(data) == 1 or data.shape[1] == 1
        self.data = data.flatten()
        self.user_num = len(self.data)
        self.domain_size = int(domain_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.search_granularity = search_granularity
        self.update_error = True
    
    def phase_1(self):
        self._underlying_user_allocation()
        self.noisy_hist_SW, self.noisy_hist_SW_woS = self._estimate_underly_hist(self.underlying_users)

    def calibrate_CPS_tree_segments(self, server: object):
        self.approximated_segments, self.approximated_intervals, self.coefficients, self.segment_to_interval_map = \
            server.approximated_segments, server.approximated_intervals, server.coefficients, server.segment_to_interval_map

    def get_del_c_err(self, c, CPS_tree, x_mean_samples_less_than_c, x_mean_samples_greater_than_c):
        err = 0
        user_number = self.user_num / (len(CPS_tree.all_nodes()) - 1)
        q=1.0 / (math.exp(self.epsilon) + 1)
        p=0.5
        var = q * (1 - q) / (user_number * (p - q) ** 2)
        if len(x_mean_samples_greater_than_c) > 0:
            for x_mean in x_mean_samples_greater_than_c:
                prob = scipy.stats.norm.cdf(c, loc=x_mean, scale=var)
                err += (prob * c)
        if len(x_mean_samples_less_than_c) > 0:
            for x_mean in x_mean_samples_less_than_c:
                for perturbed_x in np.arange(x_mean, c + 0.01, 0.01):
                    prob = scipy.stats.norm.cdf(perturbed_x, loc=x_mean, scale=var)
                    err += (prob * x_mean)
        
        return err
    
    def get_preserve_c_err(self, c, CPS_tree, x_mean_samples_less_than_c, x_mean_samples_greater_than_c):
        err = 0
        user_number = self.user_num / (len(CPS_tree.all_nodes()) - 1)
        q=1.0 / (math.exp(self.epsilon) + 1)
        p=0.5
        var = q * (1 - q) / (user_number * (p - q) ** 2)
        if len(x_mean_samples_greater_than_c) > 0:
            for x_mean in x_mean_samples_greater_than_c:
                for perturbed_x in np.arange(0.01, c + 0.01, 0.01):
                    prob = scipy.stats.norm.cdf(perturbed_x, loc=x_mean, scale=var)
                    err += (prob * (c - perturbed_x))
        if len(x_mean_samples_less_than_c) > 0:
            for x_mean in x_mean_samples_less_than_c:
                for perturbed_x in np.arange(x_mean, c + 0.01, 0.01):
                    prob = scipy.stats.norm.cdf(perturbed_x, loc=x_mean, scale=var)
                    err += (prob * abs(x_mean - perturbed_x))

        return err
    
    def select_p1_p2(self, server: object, CPS_tree):
        p1_opt, p2_opt = None, None
        err_del = 1e10
        n = len(self.noisy_hist_SW)
        indices = np.arange(n)
        x_mean_samples = np.sort(np.random.choice(indices, size=100, p=self.noisy_hist_SW))
        for idx, sample in enumerate(x_mean_samples):
            x_mean_samples[idx] = self.noisy_hist_SW[sample]
        for p1 in np.arange(0.02, 0.5 * math.exp(-self.epsilon) + 0.01, 0.02):
            for p2 in np.arange(p1, 0.5 * math.exp(-self.epsilon) + 0.01, 0.02):
                c = p1 * p2
                x_mean_samples_less_than_c = x_mean_samples[x_mean_samples < c]
                x_mean_samples_greater_than_c = x_mean_samples[x_mean_samples >= c]
                err_del_this = self.get_del_c_err(c, CPS_tree, x_mean_samples_less_than_c, x_mean_samples_greater_than_c)
                err_preserve_this = self.get_preserve_c_err(c, CPS_tree, x_mean_samples_less_than_c, x_mean_samples_greater_than_c)
                if err_del_this <= err_preserve_this and err_del_this <= err_del:
                    p1_opt = p1
                    p2_opt = p2
                    err_del = err_del_this
        
        return p1_opt, p2_opt
    
    def phase_2(self, server: object):
        CPS_tree = self._CPS_tree_construction(self.approximated_segments, self.approximated_intervals, self.coefficients, self.segment_to_interval_map)
        self._uniform_user_allocation(CPS_tree)
        p1, p2 = self.select_p1_p2(server=server, CPS_tree=CPS_tree)
        for node in CPS_tree.all_nodes():
            if not node.is_root():
                self._estimate_node_frequency(node)
                this_node_overall_freq_phase_1 = sum(server.overall_sw_hist_smooth[node.data.interval_left:node.data.interval_right+1])
                this_node_local_freq_phase_1 = sum(self.noisy_hist_SW[node.data.interval_left:node.data.interval_right+1])
                
                beta = abs(this_node_local_freq_phase_1-this_node_overall_freq_phase_1) / this_node_overall_freq_phase_1
                upper_bound = min(1, this_node_local_freq_phase_1 * (1 + beta))
                lower_bound = max(1e-10, this_node_local_freq_phase_1 * (1 - beta))
                if node.data.frequency < lower_bound and node.data.frequency > 0:
                    xx = (lower_bound - node.data.frequency) / lower_bound
                    xi = exp(- self.epsilon) * (min(xx, 1/2))
                    node.data.frequency = node.data.frequency * (1 + xi)
                elif node.data.frequency > upper_bound and node.data.frequency < 1:
                    xx = (node.data.frequency - upper_bound) / node.data.frequency
                    xi = exp(- self.epsilon) * (min(xx,1/2))
                    node.data.frequency = node.data.frequency * (1 - xi)

                if p1 != None:
                    this_node_overall_freq_phase_1 = sum(server.overall_sw_hist_smooth[node.data.interval_left:node.data.interval_right+1])
                    if this_node_overall_freq_phase_1>0:
                        if node.data.frequency / this_node_overall_freq_phase_1 <= p1 and this_node_overall_freq_phase_1 < p2:
                            node.data.frequency = 0
        self._weighted_average(CPS_tree, consider_underlying_distribution=True)
        self._frequency_consistency(CPS_tree)
        self.tree = CPS_tree
        
        return CPS_tree
    
    def _underlying_user_allocation(self):
        num = int(self.user_num * self.alpha) - 1
        self.underlying_users = [0,num]
    
    def _uniform_user_allocation(self, pl_tree):
        tree_user_num = self.user_num - (self.underlying_users[1] - self.underlying_users[0] + 1)
        for node in self._bfs(pl_tree, return_root=True):
            if node.is_root():
                node.data.set_allocated_users(self.user_num, tree_user_num, 0)
            else:
                to_be_allocated_user_num = pl_tree.parent(node.identifier).data.left_user_num
                subtree_depth = pl_tree.subtree(node.identifier).depth() + 1
                node.data.set_allocated_users(self.user_num, to_be_allocated_user_num, 1/subtree_depth)
    
    def _estimate_underly_hist(self, users):
        data = self.data[users[0]:users[1]]
        # using sw
        fo = Frequency_oracle(data, frequency_oracle_name = 'SW', epsilon=self.epsilon, domain_size=self.domain_size)
        hist_SW, hist_SW_woS = fo.get_aggregated_frequency(None)
        self.ns_frequency_hist_phase_1 = fo.ns_hist
        self.sw_transform = fo.transform
        self.sw_n = fo.n
        self.sw_max_iteration = fo.max_iteration
        self.sw_loglikelihood_threshold = fo.loglikelihood_threshold
        
        return hist_SW, hist_SW_woS

    def _estimate_node_frequency(self, node):
        data = self.data[node.data.allocated_users[0]: node.data.allocated_users[1]+1]
        subdomain = range(node.data.interval_left, node.data.interval_right+1)
        fo = Frequency_oracle(data, frequency_oracle_name='ORR', epsilon=self.epsilon, domain_size=self.domain_size, merged_domain=[subdomain])
        node.data.frequency = fo.get_aggregated_frequency()
        node.data.error = fo.get_one_theoretical_square_error()
        node.user_num = len(data)
    
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
    
    def _update_frequency_in_wa(self, CPS_tree, node, is_leaf):
        x_1 = node.data.frequency
        var_1 = node.data.error
        x_2 = 0
        var_2 = 0
        if is_leaf:
            x_2 = self.noisy_hist_SW[node.data.interval_left: node.data.interval_right+1].sum()
            var_2 = np.square(x_2 - x_1)
            var_2 = var_2 - var_1 if var_2 > var_1 else var_2
            var_2 = min(var_2, np.square(x_2))
        else:
            for child in CPS_tree.children(node.identifier):
                x_2 += child.data.frequency
                var_2 += child.data.error
        alpha = var_2 / (var_1 + var_2)
        x = alpha * x_1 + (1 - alpha) * x_2
        node.data.frequency = x
        node.data.error = var_1 * var_2 / (var_1 + var_2)
        return alpha, 1 - alpha
    
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
                    if self.update_error:
                        if i in C_plus: # f_c in C+, then we update its variance
                            coef_w = coef.copy()
                            coef_w[i] += 1
                            children[i].data.error_coef = (1 / len(C_plus)) * node.data.error_coef + np.dot(coef_w, error_coef_matrix)
                            children[i].data.error = np.dot(np.square(children[i].data.error_coef), self.error_vector)
    
    def _refine_slope(self, slope, d, count):
        assert d > 1   
        if count == 0:
            refined_slope = 0
        else:  
            refined_slope = slope
            bound = 2 * count / (d * (d - 1))
            if refined_slope > bound:
                refined_slope = bound
            elif refined_slope < - bound:
                refined_slope = - bound
        return refined_slope
    
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
    
        
    def _CPS_tree_construction(self, segments, approximated_intervals, coefficients, segment_to_interval_map, branch = 2):
        '''
        Phase 2: CPS Tree Construction

        step 1: we initially build a binary tree, where the optimized fan-out for non-leaf nodes is 2
        step 2: we adaptively reduce some non-leaf node in the tree to optimized the total range query error. 
            After this step, we derive a unbalanced tree, and each node may have different fan-out.
        '''
        
        CPS_tree = Tree()
        node_queue = queue.Queue()
        root = Node(tag='root', identifier=0, data=Node_CPS(this_node_interval_left=0, this_node_interval_right=self.domain_size-1, \
            domain_size=self.domain_size, frequency=1, error=0))
        CPS_tree.add_node(root)
        if len(segments) == 1:
            root.data.set_data(this_node_coefficients = coefficients[(0, self.domain_size-1)])
        else:
            # initial all leaves 
            for i, seg in enumerate(segments):
                this_node_coefficients = segment_to_interval_map[seg]['coefficients']
                node_data = Node_CPS(seg[0], seg[1], self.domain_size, frequency=self.noisy_hist_SW[seg[0]:seg[1]+1].sum(), \
                    this_node_coefficients=this_node_coefficients)
                CPS_tree.create_node(tag='n_'+str(int(i+1)), identifier = i+1, parent=root, data=node_data)
                node = CPS_tree.get_node(i+1)
                node_queue.put(node)
            # generate non-leaf nodes
            node_id = len(segments)
            to_be_proc_num = len(segments)
            parent_num = math.ceil(to_be_proc_num / branch) if branch > 2 or to_be_proc_num % branch != 1 \
                else math.floor(to_be_proc_num / branch)
            avg_seg_length = self.domain_size / parent_num
            while parent_num > 1:
                while parent_num != 0:
                    children_num = math.floor(to_be_proc_num / parent_num)
                    children = [node_queue.get() for _ in range(children_num)]
                    if to_be_proc_num % parent_num != 0:
                        adaptive = node_queue.queue[0]
                        reject_adap = children[-1].data.interval_right - children[0].data.interval_left + 1
                        accept_adap = reject_adap + adaptive.data.interval_right - adaptive.data.interval_left + 1
                        if (accept_adap - avg_seg_length) < (reject_adap - avg_seg_length) or parent_num == 1:
                            children.append(node_queue.get())
                    node_id += 1
                    CPS_tree.create_node(tag='n_'+str(int(node_id)), identifier=node_id, parent=root)
                    accum_fre = 0
                    for child in children:
                        accum_fre += child.data.frequency
                        CPS_tree.move_node(child.identifier, node_id)
                    parent_data = Node_CPS(children[0].data.interval_left, children[-1].data.interval_right, \
                        self.domain_size, frequency=accum_fre)
                    parent = CPS_tree.get_node(node_id)
                    parent.data = parent_data
                    node_queue.put(parent)
                    parent_num -= 1
                    to_be_proc_num -= children_num
                to_be_proc_num = node_queue.qsize()
                parent_num = math.ceil(to_be_proc_num / branch) if branch > 2 or to_be_proc_num % branch != 1 \
                    else math.floor(to_be_proc_num / branch)
                avg_seg_length = self.domain_size / parent_num
        # # step 2: reduce some non-leaf nodes by DFS
        for node in self._dfs_postorder_traveral(CPS_tree):
            if not node.is_leaf() and not node.is_root():
                if self._reduce_determine(CPS_tree, node):
                    CPS_tree.link_past_node(node.identifier)
        return CPS_tree
    
    def _dfs_postorder_traveral(self, pripl_tree):
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
    
    def _reduce_determine(self, pripl_tree, node):
        """

        Args:
            pripl_tree (_type_): _description_
            node (_type_): _description_

        Returns:
            bool: True 为应该 reduce 
        """
        pripl_tree_reduced = Tree(pripl_tree.subtree(pripl_tree.root), deep=True)
        pripl_tree_reduced.link_past_node(node.identifier)
        error_remain = error_reduce = 0
        anc_left_ratio_remain = anc_left_ratio_reduce = 1
        for anc_id in list(pripl_tree.rsearch(node.identifier))[-2::-1]:
            if anc_id != node.identifier:
                l_reduce = pripl_tree_reduced.subtree(anc_id).depth() + 1
                anc_reduce = pripl_tree_reduced.get_node(anc_id)
                error_reduce += l_reduce / anc_left_ratio_reduce * (anc_reduce.data.partial_query_weight - pripl_tree_reduced.parent(anc_id).data.partial_query_weight)
                anc_left_ratio_reduce = anc_left_ratio_reduce * (1 - 1 / l_reduce)
            l_remain = pripl_tree.subtree(anc_id).depth() + 1
            anc_remain = pripl_tree.get_node(anc_id)
            error_remain += l_remain / anc_left_ratio_remain * (anc_remain.data.partial_query_weight - pripl_tree.parent(anc_id).data.partial_query_weight)
            anc_left_ratio_remain = anc_left_ratio_remain * (1 - 1 / l_remain)
        pripl_tree_reduced.get_node(pripl_tree.parent(node.identifier).identifier).data.left_user_ratio = anc_left_ratio_reduce
        node.data.left_user_ratio = anc_left_ratio_remain
        for des in self._bfs(pripl_tree.subtree(node.identifier), return_root=False):
            if des.identifier == node.identifier:
                continue
            l_reduce = pripl_tree_reduced.subtree(des.identifier).depth() + 1
            anc_left_ratio_reduce = pripl_tree_reduced.parent(des.identifier).data.left_user_ratio
            error_reduce += l_reduce / anc_left_ratio_reduce * (des.data.partial_query_weight - pripl_tree_reduced.parent(des.identifier).data.partial_query_weight)
            pripl_tree_reduced.get_node(des.identifier).data.left_user_ratio = anc_left_ratio_reduce * (1 - 1 / l_reduce)
            l_remain = pripl_tree.subtree(des.identifier).depth() + 1
            anc_left_ratio_remain = pripl_tree.parent(des.identifier).data.left_user_ratio
            error_remain += l_remain / anc_left_ratio_remain * (des.data.partial_query_weight - pripl_tree.parent(des.identifier).data.partial_query_weight)
            des.data.left_user_ratio = anc_left_ratio_remain * (1 - 1 / l_remain)
        return error_reduce < error_remain
    
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


