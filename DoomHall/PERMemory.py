import random
import numpy as np


class SumTreeList:

    def __init__(self, capacity):
        self.capacity = capacity
        self.current_capacity = 0
        # Tree holds priority data
        # A binary tree with the lowest layer N nodes has total nodes N*(N-1)
        # Parent Nodes = capacity - 1
        # Leaf Nodes = capacity+
        self.tree = np.zeros(2 * capacity - 1)
        # Memory holds SARSA data
        self.experience = np.zeros(capacity, dtype=object)

        # Pointer to check where to update data
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1

        self.experience[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if(self.data_pointer >= self.capacity):
            self.data_pointer = 0
        self.current_capacity = min(self.current_capacity + 1, self.capacity)


    def update(self, tree_index, new_value):
        change = new_value - self.tree[tree_index]
        self.tree[tree_index] = new_value

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, seek_value):

        search_index = 0
        while search_index < self.capacity - 1:
            left_index = search_index * 2 + 1
            right_index = left_index + 1
            # Go left tree
            if seek_value < self.tree[left_index]:
                search_index = left_index
            # Go right tree
            else:
                search_index = right_index
                seek_value = seek_value - self.tree[left_index]
        """
                  0
               /     \
              1       2
             / \    /  \ 
            3   4  5    6
        """
        # Search_index will now be in leaf_nodes
        data_index = search_index - self.capacity + 1

        # Return tree index, priority, SARSA
        return search_index, self.tree[search_index], self.experience[data_index]

    @property
    def total_priority(self):
        # First value holds sum of all priorities
        return self.tree[0]


class Memory:

    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.capacity = capacity
        self.sum_tree = SumTreeList(self.capacity)

    def store(self, experience):

        max_priority = np.max(self.sum_tree.tree[(self.sum_tree.capacity - 1):])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.sum_tree.add(max_priority, experience)

    def sample(self, sample_size):
        exp_batch = []

        index_batch = np.empty((sample_size,), dtype=np.int32)
        ISWeights_batch = np.empty((sample_size, 1), dtype=np.float32)

        interval = self.sum_tree.total_priority / sample_size
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # P(j) only used to calculate w_j, P(j) updated in update_batch
        # Tree will have priority = 0 for empty records
        p_min = np.min(self.sum_tree.tree[(self.sum_tree.capacity - 1):(self.sum_tree.capacity - 1 + self.sum_tree.current_capacity)]) / self.sum_tree.total_priority
        max_weight = (p_min * sample_size) ** (-self.PER_b)

        for i in range(sample_size):
            rand_value = np.random.uniform(i * interval, (i+1)*interval)
            leaf_index, priority, sample_exp = self.sum_tree.get_leaf(rand_value)

            # Need to save index_batch to pass for updates later
            index_batch[i] = leaf_index

            # P(j)
            sample_prob = priority / self.sum_tree.total_priority
            # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            ISWeights_batch[i, 0] = np.power(sample_size * sample_prob, -self.PER_b) / max_weight
            exp_batch.append(sample_exp)

        return index_batch, ISWeights_batch, exp_batch

    def update_batch(self, tree_indices, abs_errors):
        abs_errors += self.PER_e # add extra to avoid 0 priority
        # clip errors for stability
        # use np.minimum for an element-wise operation
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for i, p in zip(tree_indices, ps):
            self.sum_tree.update(i, p)


def test_tree():

    input = [1, 4, 5]
    tree = SumTreeList(len(input))

    for i, p in enumerate(input):
        tree.add(priority=p, data=i)

    iterations = 1000000
    values = []
    randvalues = []
    for i in range(iterations):
        rand = random.uniform(0, tree.total_priority)
        randvalues.append(rand)
        idx, prior, data = tree.get_leaf(rand)
        values.append(data)
    occurrences = [values.count(i) for i in range(len(input))]
    prob = [i/sum(occurrences) for i in occurrences]
    print(prob)

def test_mem():
    input = [1, 4, 3, 2, 5, 'a', 'b']

    mem = Memory(capacity=len(input))
    for i in input:
        mem.store(i)

    indices, prior_batch, sample_batch = mem.sample(10)
    # print("Indices: ", indices)
    # print("Priorities: ", prior_batch)
    # print("Samples: ", sample_batch)
    occurrences = [sample_batch.count(i) for i in input]
    prob = [i/sum(occurrences) for i in occurrences]
    print(prob)
    print(mem.sum_tree.total_priority)
    print(mem.sum_tree.tree)

    print("\nUpdate: ")
    mem.update_batch(tree_indices=np.array([len(input)-1+i for i in range(len(input))]),
                     abs_errors=np.array([0.13, 0.23, 0.22, 0.07, 0.25, 0.06, 0.04]))
    print(mem.sum_tree.tree)
    print("\nAfter Update")
    indices, prior_batch, sample_batch = mem.sample(10)
    occurrences = [sample_batch.count(i) for i in input]
    prob = [i/sum(occurrences) for i in occurrences]
    print(prob)
    print("sample",sample_batch)
    leaf_indices = np.array([len(input)-1+i for i in range(len(input))])
    print(leaf_indices)

    print(indices)
    print(mem.sum_tree.total_priority)
    print(mem.sum_tree.tree)

if __name__ == "__main__":
    test_mem()