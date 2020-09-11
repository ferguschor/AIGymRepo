import random

class Node():

    def __init__(self, left, right, is_leaf: bool=False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

        if not is_leaf:
            self.value = self.left.value + self.right.value

        self.parent = None

        # idx used to identify the leaf node (only active for leaf node)
        self.idx = idx

        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.left.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, True, idx)
        leaf.value = value
        return leaf

    @staticmethod
    def create_tree(input: list):
        node_list = [Node.create_leaf(value=v, idx=i) for i, v in enumerate(input)]
        leaf_nodes = node_list
        while len(node_list) > 1:
            if len(node_list) % 2 == 1: # For odd lengths, create node using first 2 nodes
                inodes = iter([Node(node_list[0], node_list[1])]+node_list[2:])
            else:
                inodes = iter(node_list)
            node_list = [Node(*pair) for pair in zip(inodes, inodes)]

        return node_list[0], leaf_nodes

    def retrieve(self, seek_value):
        if self.is_leaf:
            return self
        if seek_value < self.left.value:
            return self.left.retrieve(seek_value)
        if seek_value >= self.left.value:
            return self.right.retrieve(seek_value - self.left.value)

    def update(self, new_value):
        self.value = new_value
        if self.parent is not None:
            self.parent.propagate_changes(new_value-self.value)

    def propagate_changes(self, change):
        self.value += change
        if self.parent is not None:
            self.parent.propagate_changes(change)

    @staticmethod
    def print_tree(node, level=0):
        if node is not None:
            print("\t"*level, end='')
            print(node.value)
        if node.left:
            Node.print_tree(node.left, level+1)
        if node.right:
            Node.print_tree(node.right, level+1)

def test_tree(input):
    root_node, leaf_nodes = Node.create_tree(input)
    iterations = 1000000
    values = []
    randvalues = []
    for i in range(iterations):
        rand = random.uniform(0, root_node.value)
        randvalues.append(rand)
        values.append(root_node.retrieve(rand).idx)
    occurrences = [values.count(i) for i in range(len(input))]
    prob = [i/sum(occurrences) for i in occurrences]
    print(prob)


if __name__ == "__main__":
    tmp = [2, 0, 2, 3, 1, 2]
    t, leaf_nodes = Node.create_tree(tmp)
    test_tree(tmp)