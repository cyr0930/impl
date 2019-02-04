class AVLtree:
    def __init__(self, val):
        self.root = self.TreeNode(val)

    def get_root(self):
        return self.root

    def insert(self, val):
        t = self.root.insert(val)
        if t is None:
            return False
        self.root = t[1]
        return True

    def find(self, val):
        return self.root.find(val)

    def find_min(self):
        return self.root.find_min()

    @staticmethod
    def next_larger(node):
        if node.right is None:
            while node.parent is not None and node.parent.right == node:
                node = node.parent
            return node.parent
        return node.right.find_min()

    def get_height(self):
        return self.root.get_height()

    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
            self.parent = None
            self.height = 1

        def _balancing(self):
            heights = self.get_height(False)
            if abs(heights[1] - heights[2]) != 2:
                return self
            if heights[1] > heights[2]:
                l_heights = self.left.get_height(False)
                parent, left = self.parent, self.left
                if l_heights[1] < l_heights[2]:
                    lr = left.right
                    self.parent, self.left = lr, lr.right
                    left.parent, left.right = lr, lr.left
                    if lr.right is not None:
                        lr.right.parent = self
                    if lr.left is not None:
                        lr.left.parent = left
                    lr.parent, lr.right, lr.left = parent, self, left
                    lr.height, self.height, left.height = lr.height + 1, self.height - 2, left.height - 1
                    return lr
                else:
                    self.parent, self.left = left, left.right
                    if left.right is not None:
                        left.right.parent = self
                    left.parent, left.right = parent, self
                    self.height = max(heights[1], l_heights[2]) + 1
                    left.height = max(self.height, l_heights[1]) + 1
                    return left
            else:
                r_heights = self.right.get_height(False)
                parent, right = self.parent, self.right
                if r_heights[1] > r_heights[2]:
                    rl = right.left
                    self.parent, self.right = rl, rl.left
                    right.parent, right.left = rl, rl.right
                    if rl.left is not None:
                        rl.left.parent = self
                    if rl.right is not None:
                        rl.right.parent = right
                    rl.parent, rl.left, rl.right = parent, self, right
                    rl.height, self.height, right.height = rl.height+1, self.height-2, right.height-1
                    return rl
                else:
                    self.parent, self.right = right, right.left
                    if right.left is not None:
                        right.left.parent = self
                    right.parent, right.left = parent, self
                    self.height = max(heights[1], r_heights[1]) + 1
                    right.height = max(self.height, r_heights[2]) + 1
                    return right

        def insert(self, val):
            if self.val == val:
                return
            if val < self.val:
                if self.left is None:
                    self.left = AVLtree.TreeNode(val)
                    self.left.parent = self
                    if self.right is None:
                        self.height += 1
                else:
                    self.height = self.left.insert(val)[0] + 1
            else:
                if self.right is None:
                    self.right = AVLtree.TreeNode(val)
                    self.right.parent = self
                    if self.left is None:
                        self.height += 1
                else:
                    self.height = self.right.insert(val)[0] + 1
            root = self._balancing()
            return self.height, root

        def find(self, val):
            if val < self.val and self.left is not None:
                return self.left.find(val)
            if self.val < val and self.right is not None:
                return self.right.find(val)
            if self.val == val:
                return self
            return None

        def find_min(self):
            if self.left is None:
                return self
            return self.left.find_min()

        def get_height(self, single=True):
            if single:
                return self.height
            l_height = 0 if self.left is None else self.left.height
            r_height = 0 if self.right is None else self.right.height
            return self.height, l_height, r_height


tree = AVLtree(10)
tree.insert(5)
tree.insert(7)
tree.insert(12)
tree.insert(9)
tree.insert(20)
tree.insert(11)
tree.insert(22)
print(tree.get_height())
minNode = tree.find_min()
print(minNode.val)
print(AVLtree.next_larger(minNode).val)
node20 = tree.find(20)
print(node20.val)
print(AVLtree.next_larger(node20).val)
print(tree.find(6))
print(AVLtree.next_larger(tree.find(7)).val)
print(AVLtree.next_larger(tree.find(9)).val)
print(AVLtree.next_larger(tree.find(22)))
