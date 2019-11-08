# No node has two red links connected to it
# Every path from root to null link has the same number of black links
# Red links lean left


class LLRBtree:
    def __init__(self, val):
        self.root = self.Node(val)

    class Node:
        def __init__(self, val, is_red=False):
            self.val = val
            self.left = None
            self.right = None
            self.isRed = is_red   # color of parent link

    def find(self, val):
        node = self.root
        while node is not None:
            if val < node.val:
                node = node.left
            elif node.val < val:
                node = node.right
            else:
                return node
        return None

    def insert(self, val):
        self.root = self._insert(self.root, val)
        self.root.isRed = False

    def _is_red(self, node):
        if node is None:
            return False
        return node.isRed

    def _rotate_left(self, node):  # orient a (temporarily) right-leaning red link to lean left
        r = node.right
        node.right = r.left
        r.left = node
        r.isRed = node.isRed
        node.isRed = True
        return r

    def _rotate_right(self, node):  # orient a left-leaning red link to (temporarily) lean right
        l = node.left
        node.left = l.right
        l.right = node
        l.isRed = node.isRed
        node.isRed = True
        return l

    def _flip_colors(self, node):
        node.isRed = True
        node.left.isRed = False
        node.right.isRed = False

    def _insert(self, node, val):
        if node is None:
            return LLRBtree.Node(val, True)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        if self._is_red(node.right) and not self._is_red(node.left):
            node = self._rotate_left(node)
        if self._is_red(node.left) and self._is_red(node.left.left):
            node = self._rotate_right(node)
        if self._is_red(node.left) and self._is_red(node.right):
            self._flip_colors(node)
        return node


def main():
    tree = LLRBtree(10)
    l = [5, 7, 12, 9, 20, 11, 22]
    for num in l:
        tree.insert(num)
    for num in l:
        print(tree.find(num).val)
    print(tree.find(6))


if __name__ == '__main__':
    main()
