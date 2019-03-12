# Udacity - Intro to Algorithms - Lesson 17: Final Assessment - 7. Quiz: Distance Oracle (II)
from collections import deque


def bfs_for_labels(g, tree_g, node, labels):
    dq = deque([(node, 0)])
    seen = {node}
    while dq:
        cur = dq.popleft()
        labels.setdefault(cur[0], {})[node] = cur[1]
        for key in g[cur[0]]:
            if key in seen:
                continue
            dq.append((key, cur[1] + tree_g[cur[0]][key]))
            seen.add(key)


def dfs_for_weights(g, node, g_weighted, seen):
    seen.add(node)
    count = 0
    for nei in g[node]:
        if nei in seen:
            continue
        val = dfs_for_weights(g, nei, g_weighted, seen)
        g_weighted.setdefault(node, {})[nei] = val
        count += val
    return count + 1


def calc_weight(g_weighted, node):
    dq = deque([node])
    seen = set()
    while dq:
        cur = dq.popleft()
        seen.add(cur)
        total = sum(g_weighted[cur].values())
        for key in g_weighted[cur]:
            if key in seen:
                continue
            dq.append(key)
            g_weighted.setdefault(key, {})[cur] = total + 1 - g_weighted[cur][key]


def reduce_weight(g_weighted, node, seen, reduce):
    seen.add(node)
    for nei in g_weighted[node]:
        if nei in seen:
            continue
        g_weighted[nei][node] -= reduce
        reduce_weight(g_weighted, nei, seen, reduce)


def create_labels(tree_g):
    g_weighted = {}
    start = next(iter(tree_g))
    dfs_for_weights(tree_g, start, g_weighted, set())
    calc_weight(g_weighted, start)
    labels = {}
    seen = set()
    while g_weighted:
        pick = None
        for node in g_weighted:
            cur = max(g_weighted[node].values())
            if pick is None or pick[1] > cur:
                pick = (node, cur)
        bfs_for_labels(g_weighted, tree_g, pick[0], labels)
        seen.add(pick[0])
        total = sum(g_weighted[pick[0]].values())
        for nei in g_weighted[pick[0]]:
            reduce_weight(g_weighted, nei, {pick[0]}, total + 1 - g_weighted[pick[0]][nei])
            del g_weighted[nei][pick[0]]
            if not g_weighted[nei]:
                del g_weighted[nei]
        del g_weighted[pick[0]]
    for node in tree_g:
        labels.setdefault(node, {})[node] = 0
    return labels


def make_link(g, node1, node2, weight=1):
    if node1 not in g: g[node1] = {}
    g[node1][node2] = weight
    if node2 not in g: g[node2] = {}
    g[node2][node1] = weight
    return g


def main():
    from math import log, ceil
    from random import randint

    def get_distances(g, labels):
        distances = {}
        for start in g:
            label_node = labels[start]
            s_distances = {}
            for destination in g:
                shortest = float('inf')
                label_dest = labels[destination]
                for intermediate_node, dist in label_node.items():
                    if intermediate_node == destination:
                        shortest = dist
                        break
                    other_dist = label_dest.get(intermediate_node)
                    if other_dist is None:
                        continue
                    if other_dist + dist < shortest:
                        shortest = other_dist + dist
                s_distances[destination] = shortest
            distances[start] = s_distances
        return distances

    def distance(tree, w, u):
        if w == u: return 0
        distances = {w: 0}
        frontier = deque([w])
        while frontier:
            n = frontier.popleft()
            for s in tree[n]:
                if s not in distances:
                    distances[s] = distances[n] + tree[n][s]
                    frontier.append(s)
                if s == u: return distances[u]
        return None

    def random_test():
        n, n0, n1 = 100, 20, 100
        for _ in range(n):
            tree = {}
            for w in range(1, n0):
                make_link(tree, w, w + 1, randint(1, 10))
            for w in range(n0 + 1, n1 + 1):
                make_link(tree, randint(1, w - 1), w, randint(1, 10))
            labels = create_labels(tree)
            distances = get_distances(tree, labels)
            assert max([len(labels[n]) for n in tree]) <= int(ceil(log(len(tree) + 1, 2)))
            for _ in range(n):
                w = randint(1, n1)
                u = randint(1, n1)
                assert distance(tree, w, u) == distances[w][u]
        print('random_test() passed')
    random_test()


if __name__ == '__main__':
    main()

