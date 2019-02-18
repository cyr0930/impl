def create_rooted_spanning_tree(G, root):
    def s_rec(root, S):
        for node in G[root]:
            if node in S:
                if root not in S[node]:
                    S.setdefault(root, {})[node] = 'red'
                    S[node][root] = 'red'
            else:
                S.setdefault(root, {})[node] = 'green'
                S.setdefault(node, {})[root] = 'green'
                s_rec(node, S)
        return S
    return s_rec(root, {})


def post_order(S, root):
    def po_rec(root, seen, po):
        for node in S[root]:
            if node not in seen and S[root][node] == 'green':
                seen.add(node)
                po_rec(node, seen, po)
                seen.remove(node)
        po[root] = len(po) + 1
        return po
    return po_rec(root, {root}, {})


def number_of_descendants(S, root):
    def nd_rec(root, seen, nd):
        count = 1
        for node in S[root]:
            if node not in seen and S[root][node] == 'green':
                seen.add(node)
                nd_rec(node, seen, nd)
                count += nd[node]
                seen.remove(node)
        nd[root] = count
        return nd
    return nd_rec(root, {root}, {})


def lowest_post_order(S, root, po):
    def l_rec(root, seen, l):
        low = po[root]
        for node in S[root]:
            if S[root][node] == 'green':
                if node not in seen:
                    seen.add(node)
                    l_rec(node, seen, l)
                    low = min(low, l[node])
                    seen.remove(node)
            else:
                low = min(low, po[node])
        l[root] = low
        return l
    return l_rec(root, {root}, {})


def highest_post_order(S, root, po):
    def h_rec(root, seen, h):
        high = po[root]
        for node in S[root]:
            if S[root][node] == 'green':
                if node not in seen:
                    seen.add(node)
                    h_rec(node, seen, h)
                    high = max(high, h[node])
                    seen.remove(node)
            else:
                high = max(high, po[node])
        h[root] = high
        return h
    return h_rec(root, {root}, {})


def bridge_edges(G, root):
    S = create_rooted_spanning_tree(G, root)
    po = post_order(S, root)
    nd = number_of_descendants(S, root)
    l = lowest_post_order(S, root, po)
    h = highest_post_order(S, root, po)
    bridges = set()
    for key in G:
        if key != root and h[key] == po[key] and l[key] == po[key] - nd[key] + 1:
            for node in G[key]:
                if key in G[node] and po[node] > po[key]:
                    bridges.add((node, key))
                    break
    return bridges


if __name__ == '__main__':
    G = {'a': {'c': 1, 'b': 1},
         'b': {'a': 1, 'd': 1},
         'c': {'a': 1, 'd': 1},
         'd': {'c': 1, 'b': 1, 'e': 1},
         'e': {'d': 1, 'g': 1, 'f': 1},
         'f': {'e': 1, 'g': 1},
         'g': {'e': 1, 'f': 1}}
    assert bridge_edges(G, 'a') == {('d', 'e')}
