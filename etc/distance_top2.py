# Udacity - Intro to Algorithms - Lesson 18: Extra Challenge Problems - 1. Quiz: Top Two
import heapq


def top_two(graph, start):
    output = {}
    for node in graph:
        output[node] = []
    heap = [(0, [start])]
    seen = set()
    while heap and len(seen) + 1 < len(graph):
        cost, path = heapq.heappop(heap)
        if len(output[path[-1]]) < 2:
            output[path[-1]].append((cost, path))
        else:
            seen.add(path[-1])
        for key, value in graph[path[-1]].items():
            if key not in path:
                heapq.heappush(heap, (cost + value, path + [key]))
    return output


def main():
    def test():
        graph = {'a': {'b': 3, 'c': 4, 'd': 8},
                 'b': {'a': 3, 'c': 1, 'd': 2},
                 'c': {'a': 4, 'b': 1, 'd': 2},
                 'd': {'a': 8, 'b': 2, 'c': 2}}
        result = top_two(graph, 'a')
        b = result['b']
        b_first = b[0]
        assert b_first[0] == 3
        assert b_first[1] == ['a', 'b']
        b_second = b[1]
        assert b_second[0] == 5
        assert b_second[1] == ['a', 'c', 'b']
    test()


if __name__ == '__main__':
    main()
