import random


def quick_sort(l, l_b=0, u_b=None):
    if u_b is None:
        u_b = len(l) - 1
    if l_b >= u_b:
        return
    pivot = random.randint(l_b, u_b)
    l[l_b], l[pivot] = l[pivot], l[l_b]
    start, end = l_b + 1, u_b
    while True:
        # to avoid worst case (all elem are equal), use a < b, not a <= b
        while start <= u_b and l[start] < l[l_b]:
            start += 1
        while l[l_b] < l[end]:
            end -= 1
        if start > end:
            break
        l[start], l[end] = l[end], l[start]
        start, end = start + 1, end - 1
    l[l_b], l[end] = l[end], l[l_b]
    quick_sort(l, l_b, end - 1)
    quick_sort(l, end + 1, u_b)


def quick_sort_3_way(l, l_b=0, u_b=None):
    if u_b is None:
        u_b = len(l) - 1
    if l_b >= u_b:
        return
    pivot = random.randint(l_b, u_b)
    v = l[pivot]
    start, end, i = l_b, u_b, l_b
    while i <= end:
        if l[i] < v:
            l[start], l[i] = l[i], l[start]
            start, i = start + 1, i + 1
        elif l[i] > v:
            l[end], l[i] = l[i], l[end]
            end -= 1
        else:
            i += 1
    quick_sort_3_way(l, l_b, start - 1)
    quick_sort_3_way(l, end + 1, u_b)


def main():
    import time
    n = 10000
    l1 = [random.randrange(0, 100) for _ in range(n)]
    l2 = l1.copy()

    t = time.time()
    quick_sort(l1)
    print(f'time: {time.time() - t}, {l1}')

    t = time.time()
    quick_sort_3_way(l2)
    print(f'time: {time.time() - t}, {l2}')

    assert all([l1[i] == l2[i] for i in range(n)])


if __name__ == '__main__':
    main()
