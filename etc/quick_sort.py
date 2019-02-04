import random


def _quick_sort(l, l_b, u_b):
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
    _quick_sort(l, l_b, end - 1)
    _quick_sort(l, end + 1, u_b)


def quick_sort(l):
    _quick_sort(l, 0, len(l) - 1)


l = [4, 2, 5, 1, 6, 6, 2, 3]
quick_sort(l)
print(l)
