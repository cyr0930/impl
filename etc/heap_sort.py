def siftdown(l, idx, end):
    c = 2 * idx + 1
    while c <= end:
        if c != end and l[c] < l[c + 1]:
            c += 1
        if l[c] <= l[idx]:
            break
        l[idx], l[c] = l[c], l[idx]
        idx = c
        c = 2 * idx + 1


def heapify(l):
    # heapify with siftdown takes O(n) time, while siftup takes O(n*log n)
    for i in range((len(l)-2) // 2, -1, -1):
        siftdown(l, i, len(l)-1)


def heap_sort(l):
    # in-place heap sort, time O(n*log n), space O(1)
    heapify(l)
    for i in range(len(l)-1, 0, -1):
        l[0], l[i] = l[i], l[0]
        siftdown(l, 0, i-1)


l = [4, 2, 5, 1, 6, 6, 2]
heap_sort(l)
print(l)
l = [4, 2, 5, 1, 6, 6, 2, 3]
heap_sort(l)
print(l)
