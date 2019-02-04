def sift_down(l, idx, end):
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
    end = len(l) - 1
    for i in range(end // 2, -1, -1):
        sift_down(l, i, end)


def heap_sort(l):
    heapify(l)
    for i in range(len(l)-1, 0, -1):
        l[0], l[i] = l[i], l[0]
        sift_down(l, 0, i-1)
    return l


print(heap_sort([4,2,5,1,6,6,2,3]))
