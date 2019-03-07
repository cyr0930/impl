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


def main():
    l = [4, 2, 5, 1, 6, 6, 2, 3]
    quick_sort(l)
    print(l)


if __name__ == '__main__':
    main()
