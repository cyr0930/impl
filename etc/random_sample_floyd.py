import random


def floyd_random_sample(m, n):
    # pick m random sample in range [0, n]
    s = set()
    for i in range(n-m+1, n+1):
        t = random.randint(0, i)
        s.add(i if t in s else t)
    return s


def main():
    ans = floyd_random_sample(28, 32)
    print('length:', len(ans), ', set:', ans)

    
if __name__ == '__main__':
    main()
