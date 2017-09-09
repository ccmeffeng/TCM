def count(x):
    c = [0] * 4
    c[x//1024] += 1
    c[x//256 % 4] += 1
    c[x//64 % 4] += 1
    c[x//16 % 4] += 1
    c[x//4 % 4] += 1
    c[x % 4] += 1
    a,b =0,0
    for i in c:
        if i == 1:
            a += 1
        if i == 2:
            b += 1
    if a==2 and b==2:
        return True
    else:
        return False


def num():
    k = 0
    for i in range(4096):
        if count(i):
            k += 1
    return k

print(num())
