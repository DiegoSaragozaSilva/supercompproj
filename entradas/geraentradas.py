import random

n = input()
for i in range(int(n)):
    f = open("{0:03}.txt".format(i), "w")
    f.write(str(i) + "\n")
    for j in range(i):
        f.write("{} {}\n".format(random.random() * 100, random.random() * 100))
