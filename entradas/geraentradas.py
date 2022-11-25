import random
n = input()
size = 100000
for i in range(int(n)): 
    f = open("in-{}.txt".format(i), "w")
    f.write(str(size) + "\n")
    for j in range(size):
        f.write("{} {}\n".format(random.random()*100, random.random()*100))
    size += 1
