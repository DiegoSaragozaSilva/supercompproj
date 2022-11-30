import os
import time
import subprocess
import numpy as np
from matplotlib import pyplot as plt

def runCommand(command, _stdin = ""):
    startTime = time.time() * 1000.0
    process = subprocess.run(command.split(), input = _stdin.encode('utf-8'))
    endTime = time.time() * 1000.0

    return [0, endTime - startTime]

def compileCode(compiler, args):
    command = compiler
    for arg in args:
        command += " " + arg
    print(f"Running {command}...")
    runCommand(command)

    # print(f"Command output: {0)}")
    # if (output[0][1] != None):
    #     raise Exception("Command outputed error during execution")

def getFileData(file):
    fileData = ""
    with open(file, "r") as _file:
        fileData = _file.read()
    return fileData

def getDirFiles(path):
    _files = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            _files.append(path + f"/{file}")
    return _files

def preparePerformancePlot(n):
    plt.title("Heuristic C++ execution times")
    plt.xlabel("Number of cities")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.xticks(np.arange(0, len(n), len(n) / 10))

def runTests(binaries):
    testFiles = sorted(getDirFiles(r"tests"))

    for binary in binaries:
        totalDone = 0
        testTimes = list()
        n = list()
        for file in testFiles:
            n.append(file.replace("tests/", "").replace(".txt", ""))

            output = runCommand(f"./{binary}", getFileData(file))
            testTimes.append(output[1])
            totalDone += 1
            print(f"{binary} files done: {totalDone} of {len(testFiles)}")
        legend = ""
        if binary == "cpptsp":
            legend = "C++"
        elif binary == "omptsp":
            legend = "C++ (OMP)"
        elif binary == "cudatsp":
            legend = "Cuda (GPU)"
        plt.plot(n, testTimes, label = legend)
        preparePerformancePlot(n)
    plt.savefig("final_performance_long.png")

if __name__ == "__main__":
    # Compile both cpp, openmp and cuda files
    print("Compiling cpp implementation...")
    cppFile = "TSPHeuristico.cpp"
    cppBinary = "cpptsp"
    compileCode("g++", [f"-o {cppBinary}", "-O3", cppFile])
    print("Cpp binary successfully compiled!\n")

    binaries = [cppBinary]
    runTests(binaries)

    print("Done performance testing!")
