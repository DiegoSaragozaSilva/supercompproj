import os
import time
import subprocess
import numpy as np
from matplotlib import pyplot as plt

def runCommand(command, _stdin = None):
    process = subprocess.Popen(command.split(), stdin = subprocess.PIPE, stdout = subprocess.PIPE)
    startTime = time.time() * 1000.0

    output = None
    if _stdin == None:
        output = process.communicate(input = b"")
    else:
        output = process.communicate(input = bytes(_stdin, "utf-8"))
    process.wait()
    endTime = time.time() * 1000.0

    return [output, endTime - startTime]

def compileCode(compiler, args):
    command = compiler
    for arg in args:
        command += " " + arg
    print(f"Running {command}...")
    output = runCommand(command)

    print(f"Command output: {str(output[0][0])}")
    if (output[0][1] != None):
        print(f"Command error: {output[0][1]}")
        raise Exception("Command outputed error during execution")

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
    plt.title("Performance comparison C++ vs C++ (OMP) vs Cuda")
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
        plt.plot(n, testTimes, label = legend)
        preparePerformancePlot(n)
    plt.savefig("final_performance.png")

if __name__ == "__main__":
    # Compile both cpp, openmp and cuda files
    print("Compiling cpp implementation...")
    cppFile = "TSPBuscaLocal.cpp"
    cppBinary = "cpptsp"
    compileCode("g++", [f"-o {cppBinary}", "-O3", cppFile])
    print("Cpp binary successfully compiled!\n")

    print("Compiling cpp openmp implementation...")
    ompBinary = "omptsp"
    compileCode("g++", [f"-o {ompBinary}", "-O3", "-fopenmp", cppFile])
    print("Cpp openmp binary successfully compiled!\n")

#    print("Compiling cuda implementation...")
#    cudaFile = "TSPBuscaLocal.cu"
#    cudaBinary = "cudatsp"
#    compileCode("nvcc", [f"-o {cudaBinary}", cudaFile])
#    print("Cuda binary successfully compiled!\n")

    binaries = [cppBinary, ompBinary]
    runTests(binaries)
