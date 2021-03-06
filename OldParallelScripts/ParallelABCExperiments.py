import subprocess
from multiprocessing import Pool

def runABC(runid):
    result = subprocess.check_output(['python3', 'RunABCExperiments.py'])
    return str(runid) + "," + result.decode('utf-8')

if __name__ == "__main__":

    runs = 25
    p = Pool(runs)
    outputs = p.imap(runABC, range(runs))
    for output in outputs:
        print(output)
