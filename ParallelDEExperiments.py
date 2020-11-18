import subprocess
from multiprocessing import Pool

def runDE(runid):
    result = subprocess.check_output(['python3', 'RunDEExperiments.py'])
    return str(runid) + "," + result.decode('utf-8')

if __name__ == "__main__":

    runs = 5
    p = Pool(runs)
    outputs = p.imap(runDE, range(runs))
    for output in outputs:
        print(output)
