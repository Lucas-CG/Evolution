import subprocess
from multiprocessing import Pool

def runSSO(runid):
    result = subprocess.check_output(['python3', 'RunSSOExperiments.py'])
    return str(runid) + "," + result.decode('utf-8')

if __name__ == "__main__":

    runs = 25
    p = Pool(runs)
    outputs = p.imap(runSSO, range(runs))
    for output in outputs:
        print(output)
