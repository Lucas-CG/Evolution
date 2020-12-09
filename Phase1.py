import subprocess
from multiprocessing import Pool

def runABC(runid):
    result = subprocess.check_output(['python3', 'RunABCExperiments.py'])

def runACO(runid):
    result = subprocess.check_output(['python3', 'RunACOExperiments.py'])

def runAGA(runid):
    result = subprocess.check_output(['python3', 'RunAGAExperiments.py'])

def runDE(runid):
    result = subprocess.check_output(['python3', 'RunDEExperiments.py'])

def runGA(runid):
    result = subprocess.check_output(['python3', 'RunGAExperiments.py'])

def runPSO(runid):
    result = subprocess.check_output(['python3', 'RunPSOExperiments.py'])

def runRegPSO(runid):
    result = subprocess.check_output(['python3', 'RunRegPSOExperiments.py'])

def runSSO(runid):
    result = subprocess.check_output(['python3', 'RunSSOExperiments.py'])


if __name__ == "__main__":

    runs = 25
    p = Pool(runs)
    outputs = p.imap(runACO, range(runs))
    for output in outputs:
        print(output)
