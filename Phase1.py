import subprocess
from multiprocessing import Pool

def runABC():
    result = subprocess.check_output(['python3', 'RunABCExperiments.py'])

def runACO():
    result = subprocess.check_output(['python3', 'RunACOExperiments.py'])

def runAGA():
    result = subprocess.check_output(['python3', 'RunAGAExperiments.py'])

def runDE():
    result = subprocess.check_output(['python3', 'RunDEExperiments.py'])

def runGA():
    result = subprocess.check_output(['python3', 'RunGAExperiments.py'])

def runPSO():
    result = subprocess.check_output(['python3', 'RunPSOExperiments.py'])

def runRegPSO():
    result = subprocess.check_output(['python3', 'RunRegPSOExperiments.py'])

def runSSO():
    result = subprocess.check_output(['python3', 'RunSSOExperiments.py'])

def doFunc(func):
    func()

if __name__ == "__main__":

    runs = 8
    p = Pool(runs)
    funList = [runABC, runACO, runAGA, runDE, runGA, runPSO, runRegPSO, runSSO]
    outputs = p.imap(doFunc, funList)
    for output in outputs:
        print(output)
