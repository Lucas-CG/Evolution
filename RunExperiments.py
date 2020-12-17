import subprocess
from multiprocessing import Pool

def runABC():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'ABC'])

def runACO():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'ACO'])

def runAGA():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'AGA'])

def runDE():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'DE'])

def runGA():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'GA'])

def runPSO():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'PSO'])

def runRegPSO():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'RegPSO'])

def runSSO():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'SSO'])

def doFunc(func):
    func()

if __name__ == "__main__":

    runs = 8
    p = Pool(runs)
    funList = [runABC, runACO, runAGA, runDE, runGA, runPSO, runRegPSO, runSSO]
    outputs = p.imap(doFunc, funList)
    for output in outputs:
        print(output)
