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

def runES():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'ES'])

def runCMAES():
    result = subprocess.check_output(['python3', 'Experiments.py', '--algorithm', 'CMA-ES'])

def doFunc(func):
    func()

if __name__ == "__main__":

    numAlgs = 4
    p = Pool(numAlgs)
    funList = [runAGA, runDE, runES, runCMAES]

    outputs = p.imap(doFunc, funList)
    for output in outputs:
        print(output)
