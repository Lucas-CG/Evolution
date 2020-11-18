from optproblems import cec2005

func = cec2005.F1(10)
FESCount = 0

def calc_func(func, args, FESCount):
    return func(args), FESCount+1

population = [ [1 for i in range(10)], [2 for i in range(10)], [3 for i in range(10)] ]

for i in range(len(population)):
    f, FESCount = calc_func(func, population[i], FESCount)
    print("Population and FESCount")
    print(f)
    print(FESCount)
    print()
