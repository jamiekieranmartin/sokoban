from __future__ import print_function
from __future__ import division


from sokoban import Warehouse, find_2D_iterator
from mySokobanSolver import solve_sokoban_macro
import time

t1_test = '''
####
# .#  
#  ###
#*@  #
#  $ #
#  ###
####  
'''

t1_expected = []

def same_multi_line_strings(s1,s2):
    '''
    Auxiliary function to test two multi line string representing warehouses
    '''
    L1 = [s.rstrip() for s in s1.strip().split('\n')]
    L2 = [s.rstrip() for s in s2.strip().split('\n')]
    S1 = '\n'.join(L1)
    S2 = '\n'.join(L2)
    return S1==S2

def test(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print("Testing:", problem_file)
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    time1 = time.time()
    answer = solve_sokoban_macro(wh)
    print(time.time() - time1)
    print(answer)

def test_expected(n, expected):
    wh = Warehouse()
    # removes unneccessary \n
    wh.from_lines(n.split(sep='\n'))
    answer = solve_sokoban_macro(wh)
    if answer == expected:
        print('Test Passed')
    else:
        print("Test Failed")

if __name__ == "__main__":

    # test_expected(t1_test, t1_expected)

    print('Test a Custom Puzzle')
    print("enter 'quit' to exit\n")
    c = input('Warehouse number: ')
    while c != 'quit':
        try:
            test(c)
        except FileNotFoundError as e:
            print("Warehouse %s does not exist\n" % str(c))
        c = input('Warehouse number: ')