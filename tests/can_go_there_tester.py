from __future__ import print_function
from __future__ import division


from sokoban import Warehouse
from mySokobanSolver import can_go_there


t1 = '''
####
# .#  
#  ###
#*@  #
#  $ #
#  ###
####  
'''

t1_expected = False

t2 = '''
####
# .#  
#  ###
#*@  #
#  $ #
#  ###
####  
'''

t2_expected = True

t3 ='''
#######
#@ $ .#
#. $  #
#######'''

t3_expected = False

def test_can_go_there(test, dst, expected):
    wh = Warehouse()
    print(test)
    # removes unneccessary \n
    wh.from_lines(test.split(sep='\n'))
    answer = can_go_there(wh, dst)
    print(wh.worker, '->', dst, '=', answer)
    if answer is expected:
        print("Test Passed")
    else:
        print("Test Failed")


if __name__ == "__main__":
    # out of bounds test
    test_can_go_there(t1, (0, 4), t1_expected)
    # normal valid movement
    test_can_go_there(t2, (1, 1), t2_expected)
    # blocked by box
    test_can_go_there(t3, (1, 4), t3_expected)