from __future__ import print_function
from __future__ import division

from sokoban import Warehouse, find_2D_iterator
from mySokobanSolver import solve_sokoban_elem, solve_sokoban_macro, solve_weighted_sokoban_elem
import time


def test_elem(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print(n, ": ", end="")
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    time1 = time.time()
    solve_sokoban_elem(wh)
    print('{:06.3f}s'.format(time.time() - time1))


def test_macro(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print(n, ": ", end="")
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    time1 = time.time()
    a = solve_sokoban_macro(wh)
    print('{:06.3f}s'.format(time.time() - time1))


def test_weighted(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print(n, ": ", end="")
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    time1 = time.time()
    solve_weighted_sokoban_elem(wh, [1 for box in wh.boxes])
    print('{:06.3f}s'.format(time.time() - time1))


if __name__ == "__main__":
    print("ELEMENTARY TESTS")
    # test_elem("07")
    # test_elem("09")
    # test_elem("11")
    # test_elem("47")
    # test_elem("81")
    # test_elem("147")

    print("MACRO TESTS")
    # test_macro("07")
    # test_macro("09")
    # test_macro("11")
    # test_macro("47")
    # test_macro("81")
    # test_macro("147")

    print("WEIGHTED TESTS")
    test_weighted("07")
    test_weighted("09")
    test_weighted("11")
    test_weighted("47")
    test_weighted("81")
    test_weighted("147")
