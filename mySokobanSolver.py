"""

    2020 CAB320 Sokoban assignment


The functions and classes defined in this module will be called by a marker script.
You should complete the functions and classes according to their specified interfaces.
No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.


You are NOT allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the
interface and results in a fail for the test of your code.
This is not negotiable!


"""

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search
import sokoban

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# -- Additional Import -- #

import itertools

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# -- Global Variables -- #

# sokoban squares
SPACE = ' '
WALL = '#'
BOX = '$'
TARGET_SQUARE = '.'
PLAYER = '@'
PLAYER_ON_TARGET_SQUARE = '!'
BOX_ON_TARGET = '*'
TABOO = 'X'
NEW_LINE = '\n'
EMPTY_STRING = ''

# different types of target squares
TARGETS = [TARGET_SQUARE, PLAYER_ON_TARGET_SQUARE, BOX_ON_TARGET]

# helper for corners + agent action types (x, y)
ACTIONS = {'Up': (0, -1), 'Left': (-1, 0), 'Down': (0, 1), 'Right': (1, 0)}

# game outcome
FAILED = 'Impossible'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# -- Auxiliary Functions -- #

def add_action(state, action):
    """
    adds the action tuple to the state tuple and returns a new state

    @param state: current state to act upon (x, y)
    @param action: the action to act upon (x, y)

    @return
        new state of the acted upon action taking into account the scale
    """
    return state[0] + action[0], state[1] + action[1]


def opposite_action(action):
    """
    Get the opposite of the given action

    @param action: the action to check

    @return
        the returned action
    """
    if action == 'Up':
        return 'Down'
    elif action == 'Left':
        return 'Right'
    elif action == 'Down':
        return 'Up'
    elif action == 'Right':
        return 'Left'
    else:
        return NotImplementedError()


def check_if_corner_cell(walls, dst):
    """
    checks the warehouse and determines if the cell is surrounded by a corner

    @param walls: list of walls in (x, y) format
    @param dst: the (row, col) of the position to test

    @return
        True if two surroundings diagonally adjacent to each other are both walls, thus the dst is in a corner
        False otherwise
    """
    # get a list of surroundings in (x, y) form
    surroundings = list(ACTIONS.values())
    for i, (a_x, a_y) in enumerate(surroundings):
        # gets the next cell diagonally adjacent, use of mod wraps the
        # index around back to the start for the final test
        next_corner_index = (i + 1) % len(surroundings)
        (b_x, b_y) = surroundings[next_corner_index]

        # if both are walls, as in is a corner, then return True
        if (dst[1] + a_x, dst[0] + a_y) in walls and (dst[1] + b_x, dst[0] + b_y) in walls:
            return True
    return False


def check_along_wall(walls, dst):
    """
    checks the warehouse and determines if the cell is along a wall

    @param walls: list of walls in (x, y) format
    @param dst: the (row, col) of the position to test

    @return
        True if the position is next to a wall
        False otherwise
    """
    # get a list of surroundings in (x, y) form
    surroundings = list(ACTIONS.values())
    (row, col) = dst
    for (a_x, a_y) in surroundings:
        # if next to wall then return True
        if (col + a_x, row + a_y) in walls:
            return True
    return False


def matrix_to_string(warehouse_matrix):
    """
    converts a 2D array of chars to a string

    @param warehouse_matrix: 2D array of chars representing the warehouse

    @return
        a string representing the warehouse
    """
    return NEW_LINE.join([EMPTY_STRING.join(row) for row in warehouse_matrix])


def warehouse_to_matrix(warehouse):
    """
    converts a string to a 2D array of chars

    @param warehouse: the warehouse object

    @return
        2D array of chars representing the warehouse
    """
    return [list(line) for line in warehouse.__str__().split(NEW_LINE)]


def manhattan_distance(start, end):
    """
    manhattan distance |x2 - x1| + |y2 - y1|

    @param start: the first x, y value
    @param end: the last x, y value

    @return
        the calculated manhattan distance between the two given tuples
    """
    return abs(end[0] - start[0]) + abs(end[1] - start[1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# -- Auxiliary Class -- #

class PathProblem(search.Problem):

    def __init__(self, warehouse, goal):
        """
        initialises the problem

        @param warehouse: a valid Warehouse object
        @param goal: the (x, y) location for the worker to attempt to go to
        """
        self.initial = warehouse.worker
        self.boxes_and_walls = set(itertools.chain(warehouse.walls, warehouse.boxes))
        self.goal = goal

    def actions(self, state):
        """
        yield all possible actions

        @param state: state of the worker (x, y)

        @yield
            possible actions for the worker to take given the current state
        """
        for action in ACTIONS.values():
            # check that the new state from the given action doesn't result in a wall or box collision
            if add_action(state, action) not in self.boxes_and_walls:
                yield action

    def result(self, state, action):
        """
        return the new state with the action applied

        @param state: current state of the worker
        @param action: the action to be acted upon by the worker

        @return
            the new state of the worker after acting upon the action
        """
        return add_action(state, action)

    def h(self, n):
        """
        heuristic using manhattan distance for A* graph search |x2 - x1| + |y2 - y1|

        @param n: the current node

        @return
            the manhattan distance between the worker and it's goal
        """
        return manhattan_distance(self.goal, n.state)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    """
    Identify the taboo cells of a warehouse. A cell inside a warehouse is
    called 'taboo'  if whenever a box get pushed on such a cell then the puzzle
    becomes unsolvable. Cells outside the warehouse should not be tagged as taboo.
    When determining the taboo cells, you must ignore all the existing boxes,
    only consider the walls and the target  cells.
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of
             these cells is a target.

    @param warehouse:
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the puzzle with only the wall cells marked with
       a '#' and the taboo cells marked with a 'X'.
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.
    """

    # convert warehouse into 2D array of characters
    warehouse_matrix = warehouse_to_matrix(warehouse)

    worker, walls = warehouse.worker, set(warehouse.walls)

    # ignore boxes for can_go_there method
    warehouse.boxes = []

    # iterate through rows and cols
    for row_index in range(warehouse.nrows):
        for col_index in range(warehouse.ncols):
            position = (row_index, col_index)
            cell = warehouse_matrix[row_index][col_index]

            # remove unnecessary chars
            if cell is PLAYER or cell is BOX:
                warehouse_matrix[row_index][col_index] = SPACE

            # rule 1: if a cell is a corner and not a target, then it is a taboo cell.
            if cell is not WALL and cell not in TARGETS:
                # if the position is a corner cell and is either where the worker is
                # or it can go there (we don't care about stuff outside of the playing field)
                # then set the character to a taboo cell
                if check_if_corner_cell(walls, position) and (
                        position == worker or can_go_there(warehouse.copy(), position)):
                    warehouse_matrix[row_index][col_index] = TABOO

                    # rule 2: all the cells between two corners along a wall are taboo if none of these cells is a
                    # target. from the taboo point get the rest of the row (col_index + 1) to the right of it
                    # and iterate
                    next_in_column = col_index + 1
                    for taboo_col_index in range(next_in_column, warehouse.ncols):
                        taboo_cell = warehouse_matrix[row_index][taboo_col_index]
                        # if there's any targets or walls break
                        if taboo_cell in TARGETS or taboo_cell is WALL:
                            break

                        taboo_position = (row_index, taboo_col_index)

                        # find another taboo cell or corner
                        if check_if_corner_cell(walls, taboo_position):
                            current_section = range(next_in_column, taboo_col_index)
                            # if the entire row is along a wall then the entire row is taboo
                            cells_along_wall = [check_along_wall(walls, (row_index, i)) for i in current_section]
                            if all(cells_along_wall):
                                # fill with taboo
                                for taboo_index in range(next_in_column, taboo_col_index):
                                    warehouse_matrix[row_index][taboo_index] = TABOO

                    # from the taboo point get the rest of the column (row_index + 1) below it and enumerate over
                    next_in_row = row_index + 1
                    for taboo_row_index in range(next_in_row, warehouse.nrows):
                        taboo_cell = warehouse_matrix[taboo_row_index][col_index]
                        # if there's any targets or walls break
                        if taboo_cell in TARGETS or taboo_cell is WALL:
                            break

                        taboo_position = (taboo_row_index, col_index)

                        # find another taboo cell or corner
                        if check_if_corner_cell(walls, taboo_position):
                            current_section = range(next_in_row, taboo_row_index)
                            # if the entire column is along a wall then the entire column is taboo
                            cells_along_wall = [check_along_wall(walls, (i, col_index)) for i in current_section]
                            if all(cells_along_wall):
                                # fill with taboo
                                for taboo_index in range(next_in_row, taboo_row_index):
                                    warehouse_matrix[taboo_index][col_index] = TABOO

    # return to string variable
    warehouse_string = matrix_to_string(warehouse_matrix)

    # remove target chars
    for square in TARGETS:
        warehouse_string = warehouse_string.replace(square, SPACE)

    return warehouse_string


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SokobanPuzzle(search.Problem):
    """
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of
    the provided module 'search.py'.

    Each SokobanPuzzle instance should have at least the following attributes
    - self.allow_taboo_push
    - self.macro

    When self.allow_taboo_push is set to True, the 'actions' function should
    return all possible legal moves including those that move a box on a taboo
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.

    If self.macro is set True, the 'actions' function should return
    macro actions. If self.macro is set False, the 'actions' function should
    return elementary actions.
    """

    def __init__(self, warehouse, macro=False, allow_taboo_push=False, push_costs=None):
        """
        initialisation function

        stores the state as a tuple of (worker, frozenset((box, cost), ...)
        and any other necessary information for performing the a* graph search algorithm

        @param warehouse: the warehouse object
        @param macro: default False, whether to use macro actions
        @param allow_taboo_push: default False, whether to allow boxes to be pushed onto taboo cells
        @push_costs: default None, list of integer costs following the same order of warehouse.boxes
        """
        self.initial = (warehouse.worker, frozenset(zip(warehouse.boxes, push_costs))) \
            if push_costs is not None \
            else (warehouse.worker, frozenset((box, 0) for box in warehouse.boxes))

        # custom variable inputs
        self.push_costs = push_costs
        self.macro = macro
        self.allow_taboo_push = allow_taboo_push

        # helpers
        self.taboo_cells = set(sokoban.find_2D_iterator(taboo_cells(warehouse).split(sep='\n'), "X"))
        self.walls = set(warehouse.walls)
        self.goal = set(warehouse.targets)

        # for macro actions can_go_there purposes
        self.warehouse = warehouse

    def actions(self, state):
        """
        yield of all possible actions

        @param state: state of the puzzle as a tuple of (worker, [(box, cost), ...])

        @yield
            possible actions for the worker to take given the current state
        """
        (worker, boxes) = state
        boxes = set(box for (box, _) in boxes)

        # macro actions
        if self.macro:
            # go through boxes and determine what worker can do to them
            for box in boxes:
                # enumerate through possible surroundings of each box
                for action in ACTIONS:
                    surr = ACTIONS[action]
                    # test the possible surroundings for the worker to move to
                    test_pos = add_action(box, surr)
                    # if the worker can't go there then it's not a valid move
                    if worker == test_pos or \
                            can_go_there(self.warehouse.copy(worker=worker, boxes=boxes), tuple(reversed(test_pos))):
                        # new position of the box when pushed, opposite direction of current surrounding
                        # as we're testing the X position next to the box
                        # to see if we can push it in the direction of the box
                        opp_action = opposite_action(action)
                        opposite_surr = ACTIONS[opp_action]
                        new_box_pos = add_action(box, opposite_surr)

                        # ensure the new box position doesn't merge with a wall, box and
                        # that allow taboo push is true or the test box not in taboo_cells
                        if new_box_pos not in boxes and new_box_pos not in self.walls \
                                and (self.allow_taboo_push or new_box_pos not in self.taboo_cells):
                            # get the opposite of the surrounding as in,
                            # worker goes to the 'Left' and pushes the box 'Right'
                            yield tuple(reversed(box)), opp_action
        # elementary actions
        else:
            # enumerate through possible surroundings of the worker
            for action in ACTIONS:
                surr = ACTIONS[action]
                # add the surrounding to the workers current position to test if it's viable
                test_pos = add_action(worker, surr)

                # ensure it's not in a wall
                if test_pos not in self.walls:
                    # if it's not in a box then the worker can move there
                    if test_pos not in boxes:
                        yield action

                    # if it's within a box test new position of the box
                    else:
                        # this is the position two spaces from the current worker
                        test_pos = add_action(test_pos, surr)
                        # ensure the new box position doesn't merge with a wall, box and
                        # that allow taboo push is true or the test box not in taboo_cells
                        if test_pos not in boxes and test_pos not in self.walls \
                                and (self.allow_taboo_push or test_pos not in self.taboo_cells):
                            yield action

    def path_cost(self, c, state1, movement_cost, state2):
        """
        the path cost of the change from state 1 to state 2

        @param c: the current cost
        @param state1: current state of the puzzle
        @param movement_cost: the action to get from state1 to state2
        @param state2: new state of the puzzle

        @return
            the cost of performing the action to get from state1 to state2
        """
        movement_cost = 1

        # determines if we need to worry about push_costs
        if self.push_costs is not None:
            # copy the two states into workable variables
            (old_worker, old_boxes), (new_worker, new_boxes) = state1, state2
            # set comparison is unordered + we shouldn't have a case of box_stack up as this has already been checked
            old_boxes, new_boxes = set(old_boxes), set(new_boxes)

            # if the two are different try find the box that moved
            if new_boxes != old_boxes:
                for box_index, (box, cost) in enumerate(new_boxes):
                    # assign push_cost the cost of the box movement
                    if (box, cost) not in old_boxes:
                        return c + movement_cost + cost

        # returns the current cost + 1 for an action + the push cost
        return c + movement_cost

    def goal_test(self, state):
        """
        tests if the state is in the goal position

        @param state: current state of the puzzle

        @return
            True if the goal is met
            False otherwise
        """
        (_, boxes) = state
        return set(box for (box, _) in boxes) == self.goal

    def result(self, state, action):
        """
        act upon the given action using the given state

        @param state: current state of the puzzle
        @param action: the action to act upon

        @return
            the new state
        """
        # copy the state into workable variables
        (worker, boxes) = state
        boxes = list(boxes)

        # macro result
        if self.macro:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = ACTIONS[action[1]]
            # assigns the worker their new position
            # flip the action because it's in row, col (y, x) not x, y
            worker = tuple(reversed(action[0]))
        # elementary result
        else:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = ACTIONS[action]
            # assigns the worker their new position
            worker = add_action(worker, next_pos)

        # update the box if one is pushed
        for i, (box, cost) in enumerate(boxes):
            if worker == box:
                boxes[i] = (add_action(box, next_pos), cost)

        return worker, frozenset(boxes)

    def h(self, n):
        """
        heuristic using that defines the closest box to the worker
        and also the closest box to target combination,
        incorporating push_costs if necessary

        @param n: current node

        @return
            the heuristic value to favour any paths that move the worker closer to boxes
            and push boxes closer to any targets, taking into account push_costs if any
            0 if the worker is next to a box and all boxes are on a target
        """
        # copy the state into workable variables
        (worker, boxes) = n.state
        boxes = list(boxes)

        # initialise the list of distances
        # we don't care about double ups we just want the smallest possible answer, hence using a set
        worker_to_box_distances, box_to_target_totals = set(), set()

        # iterate through boxes and append the distance for each from worker
        for (box, _) in boxes:
            distance_to_box = manhattan_distance(worker, box)
            worker_to_box_distances.add(distance_to_box)

        # iterate through each permutation of targets to find the distance between each box
        for targets_perm in itertools.permutations(self.goal):
            total_cost = 0

            # combines targets and boxes in tuples as in (target, box)
            zipped_tuples = zip(targets_perm, boxes)

            # for each target and box get the manhattan distance for each
            for target, (box, cost) in zipped_tuples:
                # cost is incorporated to ensure the worker understands
                # the effort required to push this box.
                # if it's 0 make it 1 because it still costs the worker to move there
                cost = 1 if cost == 0 else cost
                # append the calculated box_to_target cost for this permutation
                box_to_target = manhattan_distance(target, box) * cost
                total_cost += box_to_target

            # add the total so we have
            box_to_target_totals.add(total_cost)

        # return the smallest worker to box distance and smallest box to target total distance
        return min(worker_to_box_distances) + min(box_to_target_totals)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    """

    Determine if the sequence of actions listed in 'action_seq' is legal or not.

    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.

    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']

    @return
        The string 'Impossible', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    """
    # copies warehouse into a new Sokoban puzzle
    puzzle = SokobanPuzzle(warehouse.copy())

    # initialises the walls, boxes and worker for use in the action sequence
    boxes, worker = warehouse.boxes, warehouse.worker

    # iterates over the actions
    for action in action_seq:
        # we can use the result() to get the state of the acted upon result of each action
        no_cost = 0
        state = (worker, frozenset((box, no_cost) for box in boxes))
        (worker, boxes) = puzzle.result(state, action)
        # get the list of just the boxes, no costs
        boxes = list(box for (box, _) in boxes)

        # ensures the worker hasn't clipped a wall
        if worker in puzzle.walls:
            return FAILED

        # ensure boxes aren't stacked
        if len(boxes) != len(set(boxes)):
            return FAILED

        # ensures no boxes have clipped any walls
        for box in boxes:
            if box in puzzle.walls:
                return FAILED

    # return a copy of the warehouse with the new worker and boxes
    return warehouse.copy(worker=worker, boxes=boxes).__str__()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    """
    This function should solve using A* algorithm and elementary actions
    the puzzle defined in the parameter 'warehouse'.

    In this scenario, the cost of all (elementary) actions is one unit.

    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    """

    path = search.astar_graph_search(SokobanPuzzle(warehouse))

    if path is not None:
        return path.solution()

    return FAILED


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    """
    Determine whether the worker can walk to the cell dst=(row,column)
    without pushing any box.

    @param warehouse: a valid Warehouse object
    @param dst: (row, col) of the location to go to

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    """
    # separate row, col for usage below
    (row, col) = dst

    # the player is only able to move to a space and a target square
    allowed_cells = {SPACE, TARGET_SQUARE}

    # convert the warehouse to a Array<Array<char>>
    warehouse_matrix = warehouse_to_matrix(warehouse)

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    if warehouse_matrix[row][col] not in allowed_cells:
        return False

    # check if a valid path from the worker to the coordinate provided exists
    path = search.astar_graph_search(PathProblem(warehouse, (col, row)))

    return path is not None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    """
    Solve using using A* algorithm and macro actions the puzzle defined in
    the parameter 'warehouse'.

    A sequence of macro actions should be
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ]
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.

    In this scenario, the cost of all (macro) actions is one unit.

    @param warehouse: a valid Warehouse object

    @return
        If the puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    """

    path = search.astar_graph_search(SokobanPuzzle(warehouse, macro=True))

    if path is not None:
        return path.solution()

    return FAILED


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban_elem(warehouse, push_costs):
    """
    In this scenario, we assign a pushing cost to each box, whereas for the
    functions 'solve_sokoban_elem' and 'solve_sokoban_macro', we were
    simply counting the number of actions (either elementary or macro) executed.

    When the worker is moving without pushing a box, we incur a
    cost of one unit per step. Pushing the ith box to an adjacent cell
    now costs 'push_costs[i]'.

    The ith box is initially at position 'warehouse.boxes[i]'.

    This function should solve using A* algorithm and elementary actions
    the puzzle 'warehouse' while minimizing the total cost described above.

    @param
     warehouse: a valid Warehouse object
     push_costs: list of the weights of the boxes (pushing cost)

    @return
        If puzzle cannot be solved return 'Impossible'
        If a solution exists, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    """

    path = search.astar_graph_search(SokobanPuzzle(warehouse, push_costs=push_costs))

    if path is not None:
        return path.solution()

    return FAILED

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
