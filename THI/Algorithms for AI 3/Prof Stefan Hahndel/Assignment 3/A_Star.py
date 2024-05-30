# %%
import heapq

# %%
class State:
    heuristic = None
    cost = None
    arr = None
    empty_space_coord = [None, None]
    parent = None
    target = None
    
    def __init__(self, state: list, target: list, cost = 0, parent = None) -> None:
        self.heuristic = self.manhattan_distance(state, target)
        self.arr = state
        self.cost = cost
        self.parent = parent
        self.target = target
    
    # calculate heuristic value (manhattan distance)
    def manhattan_distance(self, state, goal):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:  # We don't calculate distance for the empty space (0)
                    goal_pos = [(row, col) for row in range(3) for col in range(3) if goal[row][col] == state[i][j]]
                    distance += abs(goal_pos[0][0] - i) + abs(goal_pos[0][1] - j)
                else:
                    self.empty_space_coord = [i,j]
        return distance
    

    # get all possible empty space moves
    def get_next_states(self):
        state = self.arr

        # return new pos with swapped position of two numbers
        def swap_positions(state, pos1, pos2):
            new_state = [list(row) for row in state]
            new_state[pos1[0]][pos1[1]], new_state[pos2[0]][pos2[1]] = new_state[pos2[0]][pos2[1]], new_state[pos1[0]][pos1[1]]
            return new_state
        
        next_moves = []
        row, col = self.empty_space_coord
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # apply swaps while checking for limits and store new moves
        for row_delta, column_delta in directions:
            new_row, new_col = row + row_delta, col + column_delta
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = swap_positions(state, (row, col), (new_row, new_col))
                new_cost = self.cost + 1
                next_moves.append(State(new_state, self.target, new_cost, self))
        
        return next_moves
    
    # define comparator for use in a minheap
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

# %%
def reconstruct_path(state):
    path = []
    while state:
        path.append(state.arr)
        state = state.parent
    path.reverse()
    return path

# mutable lists are unhashable
def list_to_tuple(x: list):
    return tuple(tuple(row) for row in x)

def a_star(start, target):
    # initialize a* minheap and closed set
    start_state = State(start, target)
    open_list = []
    heapq.heappush(open_list, start_state)
    closed_set = set()

    while open_list:
        # get min element (min by cost + heuristics)
        current_state = heapq.heappop(open_list)

        # if goal state is reached then print path
        if current_state.arr == target:
            return reconstruct_path(current_state)

        # if goal state is not reached, mark current state as visited
        closed_set.add(list_to_tuple(current_state.arr))

        # for all next possible states
        for next_state in current_state.get_next_states():
            # if state is visited then skip
            if list_to_tuple(next_state.arr) in closed_set:
                continue
            
            # if state is not visited, then add to minheap
            heapq.heappush(open_list, next_state)
    
    return None

# check if inversions are in even count
def compute_inversions(arr):
    flattened = [tile for row in arr for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            if flattened[i] > flattened[j]:
                inversions += 1
    return inversions

def check_solvable(start, goal):
    return compute_inversions(start) % 2 == compute_inversions(goal) % 2



# handle inputs
while(True):
    start = []
    target = []

    print("Enter Row Wise Inputs of Matrix A (Format: <x> <y> <z>) (Example: 1 2 3)")
    for i in range(3):
        start.append([int(x) for x in str(input(f"row {i}: ")).split()])
    
    print("Enter Row Wise Inputs of Matrix B (Format: <x> <y> <z>) (Example: 1 2 3)")
    for i in range(3):
        target.append([int(x) for x in str(input(f"row {i}: ")).split()])

    if(check_solvable(start, target)):
        solution_path = a_star(start, target)

        if solution_path:
            print(f"Solution found in {len(solution_path) - 1} steps")
            for step in solution_path:
                for row in step:
                    print(row)
                print()
        else:
            print("No solution found")
    
    else:
        print("no solution possible")
    
    continue_state = str(input("Continue? (Y/N): "))
    
    if(continue_state.lower() == "n"):
        exit(0)