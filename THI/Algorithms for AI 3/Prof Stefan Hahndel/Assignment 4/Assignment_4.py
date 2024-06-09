class sudoku_solver:
    def __init__(self, board, limit) -> None:
        self.board = board
        self.side_size = 9
        self.sub_square_size = 3
        self.solutions = []
        self.limit = limit
    
    # display the board
    def displayBoard(self, board) -> None:
        for i in range(self.side_size):
            # condition to print row borders
            if(i%3 == 0): print("-" * 22)

            for j in range(self.side_size):

                # condition to print column borders
                if(j%3 == 0): print("|", end="")
                print(board[i][j] if board[i][j] != 0 else ".", end=" ")
                # condition to print final column border
                if(j == self.side_size - 1): print("|")
            
            # condition to print final row border
            if(i == self.side_size - 1): print("-" * 22)
    
    # compute possible values for given position in puzzle state
    def possible_values(self, row, col, board):

        if board[row][col] != 0:
            return []  # no values are possible if already filled

        possible_vals = set(range(1, 10))

        # remove values in same row
        for i in range(self.side_size):
            if board[row][i] in possible_vals:
                possible_vals.remove(board[row][i])
        
        # remove values in same column
        for i in range(self.side_size):
            if board[i][col] in possible_vals:
                possible_vals.remove(board[i][col])
        
        # remove values in the same subgrid
        start_row = row - row % self.sub_square_size
        start_col = col - col % self.sub_square_size
        for i in range(start_row, start_row + self.sub_square_size):
            for j in range(start_col, start_col + self.sub_square_size):
                if board[i][j] in possible_vals:
                    possible_vals.remove(board[i][j])
        
        return list(possible_vals)

    # calculate possible values for all unknown
    # cells and store in a list as 2d vec
    def evaluate_branching_array(self, board) -> list:
        curr_puzzle_state = board
        branching_arr = []

        # for each unfilled cell, calculate possible values
        # after appliying row, column and subgrid constraints
        for row in range(len(curr_puzzle_state)):
            for column in range(len(curr_puzzle_state[0])):
                if(curr_puzzle_state[row][column] == 0):
                    branching_arr.append([[row, column], self.possible_values(row, column, curr_puzzle_state)])

        # sort branching array by number of
        # possible values in ascending order
        branching_arr = sorted(branching_arr, key = lambda x: len(x[1]))
        return branching_arr
    
    # reset solutions 
    def solve_sudoku(self) -> bool:
        self.solutions = []
        self.solver(self.board)
        return self.solutions
    
    # solve puzzle using backtracking and forward checking
    def solver(self, board) -> bool:
        branching_arr = self.evaluate_branching_array(board)

        if not branching_arr:
            self.solutions.append([row[:] for row in board])
            if(len(self.solutions) >= self.limit):
                return True # return True stop computing answers
            else:
                return False # return false and backtrack
                             # for next possible combination

        # get cell with the fewest possible values
        cell = branching_arr[0]
        row, col = cell[0]
        possible_vals = cell[1]

        for val in possible_vals:
            board[row][col] = val

            if self.solver(board):
                return True

            # set the current cell to 0 to undo computation for backtracking
            board[row][col] = 0
        
        return False



# read board data from file
def read_board(filename):
    with open(filename, 'r') as file:
        board = []
        for line in file:
            board.append([int(char) for char in line.strip()])
        return board


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python Assignment_4.py <input_file> [<solution_limit>]")
        sys.exit(1)

    filename = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) == 3 else sys.maxsize
    board = read_board(filename)
    solver = sudoku_solver(board, limit)

    print("Initial Sudoku Board:")
    solver.displayBoard(solver.board)

    solutions = solver.solve_sudoku()

    if(len(solutions) == 0):
        print("No Solutions Found")
    else:
        print(f"{len(solutions)} solutions found!", end = "\n\n")
        print("Solved Sudoku Board(s):")
        c = 0
        for solution in solutions:
            print(f"Solution {c}")
            c += 1
            sudoku_solver(solution, 0).displayBoard(solution)
            print()
