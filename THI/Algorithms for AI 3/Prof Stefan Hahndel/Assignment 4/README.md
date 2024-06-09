# Sudoku Solver

This Python script solves Sudoku puzzles using a backtracking algorithm with forward checking.

## Usage

To use the Sudoku solver script, follow these steps:

1. **Navigate to the Directory:**
2. **Run the Script:** python Assignment_4.py <input_file> [<solution_limit>]

-   `<input_file>`: Path to the text file containing the Sudoku puzzle. Each row of the puzzle should be represented as a string of digits without spaces.
-   `[<solution_limit>]` (Optional): Maximum number of solutions to find. If not provided, the script will find all possible solutions.

3. **Example:**
   `python sudoku_solver.py sudoku.txt 2`

4. **View Results:**

-   The initial Sudoku board will be displayed.
-   If solutions are found, the solved Sudoku boards will be displayed.
-   If no solution exists, a message will be shown.

## Sample Sudoku Puzzle

Here's an example of a Sudoku puzzle file format:

800009740
000400000
010000002
000090800
000806000
009050000
570000010
000008000
098000006
