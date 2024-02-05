# Assignment 2 Java Program

This Java program provides functionalities related to array operations, including sorting elements into even and odd arrays, finding the smallest neighboring distance, and converting arrays to ArrayLists and vice versa.

## Table of Contents

- [Overview](#overview)
- [How to Use](#how-to-use)
- [Methods Overview](#methods-overview)

## Overview

The program consists of three classes: `MainClass`, `ArrayClass`, and `InputClass`.

## How to Use

1. Run the `MainClass` to execute the program.
2. Enter the number of inputs as prompted.
3. Input individual numbers as required.
4. View the output, including even and odd arrays, smallest neighboring distance, and array conversions.

## Methods Overview

### `MainClass`

- `main(String[] args)`: Entry point for the program, handles user input and displays results.

### `ArrayClass`

- `getEven(): int[]`: Returns the even array.
- `getOdd(): int[]`: Returns the odd array.
- `getArr(): int[]`: Returns the universal array.
- `appendEven(int x)`: Appends a number to the even array.
- `appendOdd(int x)`: Appends a number to the odd array.
- `appendNums(int x)`: Appends a number to the universal array.
- `findSmallestDistance(): int[]`: Finds the smallest neighboring distance in the array.
- `arrayToArrayList(int arr[]): ArrayList<Integer>`: Converts an array to an ArrayList.
- `ArrayListToArray(ArrayList<Integer> arrList): int[]`: Converts an ArrayList to an array.

### `InputClass`

- `disposeScanner()`: Closes the static Scanner instance.
- `intInput(): int`: Takes an integer input.
- `doubleInput(): double`: Takes a double input.
- `strInput(): String`: Takes a string input.
