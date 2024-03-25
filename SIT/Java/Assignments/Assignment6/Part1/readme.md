# Fixed and Dynamic Stack Implementation

This Java project implements fixed-size and dynamic stack data structures along with a demonstration in the `MainClass`.

## Table of Contents

- [Overview](#overview)
- [FixedStack](#fixedstack)
- [DynamicStack](#dynamicstack)
- [MainClass](#mainclass)

## Overview

The project consists of three classes:

- `FixedStack`: Implements a fixed-size stack.
- `DynamicStack`: Implements a dynamic-size stack.
- `MainClass`: Contains the main method to demonstrate the usage of both stack implementations.

## FixedStack

The `FixedStack` class implements a fixed-size stack data structure. It has the following methods:

- `isOverflow()`: Checks if the stack is full.
- `isUnderflow()`: Checks if the stack is empty.
- `pop()`: Removes and returns the top element from the stack.
- `push(int x)`: Adds an element to the top of the stack.
- `display()`: Displays the elements of the stack.

## DynamicStack

The `DynamicStack` class implements a dynamic-size stack data structure using an ArrayList. It has the same methods as `FixedStack`.

## MainClass

The `MainClass` contains a demonstration of both `FixedStack` and `DynamicStack` implementations. It creates instances of both stack types and performs various operations like pushing, popping, and displaying elements.

### Usage

To use the provided classes, you can create instances of `FixedStack` and `DynamicStack` and call their methods according to your requirements.
