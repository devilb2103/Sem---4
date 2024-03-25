# Student Management System

This Java program implements a simple student management system where users can perform various operations such as adding students, displaying student records, searching for students by different criteria, updating student data, and deleting students from the database.

## Functions and Methods

### InputClass

- `showSC_Hash()`: Displays the hash code of the static Scanner instance.
- `disposeScanner()`: Closes the static Scanner instance.
- `intInput()`: Reads an integer input from the user.
- `doubleInput()`: Reads a double input from the user.
- `strInput()`: Reads a string input from the user.

### MainClass

- `main(String[] args)`: The main method that starts the program execution.

### Storage

- `addStudent()`: Adds a student record to the database.
- `displayDB()`: Displays all student records stored in the database.
- `searchStudent()`: Searches for a student by PRN, Name, or Marks Rank.
- `updateStudent()`: Updates the details of a student in the database.
- `deleteStudent()`: Deletes a student record from the database.

### Student

- Constructor `Student(int prn, String name, int marks)`: Initializes a new student object with PRN, Name, and Marks.

### Utils

- `studentInput()`: Reads input for adding a student record.
- `searchByInput()`: Reads input for searching a student by PRN, Name, or Marks Rank.
- `copyArrayList(ArrayList<Student> source, ArrayList<Student> target)`: Copies data from one student ArrayList to another.
- `SS_Instance()`: Returns an instance of the inner class `SearchAndSort`.

### SearchAndSort (Inner Class of Utils)

- `binarySearch(ArrayList<Student> arr, String attribute, int len, Student target)`: Performs binary search on an ArrayList of students.
- `quickSort(ArrayList<Student> arr, String attribute)`: Sorts the ArrayList using the quicksort algorithm.
- `compareByAttribute(Student a, Student b, String attribute)`: Compares students based on a specific attribute (PRN, Name, or Marks).

## How to Use

1. **Clone the Repository**: Clone this repository to your local machine.

   ```bash
   git clone <repository_url>
   ```

2. **Compile and Run**: Compile the Java files and execute the MainClass file to start the program.

   ```bash
   cd StudentManagementSystem
   javac Java/Assignments/Assignment3/*.java
   java Java.Assignments.Assignment3.MainClass
   ```

3. **Follow On-Screen Instructions**: Once the program starts, follow the on-screen instructions to perform various operations such as adding, displaying, searching, updating, or deleting student records.

4. **Exit Program**: To exit the program, choose the 'Exit' option from the menu.
