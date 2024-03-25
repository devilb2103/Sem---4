package Java.Assignments.Assignment3;

import java.util.ArrayList;
import Java.Assignments.Assignment3.Utils.SearchAndSort;

public class Storage{

    // singleton pattern allows only 1 storage class instance
    
    // single instance
    static Storage instance = new Storage();

    // does not allow constructor to be called
    private Storage(){}
    
    // returns instance of class for storing it in a reference
    static Storage getInstance() {
        return instance;
    }

    // ------------------------------------------------------------

    // external class instance declarations
    static Utils utils = new Utils();
    static SearchAndSort ss = utils.SS_Instance();
    InputClass input = new InputClass();
    
    // student db arrayList
    private static ArrayList<Student> studentDB_PRN = new ArrayList<Student>();
    private static ArrayList<Student> studentDB_Name = new ArrayList<Student>();
    private static ArrayList<Student> studentDB_Marks = new ArrayList<Student>();

    // initialize student db with some default values
    static{
        studentDB_PRN.add(new Student(1, "Dev", 1));
        studentDB_PRN.add(new Student(0, "Vedant", 4));
        studentDB_PRN.add(new Student(7, "Harsh", 24));
        studentDB_PRN.add(new Student(9, "Jaanvi", 6));
        studentDB_PRN.add(new Student(3, "Deepak", 26));
        studentDB_PRN.add(new Student(6, "Ishaan", 9));
        studentDB_PRN.add(new Student(13, "Luv", 14));
        studentDB_PRN.add(new Student(4, "Shruti", 35));

        utils.copyArrayList(studentDB_PRN, studentDB_Name);
        utils.copyArrayList(studentDB_PRN, studentDB_Marks);

        // System.out.println(studentDB_PRN.hashCode());
        // System.out.println(studentDB_Name.hashCode());
        // System.out.println(studentDB_Marks.hashCode());

        ss.quickSort(studentDB_PRN, "prn");
        ss.quickSort(studentDB_Name, "name");
        ss.quickSort(studentDB_Marks, "marks");
    }

    // Adds student record
    public Student addStudent(){
        Student entity = utils.studentInput();
        studentDB_PRN.add(entity);
        studentDB_Name.add(entity);
        studentDB_Marks.add(entity);
        
        System.out.println("Added students");
        ss.quickSort(studentDB_PRN, "prn");
        ss.quickSort(studentDB_Name, "name");
        ss.quickSort(studentDB_Marks, "marks");

        return entity;
    }

    // Display Students
    public void displayDB(){
        for (Student student : studentDB_PRN) {
            System.out.println(String.format("PRN: %d, Name: %s, Marks: %d", student.prn, student.name, student.marks));
        }
    }

    // search student by PRN / Name / Marks
    public void searchStudent(){
        
        int option = utils.searchByInput();

        // Search by PRN
        if(option == 1){
            System.out.println("Enter Prn: ");
            int prn = input.intInput();
            int index = ss.binarySearch(studentDB_PRN, "prn", studentDB_PRN.size() - 1, new Student(prn, "", 0));

            if(Integer.compare(index, -1) > 0){
                Student student = studentDB_PRN.get(index);
                System.out.println(String.format("PRN: %d, Name: %s, Marks: %d", student.prn, student.name, student.marks));
            }
            else{
                System.out.println(String.format("Student with PRN %d does not exist", prn));
            }
        }
        // Search by name
        else if(option == 2){
            System.out.println("Enter Name: ");
            String name = input.strInput().toLowerCase();
            int index = ss.binarySearch(studentDB_Name, "name", studentDB_Name.size() - 1, new Student(0, name, 0));

            if(Integer.compare(index, -1) > 0){
                Student student = studentDB_Name.get(index);
                System.out.println(String.format("PRN: %d, Name: %s, Marks: %d", student.prn, student.name, student.marks));
            }
            else{
                System.out.println(String.format("Student with Name %s does not exist", name));
            }
        }
        // search by marks rank
        else{
            System.out.println("Enter Rank: ");
            int rank = input.intInput();

            if(rank > studentDB_Marks.size() || rank < 1){
                System.out.println(String.format("Range of rank is between 1 and %d", studentDB_Marks.size()));    
            }
            else{
                Student student = studentDB_Marks.get(studentDB_Marks.size() - rank);
                System.out.println(String.format("Student at rank %d is: PRN: %d, Name: %s, Marks: %d", rank, student.prn, student.name, student.marks));
            }
        }
    }

    // update student details
    public void updateStudent(){
        System.out.println("Enter PRN: ");
        int prn = input.intInput();

        int prn_index = ss.binarySearch(studentDB_PRN, "prn", studentDB_PRN.size() - 1, new Student(prn, "", 0));

        if(Integer.compare(prn_index, -1) == 0){
            System.out.println(String.format("Student with PRN %d does not exist", prn));
            return;
        }

        int name_index = ss.binarySearch(studentDB_Name, "name", studentDB_Name.size() - 1, new Student(0, studentDB_PRN.get(prn_index).name, 0));
        int marks_index = ss.binarySearch(studentDB_Marks, "marks", studentDB_Marks.size() - 1, new Student(0, "", studentDB_PRN.get(prn_index).marks));

        boolean editing = true;
        Student newStudent = new Student(studentDB_PRN.get(prn_index).prn, studentDB_PRN.get(prn_index).name, studentDB_PRN.get(prn_index).marks);
        while(editing){
            System.out.println(String.format("""
                    Current Details
                    PRN: %d
                    Name: %s
                    Marks: %d
                    \n""", newStudent.prn, newStudent.name, newStudent.marks));
            
            System.out.println("""
                    1) Edit Name
                    2) Edit Marks
                    3) Apply
                    4) Cancel
                    """);
            
            System.out.println("Enter your option: ");
            int option = input.intInput();
            switch (option) {
                case 1:
                    System.out.println("Enter new name: ");
                    String name = input.strInput();
                    newStudent.name = name;
                    break;
                
                case 2:
                    System.out.println("Enter new marks: ");
                    int marks = input.intInput();
                    newStudent.marks = marks;
                    break;
                
                case 3:
                    studentDB_PRN.set(prn_index, newStudent);
                    studentDB_Name.set(name_index, newStudent);
                    studentDB_Marks.set(marks_index, newStudent);
                    return;

                case 4:
                    return;
                    
                default:
                    System.out.println("Invalid option");
                    break;
            }
        }

        studentDB_PRN.remove(prn_index);
        studentDB_Name.remove(name_index);
        studentDB_Marks.remove(marks_index);

        System.out.println(String.format("Deleted student with PRN: %d", prn));
    }

    // delete student (by PRN)
    public void deleteStudent(){

        System.out.println("Enter PRN: ");
        int prn = input.intInput();

        int prn_index = ss.binarySearch(studentDB_PRN, "prn", studentDB_PRN.size(), new Student(prn, "", 0));

        
        if(Integer.compare(prn_index, -1) == 0){
            System.out.println(String.format("Student with PRN %d does not exist", prn));
            return;
        }
        
        int name_index = ss.binarySearch(studentDB_Name, "name", studentDB_Name.size(), new Student(prn, "", 0));
        int marks_index = ss.binarySearch(studentDB_Marks, "marks", studentDB_Marks.size(), new Student(prn, "", 0));
        
        studentDB_PRN.remove(prn_index);
        studentDB_Name.remove(name_index);
        studentDB_Marks.remove(marks_index);

        System.out.println(String.format("Deleted student with PRN: %d", prn));
    }
}
