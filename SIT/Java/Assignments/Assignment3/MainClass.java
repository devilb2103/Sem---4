package Java.Assignments.Assignment3;

public class MainClass {
    public static void main(String[] args) {
        final Storage storage = Storage.getInstance();
        InputClass input = new InputClass();

        boolean exit = false;
        while (!exit) {
            System.out.println(
                """

                    1) Add student
                    2) Display Students
                    3) Search Student
                    4) Update Student Data
                    5) Delete Student
                    6) Exit program
                    
                """);
            
            System.out.println("Your option: ");
            int option = input.intInput();
            switch (option) {
                case 1:
                    // Add student
                    storage.addStudent();
                    break;

                case 2:
                    // Display students
                    storage.displayDB();
                    break;
                
                case 3:
                    // Search student
                    storage.searchStudent();
                    break;
                
                case 4:
                    // Update student data
                    storage.updateStudent();
                    break;
                
                case 5:
                    // Delete student
                    storage.deleteStudent();
                    break;
                
                case 6:
                    // exit program
                    input.disposeScanner();
                    exit = true;
                    break;
                
                default:
                    System.out.println("Invalid Option");
                    break;
            }

        }
    }
}
