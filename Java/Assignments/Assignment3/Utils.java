package Java.Assignments.Assignment3;

import java.util.ArrayList;

public class Utils{
    
    InputClass input = new InputClass();
    
    // console input - output for adding a student to storage class
    public Student studentInput() {
        System.out.println("Enter Student's Prn:");
        int prn = input.intInput();

        System.out.println("Enter Student's Name:");
        String name = input.strInput();

        System.out.println("Enter Student's Final Marks:");
        int marks = input.intInput();

        
        Student stud = new Student(prn, name, marks);
        return stud;
    }

    // console input - output for searching a student from storage class
    public int searchByInput(){
        System.out.println("""

                1) Search by PRN
                2) Search by Name
                3) Search by Marks Rank
            
                """);
        
                int option = input.intInput();

                if(option >= 1 && option <= 3){
                    return option;
                }
                else{
                    return searchByInput();
                }
    }

    // method to copy data from one student arraylist to another of the same type
    public void copyArrayList(ArrayList<Student> source, ArrayList<Student> target){
        target.clear();
        for (int i = 0; i < source.size(); i++) {
            // Student stdCopy = new Student(source.get(i).prn, source.get(i).name, source.get(i).marks)
            target.add(source.get(i));
        }
    }

    // returns instance of the SearchAndSort inner class
    public SearchAndSort SS_Instance(){
        return new SearchAndSort();
    }

    class SearchAndSort {

        // binary search for a student ArrayList
        public int binarySearch(ArrayList<Student> arr, String attribute, int len, Student target){
            int low = 0;
            int high = len;
    
            int index = -1;
    
            while(low <= high){
                
                int mid = (low + high) / 2 ;
                // System.out.println("Mid: "+mid);
                if(compareByAttribute(arr.get(mid), target, attribute) < 0){
                    low = mid + 1;
                }
                else if(compareByAttribute(arr.get(mid), target, attribute) > 0){
                    high = mid - 1;
                }
                else{
                    index = mid;
                    break;
                }
            }
    
            return index;
        }

        // ArrayList size check
        public void quickSort(ArrayList<Student> arr, String attribute) {
            if (arr == null || arr.size() <= 1) {
                return;
            }
            quickSort(arr, 0, arr.size() - 1, attribute);
        }
    
        // recursively find partition for subArrayLists
        private void quickSort(ArrayList<Student> arr, int low, int high, String attribute) {
            if (low < high) {
                int pivot = partition(arr, low, high, attribute);
    
                quickSort(arr, low, pivot - 1, attribute);
                quickSort(arr, pivot + 1, high, attribute);
            }
        }
    
        // find partitioning element for an array
        private int partition(ArrayList<Student> arr, int low, int high, String attribute) {
            Student pivot = arr.get(low);
            int start = low;
            int end = high;
    
            while(start < end){
                while(compareByAttribute(arr.get(start), pivot, attribute) <= 0 && start < high){
                    start++;
                }
    
                while(compareByAttribute(arr.get(end), pivot, attribute) > 0 && end > low){
                    end--;
                }
    
                if(start < end){
                    swap(arr, start, end);
                }
            }
    
            if(end <= start){
                swap(arr, low, end);
            }
            
            return end;
        }
    
        // swap 2 elements for a student type ArrayList
        private void swap(ArrayList<Student> arr, int i, int j) {
            Student temp = arr.get(i);
            arr.set(i, arr.get(j));
            arr.set(j, temp);
        }
        
        // method that returns values by comparing certain attributes between student objects
        private int compareByAttribute(Student a, Student b, String attribute){
            if("name".equals(attribute)){
                return a.name.toLowerCase().compareTo(b.name.toLowerCase());
            }
            else if("marks".equals(attribute)){
                return Integer.compare(a.marks, b.marks);
            }
            else{
                return Integer.compare(a.prn, b.prn);
            }
        }
    }
}