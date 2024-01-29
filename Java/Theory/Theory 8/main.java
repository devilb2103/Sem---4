/**
 * main
 */
public class main {

    public static void main(String[] args) {
        int i = 10; // stored in stack
        Integer j = 10; // stored in heap, this is called boxing

        // one is for call by value
        // other is call by reference

        int k = j; // this is called unboxing

        System.out.println(j);
        
        boolean a = true;
        Boolean x = true;

    }
}

// types of classes
// 1) member inner class
class outer{
    class inner{
        // inner is a member inner class
    }
}

// 2)