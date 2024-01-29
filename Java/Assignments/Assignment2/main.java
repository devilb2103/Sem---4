package Java.Assignments.Assignment2;

public class main {
    
    public static void main(String[] args) {
        InputClass input = new InputClass();

        System.out.println("Enter number of inputs: ");
        int numCount = input.intInput();

        arrayClass arr = new arrayClass(numCount);

        for (int i = 0; i < numCount; i++) {
            int num = input.intInput();
            arr.appendNums(num);
            // if even
            if(num%2 == 0){
                arr.appendEven(num);
            }
            // if odd
            else{
                arr.appendOdd(num);
            }
        }

        System.out.println(arr.getEven());
        System.out.println(arr.getOdd());

        int dist[] = arr.findSmallestDistance();
        System.out.println("Smallest dist is " + dist[0] + " at index " + dist[1]);
    }
}
