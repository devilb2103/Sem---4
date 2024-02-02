import java.io.BufferedReader;
import java.io.Console;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;

public class InputClass {

    // static scanner instance for entire program
    private static Scanner sc = new Scanner(System.in);
    private static Calc calc = new Calc();

    // prints address of static Scanner instance
    public void showSC_Hash(){
        System.out.println(sc.hashCode());
    }

    // disposes static scanner instance
    public void disposeScanner(){
        sc.close();
    }
    
    // integer input using scanner class
    public int intInput(){
        int num = sc.nextInt();
        return num;
    }
    
    // double input using scanner class
    public double doubleInput(){
        double num = sc.nextDouble();
        return num;
    }

    // string input using scanner class
    public String strInput(){
        String str = sc.next();
        return str;
    }

    // factorial using Command Line Args
    public void factorialCommandLineArgs(String[] args){
        try {
            if(args.length > 0 && Integer.parseInt(args[0]) >= 0){
                int factorial_num = Integer.parseInt(args[0]);
                if(factorial_num == Integer.MIN_VALUE) System.out.println("Invalid Command Line input");
                else System.out.println(String.format("Factorial of %d is: %d", factorial_num, calc.factorial(factorial_num)));
            }
            else{
                System.out.println("Invalid Command Line input");
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid Command Line input");
        }
    }

    // factorial using Scanner
    public void factorialScanner(){
        System.out.println("Enter Number 1: ");
        int factorial_num = intInput();
        System.out.println(String.format("Factorial of %d is: %d", factorial_num, calc.factorial(factorial_num)));
    }

    // factorial using Buffered Reader
    public void factorialBufferedReader(){
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            System.out.println("Enter Number 1: ");
            int factorial_num = Integer.parseInt(reader.readLine());
            System.out.println(String.format("Factorial of %d is: %d", factorial_num, calc.factorial(factorial_num)));
        } catch (IOException | NumberFormatException e) {
            System.out.println("Invalid input");
        }
    }

    // factorial using Data input stream () ====== DEPRECATED
    // public void factorialDataInputStream(){
    //     DataInputStream dis = new DataInputStream(System.in);
    //     try {
    //         System.out.println("Enter Number 1: ");
    //         int factorial_num = Integer.parseInt(dis.readLine());
    //         System.out.println(String.format("Factorial of %d is: %d", factorial_num, calc.factorial(factorial_num)));
    //     } catch (IOException | NumberFormatException e) {
    //         System.out.println("Invalid input");
    //     }
    // }

    // factorial using factorial console
    public void factorialConsole(){
        Console console = System.console();
        try {
            System.out.println("Enter Number 1: ");
            int factorial_num = Integer.parseInt(console.readLine());
            System.out.println(String.format("Factorial of %d is: %d", factorial_num, calc.factorial(factorial_num)));
        } catch (NumberFormatException e) {
            System.out.println("Invalid input");
        }
    }
}
