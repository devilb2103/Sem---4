package Java.Assignments.Assignment3;

import java.util.InputMismatchException;
import java.util.Scanner;

public class InputClass {

    // static scanner instance for entire program
    private static Scanner sc = new Scanner(System.in);

    public void showSC_Hash(){
        System.out.println(sc.hashCode());
    }

    public void disposeScanner(){
        sc.close();
    }
    
    public int intInput(){
        int num = Integer.MIN_VALUE;
        
        try {
            num = sc.nextInt();
            return num;

        } catch (InputMismatchException e) {
            System.out.println("Invalid Integer input");
            sc.nextLine();
            
            return intInput();
        }
    }
    
    public double doubleInput(){
        double num = sc.nextDouble();
        return num;
    }

    public String strInput(){
        String str = sc.next();
        return str;
    }
}
