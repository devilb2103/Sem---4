package Java.Assignments.Assignment1;

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
        int num = sc.nextInt();
        return num;
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
