package Java.Assignments.Assignment1;

import java.util.Scanner;;

public class input {
    
    public double numInput(){
        Scanner sc = new Scanner(System.in);
        double num = sc.nextDouble();

        sc.close();

        return num;
    }

    public String strInput(){
        Scanner sc = new Scanner(System.in);
        String str = sc.next();

        sc.close();

        return str;
    }
}
