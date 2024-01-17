package Java.Assignments.Assignment1;
import java.util.Scanner;

public class demo {
    
    public static void main(String[] args){
        
        input dat = new input();
        calc calc = new calc();
        double res;
        while (true) {
            System.out.println(
                """
                    1) Add
                    2) Sub
                    3) Multi
                    4) Div
                    5) Sqrt
                    6) Pow
                    7) Mean
                    8) Var
                    9) Exit
                    
                """);
            
            System.out.println("Your option: ");
            Scanner sc = new Scanner(System.in);
                int option = sc.nextInt();
            switch (option) {
                case 1:
res = calc.addition();
                case 2:
                res =           calc.subtratcion();
                case 3:
                res = calc.multiplication();
                
                case 4:
                res = calc.division();
                
                case 5:
                res = calc.squareRoot();
                
                case 6:
                res = calc.power();
                
                case 7:
                res = calc.mean();
                
                case 8:
                res = calc.variance();
                case 9:
                return;
                default:
                    System.out.println("Invalid Option");;
            }
            System.out.println();
        }
    }
}
