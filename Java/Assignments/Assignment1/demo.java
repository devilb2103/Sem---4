package Java.Assignments.Assignment1;

public class demo {
    
    public static void main(String[] args){
        
        InputClass dat = new InputClass();
        calc calc = new calc();
        
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
            int option = dat.intInput();
            switch (option) {
                case 1:
                    System.out.println(calc.addition());
                    continue;
                case 2:
                    System.out.println(calc.subtratcion());
                    continue;
                case 3:
                    System.out.println(calc.multiplication());
                    continue;
                case 4:
                    System.out.println(calc.division());
                    continue;
                case 5:
                    System.out.println(calc.squareRoot());
                    continue;
                case 6:
                    System.out.println(calc.power());
                    continue;
                case 7:
                    System.out.println(calc.mean());
                    continue;
                case 8:
                    System.out.println(calc.variance());
                    continue;
                case 9:
                    return;

                default:
                    System.out.println("Invalid Option");;
            }

            // System.out.println(res);
        }
    }
}
