package Java.Assignments.Assignment1;

public class MainClass {
    
    public static void main(String[] args){
        
        InputClass input = new InputClass();
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
            int option = input.intInput();
            switch (option) {
                case 1:
                    System.out.println(String.format("Addition Result: ", calc.addition()));
                    continue;
                case 2:
                    System.out.println(String.format("Subtraction Result: ", calc.subtraction()));
                    continue;
                case 3:
                    System.out.println(String.format("Multiplication Result: ", calc.multiplication()));
                    continue;
                case 4:
                    System.out.println(String.format("Division Result: ", calc.division()));
                    continue;
                case 5:
                    System.out.println(String.format("SquareRoot Result: ", calc.squareRoot()));
                    continue;
                case 6:
                    System.out.println(String.format("Power Result: ", calc.power()));
                    continue;
                case 7:
                    System.out.println(String.format("Mean Result: ", calc.mean()));
                    continue;
                case 8:
                    System.out.println(String.format("Variance Result: ", calc.variance()));
                    continue;
                case 9:
                    // dispose scanner class instance in InputClass object
                    input.disposeScanner();
                    return;

                default:
                    System.out.println("Invalid Option");;
            }

            // System.out.println(res);
        }
    }
}
