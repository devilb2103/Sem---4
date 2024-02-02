public class MainClass {
    
    public static void main(String[] args){
        
        InputClass input = new InputClass();
        Calc calc = new Calc();
        
        // Assignment Part 1
        boolean exitPart1 = false;
        while (!exitPart1) {
            System.out.println(
                """
                    Calculate Factorial using:-

                    1) Use Command Line Args
                    2) Use Scanner
                    3) Use BufferedReader
                    4) Use DataInputStream
                    5) Use Console
                    6) Go to part 2 of Assignment
                    
                """);
            
            System.out.println("Your option: ");
            int option = input.intInput();
            switch (option) {
                case 1:
                    // factorial using Command Line Args
                    input.factorialCommandLineArgs(args);
                    break;

                case 2:
                    // factorial using Scanner
                    input.factorialScanner();
                    break;
                
                case 3:
                    // factorial using Buffered Reader
                    input.factorialBufferedReader();
                    break;
                
                case 4:
                    // factorial using Data input stream
                    // input.factorialDataInputStream();
                    System.out.println("Code is in file but method is deprecated which causes compilation errors");
                    break;
                
                case 5:
                    // factorial using factorial console
                    input.factorialConsole();
                    break;
                
                case 6:
                    // exit part 1 and move to part 2
                    exitPart1 = true;
                    break;
                
                default:
                    System.out.println("Invalid Option");
                    break;
            }

        }

        // Assignment Part 2
        boolean exitPart2 = false;
        while (!exitPart2) {
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
                    // perform Addition
                    System.out.println(String.format("Addition Result: %.2f", calc.addition()));
                    break;
                    
                case 2:
                    // perform Subtraction
                    System.out.println(String.format("Subtraction Result: %.2f", calc.subtraction()));
                    break;
                    
                case 3:
                    // perform Multiplication
                    System.out.println(String.format("Multiplication Result: %.2f", calc.multiplication()));
                    break;

                case 4:
                    // perform Division
                    System.out.println(String.format("Division Result: %.2f", calc.division()));
                    break;

                case 5:
                    // perform SquareRoot
                    System.out.println(String.format("SquareRoot Result: %.2f", calc.squareRoot()));
                    break;

                case 6:
                    // perform Power
                    System.out.println(String.format("Power Result: %.2f", calc.power()));
                    break;

                case 7:
                    // perform Mean
                    System.out.println(String.format("Mean Result: %.2f", calc.mean()));
                    break;
                    
                case 8:
                    // perform Variance
                    System.out.println(String.format("Variance Result: %.2f", calc.variance()));
                    break;

                case 9:
                    // dispose scanner class instance in InputClass object before exitting program
                    input.disposeScanner();
                    exitPart2 = true;
                    break;

                default:
                    System.out.println("Invalid Option");
                    break;
            }

        }
    }
}
