package Java.Assignments.Assignment1;

import java.lang.Math;

public class calc {
    // input class
    InputClass input = new InputClass();
    

    // Perform addition calculation
    public double addition(){
        System.out.println("Performing addition");
        double num1 = input.doubleInput();
        double num2 = input.doubleInput();

        return num1 + num2;
    }

    // Perform subtraction calculation
    public double subtraction(){
        System.out.println("Performing subtraction");
        System.out.println("Enter Number 1: ");
        double num1 = input.doubleInput();
        System.out.println("Enter Number 2: ");
        double num2 = input.doubleInput();

        return num1 - num2;
    }

    // perform multiplication calculation
    public double multiplication(){
        System.out.println("Performing multiplication");
        System.out.println("Enter Number 1: ");
        double num1 = input.doubleInput();
        System.out.println("Enter Number 2: ");
        double num2 = input.doubleInput();

        return num1 * num2;
    }

    // perform division calculation
    public double division(){
        System.out.println("Performing division");
        System.out.println("Enter Number 1: ");
        double num1 = input.doubleInput();
        System.out.println("Enter Number 2: ");
        double num2 = input.doubleInput();

        return num1 / num2;
    }

    // perform squareRoot calculation
    public double squareRoot(){
        System.out.println("Performing root");
        System.out.println("Enter Number 1: ");
        double num1 = input.doubleInput();

        return Math.sqrt(num1);
    }

    // perform power calculation
    public double power(){
        System.out.println("Performing power");
        System.out.println("Enter Number 1: ");
        double num1 = input.doubleInput();
        System.out.println("Enter Number 2: ");
        double num2 = input.doubleInput();

        return Math.pow(num1, num2);
    }

    // perform mean calculation
    public double mean(){
        System.out.println("Performing mean (type end to stop taking inputs)");
        double sum = 0;
        int count = 0;
        while (true) {
            System.out.println(String.format("Enter Number %d: ", count + 1));
            String num = input.strInput();
            // System.out.println(num == "end");
            if(num.equals("end")){
                break;
            }
            else{
                sum += Integer.valueOf(num);
                count += 1;
            }
        }

        return sum/count;
    }

    // perform variance calculation
    public double variance(){
        System.out.println("Performing variance");
        System.out.println("Enter number of inputs");
        int len = Integer.valueOf(input.strInput());

        double[] nums = new double[len];

        // store inputs
        for (int i = 0; i < len; i++) {
            System.out.println(String.format("Enter Number %d: ", i + 1));
            nums[i] = input.doubleInput();
        }

        // compute mean
        double mean = 0;
        for (double d : nums) {
            mean += d;
        }
        mean /= len;

        // calculate variance numerator
        int var_numerator = 0;
        for (double d : nums) {
            var_numerator += Math.pow(d - mean, 2);
        }

        // return variance
        return var_numerator / (len - 1);
    }
}
