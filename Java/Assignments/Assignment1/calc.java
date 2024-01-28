package Java.Assignments.Assignment1;

import java.lang.Math;

public class calc {
    input dat = new input();

    public double addition(){
        System.out.println("Performing addition");
        double num1 = dat.doubleInput();
        double num2 = dat.doubleInput();

        return num1 + num2;
    }

    public double subtratcion(){
        System.out.println("Performing subtraction");
        double num1 = dat.doubleInput();
        double num2 = dat.doubleInput();

        return num1 - num2;
    }

    public double multiplication(){
        System.out.println("Performing multiplication");
        double num1 = dat.doubleInput();
        double num2 = dat.doubleInput();

        return num1 * num2;
    }

    public double division(){
        System.out.println("Performing division");
        double num1 = dat.doubleInput();
        double num2 = dat.doubleInput();

        return num1 / num2;
    }

    public double squareRoot(){
        System.out.println("Performing root");
        double num1 = dat.doubleInput();

        return Math.sqrt(num1);
    }

    public double power(){
        System.out.println("Performing power");
        double num1 = dat.doubleInput();
        double num2 = dat.doubleInput();

        return Math.pow(num1, num2);
    }

    public double mean(){
        System.out.println("Performing mean (type end to stop taking inputs)");
        double sum = 0;
        int count = 0;
        while (true) {
            String num = dat.strInput();
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

    public double variance(){
        System.out.println("Performing variance");
        System.out.println("Enter number of inputs");
        int len = Integer.valueOf(dat.strInput());

        double[] nums = new double[len];

        // store inputs
        for (int i = 0; i < len; i++) {
            nums[i] = dat.doubleInput();
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
