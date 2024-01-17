package Java.Assignments.Assignment1;

import java.lang.Math;

public class calc {
    input dat = new input();

    public double addition(){
        double num1 = dat.numInput();
        double num2 = dat.numInput();

        return num1 + num2;
    }

    public double subtratcion(){
        double num1 = dat.numInput();
        double num2 = dat.numInput();

        return ;
    }

    public double multiplication(){
        double num1 = dat.numInput();
        double num2 = dat.numInput();

        return num1 * num2;
    }

    public double division(){
        double num1 = dat.numInput();
        double num2 = dat.numInput();

        return num1 / num2;
    }

    public double squareRoot(){
        double num1 = dat.numInput();

        return Math.sqrt(num1);
    }

    public double power(){
        double num1 = dat.numInput();
        double num2 = dat.numInput();

        return Math.pow(num1, num2);
    }

    public double mean(){
        double sum = 0;
        int count = 0;
        while (true) {
            String num = dat.strInput();
            if(num == "end"){
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
        System.out.println("Enter number of inputs.");
        int len = Integer.valueOf(dat.strInput());

        double[] nums = new double[len];

        // store inputs
        for (int i = 0; i < len; i++) {
            nums[i] = dat.numInput();
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
