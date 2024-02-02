package Java.Theory.Theory4;

public class data2 {

    int x = 10;

    static {
        System.out.println("static block");
    }

    {
        System.out.println("Instance block, " + Integer.toString(x));
        x = 100;
    }

    data2(){
        System.out.println("constructor, " + x);
    }
}