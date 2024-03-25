package Java.Theory.Theory12.SameClassConstructor;

public class Data {

    int x, y, z;

    Data(){
        System.out.println("0 param constructor");
    }

    Data(int p1){
        this.x = p1;
        System.out.println("1 param constructor");
    }

    Data(int p1, int p2){
        this(p1);
        this.y = p2;
        System.out.println("2 param constructor");
    }

    Data(int p1, int p2, int p3){
        this(p1, p2);
        this.y = p3;
        System.out.println("3 param constructor");
    }
}
