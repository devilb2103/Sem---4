package Java.Labs.Lab2;

public abstract class KadhaiPaneer implements MyKitchen {
    void a(){

    }

    // abstract void b();

    public void toCook(){
        System.out.println("Cooking " + this.getClass().toString());
    }
}
