package Java.Theory.Theory10;

class X{
    void m1(){
        System.out.println("LMAO");
    }
}

public class MainClass implements Interface2, Interface{
    
    // @Override
    protected void finalize() throws Throwable{
        // perform custom garbage collection here
        super.finalize();   
    }

    @Override
    public void m1(){
        
    }

    @Override
    public void m2() {
        
    }

    X x = new X() {
        @Override
        void m1(){
            System.out.println("LOL");
        }
    };

    public static void main(String[] args) {
        // anonymous inner class that inherits class X and overrides method m1
        // this is local inner class
        X x = new X();
        X y = new X();

        System.out.println(x.hashCode());
        System.out.println(y.hashCode());
        // x.m1();
        // MainClass poo = new MainClass();
        // System.out.println("new object");
        // if this is done in a method then it is called method local inner class
    }
}