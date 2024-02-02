package Java.Labs.Lab2;

public class Car {
    
    Car(){
        new Engine();
    }
    
    public class Engine {

        Engine(){
            new Piston().run();
        }

        public class Piston{
            public void run(){
                System.out.println("LOL");
            }
        }
        
    }
}
