package Java.Theory.Theory11;

public class MySql implements MyDatabase {
    @Override
    public void connectToDatabase() {
        System.out.println("Connecting to MySql database");
    }
}
