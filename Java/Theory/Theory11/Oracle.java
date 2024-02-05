package Java.Theory.Theory11;

public class Oracle implements MyDatabase {
    @Override
    public void connectToDatabase() {
        System.out.println("Connecting to Oracle database");
    }
}
