package Java.Theory.Theory11;

public class Postgres implements MyDatabase{
    @Override
    public void connectToDatabase() {
        System.out.println("Connecting to Postgres database");
    }
}
