package Java.Theory.Theory11;

public class MainClass {
    public static void main(String[] args) {
        MyDatabase db;

        // upcasting
        
        db = new Oracle();
        db.connectToDatabase();

        db = new MySql();
        db.connectToDatabase();

        db = new Postgres();
        db.connectToDatabase();

        /*
         * This is called dynamic memory dispatch
         */
    }

}
