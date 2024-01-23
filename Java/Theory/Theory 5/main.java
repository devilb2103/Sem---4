public class main{
    public static void main(String[] args){
        data obj = new data();

        // print address of object and the object's "this" reference
        System.out.println(obj.hashCode());
        obj.printSelf();

        // change instance variable using "this"
        obj.setX(100);
    }
}