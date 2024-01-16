public class student{
    int prn;
    int age;
    String name;
    static String collegeName;

    void student(){
        System.out.println("your mom");
    }

    void collectInfo(int _prn, int _age, String _name) {
        prn = _prn;
        age = _age;
        name = _name;
    }

    void displayInfo() {
        System.out.println("PRN: ".concat(Integer.toString(prn)));
        System.out.println("Age: ".concat(Integer.toString(age)));
        System.out.println("Name: ".concat(name));
    }
}