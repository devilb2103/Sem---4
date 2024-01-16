import java.lang.*;

public class main{
    
    public static void main(String[] args){
        student s1 = new student();
        s1.collectInfo(1, 1, "Dev");
        s1.displayInfo();

        student.collegeName = "lmao";
    }

}