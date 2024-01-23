public class data{

    int x;

    public void printSelf(){
        System.out.println(this.hashCode());
    }

    void setX(int x){
        // x = x;
        // dont do above
        // it sets value to local variable and not instance variable;
        // aka instance variable hiding
        this.x = x;
        System.out.println(this.x);
    }
}

