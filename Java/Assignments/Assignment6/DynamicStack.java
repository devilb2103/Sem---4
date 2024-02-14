package Java.Assignments.Assignment6;

import java.util.ArrayList;

public class DynamicStack implements StackInterface{

    private ArrayList<Integer> arr = new ArrayList<Integer>();

    @Override
    public boolean isOverflow() {
        return false;
    }

    @Override
    public boolean isUnderflow() {
        if(!(this.arr.size() > 0)){
            return true;
        }
        return false;
    }

    @Override
    public void pop() {
        if(this.arr.size() > 0){
            this.arr.remove(this.arr.size() - 1);
        }
        else{
            throw new RuntimeException("Stack Underflow");
        }
    }

    @Override
    public void push(int value) {
        this.arr.add(value);
    }

    @Override
    public void display() {
        for (int i = 0; i < this.arr.size(); i++) {
            System.out.println(String.format("Element %d: %d", i, this.arr.get(i)));
        }
        System.out.println("\n");
    }

    
}
