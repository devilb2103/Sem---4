package Java.Assignments.Assignment6;

public class FixedStack implements StackInterface{

    private int sizeState = 0;
    // sizeState = 0 means normal stack state
    // sizeState = -1 means stack underflow
    // sizeState = 1 means stack overflow
    private int insertPos = 0;
    private int[] arr;

    FixedStack(int size){
        if(size < 1){
            throw new IllegalArgumentException("Fixed Stack must have size greater than 0");
        }

        // initialize array and values
        this.arr = new int[size];
        for (int i = 0; i < this.arr.length; i++) {
            this.arr[i] = Integer.MIN_VALUE;
        }
    }

    @Override
    public boolean isUnderflow(){
        if(sizeState == -1) return true;
        return false;
    }

    @Override
    public boolean isOverflow(){
        if(sizeState == 1) return true;
        return false;
    }

    @Override
    public void pop() {
        if(sizeState > -1){
            this.arr[insertPos - 1] = Integer.MIN_VALUE;
            this.insertPos--;

            if(this.insertPos == 0){
                this.sizeState = -1;
            }
        }
        else{
            throw new RuntimeException("Stack Underflow");
        }
    }

    @Override
    public void push(int value) {
        if(sizeState < 1){
            this.arr[insertPos] = value;
            this.insertPos++;

            if(this.insertPos >= this.arr.length){
                this.sizeState = 1;
            }
        }
        else{
            throw new RuntimeException("Stack Overflow");
        }
    }

    @Override
    public void display() {
        for (int index = 0; index < this.arr.length; index++) {
            if(this.arr[index] != Integer.MIN_VALUE)
            System.out.println(String.format("Element %d: %d", index, this.arr[index]));
        }
        System.out.println("\n");
    }
    
}
