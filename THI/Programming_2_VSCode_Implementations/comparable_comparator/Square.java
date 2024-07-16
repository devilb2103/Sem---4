package THI.Programming_2_VSCode_Implementations.comparable_comparator;

import java.util.Collection;

public class Square implements Comparable {
    
    public int side;
    
    Square(int side){
        this.side = side;
    }
    
    @Override
    public int compareTo(Object o){
        if(this.side < ((Square) o).side){
            return -1;
        }
        else if (this.side > ((Square) o).side){
            return 1;
        }
        else{
            return 0;
        }
    }

    public static void main(String[] args){
        Square[] arr = new Square[] {
            new Square(5),
            new Square(4),
            new Square(3),
            new Square(2)
        };
        java.util.Arrays.sort(arr, new SideLengthComparator());
        System.out.println(arr);
        for(Square i: arr){
            System.out.println(i.side);
        }
    }
}
