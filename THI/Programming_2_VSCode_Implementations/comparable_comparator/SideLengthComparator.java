package THI.Programming_2_VSCode_Implementations.comparable_comparator;

import java.util.Comparator;

public class SideLengthComparator implements Comparator {
    @Override
    public int compare(Object o1, Object o2) {
        int length_1 = ((Square) o1).side;
        int length_2 = ((Square) o2).side;

        if(length_1 < length_2){
            return 1;
        }
        else if(length_1 > length_2){
            return -1;
        }
        else{
            return 0;
        }
    }
}