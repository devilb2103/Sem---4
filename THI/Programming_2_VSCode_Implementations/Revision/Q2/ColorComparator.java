package THI.Programming_2_VSCode_Implementations.Revision.Q2;

import java.util.Comparator;

public class ColorComparator implements Comparator<Color>{
    @Override
    public int compare(Color o1, Color o2){
        if(o1 == o2 || o1 == null && o2 == null){ return 0; }
        if(o1 != null && o2 == null){ return 1; }
        if(o1 == null && o2 != null){ return -1; }
        if(o1.getRot() != o2.getRot()) { return o1.getRot() - o2.getRot(); }
        if(o1.getBlau() != o2.getBlau()) { return o1.getBlau() - o2.getBlau(); }
        return o1.getGruen() - o2.getGruen();
    }
}
