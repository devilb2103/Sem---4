package THI.Programming_2_VSCode_Implementations.Revision.Q2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Color {
    private int rot;
    private int gruen;
    private int blau;
    public Color(int rot, int gruen, int blau) {
    this.rot = rot;
    this.gruen = gruen;
    this.blau = blau;
    }
    public int getRot() { return rot; }
    public int getBlau() { return blau; }
    public int getGruen() { return gruen; }
    // public int hashCode() { /*…*/ }
    // public boolean equals(Object o) { /*…*/ }

    public static Map<Color, String> verarbeiten(String[] tokens) {
        Map<Color, String> map = new HashMap<Color,String>();

        for(int i = 0; i < tokens.length; i+=4){
            map.put(new Color(Integer.parseInt(tokens[i+1]),
            Integer.parseInt(tokens[i+2]),
            Integer.parseInt(tokens[i+3])), tokens[i]);
        }

        return map;
    }

    public static void ausgeben(Map<Color, String> map){
        Set<Color> keys = map.keySet();
        List<Color> colorList = new ArrayList<Color>(keys);
        colorList.sort(new ColorComparator());

        for(Color i: colorList){
            System.out.println(String.format("%d,%d,%d = %s", i.getRot(), i.getGruen(), i.getBlau(), map.get(i)));
        }
    }

    public static void main(String[] args) {
        String[] tokens = new String[] {
        "rot", "255", "0", "0",
        "weiss", "255", "255", "255",
        "türkis", "0", "255", "255",
        "orange", "255", "165", "0",
        "grau", "128", "128", "128" };
        Map<Color, String> map = verarbeiten(tokens);
        ausgeben(map);
    }
   }