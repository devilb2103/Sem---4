package THI.Prog2Examples;

import java.util.HashSet;

class Lottery {
    public static void main(String[] args) {
        HashSet<Integer> drawing = new HashSet<Integer>();
        // 6 (out of 49) numbers are required
        while (drawing.size() < 6) {
            // only new numbers will be added!
            drawing.add((int) (Math.random() * 49) + 1);
        }
        // Output all 6 numbers
        System.out.println("Lotto numbers: " + drawing);
    }
}