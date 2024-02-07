package Java.Assignments.Assignment6;

public class MainClass {
    public static void main(String[] args) {
        // Fixed f_stack
        System.out.println("Fixed Stack Output ------------------------------");
        FixedStack f_stack = new FixedStack(12);
        f_stack.push(0);
        f_stack.push(2);
        f_stack.push(4);
        f_stack.push(6);
        f_stack.push(8);
        f_stack.display();
        f_stack.pop();
        f_stack.push(10);
        f_stack.pop();
        f_stack.pop();
        f_stack.pop();
        f_stack.display();
        f_stack.pop();
        f_stack.pop();
        f_stack.display();
        try {
            f_stack.pop();
        } catch (RuntimeException e) {
            System.out.println(e.getMessage());
        }

        System.out.println("\n");

        // Dynamic f_stack
        System.out.println("Dynamic Stack Output ------------------------------");
        DynamicStack d_stack = new DynamicStack();
        d_stack.push(0);
        d_stack.push(2);
        d_stack.push(4);
        d_stack.push(6);
        d_stack.push(8);
        d_stack.display();
        d_stack.pop();
        d_stack.push(10);
        d_stack.pop();
        d_stack.pop();
        d_stack.pop();
        d_stack.display();
        d_stack.pop();
        d_stack.pop();
        d_stack.display();
        try {
            d_stack.pop();
        } catch (RuntimeException e) {
            System.out.println(e.getMessage());
        }
    }
}
