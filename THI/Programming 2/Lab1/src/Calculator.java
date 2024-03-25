
public class Calculator {
	public int add(int x, int y) {
		return x + y;
	}
	
	public int sub(int x, int y) {
		return x - y;
	}
	
	public static void main(String[] args) {
		int a = 4, b = 3;
		Calculator calc = new Calculator();
		System.out.printf("a + b = %d\n", calc.add(a, b));
		System.out.printf("a - b = %d", calc.sub(a, b));
	}
}