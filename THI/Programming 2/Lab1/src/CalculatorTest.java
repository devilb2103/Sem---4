import static org.junit.Assert.*;

import org.junit.Test;

public class CalculatorTest {

	Calculator calc = new Calculator();

	@Test
	public void testAdd() {
		assertEquals("Add function does not work", 3, calc.add(1, 2));
	}

	@Test
	public void testSub() {
		assertEquals("Add function does not work", 1, calc.sub(3, 2));
	}

}