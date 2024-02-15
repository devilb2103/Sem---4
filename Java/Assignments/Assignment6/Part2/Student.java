package Java.Assignments.Assignment6.Part2;

public class Student 
{
	String name;
	int roll_no;
	double mark1, mark2;
	
	public Student(String name, int roll_no, double mark1, double mark2) {
		super();
		this.name = name;
		this.roll_no = roll_no;
		this.mark1 = mark1;
		this.mark2 = mark2;
	}

	@Override
	public String toString() {
		return "Student [name=" + name + ", roll_no=" + roll_no + ", mark1=" + mark1 + ", mark2=" + mark2 + "]";
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getRoll_no() {
		return roll_no;
	}

	public void setRoll_no(int roll_no) {
		this.roll_no = roll_no;
	}

	public double getMark1() {
		return mark1;
	}

	public void setMark1(double mark1) {
		this.mark1 = mark1;
	}

	public double getMark2() {
		return mark2;
	}

	public void setMark2(double mark2) {
		this.mark2 = mark2;
	}
}
