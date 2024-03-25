package Java.Assignments.Assignment6.Part2;

public class Result extends Student implements Exam
{
	double percentage;
	
	public Result(String name, int roll_no, double mark1, double mark2) {
		super(name, roll_no, mark1, mark2);
	}

	@Override
	public void Percent_cal() {
		this.percentage = (mark1 + mark2) /2;	
	}
	
	public String display()
	{
		Percent_cal();
		return String.format("""
            Roll No: %d
            Name: %s
            Percentage: %.2f%%
        """, roll_no, name, percentage).toString();
	}
}
