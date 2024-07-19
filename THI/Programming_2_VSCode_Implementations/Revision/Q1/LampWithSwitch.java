package THI.Programming_2_VSCode_Implementations.Revision.Q1;

public class LampWithSwitch extends Device implements Switch, Lamp {
    private Double status;
    private int color;

    LampWithSwitch(String name) {
        super(name);
    }
    
    public int getColor(){
        return this.color;
    }

    public double[] getValues(){
        return new double[] {this.status, this.color};
    }

    @Override
    public void setStatus(double value) {
        this.status = value;
    }

    @Override
    public void setColor(int value) {
        this.color = value;
    }

    @Override
    public double getStatus() {
        return this.status;
    }

}
