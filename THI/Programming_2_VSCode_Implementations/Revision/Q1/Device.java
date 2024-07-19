package THI.Programming_2_VSCode_Implementations.Revision.Q1;


public abstract class Device {
    private String name;
    
    public Device(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
    
    public abstract double[] getValues();
}
