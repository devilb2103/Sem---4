package THI.Programming_2_VSCode_Implementations.Revision.Q1;

public class SmartRubberDuck extends LampWithSwitch implements TemperatureSensor {
    public double temperature;

    SmartRubberDuck(String name){
        super(name);
    }

    @Override
    public double getTemperature() {
        return this.temperature;
    }

    @Override
    public double[] getValues() {
        return new double[] {this.getStatus(), this.getColor(), this.temperature};
    }
}
