import org.opencv.core.RotatedRect;

public class GoalTarget
{
    private RotatedRect leftTape;
    private RotatedRect rightTape;
    
    public GoalTarget(RotatedRect leftTape, RotatedRect rightTape)
    {
        this.leftTape = leftTape;
        this.rightTape = rightTape;
    }

    public double centerX()
    {
        return (leftTape.center.x + rightTape.center.x) / 2;
    }

    public double centerY()
    {
        return (leftTape.center.y + rightTape.center.y) / 2;
    }

    public double targetWidth()
    {
        return rightTape.center.x - leftTape.center.x;
    }

    
}

