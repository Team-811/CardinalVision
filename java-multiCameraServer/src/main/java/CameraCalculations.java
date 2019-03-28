

public class CameraCalculations
{
  private static final double fovHorizontal = 61;
  private static final double pixelsHorizontal = 320;
  private static final double pixelsVertical = 240;
  private static final double centerPixelHorizontal = pixelsHorizontal/2;
  private static final double centerPixelVertical = pixelsVertical/2;
  private static final double distanceBetweenTargets = 1.1;

  public static double getFocalLength()
  {
      return pixelsHorizontal / (2 * Math.tan(fovHorizontal/2));
  }

  public static double getHorizontalDegreesToPixels(double targetXpixels)
  {
      return -1 * Math.atan((targetXpixels - centerPixelHorizontal) / getFocalLength());
  }

  public static double getVerticalDegreesToPixels(double targetYpixels)
  {
      return Math.atan((targetYpixels - centerPixelVertical) / getFocalLength());
  }

  public static double getMetersPerPixel(double targetWidthPixels)
  {
      return distanceBetweenTargets / targetWidthPixels;
  }

  public static double getXOffset(double targetWidthPixels, double cenX)
  {
      return getMetersPerPixel(targetWidthPixels) * (centerPixelHorizontal - cenX);
  }

  public static double getDistance(double targetWidthPixels, double cenX)
  {
      return getXOffset(targetWidthPixels, cenX) / Math.tan(getHorizontalDegreesToPixels(cenX));
  } 


}