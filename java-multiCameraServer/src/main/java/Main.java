
/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfKeyPoint;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

public final class Main {
  private static String configFile = "/boot/frc.json";

  @SuppressWarnings("MemberName")
  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();

  

  private Main() {
  }

  /**
   * Report parse error.
   */
  public static void parseError(String str) {
    System.err.println("config error in '" + configFile + "': " + str);
  }

  /**
   * Read single camera configuration.
   */
  public static boolean readCameraConfig(JsonObject config) {
    CameraConfig cam = new CameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement pathElement = config.get("path");
    if (pathElement == null) {
      parseError("camera '" + cam.name + "': could not read path");
      return false;
    }
    cam.path = pathElement.getAsString();

    // stream properties
    cam.streamConfig = config.get("stream");

    cam.config = config;

    cameraConfigs.add(cam);
    return true;
  }

  /**
   * Read configuration file.
   */
  @SuppressWarnings("PMD.CyclomaticComplexity")
  public static boolean readConfig() {
    // parse file
    JsonElement top;
    try {
      top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
    } catch (IOException ex) {
      System.err.println("could not open '" + configFile + "': " + ex);
      return false;
    }

    // top level must be an object
    if (!top.isJsonObject()) {
      parseError("must be JSON object");
      return false;
    }
    JsonObject obj = top.getAsJsonObject();

    // team number
    JsonElement teamElement = obj.get("team");
    if (teamElement == null) {
      parseError("could not read team number");
      return false;
    }
    team = teamElement.getAsInt();

    // ntmode (optional)
    if (obj.has("ntmode")) {
      String str = obj.get("ntmode").getAsString();
      if ("client".equalsIgnoreCase(str)) {
        server = false;
      } else if ("server".equalsIgnoreCase(str)) {
        server = true;
      } else {
        parseError("could not understand ntmode value '" + str + "'");
      }
    }

    // cameras
    JsonElement camerasElement = obj.get("cameras");
    if (camerasElement == null) {
      parseError("could not read cameras");
      return false;
    }
    JsonArray cameras = camerasElement.getAsJsonArray();
    for (JsonElement camera : cameras) {
      if (!readCameraConfig(camera.getAsJsonObject())) {
        return false;
      }
    }

    return true;
  }

  /**
   * Start running the camera.
   */
  public static VideoSource startCamera(CameraConfig config) {
    System.out.println("Starting camera '" + config.name + "' on " + config.path);
    CameraServer inst = CameraServer.getInstance();
    UsbCamera camera = new UsbCamera(config.name, config.path);
    MjpegServer server = inst.startAutomaticCapture(camera);

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

    if (config.streamConfig != null) {
      server.setConfigJson(gson.toJson(config.streamConfig));
    }

    return camera;
  }

  /**
   *  pipeline.
   */
  public static class CardinalPipeline implements VisionPipeline {

    //Outputs
    private static Mat blurOutput = new Mat();
    private static Mat hslThreshold0Output = new Mat();
    private static ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
    private static ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();
    private static Mat hslThreshold1Output = new Mat();
   // private static MatOfKeyPoint findBlobsOutput = new MatOfKeyPoint();
  
    static {
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
  
    /**
     * This is the primary method that runs the entire pipeline and updates the outputs.
     */
    @Override	public void process(Mat source0) {
      // Step Blur0:
      Mat blurInput = source0;
      BlurType blurType = BlurType.get("Box Blur");
      double blurRadius = 2.7451980282168957;
      blur(blurInput, blurType, blurRadius, blurOutput);
  
      // Step HSL_Threshold0:
      Mat hslThreshold0Input = blurOutput;
      double[] hslThreshold0Hue = {59.89208633093525, 121.63822525597269};
      double[] hslThreshold0Saturation = {91.72661870503596, 255.0};
      double[] hslThreshold0Luminance = {123.83093525179856, 239.76962457337885};
      hslThreshold(hslThreshold0Input, hslThreshold0Hue, hslThreshold0Saturation, hslThreshold0Luminance, hslThreshold0Output);
  
      // Step Find_Contours0:
      Mat findContoursInput = hslThreshold0Output;
      boolean findContoursExternalOnly = true; //Originally set as false
      findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);
  
      // Step Filter_Contours0:
      ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
      double filterContoursMinArea = 20.0;
      double filterContoursMinPerimeter = 0.0;
      double filterContoursMinWidth = 0.0;
      double filterContoursMaxWidth = 1000.0;
      double filterContoursMinHeight = 0.0;
      double filterContoursMaxHeight = 1000.0;
      double[] filterContoursSolidity = {60.251798561151084, 100};
      double filterContoursMaxVertices = 1000000.0;
      double filterContoursMinVertices = 0.0;
      double filterContoursMinRatio = 0.0;
      double filterContoursMaxRatio = 1000.0;
      filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);
  
      // Step HSL_Threshold1:
      /*
      Mat hslThreshold1Input = blurOutput;
      double[] hslThreshold1Hue = {0.0, 12.525414757898195};
      double[] hslThreshold1Saturation = {255.0, 255.0};
      double[] hslThreshold1Luminance = {78.29151932691136, 159.65404902264973};
      hslThreshold(hslThreshold1Input, hslThreshold1Hue, hslThreshold1Saturation, hslThreshold1Luminance, hslThreshold1Output);
  
      // Step Find_Blobs0:
      Mat findBlobsInput = hslThreshold1Output;
      double findBlobsMinArea = 60.0;
      double[] findBlobsCircularity = {0.6026365348399246, 1.0};
      boolean findBlobsDarkBlobs = false;
      findBlobs(findBlobsInput, findBlobsMinArea, findBlobsCircularity, findBlobsDarkBlobs, findBlobsOutput);
      */
    }
  
    /**
     * This method is a generated getter for the output of a Blur.
     * @return Mat output from Blur.
     */
    public Mat blurOutput() {
      return blurOutput;
    }
  
    /**
     * This method is a generated getter for the output of a HSL_Threshold.
     * @return Mat output from HSL_Threshold.
     */
    public Mat hslThreshold0Output() {
      return hslThreshold0Output;
    }
  
    /**
     * This method is a generated getter for the output of a Find_Contours.
     * @return ArrayList<MatOfPoint> output from Find_Contours.
     */
    public static ArrayList<MatOfPoint> findContoursOutput() {
      return findContoursOutput;
    }
  
    /**
     * This method is a generated getter for the output of a Filter_Contours.
     * @return ArrayList<MatOfPoint> output from Filter_Contours.
     */
    public ArrayList<MatOfPoint> filterContoursOutput() {
      return filterContoursOutput;
    }
  
    /**
     * This method is a generated getter for the output of a HSL_Threshold.
     * @return Mat output from HSL_Threshold.
     */
    public Mat hslThreshold1Output() {
      return hslThreshold1Output;
    }
  
    /**
     * This method is a generated getter for the output of a Find_Blobs.
     * @return MatOfKeyPoint output from Find_Blobs.
     */
    
     /*
    public MatOfKeyPoint findBlobsOutput() {
      return findBlobsOutput;
    }
    */
  
    /**
     * An indication of which type of filter to use for a blur.
     * Choices are BOX, GAUSSIAN, MEDIAN, and BILATERAL
     */
    enum BlurType{
      BOX("Box Blur"), GAUSSIAN("Gaussian Blur"), MEDIAN("Median Filter"),
        BILATERAL("Bilateral Filter");
  
      private final String label;
  
      BlurType(String label) {
        this.label = label;
      }
  
      public static BlurType get(String type) {
        if (BILATERAL.label.equals(type)) {
          return BILATERAL;
        }
        else if (GAUSSIAN.label.equals(type)) {
        return GAUSSIAN;
        }
        else if (MEDIAN.label.equals(type)) {
          return MEDIAN;
        }
        else {
          return BOX;
        }
      }
  
      @Override
      public String toString() {
        return this.label;
      }
    }
  
    /**
     * Softens an image using one of several filters.
     * @param input The image on which to perform the blur.
     * @param type The blurType to perform.
     * @param doubleRadius The radius for the blur.
     * @param output The image in which to store the output.
     */
    private void blur(Mat input, BlurType type, double doubleRadius,
      Mat output) {
      int radius = (int)(doubleRadius + 0.5);
      int kernelSize;
      switch(type){
        case BOX:
          kernelSize = 2 * radius + 1;
          Imgproc.blur(input, output, new Size(kernelSize, kernelSize));
          break;
        case GAUSSIAN:
          kernelSize = 6 * radius + 1;
          Imgproc.GaussianBlur(input,output, new Size(kernelSize, kernelSize), radius);
          break;
        case MEDIAN:
          kernelSize = 2 * radius + 1;
          Imgproc.medianBlur(input, output, kernelSize);
          break;
        case BILATERAL:
          Imgproc.bilateralFilter(input, output, -1, radius, radius);
          break;
      }
    }
  
    /**
     * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
     * @param input The image on which to perform the Distance Transform.
     * @param type The Transform.
     * @param maskSize the size of the mask.
     * @param output The image in which to store the output.
     */
    private void findContours(Mat input, boolean externalOnly,
      List<MatOfPoint> contours) {
      Mat hierarchy = new Mat();
      contours.clear();
      int mode;
      if (externalOnly) {
        mode = Imgproc.RETR_EXTERNAL;
      }
      else {
        mode = Imgproc.RETR_LIST;
      }
      int method = Imgproc.CHAIN_APPROX_SIMPLE;
      Imgproc.findContours(input, contours, hierarchy, mode, method);
    }
  
  
    /**
     * Filters out contours that do not meet certain criteria.
     * @param inputContours is the input list of contours
     * @param output is the the output list of contours
     * @param minArea is the minimum area of a contour that will be kept
     * @param minPerimeter is the minimum perimeter of a contour that will be kept
     * @param minWidth minimum width of a contour
     * @param maxWidth maximum width
     * @param minHeight minimum height
     * @param maxHeight maximimum height
     * @param Solidity the minimum and maximum solidity of a contour
     * @param minVertexCount minimum vertex Count of the contours
     * @param maxVertexCount maximum vertex Count
     * @param minRatio minimum ratio of width to height
     * @param maxRatio maximum ratio of width to height
     */
    private void filterContours(List<MatOfPoint> inputContours, double minArea,
      double minPerimeter, double minWidth, double maxWidth, double minHeight, double
      maxHeight, double[] solidity, double maxVertexCount, double minVertexCount, double
      minRatio, double maxRatio, List<MatOfPoint> output) {
      final MatOfInt hull = new MatOfInt();
      output.clear();
      //operation
      for (int i = 0; i < inputContours.size(); i++) {
        final MatOfPoint contour = inputContours.get(i);
        final Rect bb = Imgproc.boundingRect(contour);
        if (bb.width < minWidth || bb.width > maxWidth) continue;
        if (bb.height < minHeight || bb.height > maxHeight) continue;
        final double area = Imgproc.contourArea(contour);
        if (area < minArea) continue;
        if (Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true) < minPerimeter) continue;
        Imgproc.convexHull(contour, hull);
        MatOfPoint mopHull = new MatOfPoint();
        mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
        for (int j = 0; j < hull.size().height; j++) {
          int index = (int)hull.get(j, 0)[0];
          double[] point = new double[] { contour.get(index, 0)[0], contour.get(index, 0)[1]};
          mopHull.put(j, 0, point);
        }
        final double solid = 100 * area / Imgproc.contourArea(mopHull);
        if (solid < solidity[0] || solid > solidity[1]) continue;
        if (contour.rows() < minVertexCount || contour.rows() > maxVertexCount)	continue;
        final double ratio = bb.width / (double)bb.height;
        if (ratio < minRatio || ratio > maxRatio) continue;
        output.add(contour);
      }
    }
  
    /**
     * Segment an image based on hue, saturation, and luminance ranges.
     *
     * @param input The image on which to perform the HSL threshold.
     * @param hue The min and max hue
     * @param sat The min and max saturation
     * @param lum The min and max luminance
     * @param output The image in which to store the output.
     */
    private void hslThreshold(Mat input, double[] hue, double[] sat, double[] lum,
      Mat out) {
      Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HLS);
      Core.inRange(out, new Scalar(hue[0], lum[0], sat[0]),
        new Scalar(hue[1], lum[1], sat[1]), out);
    }
    
    /**
     * Detects groups of pixels in an image.
     * @param input The image on which to perform the find blobs.
     * @param minArea The minimum size of a blob that will be found
     * @param circularity The minimum and maximum circularity of blobs that will be found
     * @param darkBlobs The boolean that determines if light or dark blobs are found.
     * @param blobList The output where the MatOfKeyPoint is stored.
     */

     /*
    private void findBlobs(Mat input, double minArea, double[] circularity,
      Boolean darkBlobs, MatOfKeyPoint blobList) {
      FeatureDetector blobDet = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
      try {
        File tempFile = File.createTempFile("config", ".xml");
  
        StringBuilder config = new StringBuilder();
  
        config.append("<?xml version=\"1.0\"?>\n");
        config.append("<opencv_storage>\n");
        config.append("<thresholdStep>10.</thresholdStep>\n");
        config.append("<minThreshold>50.</minThreshold>\n");
        config.append("<maxThreshold>220.</maxThreshold>\n");
        config.append("<minRepeatability>2</minRepeatability>\n");
        config.append("<minDistBetweenBlobs>10.</minDistBetweenBlobs>\n");
        config.append("<filterByColor>1</filterByColor>\n");
        config.append("<blobColor>");
        config.append((darkBlobs ? 0 : 255));
        config.append("</blobColor>\n");
        config.append("<filterByArea>1</filterByArea>\n");
        config.append("<minArea>");
        config.append(minArea);
        config.append("</minArea>\n");
        config.append("<maxArea>");
        config.append(Integer.MAX_VALUE);
        config.append("</maxArea>\n");
        config.append("<filterByCircularity>1</filterByCircularity>\n");
        config.append("<minCircularity>");
        config.append(circularity[0]);
        config.append("</minCircularity>\n");
        config.append("<maxCircularity>");
        config.append(circularity[1]);
        config.append("</maxCircularity>\n");
        config.append("<filterByInertia>1</filterByInertia>\n");
        config.append("<minInertiaRatio>0.1</minInertiaRatio>\n");
        config.append("<maxInertiaRatio>" + Integer.MAX_VALUE + "</maxInertiaRatio>\n");
        config.append("<filterByConvexity>1</filterByConvexity>\n");
        config.append("<minConvexity>0.95</minConvexity>\n");
        config.append("<maxConvexity>" + Integer.MAX_VALUE + "</maxConvexity>\n");
        config.append("</opencv_storage>\n");
        FileWriter writer;
        writer = new FileWriter(tempFile, false);
        writer.write(config.toString());
        writer.close();
        blobDet.read(tempFile.getPath());
      } catch (IOException e) {
        e.printStackTrace();
      }
  
      blobDet.detect(input, blobList);
    }

    */
  
  }


  public static void sortCenX(ArrayList<RotatedRect> targets)
  {
    for(int i = 1; i < targets.size(); i++)
    {
        RotatedRect key = targets.get(i);
        int index = i - 1;
        while(key.center.x > targets.get(index).center.x && index >= 0)
        {
            targets.set(index + 1, targets.get(index));
            index--;
        }
        targets.set(index + 1, key);
    }

  }

  


  static int counter = 0; // this is just a temp, not to spam the console
  static boolean outputInfo = false;

  private static void WriteRoiToNetworkTable(NetworkTable table, double[] xOffset, double[] distance, double[]angle)
  {
    try {
      
      table.getEntry("xOffset").setDoubleArray(xOffset);
      table.getEntry("distance").setDoubleArray(distance);        
      table.getEntry("angle").setDoubleArray(angle);
    } catch (Exception e) {
      System.out.println("Exception writing NT");
    }
  }

  /**
   * Main.
   */

  
  public static void main(String... args) {
    if (args.length > 0) {
      configFile = args[0];
    }

    // read configuration
    if (!readConfig()) {
      return;
    }

    // start NetworkTables
    NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
    //set up the entries
    NetworkTable roiTable = ntinst.getTable("VisionTarget");

    if (server) {
      System.out.println("Setting up NetworkTables server");
      ntinst.startServer();
    } else {
      System.out.println("Setting up NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
    }
    //System.out.println("Not writing network tables");

    // start cameras
    List<VideoSource> cameras = new ArrayList<>();
    for (CameraConfig cameraConfig : cameraConfigs) {
      cameras.add(startCamera(cameraConfig));
    }

    // start image processing on camera 0 if present
    if (cameras.size() >= 1) {
      /*
      VisionThread visionThread = new VisionThread(cameras.get(0),
              new MyPipeline(), pipeline -> {
        // do something with pipeline results
      });
      */
      /* something like this for GRIP: */
      VisionThread visionThread = new VisionThread(cameras.get(0),
           new CardinalPipeline(), pipeline -> {


          // publish the result to the
          ArrayList<MatOfPoint> tapeContours = pipeline.findContoursOutput();
          //MatOfKeyPoint cargoTargets = pipeline.findBlobsOutput();
          ArrayList<RotatedRect> individualTapeTargets = new ArrayList<>();

          //adds targets to individual targets
          for (int index = 0; index < tapeContours.size(); index++)
          {
              MatOfPoint contour = tapeContours.get(index);
              individualTapeTargets.add(Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray())));
          }

          //sort
          //sortCenX(individualTapeTargets);
          
          for(int index = 0; index < individualTapeTargets.size(); index++)
          {
              if(counter == 20)
              {
                System.out.println(individualTapeTargets.get(index).center.x);
                System.out.println(individualTapeTargets.get(index).center.y);
                System.out.println(individualTapeTargets.get(index).angle);
              }
          }
          if(counter == 20)
            counter = 0;
          else
            counter ++;

          //Finds the grouping of vision targets
          // ArrayList<GoalTarget> fullTapeTargets = new ArrayList<>();

          // for (int index = 0; index < individualTapeTargets.size(); index+=2)
          // {
          //     if(individualTapeTargets.get(index).angle > 320 && individualTapeTargets.get(index + 1).angle < 40)
          //     {
          //         fullTapeTargets.add(new GoalTarget(individualTapeTargets.get(index  + 1), individualTapeTargets.get(index)));
          //     }
          //     if(individualTapeTargets.get(index).angle < 40 && individualTapeTargets.get(index + 1).angle > 320)
          //     {
          //         fullTapeTargets.add(new GoalTarget(individualTapeTargets.get(index), individualTapeTargets.get(index + 1)));
          //     }
          // }

          // double[] xOffset = new double[fullTapeTargets.size()];
          // double[] distance = new double[fullTapeTargets.size()];
          // double[] angle = new double[fullTapeTargets.size()];

          // for(int index = 0; index < fullTapeTargets.size(); index++)
          // {
          //   GoalTarget target = fullTapeTargets.get(index);
          //   xOffset[index] = CameraCalculations.getXOffset(target.targetWidth(), target.centerX());
          //   distance[index] = CameraCalculations.getDistance(target.targetWidth(), target.centerX());
          //   angle[index] = CameraCalculations.getHorizontalDegreesToPixels(target.centerX());
          // }

          //WriteRoiToNetworkTable(roiTable, xOffset, distance, angle);
         
      });
      
      visionThread.start();
    }

    // loop forever
    for (;;) {
      try {
        Thread.sleep(10000);
      } catch (InterruptedException ex) {
        return;
      }
    }
  }

}

