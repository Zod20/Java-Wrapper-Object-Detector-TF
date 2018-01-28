# Java-Wrapper-Object-Detector-TF
Ported the Android Tensorflow Object Detector to Java by replacing various android classes with native java classes.


<img src="https://github.com/Zod20/Java-Wrapper-Object-Detector-TF/blob/master/Java-Wrapper-Object-Detector-TF/src/main/resources/2.jpg?raw=true" width="300" height="300"><img src="https://github.com/Zod20/Java-Wrapper-Object-Detector-TF/blob/master/Java-Wrapper-Object-Detector-TF/src/main/resources/1.jpg?raw=true" width="300" height="300">          


The main points of changes from the Android version is that the android bitmap class has been converted to the BufferedImage class and also the android RectF class has been switched to Java Rectangle class. This has caused some issues particularly in the coordinate system but hopefully all that has been smoothed out. There are 3 main pathways that should be changed before running the program - 

      private static final String TF_OD_API_MODEL_FILE = "D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\ssd_mobilenet_v1_android_export.pb";
      
      private static final String TF_OD_API_LABELS_FILE = "D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\coco_labels_list.txt";
        
      inputImage = ImageIO.read(new File("D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\6.jpg"));
      
Once these have been sorted out, you can try out different images and see the bounding boxes detection as well as the percentages in console. The model used is the ssd mobilenet android coco trained model but you can experiment with rcc trained and even the inception models. Have fun!
