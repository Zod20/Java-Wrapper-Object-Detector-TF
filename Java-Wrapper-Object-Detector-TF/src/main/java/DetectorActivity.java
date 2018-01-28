import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class DetectorActivity {


	private static BufferedImage inputImage;
	private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE =
    		"D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = 
    		"D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\coco_labels_list.txt";
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static Classifier detector;
    private long lastProcessingTimeMs;
    private List<Classifier.Recognition> mappedRecognitions =
            new LinkedList<>();
	
	public static void main(String [] args) throws Exception
	{
		
		inputImage = ImageIO.read(new File(
				"D:\\Documents\\EclipseProjects\\Java-Wrapper-Object-Detector-TF\\src\\main\\resources\\6.jpg"));
		BufferedImage scaledImage = resize(inputImage, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE);
		
		try {
            detector = TensorFlowObjectDetectionAPIModel.create(
                    TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        } catch (final IOException e) {
        	System.out.println("Exception initializing classifier! " +  e);
           
        }
		
		//main detection on input image
        final List<Classifier.Recognition> results = detector.recognizeImage(scaledImage);
        System.out.println("Detect: %s" +  results);

        final JPanel panel = new JPanel();
		final JFrame frame = new JFrame("User Interface"){
			@Override
			public void paint(Graphics g) {
				super.paint(g);
				g.setColor(Color.GREEN);
				for (final Classifier.Recognition result : results) {
					final Rectangle location = result.getLocation();
					if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
						g.drawRect(location.x+15,location.y+44,location.width, location.height);

					}

				}

			}
		};
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//		frame.getContentPane().setLayout(new FlowLayout());
//		frame.getContentPane().add(new JLabel(new ImageIcon(img1)));
//		frame.pack();
		frame.add(panel);
		JLabel wIcon = new JLabel(new ImageIcon(scaledImage.getScaledInstance(scaledImage.getWidth(), scaledImage.getHeight(), Image.SCALE_FAST)));
		//JLabel wIcon = new JLabel(new ImageIcon(inputImage.getScaledInstance(inputImage.getWidth(), inputImage.getHeight(), Image.SCALE_FAST)));
		panel.add(wIcon);
		frame.setVisible(true);
		frame.setSize(scaledImage.getWidth(), scaledImage.getHeight());
		frame.pack();
		
	}
	
	
	//resizes bufferedimage
		public static BufferedImage resize(BufferedImage img, int newW, int newH) { 
		    Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
		    BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);

		    Graphics2D g2d = dimg.createGraphics();
		    g2d.drawImage(tmp, 0, 0, null);
		    g2d.dispose();

		    return dimg;
		}  

}
