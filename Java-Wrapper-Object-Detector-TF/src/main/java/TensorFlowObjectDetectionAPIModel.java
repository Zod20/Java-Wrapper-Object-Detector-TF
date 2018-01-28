import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.Graph;
import org.tensorflow.Operation;

public class TensorFlowObjectDetectionAPIModel implements Classifier {
	
	
	// Only return this many results.
	  private static final int MAX_RESULTS = 100;

	  // Config values.
	  private String inputName;
	  private int inputSize;

	  // Pre-allocated buffers.
	  private Vector<String> labels = new Vector<String>();
	  private int[] intValues;
	  private byte[] byteValues;
	  private float[] outputLocations;
	  private float[] outputScores;
	  private float[] outputClasses;
	  private float[] outputNumDetections;
	  private String[] outputNames;

	  private boolean logStats = false;

	  private TensorFlowInferenceInterface inferenceInterface;
	  
	  
	  public static Classifier create( final String modelFilename,
		      final String labelFilename,
		      final int inputSize) throws Exception{
		  
		  final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();

		    InputStream labelsInput = null;
		    //String actualFilename = labelFilename.split("file:///android_asset/")[1];
		    InputStream labelsInput1   = new FileInputStream(labelFilename);
		    BufferedReader br = null;
		    br = new BufferedReader(new InputStreamReader(labelsInput1));
		    String line;
		    while ((line = br.readLine()) != null) {
		      d.labels.add(line);
		    }
		    br.close();


		    d.inferenceInterface = new TensorFlowInferenceInterface(modelFilename);

		    final Graph g = d.inferenceInterface.graph();

		    d.inputName = "image_tensor";
		    // The inputName node has a shape of [N, H, W, C], where
		    // N is the batch size
		    // H = W are the height and width
		    // C is the number of channels (3 for our purposes - RGB)
		    final Operation inputOp = g.operation(d.inputName);
		    if (inputOp == null) {
		      throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
		    }
		    d.inputSize = inputSize;
		    // The outputScoresName node has a shape of [N, NumLocations], where N
		    // is the batch size.
		    final Operation outputOp1 = g.operation("detection_scores");
		    if (outputOp1 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_scores'");
		    }
		    final Operation outputOp2 = g.operation("detection_boxes");
		    if (outputOp2 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_boxes'");
		    }
		    final Operation outputOp3 = g.operation("detection_classes");
		    if (outputOp3 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_classes'");
		    }

		    // Pre-allocate buffers.
		    d.outputNames = new String[] {"detection_boxes", "detection_scores",
		                                  "detection_classes", "num_detections"};
		    d.intValues = new int[d.inputSize * d.inputSize];
		    d.byteValues = new byte[d.inputSize * d.inputSize * 3];
		    d.outputScores = new float[MAX_RESULTS];
		    d.outputLocations = new float[MAX_RESULTS * 4];
		    d.outputClasses = new float[MAX_RESULTS];
		    d.outputNumDetections = new float[1];
		    return d;
		  
	  }
	
	  private TensorFlowObjectDetectionAPIModel() {}


	public List<Recognition> recognizeImage(BufferedImage bitmap) {

	    // Preprocess the image data from 0-255 int to normalized float based
	    // on the provided parameters.
	   // bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());  //android bitmap
	    
	    bitmap.getRGB(0, 0, bitmap.getWidth(), bitmap.getHeight(), intValues, 0, bitmap.getWidth());
	    
	    for (int i = 0; i < intValues.length; ++i) {
	      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
	      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
	      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
	    }
	  
	    inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
	    

	   
	    inferenceInterface.run(outputNames, logStats);

	    // Copy the output Tensor back into the output array.
	    outputLocations = new float[MAX_RESULTS * 4];
	    outputScores = new float[MAX_RESULTS];
	    outputClasses = new float[MAX_RESULTS];
	    outputNumDetections = new float[1];
	    inferenceInterface.fetch(outputNames[0], outputLocations);
	    inferenceInterface.fetch(outputNames[1], outputScores);
	    inferenceInterface.fetch(outputNames[2], outputClasses);
	    inferenceInterface.fetch(outputNames[3], outputNumDetections);

	    // Find the best detections.
	    final PriorityQueue<Recognition> pq = new PriorityQueue<Classifier.Recognition>(1, new Comparator<Recognition>() {

			public int compare(Recognition lhs, Recognition rhs) {
				// Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
			}
		});

	    // Scale them back to the input size.
	    for (int i = 0; i < outputScores.length; ++i) {
	    	
	    	float x1 = outputLocations[4 * i + 1] * inputSize;
	    	float x2 = outputLocations[4 * i + 3] * inputSize;
	    	float y1 = outputLocations[4 * i] * inputSize;
	    	float y2 = outputLocations[4 * i + 2] * inputSize;
	    	
	    	
	      final Rectangle detection =
	          new Rectangle(
	              (int)(x1),
	              (int)(y1),
	              (int)(x2-x1),
	              (int)(y2-y1));
	      pq.add(
	          new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
	    }

	    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
	    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
	      recognitions.add(pq.poll());
	    }
	    //Trace.endSection(); // "recognizeImage"
	    return recognitions;
	}

	public void enableStatLogging(boolean debug) {
		this.logStats = logStats;		
	}

	public String getStatString() {
		// TODO Auto-generated method stub
		return null;
	}

	public void close() {
		inferenceInterface.close();
		
	}

}
