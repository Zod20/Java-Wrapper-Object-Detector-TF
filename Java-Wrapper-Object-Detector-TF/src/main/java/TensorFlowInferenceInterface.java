import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Tensors;
import org.tensorflow.types.UInt8;

public class TensorFlowInferenceInterface {
	
	private static final String TAG = "TensorFlowInferenceInterface";
	
	
	 // Immutable state.
	  private final String modelName;
	  private final Graph g;
	  private final Session sess;

	  // State reset on every call to run.
	  private Session.Runner runner;
	  private List<String> feedNames = new ArrayList<String>();
	  private List<Tensor<?>> feedTensors = new ArrayList<Tensor<?>>();
	  private List<String> fetchNames = new ArrayList<String>();
	  private List<Tensor<?>> fetchTensors = new ArrayList<Tensor<?>>();

	public TensorFlowInferenceInterface(String model) throws Exception{
		
		this.modelName = model;
	    this.g = new Graph();
	    this.sess = new Session(g);
	    this.runner = sess.runner();
		
		InputStream is = null;
		 try {
		     is = new FileInputStream(model);
		 } catch (IOException e2) {
		     throw new RuntimeException("Failed to load model from '" + model + "'", e2);
		 }
		
		 byte[] graphDef = new byte[is.available()];
	     final int numBytesRead = is.read(graphDef);
	      if (numBytesRead != graphDef.length) {
	        throw new IOException(
	            "read error: read only "
	                + numBytesRead
	                + " of the graph, expected to read "
	                + graphDef.length);
	      }
	      
	      loadGraph(graphDef, g);
	      is.close();
	      System.out.println("Successfully loaded model from '" + model + "'");

		
	}
	
	
	 /**
	   * Runs inference between the previously registered input nodes (via feed*) and the requested
	   * output nodes. Output nodes can then be queried with the fetch* methods.
	   *
	   * @param outputNames A list of output nodes which should be filled by the inference pass.
	   */
	  public void run(String[] outputNames) {
	    run(outputNames, false);
	  }

	 
	  /**
	   * Runs inference between the previously registered input nodes (via feed*) and the requested
	   * output nodes. Output nodes can then be queried with the fetch* methods.
	   *
	   * @param outputNames A list of output nodes which should be filled by the inference pass.
	   */
	  public void run(String[] outputNames, boolean enableStats) {
	    run(outputNames, enableStats, new String[] {});
	  }
	

	  /**
	   * An overloaded version of runInference that allows supplying targetNodeNames as well
	   */
	  public void run(String[] outputNames, boolean enableStats, String[] targetNodeNames) {
	    // Release any Tensors from the previous run calls.
	    closeFetches();

	    // Add fetches.
	    for (String o : outputNames) {
	      fetchNames.add(o);
	      TensorId tid = TensorId.parse(o);
	      runner.fetch(tid.name, tid.outputIndex);
	    }

		// Add targets.
	    for (String t : targetNodeNames) {
	      runner.addTarget(t);
	    }

	    // Run the session.
	    try {
	     
	        fetchTensors = runner.run();
	      
	    } catch (RuntimeException e) {
	      // Ideally the exception would have been let through, but since this interface predates the
	      // TensorFlow Java API, must return -1.

	      throw e;
	    } finally {
	      // Always release the feeds (to save resources) and reset the runner, this run is
	      // over.
	      closeFeeds();
	      runner = sess.runner();
	    }
	  }
	  
	  
	  /** Returns a reference to the Graph describing the computation run during inference. */
	  public Graph graph() {
	    return g;
	  }

	  public Operation graphOperation(String operationName) {
	    final Operation operation = g.operation(operationName);
	    if (operation == null) {
	      throw new RuntimeException(
	          "Node '" + operationName + "' does not exist in model '" + modelName + "'");
	    }
	    return operation;
	  }
	  
	  
	  /**
	   * Cleans up the state associated with this Object.
	   *
	   * <p>The TenosrFlowInferenceInterface object is no longer usable after this method returns.
	   */
	  public void close() {
	    closeFeeds();
	    closeFetches();
	    sess.close();
	    g.close();
	    
	  }

	  @Override
	  protected void finalize() throws Throwable {
	    try {
	      close();
	    } finally {
	      super.finalize();
	    }
	  }

	private void loadGraph(byte[] graphDef, Graph g) throws IOException {
	    final long startMs = System.currentTimeMillis();
		
	    try {
	        g.importGraphDef(graphDef);
	      } catch (IllegalArgumentException e) {
	        throw new IOException("Not a valid TensorFlow Graph serialization: " + e.getMessage());
	      }
	    
	    final long endMs = System.currentTimeMillis();

	    System.out.println("Model load took " + (endMs - startMs) + "ms, TensorFlow version: " + TensorFlow.version());
	    
	}
	
	
	
	// Methods for taking a native Tensor and filling it with values from Java arrays.

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, boolean[] src, long... dims) {
	    byte[] b = new byte[src.length];

	    for (int i = 0; i < src.length; i++) {
	      b[i] = src[i] ? (byte) 1 : (byte) 0;
	    }

	    addFeed(inputName, Tensor.create(Boolean.class, dims, ByteBuffer.wrap(b)));
	  }

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, float[] src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, FloatBuffer.wrap(src)));
	  }

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, int[] src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, IntBuffer.wrap(src)));
	  }

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, long[] src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, LongBuffer.wrap(src)));
	  }

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, double[] src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, DoubleBuffer.wrap(src)));
	  }

	  /**
	   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
	   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
	   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, byte[] src, long... dims) {
	    addFeed(inputName, Tensor.create(UInt8.class, dims, ByteBuffer.wrap(src)));
	  }

	  /**
	   * Copy a byte sequence into the input Tensor with name {@link inputName} as a string-valued
	   * scalar tensor. In the TensorFlow type system, a "string" is an arbitrary sequence of bytes, not
	   * a Java {@code String} (which is a sequence of characters).
	   */
	  public void feedString(String inputName, byte[] src) {
	    addFeed(inputName, Tensors.create(src));
	  }

	  /**
	   * Copy an array of byte sequences into the input Tensor with name {@link inputName} as a
	   * string-valued one-dimensional tensor (vector). In the TensorFlow type system, a "string" is an
	   * arbitrary sequence of bytes, not a Java {@code String} (which is a sequence of characters).
	   */
	  public void feedString(String inputName, byte[][] src) {
	    addFeed(inputName, Tensors.create(src));
	  }

	  // Methods for taking a native Tensor and filling it with src from Java native IO buffers.

	  /**
	   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
	   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
	   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
	   * elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, FloatBuffer src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, src));
	  }

	  /**
	   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
	   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
	   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
	   * elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, IntBuffer src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, src));
	  }

	  /**
	   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
	   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
	   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
	   * elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, LongBuffer src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, src));
	  }

	  /**
	   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
	   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
	   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
	   * elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, DoubleBuffer src, long... dims) {
	    addFeed(inputName, Tensor.create(dims, src));
	  }

	  /**
	   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
	   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
	   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
	   * elements as that of the destination Tensor. If {@link src} has more elements than the
	   * destination has capacity, the copy is truncated.
	   */
	  public void feed(String inputName, ByteBuffer src, long... dims) {
	    addFeed(inputName, Tensor.create(UInt8.class, dims, src));
	  }


	  private void addFeed(String inputName, Tensor<?> t) {
		    // The string format accepted by TensorFlowInferenceInterface is node_name[:output_index].
		    TensorId tid = TensorId.parse(inputName);
		    runner.feed(tid.name, tid.outputIndex, t);
		    feedNames.add(inputName);
		    feedTensors.add(t);
		  }

		  private static class TensorId {
		    String name;
		    int outputIndex;

		    // Parse output names into a TensorId.
		    //
		    // E.g., "foo" --> ("foo", 0), while "foo:1" --> ("foo", 1)
		    public static TensorId parse(String name) {
		      TensorId tid = new TensorId();
		      int colonIndex = name.lastIndexOf(':');
		      if (colonIndex < 0) {
		        tid.outputIndex = 0;
		        tid.name = name;
		        return tid;
		      }
		      try {
		        tid.outputIndex = Integer.parseInt(name.substring(colonIndex + 1));
		        tid.name = name.substring(0, colonIndex);
		      } catch (NumberFormatException e) {
		        tid.outputIndex = 0;
		        tid.name = name;
		      }
		      return tid;
		    }
		  }

		  private Tensor<?> getTensor(String outputName) {
		    int i = 0;
		    for (String n : fetchNames) {
		      if (n.equals(outputName)) {
		        return fetchTensors.get(i);
		      }
		      ++i;
		    }
		    throw new RuntimeException(
		        "Node '" + outputName + "' was not provided to run(), so it cannot be read");
		  }

		  private void closeFeeds() {
		    for (Tensor<?> t : feedTensors) {
		      t.close();
		    }
		    feedTensors.clear();
		    feedNames.clear();
		  }

		  private void closeFetches() {
		    for (Tensor<?> t : fetchTensors) {
		      t.close();
		    }
		    fetchTensors.clear();
		    fetchNames.clear();
		  }

		 
		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
		   * dst} must have length greater than or equal to that of the source Tensor. This operation will
		   * not affect dst's content past the source Tensor's size.
		   */
		  public void fetch(String outputName, float[] dst) {
		    fetch(outputName, FloatBuffer.wrap(dst));
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
		   * dst} must have length greater than or equal to that of the source Tensor. This operation will
		   * not affect dst's content past the source Tensor's size.
		   */
		  public void fetch(String outputName, int[] dst) {
		    fetch(outputName, IntBuffer.wrap(dst));
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
		   * dst} must have length greater than or equal to that of the source Tensor. This operation will
		   * not affect dst's content past the source Tensor's size.
		   */
		  public void fetch(String outputName, long[] dst) {
		    fetch(outputName, LongBuffer.wrap(dst));
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
		   * dst} must have length greater than or equal to that of the source Tensor. This operation will
		   * not affect dst's content past the source Tensor's size.
		   */
		  public void fetch(String outputName, double[] dst) {
		    fetch(outputName, DoubleBuffer.wrap(dst));
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
		   * dst} must have length greater than or equal to that of the source Tensor. This operation will
		   * not affect dst's content past the source Tensor's size.
		   */
		  public void fetch(String outputName, byte[] dst) {
		    fetch(outputName, ByteBuffer.wrap(dst));
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
		   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
		   * or equal to that of the source Tensor. This operation will not affect dst's content past the
		   * source Tensor's size.
		   */
		  public void fetch(String outputName, FloatBuffer dst) {
		    getTensor(outputName).writeTo(dst);
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
		   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
		   * or equal to that of the source Tensor. This operation will not affect dst's content past the
		   * source Tensor's size.
		   */
		  public void fetch(String outputName, IntBuffer dst) {
		    getTensor(outputName).writeTo(dst);
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
		   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
		   * or equal to that of the source Tensor. This operation will not affect dst's content past the
		   * source Tensor's size.
		   */
		  public void fetch(String outputName, LongBuffer dst) {
		    getTensor(outputName).writeTo(dst);
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
		   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
		   * or equal to that of the source Tensor. This operation will not affect dst's content past the
		   * source Tensor's size.
		   */
		  public void fetch(String outputName, DoubleBuffer dst) {
		    getTensor(outputName).writeTo(dst);
		  }

		  /**
		   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
		   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
		   * or equal to that of the source Tensor. This operation will not affect dst's content past the
		   * source Tensor's size.
		   */
		  public void fetch(String outputName, ByteBuffer dst) {
		    getTensor(outputName).writeTo(dst);
		  }
	  
}
