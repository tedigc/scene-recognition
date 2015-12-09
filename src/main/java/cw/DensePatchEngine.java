package cw;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;


public class DensePatchEngine implements Engine<FloatKeypoint, FImage>{
	
	private int stepSize;
	private int windowSize;
	
	public DensePatchEngine(int stepSize, int windowSize) {
		
		super();
		this.stepSize = stepSize;
		this.windowSize = windowSize;
		
	}
	
	@Override
	public LocalFeatureList<FloatKeypoint> findFeatures(FImage image) {
		
		// Transform sampled patches into a feature vector and add to a list
		ArrayList<FloatKeypoint> features = new ArrayList<FloatKeypoint>();
		for(Rectangle rec: getPatches(image)){
			float[] featureVector = imageToFloatVector(getSample(image, rec));
			features.add(new FloatKeypoint(rec.x, rec.y, 0, 1, featureVector));
		}
		MemoryLocalFeatureList<FloatKeypoint> featureList = new MemoryLocalFeatureList<FloatKeypoint>(features);
		
		// Return a sample of the features
		return featureList.randomSubList((int) (featureList.size()*0.2));
		
	}
	
	// Returns a list of rectangles sampled from the image
	public List<Rectangle> getPatches(FImage img){

		RectangleSampler sampler = new RectangleSampler(img.normalise(), stepSize, stepSize, windowSize, windowSize);
		return sampler.allRectangles();
		
	}
	
	// Returns the actual sampled patch given the image and rectangle
	public FImage getSample(FImage img, Rectangle rect) {

		return img.normalise().extractROI(rect);
		
	}
	
	// Transforms image pixels into a vector
	private static float[] imageToFloatVector(FImage img) {
		
		float[] floatVector = null;
		for(int i=0; i< img.height; i++){
			floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
		}
		return floatVector;
		
	}

	
}