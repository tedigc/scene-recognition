package cw;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;

public class DenseSamplingEngine implements Engine {
	
	
	private final float sampleStep = 4;
	private final float sampleSize = 8;
	

	@Override
	public LocalFeatureList findFeatures(Image image) {
		
		List<Rectangle> patches = getPatches(image);
		for(Rectangle rect : patches) {
			Image patch = image.extractROI(rect);
		}
		return null;
	}


	// Returns all patch rectangles for an image
	private List<Rectangle> getPatches(Image img) {

		RectangleSampler sampler = new RectangleSampler(img, sampleStep, sampleStep, sampleSize, sampleSize);
		return sampler.allRectangles();
	}
	
	
	// Pack image pixels into a vector
	private float[] imageToFloatVector(FImage img) {
		
		float[] floatVector = null;
		for(int i=0; i< img.height; i++){
			floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
		}
		return floatVector;
	}

}
