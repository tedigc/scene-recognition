package cw;

import java.util.List;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.Image;
import org.openimaj.image.analyser.ImageAnalyser;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;

public class DensePatch implements ImageAnalyser {

	
	LocalFeatureList<FloatKeypoint> keypoints = null;
	
	
	public DensePatch() {
		
		keypoints = new MemoryLocalFeatureList<FloatKeypoint>();
	}
	
	
	@Override
	public void analyseImage(Image image) {
		
		// Find image features, stick them in keypoints field
		List<Rectangle> rectangles = getPatchRectangles(image);
		for(Rectangle rect : rectangles) {
			Image patch = image.extractROI(rect).normalise();
			Keypoint keypoint = new Keypoint();
		}
	}
	
	
	public LocalFeatureList<FloatKeypoint> getKeypoints() {
		
		return this.keypoints;
	}
	
	
	// Returns all patch rectangles for an image
	private List<Rectangle> getPatchRectangles(Image img) {
		
		RectangleSampler sampler = new RectangleSampler(img, 4, 4, 8, 8);
		return sampler.allRectangles();
	}

}
