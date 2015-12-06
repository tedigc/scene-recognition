package cw;

import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;


public class Run1 extends Run {
	
	
	private int K = 3;
	private final int resolution = 16;
	private final int imgSize = resolution * resolution;
	
	@Override
	public void run() {
		
		loadImages("/Users/tedigc/Documents/University/Computer Vision/Scene Recognition/SceneRecognition/training");    	
    	
    	// -- Training Data
		//
    	float[][] trData = new float[nTraining][imgSize]; // Array of vectors describing each image
    	String[] trClass = new String[nTraining];		  // Array of each image's classification
    	int idx = 0;
    	
    	// For every instance of data, within every group, turn the img into a feature vector and record its classification
    	for(String groupName : training.getGroups()) {
    		ListDataset<FImage> groupInstances = training.get(groupName);
    		for(int i = 0; i < groupInstances.size(); i++) {
    			trData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			trClass[idx] = groupName;
    			idx++;
    		}
    	}
    	
    	// -- Test Data
    	//
    	float[][] tsData = new float[nTest][imgSize];
    	String[] tsClass = new String[nTest];
    	idx = 0;
    	
    	// For every instance of data, within every group, turn the img into a feature vector and record it's classification
    	for(String groupName : test.getGroups()) {
    		ListDataset<FImage> groupInstances = test.get(groupName);    		
    		for(int i=0; i<groupInstances.size(); i++) {
    			tsData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			tsClass[idx] = groupName;
    			idx++;
    		}	
    	}
    	
    	// -- KNN Classification
    	//
		final FloatNearestNeighboursExact nn = new FloatNearestNeighboursExact(trData);    	    	
    	int correct   = 0;
    	int incorrect = 0;
    	
    	// For every instance of test data within every group, find the KNN from the training data and its average class.
		for(int i = 0; i < tsData.length-1; i++) {
			List<IntFloatPair> neighbours = nn.searchKNN(tsData[i], K);
    		
    		// Predict class and record check whether or not it was correct
    		for(IntFloatPair pair : neighbours) {
    			String predictedClass = trClass[pair.getFirst()];
    			String actualClass = tsClass[i];
    			
    			if(predictedClass == actualClass) 
    				correct++;
    			else
    				incorrect++;	
    		}
    	}
		
		double accuracy = (double) correct / (double) (correct + incorrect);
		System.out.println("Accuracy : " + accuracy);
    }
	
	// Crop image to a square about the center and resize
	private FImage cropCentre(FImage original){
		
		int resolution = 0;
		if(original.width < original.height)
			resolution = original.width;
		else
			resolution = original.height;
		
		FImage img = ResizeProcessor.resample(original.extractCenter(resolution, resolution), this.resolution, this.resolution);
		return img.normalise();
	}
	
	// Pack image pixels into a row vector
	private float[] imageToFloatVector(FImage img) {
		
		float[] floatVector = null;
		
		for(int i=0; i< img.height; i++){
			floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
		}
		return floatVector;
	}


}
