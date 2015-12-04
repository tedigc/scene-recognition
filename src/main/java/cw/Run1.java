package cw;

import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;


/*
 *  - Run1: K-Nearest Neighbors
 *  
 *  
 */


public class Run1 {
	
	
	private int K = 3;
	private final int resolution = 16;
	private final int imgSize = resolution * resolution;
	
	
	public void run() {

		// Load dataset
    	VFSGroupDataset<FImage> groupedImages = null;
    	try {
			groupedImages = new VFSGroupDataset<FImage>("/home/ec7g13/Documents/Computer Vision/scene-recognition/training", ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
    	
    	// Split into training and test data
    	// TODO replace 90 and 10 magic numbers with variable
    	GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedImages, 90, 0, 10);
    	
    	// Define variables for training and test datasets
    	GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
    	GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
    	int nTraining = training.numInstances();
    	int nTest     = test.numInstances();
    	  	
    	System.out.println("nTraining: " + nTraining);
    	System.out.println("nTest    : " + nTest);
    	
    	/************* TRAINING DATA *************/
    	
    	float[][] trData = new float[nTraining][imgSize];
    	String[] trClass = new String[nTraining];
    	
    	int idx = 0;
    	
    	// For every instance of data, within every group, turn the img into a feature vector and record its classification
    	for(String groupName : training.getGroups()) {
    		
    		// Current scene
    		ListDataset<FImage> groupInstances = training.get(groupName);
    		
    		// Store vector of each image's pixels in an array
    		for(int i = 0; i < groupInstances.size(); i++) {
    			trData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			trClass[idx] = groupName;
    			idx++;
    		}
    	}
    	
    	/************** TEST DATA **************/
    	
    	float[][] tsData = new float[nTest][imgSize];
    	String[] tsClass = new String[nTest];
    	
    	idx = 0;
    	
    	// For every instance of data, within every group, turn the img into a feature vector and record it's classification
    	for(String groupName : test.getGroups()) {
    		
    		// Current scene
    		ListDataset<FImage> groupInstances = test.get(groupName);
    		
    		// Store vector of each image's pixels in an array
    		for(int i=0; i<groupInstances.size(); i++) {
    			tsData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			tsClass[idx] = groupName;
    			idx++;
    		}	
    	}
    	
    	/************ KNN ClASSIFIER ************/
    	
		// Define a NN object
		final FloatNearestNeighboursExact nn = new FloatNearestNeighboursExact(trData);
    	    	
    	int correct   = 0;
    	int incorrect = 0;
    	
    	// For every instance of test data within every group, find the KNN from the training data and its average class.
		for(int i = 0; i < tsData.length-1; i++) {
			
			// Get ordered list of pairs containing the index and distance of each neighbour
    		List<IntFloatPair> neighbours = nn.searchKNN(tsData[i], K);
    		
    		// Evaluate if prediction matches the actual class
    		for(IntFloatPair pair : neighbours) {
    			
    			String predictedClass = trClass[pair.getFirst()];
    			String actualClass = tsClass[i];
    			
    			if(predictedClass == actualClass) 
    				correct++;
    			else
    				incorrect++;	
    		}
    	}
		
		// Print evaluated accuracy
		double accuracy = (double) correct / (double) (correct + incorrect);
		System.out.println("Accuracy : " + accuracy);
    }
	
	// Crop image to a square about the centre and resize
	private FImage cropCentre(FImage original){
		
		int resolution = 0;
		
		if(original.width < original.height)
			resolution = original.width;
		else
			resolution = original.height;
		
		FImage img = ResizeProcessor.resample(original.extractCenter(resolution, resolution), this.resolution, this.resolution);
		return img.normalise();
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
