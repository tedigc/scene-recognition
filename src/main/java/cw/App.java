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


public class App {


	public static void main( String[] args ) {

    	VFSGroupDataset<FImage> groupedImages = null;
    	try {
			groupedImages = new VFSGroupDataset<FImage>("/Users/tedigc/Desktop/training", ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
    	
    	// Split into training and test data
    	GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedImages, 45, 0, 5);
    	
    	// Define variables for training and test data
    	GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
    	GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
    	int nTraining = training.numInstances();
    	int nTest     = test.numInstances();
    	
    	
    	System.out.println("nTraining: " + nTraining);
    	System.out.println("nTest    : " + nTest);
    	
    	// TRAINING DATA
    	// - For every instance of data, within every group, turn the img into a feature vector and record it's classification
    	int idx = 0;
    	float[][] trData = new float[nTraining][16*16];
    	String[] trClass = new String[nTraining];
    	for(String groupName : training.getGroups()) {
    		ListDataset<FImage> groupInstances = training.get(groupName);
    		
    		for(int i=0; i<groupInstances.size(); i++) {
    			float[] vectorInstance = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			trData[i] = vectorInstance;
    			trClass[i] = groupName;
    			idx++;
    		}
    	}
    	
    	for(int i=0; i<nTraining; i++) {
    		for(int j=0; j<256; j++) {
    			System.out.print(trData[i][j] + ", ");
    		}
    		System.out.println();
    	}
    	
    	// TEST DATA
    	// - For every instance of data, within every group, turn the img into a feature vector and record it's classification
    	idx = 0;
    	float[][] tsData = new float[nTest][16*16];
    	String[] tsClass = new String[nTest];
    	for(String groupName : test.getGroups()) {
    		ListDataset<FImage> groupInstances = test.get(groupName);
    		
    		for(int i=0; i<groupInstances.size(); i++) {
    			float[] vectorInstance = imageToFloatVector(cropCentre(groupInstances.get(i)));
    			tsData[i] = vectorInstance;
    			tsClass[i] = groupName;
    			idx++;
    		}
    	}
    	
		// Define a NN object.
		int k = 3;
		final FloatNearestNeighboursExact nn = new FloatNearestNeighboursExact(trData);
    	    	
		// For every instance of test data within every group, find the KNN from the training data and it's average class.
    	for(int i=0; i<tsData.length-1; i++) {
    		List<IntFloatPair> neighbors = nn.searchKNN(tsData[i], k);
    		for(IntFloatPair pair : neighbors) {
    			System.out.println(pair);
    			String predictedClass = trClass[pair.getFirst()];
    			String actualClass = tsClass[i];
    			System.out.println(predictedClass);
    			System.out.println(actualClass);
    		}
    		System.out.println();
    	}
    }
	
	
	public static FImage cropCentre(FImage original){
		
		int resolution = 0;
		
		if(original.width < original.height)
			resolution = original.width;
		else
			resolution = original.height;
		
		return ResizeProcessor.resample(original.extractCenter(resolution, resolution), 16, 16);
	}
	
	
	public static float[] imageToFloatVector(FImage img) {
		
		float[] floatVector = null;
		for(int i=0; i< img.height; i++){
			floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
		}
		return floatVector;
	}


}
