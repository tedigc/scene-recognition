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
    	
    	// Crop every image
    	for(String groupName : groupedImages.getGroups()) {
    		ListDataset<FImage> groupData = groupedImages.getInstances(groupName);
    		for(int i=0; i<groupData.size(); i++) {
    			FImage img = cropCentre(groupData.get(i));
    		}
    	}
    		
    	// Split into training and test data
    	GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedImages, 45, 0, 5);
    	
    	// Define variables for training and test data
    	int nTraining = splits.getTrainingDataset().numInstances();
    	int nTest     = splits.getTestDataset().numInstances();
    	GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
    	GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
    	
    	
    	// For every instance of data, within every group, turn the img into a feature vector and record it's classification
    	int idx = 0;
    	float[][] trData = new float[nTraining][16*16];
    	String[] trClass = new String[nTraining];
    	for(String groupName : training.getGroups()) {
    		ListDataset<FImage> groupInstances = training.get(groupName);
    		
    		for(int i=0; i<groupInstances.size(); i++) {
    			float[] vectorInstance = imageToFloatVector(groupInstances.get(i));
    			trData[i] = vectorInstance;
    			trClass[i] = groupName;
    			idx++;
    		}
    	}
    	
    	
		
		int k = 3;
		final FloatNearestNeighboursExact nn = new FloatNearestNeighboursExact(trData);
    	    	
		// Do KNN on tstData
    	for(String groupName : testData.getGroups()) {
    		ListDataset<FImage> groupData = testData.getInstances(groupName);
    		for(int i=0; i<groupData.size(); i++) {
    			float[] tstVector = imageToFloatVector(cropCentre(groupData.get(i)));
    			List<IntFloatPair> neighbors = nn.searchKNN(tstVector, k);

    		}
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
		for(int i=0; i< img.width; i++){
			floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
		}
		return floatVector;
	}


}
