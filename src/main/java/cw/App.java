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
import org.openimaj.knn.ObjectNearestNeighboursExact;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;


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
    	
    	float[][] trData = new float[splits.getTrainingDataset().numInstances()][16*16];
    	int idx = 0;
    	for(String groupName : splits.getTrainingDataset().getGroups()) {
    		ListDataset<FImage> groupData = groupedImages.getInstances(groupName);
    		for(int i=0; i<groupData.size(); i++) {
    			float[] vectorInstance = imageToFloatVector(groupData.get(i));
    			trData[i] = vectorInstance;
    			idx++;
    		}
    	}
		
		
		final FloatNearestNeighboursExact nn = new FloatNearestNeighboursExact(trData);
		final List<IntDoublePair> neighbours = nn.searchKNN(mean, k);
    	
    	
    	GroupedDataset<String, ListDataset<FImage>, FImage> testData = splits.getTestDataset();
    	for(String groupName : testData.getGroups()) {
    		ListDataset<FImage> groupData = testData.getInstances(groupName);
    		for
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
