package cw;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.lang.ArrayUtils;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;


public class Run1 extends Run {

	private int K = 1;
	private final int resolution = 16;
	private final int imgSize = resolution * resolution;

	@Override
	public void run() {

		realDataset(Run.TRAINING_PATH_MARCOS, Run.TESTING_PATH_MARCOS);
		//splitDataset(Run.TRAINING_PATH_MARCOS);

		// -- Training Data
		//
		String[] trIDs = new String[nTraining];
		float[][] trData = new float[nTraining][imgSize]; // Array of vectors describing each image
		String[] trClass = new String[nTraining];		  // Array of each image's classification
		int idx = 0;

		// For every instance of data, within every group, turn the img into a feature vector and record its classification
		for(String groupName : training.getGroups()) {
			ListDataset<Record> groupInstances = training.get(groupName);
			for(int i = 0; i < groupInstances.size(); i++) {
				trIDs[idx] = groupInstances.get(i).getID();
				trData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i).getImage()));
				trClass[idx] = groupName;
				idx++;
			}
		}

		// -- Test Data
		//
		String[] tsIDs = new String[nTest];
		float[][] tsData = new float[nTest][imgSize];
		String[] tsClass = new String[nTest];
		idx = 0;

		// For every instance of data, within every group, turn the img into a feature vector and record its classification
		for(String groupName : test.getGroups()) {
			ListDataset<Record> groupInstances = test.get(groupName);    		
			for(int i=0; i<groupInstances.size(); i++) {
				tsIDs[idx] = groupInstances.get(i).getID();
				//System.out.println(tsIDs[idx]);
				tsData[idx] = imageToFloatVector(cropCentre(groupInstances.get(i).getImage()));
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
		Map<Integer, String> allPredictions = new TreeMap<Integer, String>();
		for(int i = 0; i < tsData.length-1; i++) {
			List<IntFloatPair> neighbours = nn.searchKNN(tsData[i], K);

			// Check if the predicted (mode) class is equal to the actual class.
			String predictedClass = findModeClass(neighbours, trClass);
			String actualClass = tsClass[i];

			if(predictedClass == actualClass) 
				correct++;
			else
				incorrect++;

			allPredictions.put(Integer.valueOf(tsIDs[i]), predictedClass);
		}
		
		printPredictions(allPredictions, "run1.txt");
		System.out.println();
		System.out.println("nTraining: " + nTraining);
		System.out.println("nTest    : " + nTest);	
		double accuracy = (double) correct / (double) (correct + incorrect);
		System.out.println("Accuracy : " + round(accuracy*100));
		
	}

	String round(double d){

		return String.format("%.2f", d);
		
	}

	// Given a list of neighbours, return the most common one
	private String findModeClass(List<IntFloatPair> neighbours, String[] trClass) {

		Map<String, Integer> occurences = new HashMap<String, Integer>();
		int modeOccurences = 0;
		String modeClass = null;
		
		for(IntFloatPair pair : neighbours) {
			
			String neighbourClass = trClass[pair.getFirst()];
			int nOccurences;
			
			if(occurences.containsKey(neighbourClass)) {
				nOccurences = occurences.get(neighbourClass) + 1;
			} 
			else {
				nOccurences = 1;
			}
			
			occurences.put(neighbourClass, nOccurences);
			
			if(nOccurences > modeOccurences) {
				modeClass = neighbourClass;
			}
			
		}
		
		return modeClass;
		
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
