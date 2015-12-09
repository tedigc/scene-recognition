package cw;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;


public abstract class Run {


	public static final String TRAINING_PATH_TED    = "/Users/tedigc/Documents/University/Computer Vision/Scene Recognition/SceneRecognition/training";
	public static final String TESTING_PATH_TED     = "/Users/tedigc/Documents/University/Computer Vision/Scene Recognition/SceneRecognition/testing";

	public static final String TRAINING_PATH_MARCOS = "/Users/marcosss3/Downloads/training";
	public static final String TESTING_PATH_MARCOS  = "/Users/marcosss3/Downloads/testing";
	
	// Store the full dataset
	protected GroupedDataset<String, ListDataset<Record>, Record> allData;
	
	// Store training and testing datasets
	protected GroupedDataset<String, ListDataset<Record>, Record> training;
	protected GroupedDataset<String, ListDataset<Record>, Record> test;
	
	// Store number of training and testing instances
	protected int nTraining;
	protected int nTest;

	public abstract void run();	

	// Load the images from the given path
	public void loadImages(String path) {

		System.out.println("Loading images...");
		
		VFSGroupDataset<FImage> groupedImages = null;
		
		try {
			groupedImages = new VFSGroupDataset<FImage>(path, ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
		
		System.out.println("Finished loading images.");
		System.out.println("Transforming images into records...");
		
		// Store dataset of records instead of images
		allData = new MapBackedDataset<String, ListDataset<Record>, Record>();

		// Transform the groups of images into groups of records
		for(String groupName : groupedImages.getGroups()) {
			
			// List of images in the current group
			ListDataset<FImage> groupInstances = groupedImages.get(groupName);
			// List of records
			ListDataset<Record> recordList = new ListBackedDataset<Record>();
			
			// Images in the current group
			FileObject[] files = null;
			
			try {
				files = groupedImages.getGroupDirectories().get(groupName).getChildren();
			} catch (FileSystemException e) {
				e.printStackTrace();
			}
			
			// Transform every file in the current group into a record
			for(int i=0; i<groupInstances.size(); i++) {
				
				String filename = files[i].getName().getBaseName();
				
				// Only add .jpg images to the record list
				if(filename.toLowerCase().contains(".j"))
					recordList.add(new Record(filename, groupInstances.get(i), groupName));
				
			}
			
			// Set the full dataset to be the list of records
			allData.put(groupName, recordList);
			
		}
		
		System.out.println("Finished transforming images into records.");
		
	}

	// Splits a single training set into training and test subsets.
	public void splitDataset(String trainingPath){

		loadImages(trainingPath);

		System.out.println("Splitting dataset into training and testing sets...");
		GroupedRandomSplitter<String, Record> splits = new GroupedRandomSplitter<String, Record>(allData, 90, 0, 10);	
		training = splits.getTrainingDataset();
		test 	 = splits.getTestDataset();

		nTraining = training.numInstances();
		nTest = test.numInstances();
		System.out.println("Dataset split into training and testing sets.");
		
	}

	// Load two separate training and test sets from the given paths
	public void realDataset(String trainingPath, String testingPath){

		System.out.println(trainingPath);
		System.out.println(testingPath);
		loadTraining(trainingPath);
		loadTesting(testingPath);
		
	}

	// Load training set from the given path and update dataset
	public void loadTraining(String path){

		loadImages(path);

		training = allData;
		nTraining = training.numInstances();
		System.out.println("Training dataset loaded.");
		
	}

	// Load testing set from the given path and update dataset
	public void loadTesting(String path){

		loadImages(path);

		test = allData;
		nTest = test.numInstances();
		System.out.println(test.get("test").numInstances());
		System.out.println("Testing dataset loaded.");
		
	}


	protected void printPredictions(Map<Integer, String> predictions, String filePath) {

		try {
			
			File file = new File(filePath);
			if (!file.exists()) {
				file.createNewFile();
			}

			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);

			for(Integer id : predictions.keySet()) {
				System.out.println(id + ".jpg" + " " + predictions.get(id));
				bw.write(id + ".jpg " + predictions.get(id) + "\n");
			}
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}


}
