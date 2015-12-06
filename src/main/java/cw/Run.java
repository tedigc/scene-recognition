package cw;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;


public abstract class Run {


	protected VFSGroupDataset<FImage> groupedImages;
	GroupedDataset<String, ListDataset<Record>, Record> allData;

	protected GroupedDataset<String, ListDataset<Record>, Record> training;
	protected GroupedDataset<String, ListDataset<Record>, Record> test;
	protected int nTraining;
	protected int nTest;

	public abstract void run();	

	public void loadImages(String path) {

		System.out.println("Loading images...");
		try {
			this.groupedImages = new VFSGroupDataset<FImage>(path, ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
		System.out.println("Finished loading images.");

	}

	public void imagesToRecords(){

		System.out.println("Transforming images into records...");

		// Turn the groups of images into groups of records
		allData = new MapBackedDataset<String, ListDataset<Record>, Record>();

		for(String groupName : groupedImages.getGroups()) {
			ListDataset<FImage> groupInstances = groupedImages.get(groupName); 
			ListDataset<Record> recordList = new ListBackedDataset<Record>();
			for(int i=0; i<groupInstances.size(); i++) {
				recordList.add(new Record(String.valueOf(i), groupInstances.get(i), groupName));
			}
			allData.put(groupName, recordList);
		}

		System.out.println("Finished transforming images into records.");

	}

	public void splitDataset(){

		loadImages("/Users/marcosss3/Downloads/training");
		imagesToRecords();

		System.out.println("Splitting dataset into training and testing sets...");
		GroupedRandomSplitter<String, Record> splits = new GroupedRandomSplitter<String, Record>(allData, 90, 0, 10);	
		training = splits.getTrainingDataset();
		test 	 = splits.getTestDataset();

		nTraining = training.numInstances();
		nTest = test.numInstances();
		System.out.println("Dataset split into training and testing sets.");

	}

	public void loadTraining(){

		loadImages("/Users/marcosss3/Downloads/training");
		imagesToRecords();

		training = allData;
		nTraining = training.numInstances();
		System.out.println("Training dataset loaded.");

	}

	public void loadTesting(){

		loadImages("/Users/marcosss3/Downloads/testing");

		// Turn the groups of images into groups of records
		allData = new MapBackedDataset<String, ListDataset<Record>, Record>();

		for(String groupName : groupedImages.getGroups()) {
			ListDataset<FImage> groupInstances = groupedImages.get(groupName); 
			ListDataset<Record> recordList = new ListBackedDataset<Record>();
			for(int i=0; i<groupInstances.size(); i++) {
				recordList.add(new Record(String.valueOf(i), groupInstances.get(i), groupName));
			}
			allData.put(groupName, recordList);
		}

		System.out.println("Finished transforming images into records.");

		test = allData;
		nTest = test.numInstances();
		System.out.println(test.size());
		System.out.println("Testing dataset loaded.");

	}


}
