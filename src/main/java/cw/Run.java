package cw;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;


public abstract class Run {
	
	
	protected VFSGroupDataset<FImage> groupedImages;
	protected GroupedDataset<String, ListDataset<FImage>, FImage> training;
	protected GroupedDataset<String, ListDataset<FImage>, FImage> test;
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
    	System.out.println("Splitting into training and test sets.");
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedImages, 9, 0, 1);
		
    	this.training = splits.getTrainingDataset();
    	this.test     = splits.getTestDataset();
    	this.nTraining = training.numInstances();
    	this.nTest     = test.numInstances();
    	System.out.println("Finished splitting.");
    	System.out.println("Training set: " + training.numInstances());
    	System.out.println("Test set    : " + test.numInstances());
	}
	

}
