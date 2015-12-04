package cw;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.vfs2.FileSystemException;
import org.codehaus.jackson.JsonParser.Feature;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.ArrayFeatureVector;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.io.IOUtils;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class Run2 {

	public static void main(String[] args) throws IOException{

		Timer t1 = Timer.timer();

		VFSGroupDataset<FImage> groupedImages = null;
		try {
			groupedImages = new VFSGroupDataset<FImage>("/Users/tedigc/Documents/University/Computer Vision/Scene Recognition/SceneRecognition/training", ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}

		GroupedRandomSplitter<String, FImage> splits = 
				new GroupedRandomSplitter<String, FImage>(groupedImages, 90, 0, 10);

		// Define variables for training and test datasets
		GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
		int nTraining = training.numInstances();
		int nTest     = test.numInstances();
		
		File extractorDir = new File("cache");
		File assignerCache = new File("assigner");
		
		MyAnalyser analyser = new MyAnalyser();

		// Use 30 images from the training set to train the quantiser
		HardAssigner<float[], float[], IntFloatPair> assigner =
				trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 20), analyser);
		
		// Check if assigner is stored in a file
		if (assignerCache.exists()) {
			
			assigner = IOUtils.readFromFile(assignerCache);
			System.out.println("HardAssigner was loaded from cache");

		}
		// If assigner is not stored, then create a new one
		if (assigner == null) {
			
			assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 20), analyser);
			System.out.println("HardAssigner was generated");

			IOUtils.writeToFile(assigner, assignerCache);
			System.out.println("HardAssigner was saved.");

		}

		FeatureExtractor<DoubleFV, FImage> extractor = new MyExtractor(assigner, analyser);

		// Construct and train a linear classifier
		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		
		ann.train(training);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval =
				new ClassificationEvaluator<CMResult<String>, String, FImage>(
						ann, test, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);
		
		System.out.println("nTraining: " + nTraining);
		System.out.println("nTest    : " + nTest);

		System.out.println(result);

	}

	// Extracts the first 10000 dense SIFT features from the images in the given dataset
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
			GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, MyAnalyser analyser){

		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();

		for (FImage rec : groupedDataset) {
			FImage img = rec.getImage();
			//dsift.analyseImage(img);
			allkeys.add(analyser.findFeatures(img));
		}

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		FloatCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();

	}

	// Used to train classifier - computes 4 histograms across an image
	static class MyExtractor implements FeatureExtractor<DoubleFV, FImage> {

		MyAnalyser analyser;
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public MyExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, MyAnalyser analyser){

			this.analyser = analyser;
			this.assigner = assigner;

		}

		public DoubleFV extractFeature(FImage image) {

			// Assign each dense SIFT feature to a visual word
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

			// Compute spatial histograms
			BlockSpatialAggregator<float[], SparseIntFV> spatial = 
					new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			
			// Append and normalise the resultant spatial histograms
			return spatial.aggregate(analyser.findFeatures(image), image.getBounds()).normaliseFV();

		}

	}
	
	public static class MyAnalyser implements Engine<FloatKeypoint, FImage>{

		@Override
		public LocalFeatureList<FloatKeypoint> findFeatures(FImage image) {
			
			ArrayList<FloatKeypoint> features = new ArrayList<FloatKeypoint>();
			

			for(Rectangle rec: getPatches(image)){
				float[] featureVector = imageToFloatVector(getSample(image, rec));
				features.add(new FloatKeypoint(rec.x, rec.y, 0, 1, featureVector));
			}
			
			MemoryLocalFeatureList featureList = new MemoryLocalFeatureList(features);
			
			return featureList.randomSubList((int) (featureList.size()*0.1));
			
		}
		
		public static List<Rectangle> getPatches(FImage img){

			RectangleSampler sampler = new RectangleSampler(img.normalise(), 4, 4, 8, 8);

			return sampler.allRectangles();

		}

		public static FImage getSample(FImage img, Rectangle rect) {

			return img.normalise().extractROI(rect);

		}
		
		// Pack image pixels into a vector
		private static float[] imageToFloatVector(FImage img) {
			
			float[] floatVector = null;
			
			for(int i=0; i< img.height; i++){
				floatVector = ArrayUtils.addAll(floatVector, img.pixels[i]);
			}
			
			return floatVector;
			
		}
		
	}


}