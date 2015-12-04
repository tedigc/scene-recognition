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
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
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
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.FloatDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class Run3 {

	public static void main(String[] args) throws IOException{
		
		Timer t1 = Timer.timer();

		VFSGroupDataset<FImage> groupedImages = null;
		try {
			groupedImages = new VFSGroupDataset<FImage>("/Users/tedigc/Documents/University/Computer Vision/Scene Recognition/SceneRecognition/training", ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}

		GroupedRandomSplitter<String, FImage> splits = 
				new GroupedRandomSplitter<String, FImage>(groupedImages, 50, 0, 50);

		// Define variables for training and test datasets
		GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		GroupedDataset<String, ListDataset<FImage>, FImage> test = splits.getTestDataset();
		int nTraining = training.numInstances();
		int nTest     = test.numInstances();
		
		File extractorDir = new File("cache");
		File assignerCache = new File("assigner");

		// Extracts upright SIFT features at a single scale on a grid
		DenseSIFT dsift = new DenseSIFT(4, 8);
		// Dense sift features are extracted for the given bin sizes
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 4f, 8);
		// Use 30 images from the training set to train the quantiser
		HardAssigner<float[], float[], IntFloatPair> assigner =
				trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);
		
		// Check if assigner is stored in a file
		if (assignerCache.exists()) {
			
			assigner = IOUtils.readFromFile(assignerCache);
			System.out.println("HardAssigner was loaded from cache");

		}
		// If assigner is not stored, then create a new one
		if (assigner == null) {
			
			assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 20), pdsift);
			System.out.println("HardAssigner was generated");

			IOUtils.writeToFile(assigner, assignerCache);
			System.out.println("HardAssigner was saved.");

		}

		HomogeneousKernelMap map = new HomogeneousKernelMap(
				KernelType.Chi2, org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType.Rectangular);
		
		// Cache features to disk
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
		
		extractor = map.createWrappedExtractor(extractor);

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
			Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift){

		List<LocalFeatureList<FloatDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatDSIFTKeypoint>>();

		for (FImage rec : sample) {
			FImage img = rec.getImage();
			pdsift.analyseImage(img);
			allkeys.add(pdsift.getFloatKeypoints(0.005f));
		}

		if (allkeys.size() > 1000)
			allkeys = allkeys.subList(0, 1000);

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatDSIFTKeypoint, float[]>(allkeys);
		FloatCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();

	}

	// Used to train classifier - computes 4 histograms across an image
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<float[], float[], IntFloatPair> assigner){

			this.pdsift = pdsift;
			this.assigner = assigner;

		}

		public DoubleFV extractFeature(FImage object) {

			FImage image = object.getImage();
			pdsift.analyseImage(image);

			// Assign each dense SIFT feature to a visual word
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

			// Compute spatial histograms
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(
					bovw, 2, 2);

			// Append and normalise the resultant spatial histograms
			return spatial.aggregate(pdsift.getFloatKeypoints(0.015f), image.getBounds()).normaliseFV();

		}

	}


}