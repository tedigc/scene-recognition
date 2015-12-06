package cw;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.FloatDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.Annotated;
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

public class Run3 extends Run {


	File assignerCache = new File("run3_assigner");


	@Override
	public void run() {

		loadImages("/Users/marcosss3/Downloads/training");

		// Turn the groups of images into groups of records
		GroupedDataset<String, ListDataset<Record>, Record> allData = new MapBackedDataset<String, ListDataset<Record>, Record>();;
		for(String groupName : groupedImages.getGroups()) {
			ListDataset<FImage> groupInstances = groupedImages.get(groupName); 
			ListDataset<Record> recordList = new ListBackedDataset<Record>();
			for(int i=0; i<groupInstances.size(); i++) {
				recordList.add(new Record(String.valueOf(i), groupInstances.get(i), groupName));
			}
			allData.put(groupName, recordList);
		}

		// Split into training and test data
		GroupedRandomSplitter<String, Record> splits = new GroupedRandomSplitter<String, Record>(allData, 1, 0, 1);
		GroupedDataset<String, ListDataset<Record>, Record> training = splits.getTrainingDataset();
		GroupedDataset<String, ListDataset<Record>, Record> test 	 = splits.getTestDataset();;
		int nTraining = training.numInstances();
		int nTest = test.numInstances();

		Timer t1 = Timer.timer();

		// Extracts upright SIFT features at a single scale on a grid
		DenseSIFT dsift = new DenseSIFT(3, 8);
		// Dense sift features are extracted for the given bin sizes
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 8);
		HardAssigner<float[], float[], IntFloatPair> assigner = readOrTrainAssigner(pdsift, 45);

		HomogeneousKernelMap map = new HomogeneousKernelMap(
				KernelType.Chi2, org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType.Rectangular);

		FeatureExtractor<DoubleFV, Record> extractor = new PHOWExtractor(pdsift, assigner);
		extractor = map.createWrappedExtractor(extractor);

		// Construct and train a linear classifier
		LiblinearAnnotator<Record, String> ann = new LiblinearAnnotator<Record, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

		ann.train((List<? extends Annotated<Record, String>>) training);

		ClassificationEvaluator<CMResult<String>, String, Record> eval =
				new ClassificationEvaluator<CMResult<String>, String, Record>(
						ann, 
						test, 
						new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

		Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		TreeMap<Record, ClassificationResult<String>> sortedGuesses = new TreeMap<Record, ClassificationResult<String>>();
		sortedGuesses.putAll(guesses);

		Iterator it = sortedGuesses.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<Record, ClassificationResult<String>> pair = (Map.Entry<Record, ClassificationResult<String>>)it.next();
			String imgClass = pair.getValue().getPredictedClasses().toString();
			System.out.println(pair.getKey().getID() + " " + imgClass.substring(1, imgClass.length()-1));
			it.remove(); // avoids a ConcurrentModificationException
		}

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

		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(600);
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatDSIFTKeypoint, float[]>(allkeys);
		FloatCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}

	// Attempts to read the HardAssigner from the cache, or trains one if this can't be done.
	private HardAssigner<float[], float[], IntFloatPair> readOrTrainAssigner(PyramidDenseSIFT<FImage> pdsift, int nSamples) {

		HardAssigner<float[], float[], IntFloatPair> assigner = null;

		// Check if assigner is stored in a file, and try to read it.
		System.out.println("Attempting to read assigner from cache...");
		if(this.assignerCache.exists()) {
			try {
				assigner = IOUtils.readFromFile(this.assignerCache);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		// If the assigner wasn't read (successfully), train a new one.
		if(assigner == null) {
			System.out.println("Generating new HardAssigner...");
			assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, nSamples), pdsift);
			System.out.println("HardAssigner generated.");
			try {
				System.out.println("Writing HardAssigner to cache...");
				IOUtils.writeToFile(assigner, this.assignerCache);
				System.out.println("HardAssigner successfully written to cache.");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return assigner;
	}


}