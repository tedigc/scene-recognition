package cw;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
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

		System.out.println("Performing Run3...");

		realDataset(Run.TRAINING_PATH_MARCOS, Run.TESTING_PATH_MARCOS);
		//splitDataset(Run.TRAINING_PATH_MARCOS);

		int nTraining = training.numInstances();
		int nTest = test.numInstances();

		Timer t1 = Timer.timer();

		// Extracts upright SIFT features at a single scale on a grid
		DenseSIFT dsift = new DenseSIFT(3, 8);

		// Dense sift features are extracted for the given bin sizes
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4, 8, 16, 32);

		// Sample a subset from the training set to train the quantiser
		HardAssigner<float[], float[], IntFloatPair> assigner = readOrTrainAssigner(pdsift, 10);

		// Appends spatial histograms computed from the collection of visual words
		FeatureExtractor<DoubleFV, Record> extractor = new PHOWExtractor(pdsift, assigner);

		// Transforms data into a linear representation for higher accuracy
		HomogeneousKernelMap map = new HomogeneousKernelMap(
				KernelType.Chi2, org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType.Rectangular);

		// Create new feature extractor around the PHOWExtractor that applies the map
		extractor = map.createWrappedExtractor(extractor);

		// Construct and train a linear classifier
		LiblinearAnnotator<Record, String> ann = new LiblinearAnnotator<Record, String>(
				extractor,
				Mode.MULTICLASS,
				SolverType.L2R_L2LOSS_SVC,
				1.0,
				0.00001);
		ann.train(training);

		// Use OpenIMAJ evaluation framework to assess classifier accuracy
		ClassificationEvaluator<CMResult<String>, String, Record> eval =
				new ClassificationEvaluator<CMResult<String>, String, Record>(
						ann, 
						test, 
						new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

		// Store guess for each image/record
		Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		// Store ordered guesses
		TreeMap<Record, ClassificationResult<String>> sortedGuesses = new TreeMap<Record, ClassificationResult<String>>();
		sortedGuesses.putAll(guesses);

		Iterator it = sortedGuesses.entrySet().iterator();

		File file = new File("NEWrun3.txt");

		// Write predictions to a text file in the corresponding format
		try {

			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}

			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);

			while (it.hasNext()) {
				Map.Entry<Record, ClassificationResult<String>> pair = (Map.Entry<Record, ClassificationResult<String>>)it.next();
				String imgClass = pair.getValue().getPredictedClasses().toString();
				// Write file name and predicted class
				bw.write(pair.getKey().getID() + ".jpg " + imgClass.substring(1, imgClass.length()-1) + "\n");
				System.out.println(pair.getKey().getID() + ".jpg " + imgClass.substring(1, imgClass.length()-1));
				it.remove();
			}

			bw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

		// Print size of datasets, accuracy, error rate and time
		System.out.println();
		System.out.println("nTraining: " + nTraining);
		System.out.println("nTest    : " + nTest);	
		System.out.println(result);
		System.out.println("Time: " + t1.duration()/1000 + "s");

	}		


	// Extracts the first 10000 dense SIFT features from the images in the given dataset
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
			GroupedDataset<String, ListDataset<Record>, Record> groupedDataset, PyramidDenseSIFT<FImage> pdsift){

		List<LocalFeatureList<FloatDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatDSIFTKeypoint>>();

		// Record the list of features extracted from each image
		for (Record rec : groupedDataset) {
			FImage img = rec.getImage();
			pdsift.analyseImage(img);
			allkeys.add(pdsift.getFloatKeypoints(0.005f));
		}

		if (allkeys.size() > (int) allkeys.size()*0.2)
			allkeys = allkeys.subList(0, (int) (allkeys.size()*0.2));

		// Cluster features using K-Means
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