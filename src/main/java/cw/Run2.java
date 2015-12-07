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
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;


public class Run2 extends Run {


	File assignerCache = new File("run2_assigner");

	@Override
	public void run() {

		splitDataset();

		Timer t1 = Timer.timer();

		DensePatchEngine engine = new DensePatchEngine(4, 8);
		HardAssigner<float[], float[], IntFloatPair> assigner = readOrTrainAssigner(engine, (int) Math.round(0.5*nTraining));
		FeatureExtractor<DoubleFV, Record> extractor = new DensePatchFeatureExtractor(assigner, engine);

		// Construct and train a linear classifier
		LiblinearAnnotator<Record, String> annotator = new LiblinearAnnotator<Record, String>(
				extractor, 
				Mode.MULTICLASS, 
				SolverType.L2R_L2LOSS_SVC, 
				1.0, 
				0.00001
				);

		((LiblinearAnnotator<Record, String>) annotator).train(training);

		ClassificationEvaluator<CMResult<String>, String, Record> eval =
				new ClassificationEvaluator<CMResult<String>, String, Record>(
						annotator, 
						test, 
						new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

		Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		TreeMap<Record, ClassificationResult<String>> sortedGuesses = new TreeMap<Record, ClassificationResult<String>>();
		sortedGuesses.putAll(guesses);

		Iterator it = sortedGuesses.entrySet().iterator();
		
		File file = new File("run2.txt");

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
				System.out.println(pair.getKey().getID() + ".jpg " + imgClass.substring(1, imgClass.length()-1));
				it.remove(); // avoids a ConcurrentModificationException
			}

			bw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println();
		System.out.println("nTraining: " + nTraining);
		System.out.println("nTest    : " + nTest);	
		System.out.println(result);
		System.out.println("Time: " + t1.duration()/1000 + "s");
		
	}

	// Extracts the first 10000 dense SIFT features from the images in the given dataset
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
			GroupedDataset<String, ListDataset<Record>, Record> groupedDataset, Engine<FloatKeypoint, FImage> engine){

		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();

		for (Record rec: groupedDataset) {
			allkeys.add(engine.findFeatures(rec.getImage()));
		}

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
		FloatCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}

	// Attempts to read the HardAssigner from the cache, or trains one if this can't be done.
	private HardAssigner<float[], float[], IntFloatPair> readOrTrainAssigner(DensePatchEngine engine, int nSamples) {

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
			assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, nSamples), engine);
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