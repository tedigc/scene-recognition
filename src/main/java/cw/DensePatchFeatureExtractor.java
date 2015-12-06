package cw;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

class DensePatchFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

	DensePatchEngine engine;
	HardAssigner<float[], float[], IntFloatPair> assigner;

	public DensePatchFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, DensePatchEngine engine){

		this.engine = engine;
		this.assigner = assigner;
	}

	public DoubleFV extractFeature(FImage image) {

		// Assign each dense SIFT feature to a visual word
		BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

		// Compute spatial histograms
		BlockSpatialAggregator<float[], SparseIntFV> spatial = 
				new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
		
		// Append and normalise the resultant spatial histograms
		return spatial.aggregate(engine.findFeatures(image), image.getBounds()).normaliseFV();
	}
	
	
}