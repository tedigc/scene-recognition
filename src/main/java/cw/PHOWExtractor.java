package cw;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

//Used to train classifier - computes 4 histograms across an image
class PHOWExtractor implements FeatureExtractor<DoubleFV, Record> {

	
	PyramidDenseSIFT<FImage> pdsift;
	HardAssigner<float[], float[], IntFloatPair> assigner;

	
	public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<float[], float[], IntFloatPair> assigner){

		this.pdsift = pdsift;
		this.assigner = assigner;
	}
	
	@Override
	public DoubleFV extractFeature(Record object) {

		FImage image = object.getImage();
		pdsift.analyseImage(image);

		// Assign each dense SIFT feature to a visual word
		BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

		// Compute spatial histograms
		PyramidSpatialAggregator<float[], SparseIntFV> spatial = new PyramidSpatialAggregator<float[], SparseIntFV>(bovw, 4);

		// Append and normalize the resultant spatial histograms
		return spatial.aggregate(
				pdsift.getFloatKeypoints(0.015f), 
				image.getBounds()
			).normaliseFV();
	}
	

}