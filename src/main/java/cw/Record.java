package cw;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;

public class Record implements Identifiable, ImageProvider<FImage>, Comparable<Record> {
	
	
	private String id;
	private int intID;
	private FImage img;
	private String imgClass;
	
	
	public Record(String id, FImage img, String imgClass) {
		
		this.id = id;
		this.intID = Integer.valueOf(id.split("\\.")[0]);
		this.img = img;
		this.imgClass = imgClass;
	}
	
	@Override
	public FImage getImage() {
		
		return this.img;
	}

	@Override
	public String getID() {

		return String.valueOf(this.id);
	}
	
	public String getImgClass() {

		return this.imgClass;
	}

	@Override
	public int compareTo(Record o) {
		
		return this.intID - o.intID;
		
	}

}
