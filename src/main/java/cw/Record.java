package cw;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;

public class Record implements Identifiable, ImageProvider<FImage> {
	
	
	private String id;
	private FImage img;
	
	
	public Record(String id, FImage img) {
		
		this.id = id;
		this.img = img;
	}
	
	@Override
	public FImage getImage() {
		
		return this.img;
	}

	@Override
	public String getID() {

		return this.id;
	}

}
