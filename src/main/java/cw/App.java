package cw;

import java.io.IOException;


public class App {


	public static void main(String[] args) {
		
		Run3 run3 = new Run3();
		try {
			run3.run();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
