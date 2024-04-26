import studiplayer.basic.WavParamReader;

public class WavFile extends SampledFile{
	
	WavFile() {

	}
	
	WavFile(String path) {
		super(path);
		
		if(!canReadPath(pathname)) {
			throw new RuntimeException(String.format("Cannot read file with path: %s", pathname));
		}
		
		this.readAndSetDurationFromFile();
	}
	
	protected static long computeDuration(long numberOfFrames, float frameRate) {
	    return (long) ((numberOfFrames / frameRate) * 1000000);
	}

	
	private void readAndSetDurationFromFile() {
		new WavParamReader();
		WavParamReader.readParams(pathname);
		duration = computeDuration(WavParamReader.getNumberOfFrames(), WavParamReader.getFrameRate());
	}
	
	@Override
	public String toString() {
//		System.out.println(super.getTitle().split("."));
//		System.out.println(String.format("%s - %s", super.getFilename(), super.formatDuration()));
		String title_pieces[] = super.filename.split("\\.");
		String this_title = title_pieces[0];
		return String.format("%s - %s", this_title, super.formatDuration());
	}
}
