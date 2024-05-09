import studiplayer.basic.WavParamReader;

public class WavFile extends SampledFile{
	
	WavFile() {

	}
	
	WavFile(String pathname) {
		super(pathname);
		
		// throw error to terminate if file path is not readable
		if(!canReadPath(pathname)) {
			throw new RuntimeException(String.format("Cannot read file with path: %s", pathname));
		}
		else {			
			this.readAndSetDurationFromFile();
		}
		
	}
	
	// protected class that allows use of method by tests
	protected static long computeDuration(long numberOfFrames, float frameRate) {
	    return (long) ((numberOfFrames / frameRate) * 1000000);
	}

	
	// reading wavfile metadata and computing duration
	private void readAndSetDurationFromFile() {
		new WavParamReader();
		WavParamReader.readParams(super.getPathname());
		duration = computeDuration(WavParamReader.getNumberOfFrames(), WavParamReader.getFrameRate());
	}
	
	@Override
	public String toString() {
		String title = super.getFilename().split("\\.")[0];
		return String.format("%s - %s", title, super.formatDuration());
	}
}
