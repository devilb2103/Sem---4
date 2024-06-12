package studiplayer.audio;
import studiplayer.basic.WavParamReader;

public class WavFile extends SampledFile{
	
	public WavFile() {

	}
	
	public WavFile(String pathname) throws NotPlayableException {
		super(pathname);
		
		// throw error to terminate if file path is not readable
		if(!canReadPath(pathname)) {
			throw new NotPlayableException(pathname, String.format("Cannot read file with path: %s", pathname));
		}
		else {			
			this.readAndSetDurationFromFile();
		}
		
	}
	
	// protected class that allows use of method by tests
	public static long computeDuration(long numberOfFrames, float frameRate) {
	    return (long) ((numberOfFrames / frameRate) * 1000000);
	}

	
	// reading wavfile metadata and computing duration
	public void readAndSetDurationFromFile() throws NotPlayableException{
		try {
			new WavParamReader();
			WavParamReader.readParams(super.getPathname());
			duration = computeDuration(WavParamReader.getNumberOfFrames(), WavParamReader.getFrameRate());
		} catch (Exception e) {
			throw new NotPlayableException(super.getPathname(), "Error reading WAV file parameters");
		}
	}
	
	@Override
	public String toString() {
		String title = super.getFilename().split("\\.")[0];
		return String.format("%s - %s", title, super.formatDuration());
	}
}
