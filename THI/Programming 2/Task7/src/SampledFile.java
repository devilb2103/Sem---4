import java.io.File;

import studiplayer.basic.BasicPlayer;

public abstract class SampledFile extends AudioFile{
	
	protected String filename, pathname;
	protected long duration;
	
	SampledFile(){
		
	}
	
	SampledFile(String path){
		super(path);
		this.filename = super.getFilename();
		this.pathname = super.getPathname();
		
		if(!canReadPath(pathname)) {
			throw new RuntimeException(String.format("Cannot read file with path: %s", pathname));
		}
	}
	
	public static void main(String[] args) {
		System.out.println(SampledFile.timeFormatter(999999L));
	}
	
	protected static String timeFormatter(long timeInMicroSeconds) {
		if(timeInMicroSeconds < 0 || timeInMicroSeconds > Long.MAX_VALUE) {
			throw new RuntimeException("Invalid time format");
		}
		else {
			long total_seconds = timeInMicroSeconds / 1000000;
			long minutes = Math.floorDiv(total_seconds, 60);
			
			if(minutes > 99) {
				throw new RuntimeException("Invalid time format");
			}
			long seconds = total_seconds % 60;
			return String.format("%02d:%02d", minutes, seconds);
		}
	}
	
	public void play() {
		BasicPlayer.play(this.pathname);
	}
	
	public void togglePause() {
		BasicPlayer.togglePause();
	}
	
	public void stop() {
		BasicPlayer.stop();
	}
	
	public String formatDuration() {
		return timeFormatter(this.duration);
	}
	
	public String formatPosition() {
		return timeFormatter(BasicPlayer.getPosition());
	}
	
	protected long getDuration() {
		return this.duration;
	}
	
	protected boolean canReadPath(String path) {
		return new File(path).canRead();
	}
}
