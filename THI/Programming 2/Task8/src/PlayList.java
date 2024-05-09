import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;

public class PlayList {

	private LinkedList<AudioFile> playlist = new LinkedList<AudioFile>();
	private int current = 0;
	
	public static void main(String[] args) {
		PlayList pl = new PlayList();
		System.out.println(pl.current);
	}
	
	// constructors
	PlayList() {
		// TODO Auto-generated constructor stub
	}
	
	// functional base class methods ====================================================================================================
	public void add(AudioFile file) {
		this.playlist.add(file);
	}
	
	public void remove(AudioFile file) { this.playlist.remove(file); }
	
	public int size() { return playlist.size(); }
	
	public void nextSong() {
		if(this.current >= playlist.size() - 1) { current = 0; } else { current += 1; }
	}
	
	public void saveAsM3U(String path) {
		FileWriter writer = null;
		String sep = System.getProperty("line.separator");
		
		// read paths and store in list
		String lines[] = new String[this.size()];
		int iter = 0;
		for (AudioFile file : this.playlist) {
			lines[iter] = file.getPathname();
			iter++;
		}
		
		// write list with paths to m3u file
		try {
			writer = new FileWriter(path);
			for (String line : lines) { writer.write(line + sep); }
		} catch (IOException e) {
			throw new RuntimeException("Unable to write file " + path + "!");
		} finally {
			try { writer.close(); } catch (Exception e) {}
		}
	}
	
	
	
	
	// getters ====================================================================================================
	public LinkedList<AudioFile> getList(){ return this.playlist; }
	
	public int getCurrent() { return this.current; }
	
	public AudioFile currentAudioFile() {
		if(this.current < this.playlist.size()) {
			return playlist.get(this.current);
		}
		return null;
	}
	
	// setters ====================================================================================================
	public void setCurrent(int position) { this.current = position; }
	
}
