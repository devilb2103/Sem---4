package studiplayer.audio;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class PlayList implements Iterable<AudioFile> {

	private LinkedList<AudioFile> playlist = new LinkedList<AudioFile>();
	
	private String search = "";
	private SortCriterion sortCriterion = SortCriterion.DEFAULT;
	
	private ControllablePlayListIterator iter;
	private AudioFile currAudioFile;
	
	
	// constructors
	public PlayList() {
	}

	public PlayList(String path) {
		this.loadFromM3U(path);
	}

	
	// functional base class methods ====================================================================================================
	public void add(AudioFile file) {
		this.playlist.add(file);
		this.iter = new ControllablePlayListIterator(this.playlist, this.search, this.sortCriterion);
		resetAudioFile();
	}
	
	public void remove(AudioFile file) {
		this.playlist.remove(file);
		this.iter = new ControllablePlayListIterator(this.playlist, this.search, this.sortCriterion);
		resetAudioFile();
	}
	
	public int size() { return playlist.size(); }
	
	public void nextSong() {
		// check if playlist not initialized
		if(iter == null) { return; }
		
		// normal skip 0 index to prevent playing
		// first song twice when next is pressed
		if(this.iter.iterator == 0) {
			this.iter.iterator = 1;
		}
		this.currAudioFile = iter.next();
	}
	
	public void saveAsM3U(String path) throws RuntimeException {
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
	
	public void loadFromM3U(String path) {
		List<String> lines = new ArrayList<>();
		Scanner scanner = null;
		
		// read paths and store it
		try {
			
			scanner = new Scanner(new File(path));
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String lineContent = line;
				
				if (!lineContent.strip().isEmpty() && lineContent.strip().charAt(0) != '#') {
	                lines.add(lineContent);
	            }
			}
		
		} catch (Exception e) { 
			throw new RuntimeException(e); 
			} 
		
		finally {
			try {
				scanner.close();
				}
			catch (Exception e) { }
		}
		
		// load all audiofiles
		if(lines.size() > 0) {
			this.playlist.clear();
		}
		
		for (String filepath : lines) {
			try {
				this.add(AudioFileFactory.createAudioFile(filepath));

			} catch (NotPlayableException e) {
				e.printStackTrace();
	            System.out.printf("=== Skipping file %s due to NotPlayableException ===\n", filepath);

			}

		}
	}
	
	public void jumpToAudioFile(AudioFile audioFile) { this.currAudioFile = iter.jumpToAudioFile(audioFile); }
	
	
	// getters ====================================================================================================
	public LinkedList<AudioFile> getList(){ return this.playlist; }
	public AudioFile currentAudioFile() { return this.currAudioFile; }
	public String getSearch() { return this.search; }
	public SortCriterion getSortCriterion() {
		return this.sortCriterion;
	}
	
	
	
	// setters ====================================================================================================
	public void setSearch(String query) {
		this.search = query;
		this.iter = new ControllablePlayListIterator(this.playlist, this.search, this.sortCriterion);
		resetAudioFile();
	}
	public void setSortCriterion(SortCriterion sortCriterion) {
		this.sortCriterion = sortCriterion;
		this.iter = new ControllablePlayListIterator(this.playlist, this.search, this.sortCriterion);
		resetAudioFile();
	}
	
	private void resetAudioFile() {
		if(iter.files.size() > 0) {
			
			this.currAudioFile = iter.files.get(0);
		}
		else {
			this.currAudioFile = null;
		}
	}
	
	@Override
	public String toString() {
		List<AudioFile> files = new ArrayList<>();
		AudioFile first = this.currentAudioFile();
		files.add(first);
		AudioFile cur;
		do {
			this.nextSong();
			cur = this.currentAudioFile();
			if(cur != first) {
				files.add(cur);
			}
		} while (cur != first);
		
		return files.toString();
	}

	@Override
	public Iterator<AudioFile> iterator() {
		return new ControllablePlayListIterator(this.playlist, this.search, this.sortCriterion);
	}
	
}
