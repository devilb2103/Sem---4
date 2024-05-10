import java.util.Map;

import studiplayer.basic.TagReader;

public class TaggedFile extends SampledFile{
	
	private String album;
	
	TaggedFile(){
		
	}
	
	TaggedFile(String pathname) throws RuntimeException {
		super(pathname);
		
		// throw error to terminate if file path is not readable
		if(!canReadPath(pathname)) {
			throw new RuntimeException(String.format("Cannot read file with path: %s", pathname));
		}
		else {
		 	this.readAndStoreTags();
		}
		
	}
		
	public String getAlbum() {
		return this.album.strip();
	}
	
	public void readAndStoreTags() {
		Map<String, Object> tagMap = TagReader.readTags(this.getPathname());
		
		// if title and author present in metadata then replace audiofile attributes with metadata data
		// does not apply to album and duration
		
		String titleData = tagMap.containsKey("title") ? tagMap.get("title").toString().strip() : null;
		String authorData = tagMap.containsKey("author") ? tagMap.get("author").toString().strip() : null;
		String albumDat = tagMap.containsKey("album") ? tagMap.get("album").toString().strip() : null;
		String durationDat = tagMap.containsKey("duration") ? tagMap.get("duration").toString().strip() : null;
		
		if(titleData != null && titleData != "") { super.SetTitle(titleData); }
		if(authorData != null && authorData != "") { super.SetAuthor(authorData); }
		if(albumDat != null && albumDat != "") { this.album = albumDat; }
		if(durationDat != null && durationDat != "") { this.duration = Long.parseLong(durationDat); }
		
	}
	
	@Override
	public String toString() {
		
		String str = "";
		
		// fetches author and title data using getter declared in AudioFile
		if(super.getAuthor() != null && super.getAuthor().strip() != "") { str += super.getAuthor().strip() + " - "; }
		if(super.getTitle() != null && super.getTitle().strip() != "") { str += super.getTitle().strip() + " - "; }
		if(this.album != null) { str += this.album.strip() + " - "; }
		str += super.formatDuration();
		
		return str;
	}
}
