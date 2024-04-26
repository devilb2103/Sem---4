import java.util.Map;

import studiplayer.basic.TagReader;

public class TaggedFile extends SampledFile{
	
//	private String title;
//	private String author;
	private String album;
	
	TaggedFile(){
		
	}
	
	TaggedFile(String path){
		super(path);
		
		if(!canReadPath(pathname)) {
			throw new RuntimeException(String.format("Cannot read file with path: %s", pathname));
		}
		
		readAndStoreTags();
	}
	
	public static void main(String[] args) {
		TaggedFile f3 = new TaggedFile("audiofiles/beethoven-ohne-album.mp3");
		System.out.println(f3.toString());
	}
	
	public String getAlbum() {
		return this.album.strip();
	}
	
	// override to previous get author since here author is split 
	// by _ and not - which is not detectable by parsing logic in Audiofile.java
	@Override
	public String getTitle() {
		return super.getTitle();
	}
	
	// override getAuthor method since here author is split 
	// by _ and not - which is not detectable by the parsing logic in Audiofile.java
	@Override
	public String getAuthor() {
		return super.getAuthor();
	}
	
	public void readAndStoreTags() {
		Map<String, Object> tagMap = TagReader.readTags(this.getPathname());
		
		String keys[] = {"title", "author", "album", "duration"};
		
		// if title and author present in metadata then replace audiofile attributes with metadata data
		// does not apply to album and duration
		for(String attribute: keys) {
			if(attribute == "title") {
				if(tagMap.get(attribute) != null && tagMap.get(attribute) != "") {
					super.SetTitle(tagMap.get(attribute).toString().strip());
				}
			}
			else if(attribute == "author") {
				if(tagMap.get(attribute) != null && tagMap.get(attribute) != "") {					
					super.SetAuthor(tagMap.get(attribute).toString().strip());
				}
			}
			else if(attribute == "album" && tagMap.get(attribute) != null && tagMap.get(attribute) != "") {
				this.album = tagMap.get(attribute).toString().strip();
			}
			else if(attribute == "duration" && tagMap.get(attribute) != null) {
				this.duration = Long.parseLong(tagMap.get(attribute).toString());
			}
		}
		
	}
	
	@Override
	public String toString() {
		
		String str = "";
		
		// fetch data of author and title from audiofile class if not present in metadata
		
		if(super.getAuthor() != null && super.getAuthor().strip() != "") {
			str += super.getAuthor().strip() + " - ";
		}
		
		if(super.getTitle() != null && super.getTitle().strip() != "") {
			str += super.getTitle().strip() + " - ";
		}
		
		if(this.album != null) {
			str += this.album.strip() + " - ";
		}
		
		str += super.formatDuration();
		
		return str;
	}
}
