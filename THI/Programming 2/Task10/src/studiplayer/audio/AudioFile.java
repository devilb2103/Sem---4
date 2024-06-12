package studiplayer.audio;
public abstract class AudioFile {
	
	private String pathname = "";
	private String filename = "";
	private String author = "";
	private String title = "";
	
	// constructors
	AudioFile(){
		// Utils.emulateLinux();
	}
	
	protected AudioFile(String pathname) throws NotPlayableException {
		this.pathname = pathname;
		parsePathname(this.getPathname());
		parseFilename(this.getFilename());
	}
	
	// abstract methods ====================================================================================================
	public abstract void play() throws NotPlayableException;
	public abstract void togglePause();
	public abstract void stop();
	public abstract String formatDuration();
	public abstract String formatPosition();
	
	
	
	// functional base class methods ====================================================================================================
	public void parsePathname(String path) {
		
		// edge case checks
		if(path.stripTrailing() == "") {
			this.pathname = "";
			return;
		}
		
		
		// left and right stripping
		String oldPath = path.stripTrailing();
		if(oldPath.stripLeading().toCharArray()[0] != '\\' || oldPath.stripLeading().toCharArray()[0] != '/') oldPath = oldPath.stripLeading();
		
		String fileName = "";
		
		// check OS and parse appropriately
		if(isWindows()) {
			
			// Windows path parsing
  			// System.out.println("OS: Windows");
			
			// replace wrong and repeating slashes by platform
			while(oldPath.contains("/")) oldPath = oldPath.replace("/", "\\");
			while(oldPath.contains("\\\\")) oldPath = oldPath.replace("\\\\", "\\");
			
			// extract filename
			String pathElements[] = oldPath.split("\\\\");
			if(oldPath.toCharArray()[oldPath.toCharArray().length - 1] != '\\') fileName = pathElements[pathElements.length - 1];
						
			// store file path and name
			this.pathname = oldPath;
			this.filename = fileName;
			
		}
		else {

			// Linux path parsing
			// System.out.println("OS: Linux");
			
			// replace wrong and repeating slashes by platform
			while(oldPath.contains("\\")) oldPath = oldPath.replace("\\", "/");
			while(oldPath.contains("//")) oldPath = oldPath.replace("//", "/");
			
			// extract filename
			boolean endsWithSlash = false;
			String pathElements[] = oldPath.split("/");
			if(oldPath.toCharArray()[oldPath.toCharArray().length - 1] != '/') fileName = pathElements[pathElements.length - 1];
			else endsWithSlash = true;

			// handle file path format for drive when specified in global path
			for(int i = 0; i < pathElements.length; i++) {
				if(pathElements[i].contains(":")) {
					pathElements[i] = "/" + pathElements[i].split(":")[0];
				}
			}

			// rejoin path terms and add extra slash in end if present originally
			oldPath = String.join("/", pathElements);		
			if(endsWithSlash == true) oldPath += "/";
						
			this.pathname = oldPath;
			this.filename = fileName;
		}
	}
	
	public void parseFilename(String filename) {
		// handle edge case
		if(filename.strip() == "") return;

		// remove extension only if it exists
		String name = filename;
		if(name.contains(".")) name = filename.strip().substring(0, filename.lastIndexOf("."));

		// split by "-" and find out author and title
		String author = "", title = "";
		if(name.contains(" -")) {
			String dat[] = name.split(" -");
			author = dat[0].strip();
			title = dat[1].strip();
		}
		else {
			title = name.strip();
		}
		
		this.author = author.strip();
		this.title = title.strip();
		
	}
	
	// getters ====================================================================================================
	public String getPathname() {
		return pathname;
	}
	
	public String getFilename() {
		return filename;
	}
	
	public String getAuthor() {
		return author;
	}
	
	public String getTitle() {
		return title;
	}
	
	// setters ====================================================================================================
	protected void SetAuthor(String author) {
		this.author = author;
	}
	
	protected void SetTitle(String title) {
		this.title = title;
	}
	
	// miscellaneous ====================================================================================================
	public String toString() {
		String author = getAuthor();
		String title = getTitle();

		// return only title if author is non-existent
		if(author != "") {
			return String.format("%s - %s", author, title);
		}
		else {
			return title;
		}
	}
	
	// find out current working platform
	// helps write cross - platform compatible code???
	private boolean isWindows() {
		 return System.getProperty("os.name").toLowerCase()
		 .indexOf("win") >= 0;
	}
	
}
