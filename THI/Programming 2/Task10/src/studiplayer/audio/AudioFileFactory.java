package studiplayer.audio;

public class AudioFileFactory {
	
	public static AudioFile createAudioFile(String path) throws NotPlayableException {
		// find and store extension of path
		String pathSplit[] = path.split("\\.");
		String extension = pathSplit[pathSplit.length - 1].toLowerCase();

		// return taggedfile class instance if extension is mp3 or ogg
		// return wavfile class instance of extension is wav
		// throw runtime exception otherwise
		if(extension.equals("mp3") || extension.equals("ogg")) { return new TaggedFile(path); }
		else if(extension.equals("wav")) { return new WavFile(path); }
		else { throw new NotPlayableException(path, String.format("Unknown suffix for AudioFile \"%s\"", path)); }
	}

}
