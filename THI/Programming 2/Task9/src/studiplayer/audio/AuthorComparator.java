package studiplayer.audio;

import java.util.Comparator;

public class AuthorComparator implements Comparator<AudioFile> {
    @Override
    public int compare(AudioFile o1, AudioFile o2) {

    	String author1 = o1.getAuthor();
    	String author2 = o2.getAuthor();
    	
    	if (author1 == null && author2 == null) {
    		return 0;
    	} 
    	else if (author1 == null) {
    		return -1;
    	}
    	else if (author2 == null) {
    		return 1;
		}
    	
    	return author1.compareTo(author2);
    }
}
