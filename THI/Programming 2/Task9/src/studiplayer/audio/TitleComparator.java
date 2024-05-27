package studiplayer.audio;

import java.util.Comparator;

public class TitleComparator implements Comparator<AudioFile> {
    @Override
    public int compare(AudioFile o1, AudioFile o2) {
    	String title1 = o1.getTitle();
    	String title2 = o2.getTitle();
    	
    	if (title1 == null && title2 == null) {
    		return 0;
    	} 
    	else if (title1 == null) {
    		return -1;
    	}
    	else if (title2 == null) {
    		return 1;
		}
    	
    	return title1.compareTo(title2);
    }
}
