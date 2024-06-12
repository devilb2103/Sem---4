package studiplayer.audio;

import java.util.Comparator;

public class DurationComparator implements Comparator<AudioFile> {
    @Override
    public int compare(AudioFile o1, AudioFile o2) {
    	
    	Long duration1 = (o1 instanceof TaggedFile) ? TaggedFile.class.cast(o1).getDuration() : ((o1 instanceof WavFile ? WavFile.class.cast(o1).getDuration() : 0l));
    	Long duration2 = (o2 instanceof TaggedFile) ? TaggedFile.class.cast(o2).getDuration() : ((o2 instanceof WavFile ? WavFile.class.cast(o2).getDuration() : 0l));
        
        System.out.printf("\n%d - %d\n", duration1, duration2);
    	
        // when one of the albums from wav file are null
        if (duration1 == null && duration2 == null) {
            return 0;
        } else if (duration1 == null) {
            return -1;
        } else if (duration2 == null) {
            return 1;
        }

     // when both albums from wav file are stored
        return duration1.compareTo(duration2);
    }
}
