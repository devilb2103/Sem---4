package studiplayer.audio;

import java.util.Comparator;

public class AlbumComparator implements Comparator<AudioFile> {
    @Override
    public int compare(AudioFile o1, AudioFile o2) {

        // case when either of two or both are not tagged file (don't have album data stored)
    	if (!(o1 instanceof TaggedFile) || !(o2 instanceof TaggedFile)) {
            return o1 instanceof TaggedFile ? 1 : (o2 instanceof TaggedFile ? -1 : 0);
        }

        // both of them have album data
        String album1 = TaggedFile.class.cast(o1).getAlbum();
        String album2 = TaggedFile.class.cast(o2).getAlbum();

        // when one of the albums from wav file are null
        if (album1 == null && album2 == null) {
            return 0;
        } else if (album1 == null) {
            return -1;
        } else if (album2 == null) {
            return 1;
        }

        // when both albums from wav file are stored
        return album1.compareToIgnoreCase(album2);
    }
}
