package studiplayer.audio;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class ControllablePlayListIterator implements Iterator<AudioFile> {

	public List<AudioFile> files;
	public int iterator;
 	
	public ControllablePlayListIterator(List<AudioFile> files) {
		this.files = files;
		this.iterator = 0;
	}
	
	public ControllablePlayListIterator(List<AudioFile> files, String search, SortCriterion sortCriterion) {
        List<AudioFile> filteredAndSortedFiles = new ArrayList<>();

        // filter songs
        if (search != null && !search.isEmpty()) {
            for (AudioFile file : files) {
                if (file.getAuthor().toLowerCase().contains(search.toLowerCase()) ||
                    file.getTitle().toLowerCase().contains(search.toLowerCase()) ||
                    (file instanceof TaggedFile && ((TaggedFile) file).getAlbum() != null && ((TaggedFile) file).getAlbum().toLowerCase().contains(search.toLowerCase()))) {
                    filteredAndSortedFiles.add(file);
                }
            }
        } else {
            filteredAndSortedFiles.addAll(files);
        }
        
        // sort songs
        switch (sortCriterion) {
            case AUTHOR:
                filteredAndSortedFiles.sort(new AuthorComparator());
                break;
            case TITLE:
                filteredAndSortedFiles.sort(new TitleComparator());
                break;
            case ALBUM:
                filteredAndSortedFiles.sort(new AlbumComparator());
                break;
            case DURATION:
                filteredAndSortedFiles.sort(new DurationComparator());
                break;
            default:
                break;
        }

        // save sequence in memory
        this.files = filteredAndSortedFiles;
        this.iterator = 0;
	}
	

	// checks if current iterator'th element exists
	 @Override
    public boolean hasNext() {
        return this.iterator < this.files.size();
    }

	 // returns iterator'th file and increments iterator position
	 public AudioFile next() {
	    if(hasNext()) {
	    	iterator += 1;
	        return files.get(this.iterator - 1);
	    }
	    else {
	    	iterator = 0;
	        return files.get(this.iterator++);
	    }
	}

	
	public AudioFile jumpToAudioFile(AudioFile file) {
		/* let file = a
			file + 1 = b
			file + 2 = c
			
			we want to place iterator at file b
			 and return file a
		*/
		int index = this.files.indexOf(file);
		if (index == -1) {
		    return null;
		} else {
		    this.iterator = index + 1;
		    return file;
		}
	}

}
