package studiplayer.audio;

public class NotPlayableException extends Exception {

    public NotPlayableException(String pathname, String msg) {
        super("Path: " + pathname + ", Message: " + msg);
    }

    public NotPlayableException(String pathname, Throwable t) {
        super("Path: " + pathname + ", Cause: " + t.toString(), t);
    }

    public NotPlayableException(String pathname, String msg, Throwable t) {
        super("Path: " + pathname + ", Message: " + msg + ", Cause: " + t.toString(), t);
    }

}
