package studiplayer.ui;

import java.io.File;
import java.net.URL;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import studiplayer.audio.AudioFile;
import studiplayer.audio.NotPlayableException;
import studiplayer.audio.PlayList;
import studiplayer.audio.SampledFile;
import studiplayer.audio.SortCriterion;


public class Player extends Application{
	
	public static final String DEFAULT_PLAYLIST = "playlists/DefaultPlayList.m3u";
	public static final String TEST_PLAYLIST = "playlists/playList.cert.m3u";
	private static final String PLAYLIST_DIRECTORY = "playlists";
    private static final String INITIAL_PLAY_TIME_LABEL = "00:00";
    private static final String NO_CURRENT_SONG = " - ";
	
	private boolean useCertPlayList = false;
	public void setUseCertPlayList(boolean state) { this.useCertPlayList = state; }
	
	private PlayList playList;
	
	// UI
	private Button playButton;
	private Button pauseButton;
	private Button stopButton;
	private Button nextButton;
	private Label playListLabel;
	private Label playTimeLabel;
	private Label currentSongLabel;
	private ChoiceBox<SortCriterion> sortChoiceBox;
	private TextField searchTextField;
	private Button filterButton;
	private TableView<Song> songTable;
	
	private PlayerThread playerThread;
	private TimerThread timerThread;

	public Player() {
	}
	
	public static void main(String[] args) {
		launch();
	}

	@Override
	public void start(Stage primaryStage) throws Exception {

		// playlist filepicker when start up
		if (useCertPlayList) {
	        playList = new PlayList(DEFAULT_PLAYLIST);
	    } else {
	    	// playList = new PlayList(TEST_PLAYLIST);
	        FileChooser fileChooser = new FileChooser();
	        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("M3U Files", "*.m3u"));
	        File selectedFile = fileChooser.showOpenDialog(primaryStage);
	        loadPlayList(selectedFile.getPath());
	    }
		
		// load app

		BorderPane mainPane = new BorderPane();
		Scene scene = new Scene(mainPane, 600, 400);
		
		// Create Search and Filter UI
        TitledPane filterPane = filterPane();
        mainPane.setTop(filterPane);
        
        songTable = createSongTable();
        mainPane.setCenter(songTable);
        
        VBox contolPanel = createControlPanel();
        mainPane.setBottom(contolPanel);
        
        primaryStage.setScene(scene);
        primaryStage.setTitle("APA Player");
		primaryStage.show();
		
	}
	
	private TitledPane filterPane() {
        
        GridPane filterGrid = new GridPane();
        filterGrid.setHgap(10);
        filterGrid.setVgap(5);

        // Search UI
        Label searchLabel = new Label("Search text");
        searchTextField = new TextField();
        filterButton = new Button("Display");

        // Filter dropdown UI
        Label sortLabel = new Label("Sort by:");
        sortChoiceBox = new ChoiceBox<SortCriterion>();
        sortChoiceBox.setPrefWidth(200);
        sortChoiceBox.getItems().addAll(SortCriterion.values());
        sortChoiceBox.getSelectionModel().select(SortCriterion.DEFAULT);
        
        filterButton.setOnAction(e -> {
        	if(this.playList == null) {return;}
			this.playList.setSearch(searchTextField.getText());
			this.playList.setSortCriterion(sortChoiceBox.getValue());
			updateSongTable();
		});
        
        
        filterGrid.add(searchLabel, 0, 0);
        filterGrid.add(searchTextField, 1, 0);
        filterGrid.add(sortLabel, 0, 1);
        filterGrid.add(sortChoiceBox, 1, 1);
        filterGrid.add(filterButton, 2, 1);

        TitledPane filterPane = new TitledPane("Filter", filterGrid);
        return filterPane;
    }
	
	private TableView<Song> createSongTable() {
		if(this.playList == null) { return null; }
        TableView<Song> songTable = new SongTable(this.playList);
        ((SongTable) songTable).setRowSelectionHandler(this);
        return songTable;
    }
	
	private void updateSongTable() {
        if (songTable != null) {
            ((SongTable) songTable).refreshSongs();
        }
    }
	
	private VBox createControlPanel() {
		
		VBox bottomBox = new VBox();
        bottomBox.setSpacing(10);
        bottomBox.setPadding(new Insets(10));
        bottomBox.setAlignment(Pos.CENTER);

        GridPane currentSongGrid = new GridPane();
        currentSongGrid.setHgap(10);
        currentSongGrid.setVgap(5);
        currentSongGrid.setAlignment(Pos.CENTER);

        playListLabel = new Label("No Playlist Loaded");
        currentSongLabel = new Label(NO_CURRENT_SONG);
        playTimeLabel = new Label(INITIAL_PLAY_TIME_LABEL);

        currentSongGrid.add(new Label("Playlist"), 0, 0);
        currentSongGrid.add(playListLabel, 1, 0);
        currentSongGrid.add(new Label("Current Song"), 0, 1);
        currentSongGrid.add(currentSongLabel, 1, 1);
        currentSongGrid.add(new Label("Playtime"), 0, 2);
        currentSongGrid.add(playTimeLabel, 1, 2);
        currentSongGrid.setAlignment(Pos.BOTTOM_LEFT);

        HBox controlBox = new HBox();
        controlBox.setSpacing(0);
        controlBox.setAlignment(Pos.CENTER);

        playButton = createButton("play.jpg");
        pauseButton = createButton("pause.jpg"); pauseButton.setDisable(true);
        stopButton = createButton("stop.jpg"); stopButton.setDisable(true);
        nextButton = createButton("next.jpg");

        
     // In your initialization method
        playButton.setOnAction(e -> playCurrentSong(this.playList.currentAudioFile()));
        pauseButton.setOnAction(e -> pauseCurrentSong());
        stopButton.setOnAction(e -> stopCurrentSong(this.playList.currentAudioFile()));
        nextButton.setOnAction(e -> nextSong());

        controlBox.getChildren().addAll(playButton, pauseButton, stopButton, nextButton);

        bottomBox.getChildren().addAll(currentSongGrid, controlBox);
        return bottomBox;
	}
	
	private Button createButton(String iconfile) {
        Button button = null;
        try {
            URL url = getClass().getResource("/icons/" + iconfile);
            Image icon = new Image(url.toString());
            ImageView imageView = new ImageView(icon);
            imageView.setFitHeight(20);
            imageView.setFitWidth(20);
            button = new Button("", imageView);
            button.setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
            button.setStyle("-fx-background-color: #fff;");
        } catch (Exception e) {
            System.out.println("Image " + "icons/" + iconfile + " not found!");
            System.exit(-1);
        }
        return button;
    }
	
	private void playCurrentSong(AudioFile currentAudioFile) {
		if(this.playerThread == null) {	
			this.playerThread = new PlayerThread();
			this.playerThread.start();
		}

		if(this.timerThread == null) {	
			this.timerThread = new TimerThread();
			this.timerThread.start();
		}
		
		
		System.out.println("Playing " + currentAudioFile);
    	updateSongInfo(currentAudioFile);
    	System.out.println(this.timerThread);
    	setButtonStates(false, true, true, true);
	}

	private void pauseCurrentSong() {
		if(this.playerThread != null) {
			this.playerThread.currentAudioFile.togglePause();			
		}
		
		if(this.timerThread != null) {
			this.timerThread.paused = !timerThread.paused;
		}
	    System.out.println("Pausing " + playList.currentAudioFile().toString());
	}


	private void stopCurrentSong(AudioFile currentAudioFile) {
		if(this.playerThread != null) {			
			System.out.println("Stopping " + currentAudioFile.toString());
			this.playerThread.terminate();
		}
		
		if(this.timerThread != null) {			
			this.timerThread.terminate();
			timerThread = null;
		}
	    updateSongInfo(null);
	    setButtonStates(true, false, false, true);
	}


	private void nextSong() {
		AudioFile currentAudioFile = this.playList.currentAudioFile();
		stopCurrentSong(currentAudioFile);
	    playList.nextSong();
	    currentAudioFile = this.playList.currentAudioFile();
	    playCurrentSong(currentAudioFile);
	    System.out.println("Playing next song: " + currentAudioFile.toString());
	}

	private void setButtonStates(boolean play, boolean pause, boolean stop, boolean next) {
	    playButton.setDisable(!play);
	    pauseButton.setDisable(!pause);
	    stopButton.setDisable(!stop);
	    nextButton.setDisable(!next);
	}
	
	public void playSelectedSong(Song selectedSong) {
        AudioFile currentAudioFile = selectedSong.getAudioFile();
        stopCurrentSong(this.playList.currentAudioFile());
        this.playList.jumpToAudioFile(currentAudioFile);
        playCurrentSong(currentAudioFile);
        System.out.println("Playing selected song: " + currentAudioFile.toString());
    }

	private void updateSongInfo(AudioFile af) {
	    Platform.runLater(() -> {
	        if (af == null) {
	            currentSongLabel.setText("No song playing");
	            playTimeLabel.setText("00:00");
	        } else {
	            currentSongLabel.setText(af.getTitle());
	            playTimeLabel.setText("00:00");
	        }
	    });
	}

	private void updatePlayTime(int milliseconds) {
	    if (playerThread != null) {
	        int currentTimeInSeconds = milliseconds / 1000; // Assuming getPosition() returns position in milliseconds
	        int minutes = currentTimeInSeconds / 60;
	        int seconds = currentTimeInSeconds % 60;
	        String formattedTime = String.format("%02d:%02d", minutes, seconds);
	        Platform.runLater(() -> playTimeLabel.setText(formattedTime));
	    }
	}
	
	public void loadPlayList(String pathname) {
		if(pathname == null || pathname.strip().isEmpty()) {
			this.playList = new PlayList(Player.DEFAULT_PLAYLIST);
		}
		else {			
			PlayList tempPl = new PlayList(pathname);
			this.playList = tempPl;
		}
	}

	private class PlayerThread extends Thread {
        private boolean stopped = false;
        public AudioFile currentAudioFile;

        public void terminate() {
            this.stopped = true;
            currentAudioFile.stop();
            playerThread = null;
        }

        @Override
        public void run() {
            while (!stopped) {
                currentAudioFile = playList.currentAudioFile();
                if (currentAudioFile != null) {
                    try {
                        currentAudioFile.play();
//                        playList.nextSong();
                    } catch (NotPlayableException e) {
                        e.printStackTrace();
                    }
                }
                if (stopped) break;
            }
            
            System.out.println("Player thread stopped ");
        }
    }
	
	private class TimerThread extends Thread {
	    private boolean stopped = false;
	    private boolean paused = false;
	    private final int updateInterval = 300; // Update interval in milliseconds
	    private int millisecondCounter = 0;

	    public void terminate() {
	        stopped = true;
	    }

	    @Override
	    public void run() {
	        while (!stopped) {
	            if (playList != null && playList.currentAudioFile() != null) {
	                if(!paused) {
	                	millisecondCounter += updateInterval;
	                	updatePlayTime(millisecondCounter); // Update the playback time
                	}
	                
	                // Sleep for the update interval
	                try {
	                    Thread.sleep(updateInterval);
	                } catch (InterruptedException e) {
	                    e.printStackTrace();
	                }
	            }
	        }
	        System.out.println("Timer thread stopped");
	    }
	}

	



}


