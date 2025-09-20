#!/usr/bin/env python3
# News Classifier GUI Application
import csv
import os
import logging
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import string
import platform

import networkx as nx
import pandas as pd
import matplotlib

from shared_logic import NewsClassifier
from src.page_rank import extract_domain, build_graph_from_edges, scrape_outlinks_one

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import warnings

# Setup logging
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'classifier_app_ui.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATS_DIR = os.path.join(BASE_DIR, 'data/stats')

# Create assets directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# Color palette
COLORS = {
    'background': '#f0f4f8',
    'header': '#3a86ff',
    'text': '#2b2d42',
    'button': '#3a86ff',
    'button_hover': '#4361ee',
    'real': '#38b000',
    'fake': '#d90429',
    'neutral': '#adb5bd',
    'human': '#38b000',
    'bot': '#d90429'
}

# Setup macOS configuration (since some of us worked on macs)
if platform.system() == 'Darwin':
    os.environ['PYTHONHASHSEED'] = '0'
    matplotlib.use('Agg')


class ContentClassifierUI:
    """
    GUI application for content classification (fake news and bot detection).

    A comprehensive Tkinter-based application that provides an intuitive
    interface for analyzing news articles and tweets using machine learning
    models. Supports fake news detection and bot detection capabilities
    with visual result presentation.
    """

    def __init__(self, root):
        """
        Initialize the UI and load machine learning models.

        Sets up the main application window, initializes UI components,
        and loads the necessary machine learning models for content
        classification and bot detection.

        Parameters
        ----------
        root : tkinter.Tk
            The root Tkinter window object.

        Returns
        -------
        None
            Initializes the application and displays loading screen.
        """
        self.root = root
        self.root.title("🔍 Content Analysis System")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS['background'])
        self.root.resizable(True, True)

        # Set minimum window size
        self.root.minsize(800, 600)

        self.root.iconbitmap(os.path.join(ASSETS_DIR, "icon.ico"))

        logging.info("Starting content classifier UI application...")

        # Current content type (news or tweet)
        self.current_content_type = None

        # Initialize UI frames that will be shown/hidden
        self.content_selection_frame = None
        self.news_frame = None
        self.tweet_frame = None

        # Initialize dictionary for metric entries
        self.metric_entries = {}

        # Load models in a separate thread to keep UI responsive
        self.models_loaded = False
        self.loading_error = None
        self.init_ui()
        self.root.after(100, self.load_models)

    def init_ui(self):
        """
        Set up the user interface elements.

        Creates the main UI framework including header, status bar,
        loading screen, and initializes all content frames for
        news analysis and tweet analysis.

        Returns
        -------
        None
            Initializes all UI components and displays loading screen.
        """
        # Main frame with padding
        self.main_frame = tk.Frame(self.root, bg=COLORS['background'], padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header label
        header_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        self.header_label = tk.Label(
            header_frame,
            text="Content Analysis System 🕵",
            font=("Arial", 24, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        self.header_label.pack(side=tk.LEFT)

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Loading...",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress indicator during loading
        self.loading_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        self.loading_frame.pack(fill=tk.BOTH, expand=True)

        loading_label = tk.Label(
            self.loading_frame,
            text="Loading models...",
            font=("Arial", 14),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        loading_label.pack(pady=(100, 10))

        self.progress = ttk.Progressbar(
            self.loading_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='indeterminate'
        )
        self.progress.pack()
        self.progress.start()

        # Create content selection frame
        self.create_content_selection_frame()

        # Create news and tweet frames
        self.create_news_frame()
        self.create_tweet_frame()

        # Results frame
        self.results_frame = tk.Frame(self.main_frame, bg=COLORS['background'])

        # Separator for results
        self.separator = ttk.Separator(self.main_frame, orient='horizontal')

    def create_content_selection_frame(self):
        """
        Create the frame for selecting content type (news article or tweet).

        Builds the main menu interface that allows users to choose between
        news article analysis and tweet bot detection, including descriptive
        text and navigation buttons.

        Returns
        -------
        None
            Creates and configures the content selection interface.
        """
        self.content_selection_frame = tk.Frame(self.main_frame, bg=COLORS['background'])

        # Selection label
        selection_label = tk.Label(
            self.content_selection_frame,
            text="Select content type to analyze:",
            font=("Arial", 16, "bold"),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        selection_label.pack(pady=(50, 30))

        # Button frame
        button_frame = tk.Frame(self.content_selection_frame, bg=COLORS['background'])
        button_frame.pack()

        # News article button
        news_button = tk.Button(
            button_frame,
            text="News Article Analysis",
            font=("Arial", 14),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=20,
            pady=15,
            width=20,
            command=self.show_news_frame
        )
        news_button.pack(side=tk.LEFT, padx=10)

        # Tweet button
        tweet_button = tk.Button(
            button_frame,
            text="Tweet Bot Detection",
            font=("Arial", 14),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=20,
            pady=15,
            width=20,
            command=self.show_tweet_frame
        )
        tweet_button.pack(side=tk.LEFT, padx=10)

        # Description text
        description_frame = tk.Frame(self.content_selection_frame, bg=COLORS['background'])
        description_frame.pack(pady=30, fill=tk.X)

        description_text = """
        This application provides two types of content analysis:

        1. News Article Analysis: Determines if a news article is likely real or fake based on its content and source 
        domain. 

        2. Tweet Bot Detection: Analyzes if a tweet likely comes from a bot or human account based on the content and 
        account metrics. 

        Select the appropriate option for your content.
        """

        description_label = tk.Label(
            description_frame,
            text=description_text,
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            justify=tk.LEFT,
            wraplength=700
        )
        description_label.pack()

    def create_news_frame(self):
        """
        Create the frame for news article analysis.

        Builds the news analysis interface including text input area,
        domain and URL input fields, example loading functionality,
        and control buttons for analysis and clearing.

        Returns
        -------
        None
            Creates and configures the news analysis interface.
        """
        self.news_frame = tk.Frame(self.main_frame, bg=COLORS['background'])

        # Back button
        back_button = tk.Button(
            self.news_frame,
            text="← Back",
            font=("Arial", 10),
            bg=COLORS['neutral'],
            fg="black",
            command=self.show_content_selection
        )
        back_button.pack(anchor=tk.W, pady=(0, 10))

        # News header
        news_header = tk.Label(
            self.news_frame,
            text="News Article Analysis",
            font=("Arial", 18, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        news_header.pack(fill=tk.X, pady=(0, 20))

        # Instructions
        instruction_label = tk.Label(
            self.news_frame,
            text="Enter news text to analyze:",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        instruction_label.pack(fill=tk.X, pady=(0, 10))

        # Text entry area with scrollbar
        self.text_frame = tk.Frame(self.news_frame, bg=COLORS['background'])
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        self.news_text = scrolledtext.ScrolledText(
            self.text_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            height=10,
            bg="white",
            fg=COLORS['text']
        )
        self.news_text.pack(fill=tk.BOTH, expand=True)

        # Domain input
        domain_frame = tk.Frame(self.news_frame, bg=COLORS['background'])
        domain_frame.pack(fill=tk.X, pady=(0, 10))

        domain_label = tk.Label(
            domain_frame,
            text="Enter domain (optional):",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        domain_label.pack(side=tk.LEFT)

        self.domain_entry = tk.Entry(
            domain_frame,
            font=("Arial", 12),
            bg="white",
            fg=COLORS['text'],
            width=30
        )
        self.domain_entry.pack(side=tk.LEFT, padx=(5, 0))

        # URL input
        url_frame = tk.Frame(self.news_frame, bg=COLORS['background'])
        url_frame.pack(fill=tk.X, pady=(0, 10))

        url_label = tk.Label(
            url_frame,
            text="Enter URL (optional):",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        url_label.pack(side=tk.LEFT)

        self.url_entry = tk.Entry(
            url_frame,
            font=("Arial", 12),
            bg="white",
            fg=COLORS['text'],
            width=30
        )
        self.url_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Example button
        self.example_button = tk.Button(
            self.news_frame,
            text="Load Example Text",
            font=("Arial", 10),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=10,
            command=self.load_example_news
        )
        self.example_button.pack(anchor=tk.W, pady=(0, 10))

        # Button frame
        news_button_frame = tk.Frame(self.news_frame, bg=COLORS['background'])
        news_button_frame.pack(fill=tk.X, pady=(10, 20))

        # Analyze button
        self.analyze_news_button = tk.Button(
            news_button_frame,
            text="Analyze",
            font=("Arial", 12, "bold"),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.analyze_news
        )
        self.analyze_news_button.pack(side=tk.LEFT)

        # Clear button
        self.clear_news_button = tk.Button(
            news_button_frame,
            text="Clear",
            font=("Arial", 12),
            bg=COLORS['neutral'],
            fg="black",
            activebackground="#999999",
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.clear_news
        )
        self.clear_news_button.pack(side=tk.LEFT, padx=(10, 0))

    def create_tweet_frame(self):
        """
        Create the frame for tweet bot detection.

        Builds the tweet analysis interface including tweet text input,
        comprehensive account metrics collection form, example loading
        functionality, and control buttons for analysis and clearing.

        Returns
        -------
        None
            Creates and configures the tweet bot detection interface.

        Notes
        -----
        Creates input fields for the following user metrics:
        - followers_count, friends_count, statuses_count
        - favourites_count, listed_count, screen_name
        - account_date, verified status
        """
        self.tweet_frame = tk.Frame(self.main_frame, bg=COLORS['background'])

        # Back button
        back_button = tk.Button(
            self.tweet_frame,
            text="← Back",
            font=("Arial", 10),
            bg=COLORS['neutral'],
            fg="black",
            command=self.show_content_selection
        )
        back_button.pack(anchor=tk.W, pady=(0, 10))

        # Tweet header
        tweet_header = tk.Label(
            self.tweet_frame,
            text="Tweet Bot Detection",
            font=("Arial", 18, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        tweet_header.pack(fill=tk.X, pady=(0, 20))

        # Tweet text input
        tweet_instruction = tk.Label(
            self.tweet_frame,
            text="Enter tweet text:",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        tweet_instruction.pack(fill=tk.X, pady=(0, 5))

        self.tweet_text = tk.Text(
            self.tweet_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            height=4,
            bg="white",
            fg=COLORS['text']
        )
        self.tweet_text.pack(fill=tk.X, pady=(0, 15))

        # Account metrics section
        metrics_label = tk.Label(
            self.tweet_frame,
            text="Account Metrics:",
            font=("Arial", 14, "bold"),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        metrics_label.pack(fill=tk.X, pady=(5, 10))

        # Create frames for each metric
        metrics_frame = tk.Frame(self.tweet_frame, bg=COLORS['background'])
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        # Left metrics column
        left_metrics = tk.Frame(metrics_frame, bg=COLORS['background'])
        left_metrics.pack(side=tk.LEFT, fill=tk.Y)

        # Right metrics column
        right_metrics = tk.Frame(metrics_frame, bg=COLORS['background'])
        right_metrics.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0))

        # Third metrics column (for additional indicators)
        third_metrics = tk.Frame(metrics_frame, bg=COLORS['background'])
        third_metrics.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0))

        # Create input fields for the metrics
        # Left column metrics
        self.create_metric_input(left_metrics, "Followers count:", "followers_count", "0")
        self.create_metric_input(left_metrics, "Friends/Following count:", "friends_count", "0")
        self.create_metric_input(left_metrics, "Total tweets/statuses:", "statuses_count", "0")

        # Right column metrics
        self.create_metric_input(right_metrics, "Favorites/Likes received:", "favourites_count", "0")
        self.create_metric_input(right_metrics, "Username:", "screen_name", "", is_text=True)
        self.create_metric_input(right_metrics, "Account creation date (YYYY-MM):", "account_date", "", is_text=True)

        # New human likelihood indicator metrics
        self.create_metric_input(third_metrics, "Listed count:", "listed_count", "0")

        # Create a checkbox for verified status
        verified_frame = tk.Frame(third_metrics, bg=COLORS['background'])
        verified_frame.pack(fill=tk.X, pady=5)

        verified_label = tk.Label(
            verified_frame,
            text="Verified account:",
            font=("Arial", 11),
            fg=COLORS['text'],
            bg=COLORS['background'],
            width=20,
            anchor="w"
        )
        verified_label.pack(side=tk.LEFT)

        # Create a StringVar for the Checkbutton
        self.verified_var = tk.StringVar(value="no")

        verified_check = tk.Checkbutton(
            verified_frame,
            text="Yes",
            variable=self.verified_var,
            onvalue="yes",
            offvalue="no",
            font=("Arial", 11),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        verified_check.pack(side=tk.LEFT)

        # Store the variable in the metric_entries dictionary
        self.metric_entries['verified'] = self.verified_var

        # Example button
        self.tweet_example_button = tk.Button(
            self.tweet_frame,
            text="Load Example Tweet",
            font=("Arial", 10),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=10,
            command=self.load_example_tweet
        )
        self.tweet_example_button.pack(anchor=tk.W, pady=(10, 10))

        # Button frame
        tweet_button_frame = tk.Frame(self.tweet_frame, bg=COLORS['background'])
        tweet_button_frame.pack(fill=tk.X, pady=(10, 20))

        # Analyze button
        self.analyze_tweet_button = tk.Button(
            tweet_button_frame,
            text="Analyze",
            font=("Arial", 12, "bold"),
            bg=COLORS['button'],
            fg="black",
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.analyze_tweet
        )
        self.analyze_tweet_button.pack(side=tk.LEFT)

        # Clear button
        self.clear_tweet_button = tk.Button(
            tweet_button_frame,
            text="Clear",
            font=("Arial", 12),
            bg=COLORS['neutral'],
            fg="black",
            activebackground="#999999",
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.clear_tweet
        )
        self.clear_tweet_button.pack(side=tk.LEFT, padx=(10, 0))

    def create_metric_input(self, parent, label_text, var_name, default_value, is_text=False):
        """
        Create a labeled input field for a metric.

        Creates a formatted input field with label for collecting user
        account metrics used in bot detection analysis.

        Parameters
        ----------
        parent : tkinter.Widget
            Parent widget to contain the input field.
        label_text : str
            Display text for the input field label.
        var_name : str
            Variable name identifier for storing the input widget.
        default_value : str
            Default value to populate in the input field.
        is_text : bool, default=False
            Whether the field accepts text input

        Returns
        -------
        None
            Creates and stores the input field widget in self.metric_entries.
        """
        frame = tk.Frame(parent, bg=COLORS['background'])
        frame.pack(fill=tk.X, pady=5)

        label = tk.Label(
            frame,
            text=label_text,
            font=("Arial", 11),
            fg=COLORS['text'],
            bg=COLORS['background'],
            width=20,
            anchor="w"
        )
        label.pack(side=tk.LEFT)

        entry = tk.Entry(
            frame,
            font=("Arial", 11),
            bg="white",
            fg=COLORS['text'],
            width=15
        )
        entry.pack(side=tk.LEFT)
        entry.insert(0, default_value)

        # Store the entry widget in the dictionary
        self.metric_entries[var_name] = entry

    def load_models(self):
        """
        Load machine learning models and domain data using NewsClassifier.

        Initializes the NewsClassifier instance which loads all necessary
        models for content classification and bot detection. Updates UI
        state based on loading success or failure.

        Returns
        -------
        None
            Sets self.models_loaded flag and updates UI to show content
            selection or error message.

        Raises
        ------
        Exception
            If model loading fails, sets self.loading_error and shows
            error dialog.
        """
        try:
            logging.info("Initializing NewsClassifier to load models")
            # Create an instance of NewsClassifier which will load all models
            self.classifier = NewsClassifier()

            # Store references to models that the UI needs to access
            self.vectorizer = self.classifier.vectorizer
            self.models = self.classifier.news_models
            self.bot_model = self.classifier.bot_model

            # Load domain data
            domain_stats = self.classifier.get_domain_stats()
            if domain_stats:
                # Convert domain stats to DataFrame format for compatibility
                self.domains_data = pd.DataFrame([
                    {'domain': domain, 'fake_ratio': ratio}
                    for domain, ratio in domain_stats.items()
                ])
                logging.info(f"Loaded data for {len(self.domains_data)} domains")
            else:
                self.domains_data = None
                logging.warning("No domain data was loaded")

            logging.info("All models loaded successfully")
            self.models_loaded = True

            # Update UI
            self.root.after(0, self.show_content_selection)

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            self.loading_error = str(e)
            self.root.after(0, self.show_loading_error)

    def show_content_selection(self):
        """
        Show the content type selection screen.

        Displays the main menu where users choose between news article
        analysis and tweet bot detection. Hides all other frames and
        updates the interface state.

        Returns
        -------
        None
            Updates UI to show content selection interface.
        """
        # Hide all frames
        self.hide_all_frames()

        # Update header
        self.header_label.config(text="Content Analysis System 🕵️")

        # Show content selection frame
        self.content_selection_frame.pack(fill=tk.BOTH, expand=True)

        # Update status
        self.status_bar.config(text="Select content type to analyze")

        # Reset current content type
        self.current_content_type = None

    def get_url_pagerank_score(self, user_url, graph=None, alpha=0.85):
        """
        Get PageRank score for a user-provided URL.

        Calculates PageRank score by extracting domain from URL and either
        using existing graph data or scraping outlinks to compute a
        temporary score.

        Parameters
        ----------
        user_url : str
            The URL to analyze for PageRank scoring.
        graph : networkx.Graph, optional
            Existing graph to use for PageRank calculation. If None,
            loads from domain_edges.csv.
        alpha : float, default=0.85
            Damping parameter for PageRank algorithm.

        Returns
        -------
        score : float or None
            PageRank score for the domain, None if calculation fails.
        message : str
            Descriptive message about the scoring process and result.

        Notes
        -----
        Process:
        1. Extract domain from URL
        2. Check if domain exists in existing graph
        3. If not, scrape outlinks and calculate temporary score
        4. Return score with ranking information
        """
        # Extract domain from URL
        domain = extract_domain(user_url)
        if not domain:
            return None, "Could not extract a valid domain from the URL."

        # Load existing graph if not provided
        if graph is None:
            try:
                # Try to load existing edges
                edges = []
                with open(STATS_DIR / "domain_edges.csv", "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for src, dst, label in reader:
                        edges.append((src, dst, label))
                graph = build_graph_from_edges(edges)
            except Exception as e:
                return None, f"Error loading existing graph: {str(e)}"

        # If domain already in graph, return its PageRank score
        pr = nx.pagerank(graph, alpha=alpha)
        if domain in pr:
            rank_position = sorted(pr.values(), reverse=True).index(pr[domain]) + 1
            return pr[domain], f"Domain {domain} exists in our database (rank {rank_position}/{len(pr)})"

        # If domain not in graph, fetch its outlinks and calculate temporary score
        print(f"Domain {domain} not in existing graph. Fetching outlinks...")
        try:
            # Get outlinks for this domain
            outlinks = scrape_outlinks_one(user_url)
            if not outlinks:
                return None, f"Could not fetch any outlinks for {domain}"

            # Create temporary graph with new domain and its connections
            temp_graph = graph.copy()
            for src, dst in outlinks:
                temp_graph.add_edge(src, dst)

            # Calculate new PageRank scores
            new_pr = nx.pagerank(temp_graph, alpha=alpha)

            # Return the score for our domain
            if domain in new_pr:
                rank_position = sorted(new_pr.values(), reverse=True).index(new_pr[domain]) + 1
                return new_pr[domain], f"Temporary score for {domain} (rank {rank_position}/{len(new_pr)})"
            else:
                return None, f"Domain {domain} has no connections in the graph"

        except Exception as e:
            return None, f"Error calculating PageRank: {str(e)}"

    def show_news_frame(self):
        """
        Show the news article analysis frame.

        Switches the interface to news article analysis mode, hiding
        other frames and updating the header and status.

        Returns
        -------
        None
            Updates UI to show news analysis interface.
        """
        self.hide_all_frames()
        self.header_label.config(text="News Article Analysis 📰")
        self.news_frame.pack(fill=tk.BOTH, expand=True)
        self.current_content_type = "news"
        self.status_bar.config(text="Enter a news article to analyze")

    def show_tweet_frame(self):
        """
        Show the tweet bot detection frame.

        Switches the interface to tweet bot detection mode, hiding
        other frames and updating the header and status.

        Returns
        -------
        None
            Updates UI to show tweet analysis interface.
        """
        self.hide_all_frames()
        self.header_label.config(text="Tweet Bot Detection 🤖")
        self.tweet_frame.pack(fill=tk.BOTH, expand=True)
        self.current_content_type = "tweet"
        self.status_bar.config(text="Enter tweet text and account metrics")

    def hide_all_frames(self):
        """
        Hide all content frames.

        Removes all visible frames from the display to prepare for
        showing a different interface section.

        Returns
        -------
        None
            Hides all UI frames including loading, content selection,
            news, tweet, and results frames.
        """
        # Hide the loading frame if it's showing
        if hasattr(self, 'loading_frame'):
            self.loading_frame.pack_forget()

        # Hide the content selection frame if it exists
        if hasattr(self, 'content_selection_frame'):
            self.content_selection_frame.pack_forget()

        # Hide the news frame if it exists
        if hasattr(self, 'news_frame'):
            self.news_frame.pack_forget()

        # Hide the tweet frame if it exists
        if hasattr(self, 'tweet_frame'):
            self.tweet_frame.pack_forget()

        # Hide the results frame and separator
        if hasattr(self, 'results_frame'):
            self.results_frame.pack_forget()

        if hasattr(self, 'separator'):
            self.separator.pack_forget()

    def show_loading_error(self):
        """
        Show error message if models failed to load.

        Displays an error dialog when model loading fails and closes
        the application since it cannot function without models.

        Returns
        -------
        None
            Shows error dialog and destroys the application window.
        """
        self.hide_all_frames()
        messagebox.showerror("Loading Error",
                             f"Failed to load models: {self.loading_error}\n\nThe application will close.")
        self.root.destroy()

    def clean_text(self, text):
        """
        Clean input text with the same preprocessing as training data.

        Applies text preprocessing including lowercasing, punctuation
        removal, and whitespace normalization to match training data
        preprocessing.

        Parameters
        ----------
        text : str or other
            Input text to clean. Non-string inputs are converted to string.

        Returns
        -------
        cleaned_text : str
            Preprocessed text with consistent formatting for model input.
        """
        logging.info("Cleaning input text...")

        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def analyze_news(self):
        """
        Analyze the news article and display results.

        Processes the news article text input along with optional domain
        and URL information through the classifier and displays the
        fake news detection results.

        Returns
        -------
        None
            Displays analysis results in the UI or shows error message
            if analysis fails. Updates status bar with progress information.

        Notes
        -----
        Performs domain extraction from URL if URL is provided but
        domain is not. Uses the NewsClassifier to predict article
        authenticity and displays formatted results with probabilities.
        """
        if not self.models_loaded:
            messagebox.showinfo("Please Wait", "Models are still loading. Please try again in a moment.")
            return

        # Get text from input area
        news_text = self.news_text.get("1.0", tk.END).strip()

        # Check if text is empty
        if not news_text:
            messagebox.showwarning("Empty Input", "Please enter some news text to analyze.")
            return

        # Get domain and URL from input fields
        domain = self.domain_entry.get().strip()
        url = self.url_entry.get().strip()

        # Extract domain from URL if URL is provided but domain is not
        if url and not domain:
            # Use the extract_domain function from shared_logic (via our classifier)
            extracted_domain = self.classifier.extract_domain(url) if hasattr(self.classifier,
                                                                              'extract_domain') else None
            if extracted_domain:
                domain = extracted_domain
                self.domain_entry.delete(0, tk.END)
                self.domain_entry.insert(0, domain)
                logging.info(f"Domain extracted from URL: {domain}")

        # Update status
        self.status_bar.config(text="Analyzing news article...")

        try:
            # Use the classifier to analyze the text
            account_date = ""
            result = self.classifier.predict(news_text, domain, account_date, url)

            if result:
                # Add the result type for UI processing
                result['result_type'] = 'news'

                # Display results
                self.display_results(result)
                self.status_bar.config(text="News article analysis complete")
            else:
                raise Exception("Analysis returned no results")

        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
            self.status_bar.config(text="Analysis failed")

    def analyze_tweet(self):
        """
        Analyze a tweet for bot detection and content classification.

        Processes tweet text and user account metrics to perform both
        bot detection and content classification analysis, displaying
        comprehensive results for both analyses.

        Returns
        -------
        None
            Displays combined analysis results or shows error message
            if analysis fails. Updates status bar with progress information.

        Notes
        -----
        Collects and processes user metrics including:
        - Basic counts (followers, friends, statuses, favorites)
        - Account information (screen name, creation date, verification)
        - Calculated metrics (ratios, age, etc.)

        Performs both bot detection and content analysis using the
        NewsClassifier and displays results in a combined format.
        """
        if not self.models_loaded:
            messagebox.showinfo("Please Wait", "Models are still loading. Please try again in a moment.")
            return

        # Get tweet text
        tweet_text = self.tweet_text.get("1.0", tk.END).strip()

        # Check if text is empty
        if not tweet_text:
            messagebox.showwarning("Empty Input", "Please enter tweet text to analyze.")
            return

        # Update status
        self.status_bar.config(text="Analyzing tweet and account metrics...")

        try:
            # Collect user account metrics
            user_data = {}

            # Get numeric metrics
            for metric in ['followers_count', 'friends_count', 'statuses_count', 'favourites_count']:
                try:
                    value = self.metric_entries[metric].get().strip()
                    user_data[metric] = float(value) if value else 0.0
                except ValueError:
                    user_data[metric] = 0.0

            # Get text metrics
            user_data['screen_name'] = self.metric_entries['screen_name'].get().strip()
            account_date = self.metric_entries['account_date'].get().strip()

            # Calculate account age in days if date is provided
            user_data['account_age_days'] = 0
            if account_date:
                try:
                    # Parse YYYY-MM format
                    year, month = account_date.split('-')
                    from datetime import datetime, date
                    creation_date = datetime(int(year), int(month),
                                             1)  # Use first day of month since we don't have the day
                    today = datetime.now()
                    user_data['account_age_days'] = (today - creation_date).days
                    logging.info(f"Calculated account age: {user_data['account_age_days']} days")
                except Exception as e:
                    logging.warning(f"Could not parse account date: {str(e)}")
                    user_data['account_age_days'] = 0

            # Calculate screen_name_length
            user_data['screen_name_length'] = len(user_data['screen_name'])

            # Add verification status
            user_data['verified'] = False
            if 'verified' in self.metric_entries:
                verified_value = self.metric_entries['verified'].get().strip().lower()
                user_data['verified'] = verified_value in ('yes', 'true', '1', 'y')

            # Calculate followers-to-friends ratio (strong human indicator)
            user_data['followers_to_friends_ratio'] = user_data['followers_count'] / (user_data['friends_count'] + 1)

            # Listed count (number of public lists the account is on)
            user_data['listed_count'] = 0
            if 'listed_count' in self.metric_entries:
                try:
                    value = self.metric_entries['listed_count'].get().strip()
                    user_data['listed_count'] = float(value) if value else 0.0
                except ValueError:
                    user_data['listed_count'] = 0.0

            # Set default values for other required metrics
            user_data['retweets'] = 0.2
            user_data['replies'] = 0.1
            user_data['favoriteC'] = 0.3
            user_data['hashtag'] = 0.3
            user_data['mentions'] = 0.4
            user_data['intertime'] = 3600
            user_data['favorites'] = 100
            user_data['uniqueHashtags'] = 0.5
            user_data['uniqueMentions'] = 0.6
            user_data['uniqueURL'] = 0.7

            # Calculate friends-to-followers ratio
            if user_data['followers_count'] > 0:
                user_data['ffratio'] = user_data['friends_count'] / user_data['followers_count']
            else:
                user_data['ffratio'] = user_data['friends_count'] if user_data['friends_count'] > 0 else 1.0

            # Perform bot detection with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message="X has feature names, but RandomForestClassifier was fitted without "
                                                "feature names")

                # Use the shared logic's predict_bot method
                bot_result = self.classifier.predict_bot(tweet_text, user_data)

                # Also analyze the tweet content using the shared logic's predict method
                content_result = self.classifier.predict(tweet_text, "", account_date)

            # Format combined results
            combined_result = {
                'bot_result': bot_result,
                'content_result': content_result,
                'result_type': 'tweet'
            }

            # Display results
            self.display_results(combined_result)
            self.status_bar.config(text="Tweet analysis complete")

        except Exception as e:
            logging.error(f"Error analyzing tweet: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
            self.status_bar.config(text="Analysis failed")

    def predict_bot(self, tweet_text, user_data):
        """
        Analyze if a tweet is from a bot using the classifier from shared_logic.

        This method is deprecated and redirects to the shared implementation.
        Use self.classifier.predict_bot() directly instead.

        Parameters
        ----------
        tweet_text : str
            The tweet text to analyze.
        user_data : dict
            User account metrics and information.

        Returns
        -------
        result : dict
            Bot detection result from NewsClassifier.predict_bot().

        Notes
        -----
        This method exists for backwards compatibility but should not be
        called directly. Use the NewsClassifier instance instead.
        """
        return self.classifier.predict_bot(tweet_text, user_data)

    def display_results(self, result):
        """
        Display the analysis results.

        Shows the analysis results in a formatted display based on the
        type of analysis performed (news or tweet). Clears previous
        results and creates appropriate visualization.

        Parameters
        ----------
        result : dict
            Analysis result dictionary containing:
            - 'result_type': str, type of analysis ('news' or 'tweet')
            - Additional keys depending on analysis type

        Returns
        -------
        None
            Updates the UI to show formatted results with appropriate
            visualizations and probability charts.
        """
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Show separator
        self.separator.pack(fill=tk.X, pady=10)

        # Show results frame
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Check result type
        if result['result_type'] == 'news':
            self.display_news_results(result)
        elif result['result_type'] == 'tweet':
            self.display_tweet_results(result)

    def display_news_results(self, result):
        """Display news article classification results"""
        # Results header
        results_header = tk.Frame(self.results_frame, bg=COLORS['background'])
        results_header.pack(fill=tk.X)

        results_label = tk.Label(
            results_header,
            text="News Article Analysis Results",
            font=("Arial", 18, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        results_label.pack(side=tk.LEFT)

        # Main prediction
        prediction_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
        prediction_frame.pack(fill=tk.X, pady=15)

        prediction_color = COLORS['real'] if result['label'] == 'REAL' else COLORS['fake']

        prediction_label = tk.Label(
            prediction_frame,
            text=f"Prediction:",
            font=("Arial", 14),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        prediction_label.pack(side=tk.LEFT)

        prediction_value = tk.Label(
            prediction_frame,
            text=result['label'],
            font=("Arial", 14, "bold"),
            fg=prediction_color,
            bg=COLORS['background']
        )
        prediction_value.pack(side=tk.LEFT, padx=(5, 0))

        confidence_label = tk.Label(
            prediction_frame,
            text=f"(Confidence: {result['confidence']:.1f}%)",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        confidence_label.pack(side=tk.LEFT, padx=(10, 0))

        # Show domain information if available
        if 'domain_info' in result and result['domain_info'] is not None:
            domain_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
            domain_frame.pack(fill=tk.X, pady=10)

            domain_info = result['domain_info']

            domain_label = tk.Label(
                domain_frame,
                text=f"Domain Information:",
                font=("Arial", 12, "bold"),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            domain_label.pack(anchor=tk.W)

            domain_name = domain_info.get('domain', 'Unknown')
            fake_ratio = domain_info.get('fake_ratio', 'N/A')

            domain_info_text = f"Domain: {domain_name}"
            if fake_ratio != 'N/A':
                fake_ratio_pct = float(fake_ratio) * 100
                domain_info_text += f" (Historical fake news ratio: {fake_ratio_pct:.1f}%)"

            domain_detail = tk.Label(
                domain_frame,
                text=domain_info_text,
                font=("Arial", 11),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            domain_detail.pack(anchor=tk.W, padx=15)

            domain_note = tk.Label(
                domain_frame,
                text="Note: Domain reputation has been factored into the analysis.",
                font=("Arial", 10, "italic"),
                fg=COLORS['neutral'],
                bg=COLORS['background']
            )
            domain_note.pack(anchor=tk.W, padx=15)

        # Create probability visualization
        self.create_probability_chart(result)

    def display_tweet_results(self, result):
        """
        Display tweet analysis results (both bot detection and content).

        Creates a comprehensive display showing both bot detection results
        and content classification results for tweet analysis, including
        probabilities and confidence measures.

        Parameters
        ----------
        result : dict
            Tweet analysis result dictionary containing:
            - 'bot_result': dict, bot detection analysis results
            - 'content_result': dict, content classification results
            - 'result_type': str, should be 'tweet'

        Returns
        -------
        None
            Updates the results display with tweet-specific formatting
            showing both bot detection and content analysis results.
        """
        # Get the bot and content results
        bot_result = result['bot_result']
        content_result = result['content_result']

        # Bot detection results header
        bot_header = tk.Frame(self.results_frame, bg=COLORS['background'])
        bot_header.pack(fill=tk.X, pady=(0, 10))

        bot_label = tk.Label(
            bot_header,
            text="Bot Detection Results",
            font=("Arial", 16, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        bot_label.pack(side=tk.LEFT)

        # Bot detection prediction
        bot_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
        bot_frame.pack(fill=tk.X, pady=(0, 5))

        # Determine color based on prediction
        if bot_result['label'] == 'HUMAN':
            pred_color = COLORS['human']
        elif bot_result['label'] == 'BOT':
            pred_color = COLORS['bot']
        else:
            pred_color = COLORS['neutral']

        bot_pred_label = tk.Label(
            bot_frame,
            text=f"Account Type:",
            font=("Arial", 14),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        bot_pred_label.pack(side=tk.LEFT)

        bot_pred_value = tk.Label(
            bot_frame,
            text=bot_result['label'],
            font=("Arial", 14, "bold"),
            fg=pred_color,
            bg=COLORS['background']
        )
        bot_pred_value.pack(side=tk.LEFT, padx=(5, 0))

        # Show confidence if available
        if 'confidence' in bot_result:
            conf_label = tk.Label(
                bot_frame,
                text=f"(Confidence: {bot_result['confidence']:.1f}%)",
                font=("Arial", 12),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            conf_label.pack(side=tk.LEFT, padx=(10, 0))

        # Show error message if there was a problem
        if 'message' in bot_result:
            error_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
            error_frame.pack(fill=tk.X, pady=(0, 10))

            error_label = tk.Label(
                error_frame,
                text=f"Error: {bot_result['message']}",
                font=("Arial", 11, "italic"),
                fg="red",
                bg=COLORS['background'],
                wraplength=700
            )
            error_label.pack(anchor=tk.W, padx=10)

        # Show bot/human probabilities if available
        if 'bot_probability' in bot_result and 'human_probability' in bot_result:
            prob_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
            prob_frame.pack(fill=tk.X, pady=(0, 10))

            prob_text = f"BOT probability: {bot_result['bot_probability']:.1f}%, HUMAN probability: {bot_result['human_probability']:.1f}%"
            prob_label = tk.Label(
                prob_frame,
                text=prob_text,
                font=("Arial", 12),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            prob_label.pack(anchor=tk.W, padx=10)

            # Create horizontal separator
            ttk.Separator(self.results_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Content analysis results
        content_header = tk.Frame(self.results_frame, bg=COLORS['background'])
        content_header.pack(fill=tk.X, pady=(10, 10))

        content_label = tk.Label(
            content_header,
            text="Tweet Content Analysis",
            font=("Arial", 16, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        content_label.pack(side=tk.LEFT)

        # Show content analysis results
        if content_result:
            content_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
            content_frame.pack(fill=tk.X, pady=(0, 5))

            content_color = COLORS['real'] if content_result['label'] == 'REAL' else COLORS['fake']

            content_pred_label = tk.Label(
                content_frame,
                text=f"Content Classification:",
                font=("Arial", 14),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            content_pred_label.pack(side=tk.LEFT)

            content_pred_value = tk.Label(
                content_frame,
                text=content_result['label'],
                font=("Arial", 14, "bold"),
                fg=content_color,
                bg=COLORS['background']
            )
            content_pred_value.pack(side=tk.LEFT, padx=(5, 0))

            conf_label = tk.Label(
                content_frame,
                text=f"(Confidence: {content_result['confidence']:.1f}%)",
                font=("Arial", 12),
                fg=COLORS['text'],
                bg=COLORS['background']
            )
            conf_label.pack(side=tk.LEFT, padx=(10, 0))

            # Create content probability chart
            self.create_probability_chart(content_result)
        else:
            error_label = tk.Label(
                self.results_frame,
                text="Could not analyze tweet content. Bot detection results are still valid.",
                font=("Arial", 12, "italic"),
                fg="red",
                bg=COLORS['background']
            )
            error_label.pack(anchor=tk.W, padx=10, pady=10)

    def create_probability_chart(self, result):
        """
        Create a visual chart of the prediction probabilities.

        Generates a horizontal bar chart showing the probabilities for
        different prediction categories using matplotlib embedded in
        the Tkinter interface.

        Parameters
        ----------
        result : dict
            Analysis result containing probability data. Should have either:
            - 'fake_probability' and 'real_probability' for news analysis, or
            - 'bot_probability' and 'human_probability' for bot detection

        Returns
        -------
        None
            Embeds a matplotlib chart in the results frame showing
            probability distributions. Returns early if no probability
            data is available.
        """
        chart_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
        chart_frame.pack(fill=tk.BOTH, pady=10)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 2.5))

        # Data
        if 'fake_probability' in result and 'real_probability' in result:
            labels = ['FAKE', 'REAL']
            sizes = [result['fake_probability'], result['real_probability']]
            colors = [COLORS['fake'], COLORS['real']]
        elif 'bot_probability' in result and 'human_probability' in result:
            labels = ['BOT', 'HUMAN']
            sizes = [result['bot_probability'], result['human_probability']]
            colors = [COLORS['bot'], COLORS['human']]
        else:
            return  # No probability data available

        # Plot horizontal bar chart
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, sizes, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Probability (%)')
        ax.set_title('Prediction Probabilities')

        # Add percentage labels on bars
        for i, v in enumerate(sizes):
            ax.text(v + 1, i, f"{v:.1f}%", va='center')

        # Set x-axis limit to slightly more than 100 for label visibility
        ax.set_xlim(0, 105)

        # Make the plot look nice
        fig.tight_layout()

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_example_news(self):
        """
        Load an example text for news analysis.

        Populates the news analysis form with sample article text,
        domain, and URL for demonstration purposes.

        Returns
        -------
        None
            Clears existing inputs and loads example data into
            news text area, domain entry, and URL entry fields.
        """
        example_text = """
        Scientists have discovered a new species of deep-sea fish that can survive 
        at depths of over 8,000 meters in the Mariana Trench. The fish, named 
        Pseudoliparis swirei, has several unique adaptations including specialized 
        cell membranes and high levels of TMAO, a chemical that helps proteins 
        maintain their shape under extreme pressure. The discovery was published 
        yesterday in the journal Marine Biology.
        """

        self.news_text.delete("1.0", tk.END)
        self.news_text.insert(tk.END, example_text.strip())
        self.domain_entry.delete(0, tk.END)
        self.domain_entry.insert(0, "marinebiology.org")
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, "https://marinebiology.org/article/deep-sea-fish-discovery")

    def load_example_tweet(self):
        """
        Load example data for tweet bot detection.

        Populates the tweet analysis form with sample tweet text and
        typical bot account metrics for demonstration purposes.

        Returns
        -------
        None
            Clears existing inputs and loads example tweet text and
            account metrics that are characteristic of bot accounts.
        """
        example_tweet = "Just published 10 amazing tips for gaining followers fast! Check out the link below to learn " \
                        "more. #followers #socialmedia #growth #marketing "

        self.tweet_text.delete("1.0", tk.END)
        self.tweet_text.insert(tk.END, example_tweet)

        # Set example metrics for a typical bot account
        self.metric_entries['followers_count'].delete(0, tk.END)
        self.metric_entries['followers_count'].insert(0, "12500")

        self.metric_entries['friends_count'].delete(0, tk.END)
        self.metric_entries['friends_count'].insert(0, "15000")

        self.metric_entries['statuses_count'].delete(0, tk.END)
        self.metric_entries['statuses_count'].insert(0, "42000")

        self.metric_entries['favourites_count'].delete(0, tk.END)
        self.metric_entries['favourites_count'].insert(0, "250")

        self.metric_entries['screen_name'].delete(0, tk.END)
        self.metric_entries['screen_name'].insert(0, "social_growth_expert")

        self.metric_entries['account_date'].delete(0, tk.END)
        self.metric_entries['account_date'].insert(0, "2024-05")

    def clear_news(self):
        """
        Clear the news article input and results.

        Removes all text from news input fields and hides any displayed
        results to prepare for new analysis.

        Returns
        -------
        None
            Clears news text area, domain entry, URL entry, and hides
            results display. Updates status bar.
        """
        self.news_text.delete("1.0", tk.END)
        self.domain_entry.delete(0, tk.END)
        self.url_entry.delete(0, tk.END)

        # Hide results
        self.results_frame.pack_forget()
        self.separator.pack_forget()

        # Update status
        self.status_bar.config(text="Ready to analyze news article")

    def clear_tweet(self):
        """
        Clear the tweet input, metrics, and results.

        Removes all text from tweet input fields and metric entries,
        resets numeric fields to default values, and hides any displayed
        results to prepare for new analysis.

        Returns
        -------
        None
            Clears tweet text area and all metric entries, resets
            numeric fields to "0", hides results display, and updates
            status bar.
        """
        self.tweet_text.delete("1.0", tk.END)

        # Clear all metric entries
        for entry in self.metric_entries.values():
            entry.delete(0, tk.END)

        # Reset numeric entries to 0
        for metric in ['followers_count', 'friends_count', 'statuses_count', 'favourites_count']:
            self.metric_entries[metric].insert(0, "0")

        # Hide results
        self.results_frame.pack_forget()
        self.separator.pack_forget()

        # Update status
        self.status_bar.config(text="Ready to analyze tweet")


if __name__ == "__main__":
    root = tk.Tk()
    app = ContentClassifierUI(root)
    root.mainloop()