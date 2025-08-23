#!/usr/bin/env python3
# News Classifier GUI Application

import os
import logging
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from joblib import load
import string
import platform
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Setup logging - redirect to file to keep console clean
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'classifier_app_ui.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()  # Also log to console for debugging
    ]
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
DOMAINS_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'domains_summary.csv')

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
    'neutral': '#adb5bd'
}

# Specific macOS configuration
if platform.system() == 'Darwin':
    os.environ['PYTHONHASHSEED'] = '0'  # For reproducibility on macOS
    # Ensure matplotlib doesn't use the native GUI backend
    matplotlib.use('Agg')


class NewsClassifierUI:
    """GUI application for the fake news classifier"""

    def __init__(self, root):
        """Initialize the UI and load models"""
        self.root = root
        self.root.title("🔍 Fake News Detector")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS['background'])
        self.root.resizable(True, True)

        # Set minimum window size
        self.root.minsize(800, 600)

        # Set application icon (if available)
        try:
            self.root.iconbitmap(os.path.join(ASSETS_DIR, "icon.ico"))
        except:
            # Icon not found or not supported on this platform
            pass

        logging.info("Starting news classifier UI application...")

        # Load models in a separate thread to keep UI responsive
        self.models_loaded = False
        self.loading_error = None
        self.init_ui()
        self.root.after(100, self.load_models)

    def init_ui(self):
        """Set up the user interface elements"""
        # Main frame with padding
        self.main_frame = tk.Frame(self.root, bg=COLORS['background'], padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header label
        header_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        self.header_label = tk.Label(
            header_frame,
            text="Fake News Detector 🕵️",
            font=("Arial", 24, "bold"),
            fg=COLORS['header'],
            bg=COLORS['background']
        )
        self.header_label.pack(side=tk.LEFT)

        # Instructions
        instruction_label = tk.Label(
            self.main_frame,
            text="Enter news text to analyze:",
            font=("Arial", 12),
            fg=COLORS['text'],
            bg=COLORS['background'],
            anchor="w"
        )
        instruction_label.pack(fill=tk.X, pady=(0, 10))

        # Text entry area with scrollbar
        self.text_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
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
        domain_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
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

        # Example button
        self.example_button = tk.Button(
            self.main_frame,
            text="Load Example Text",
            font=("Arial", 10),
            bg=COLORS['button'],
            fg="black",  # Changed from white to black
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=10,
            command=self.load_example_text
        )
        self.example_button.pack(anchor=tk.W, pady=(0, 10))

        # Button frame
        button_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        button_frame.pack(fill=tk.X, pady=(0, 20))

        # Analyze button
        self.analyze_button = tk.Button(
            button_frame,
            text="Analyze",
            font=("Arial", 12, "bold"),
            bg=COLORS['button'],
            fg="black",  # Changed from white to black
            activebackground=COLORS['button_hover'],
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.analyze_text
        )
        self.analyze_button.pack(side=tk.LEFT)

        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 12),
            bg=COLORS['neutral'],
            fg="black",  # Changed from white to black
            activebackground="#999999",
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.clear_text
        )
        self.clear_button.pack(side=tk.LEFT, padx=(10, 0))

        # Loading indicator
        self.loading_label = tk.Label(
            button_frame,
            text="Loading models...",
            font=("Arial", 10, "italic"),
            fg=COLORS['neutral'],
            bg=COLORS['background']
        )
        self.loading_label.pack(side=tk.RIGHT)

        # Progress bar for loading models
        self.progress = ttk.Progressbar(
            button_frame,
            orient=tk.HORIZONTAL,
            length=200,
            mode='indeterminate'
        )
        self.progress.pack(side=tk.RIGHT, padx=(0, 10))
        self.progress.start()

        # Results frame (hidden initially)
        self.results_frame = tk.Frame(self.main_frame, bg=COLORS['background'])

        # Separator
        self.separator = ttk.Separator(self.main_frame, orient='horizontal')

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_models(self):
        """Load machine learning models and domain data"""
        try:
            # Load vectorizer
            self.vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
            logging.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = load(self.vectorizer_path)

            # Load all models
            self.models = {}
            model_files = {
                'naive_bayes': 'naive_bayes_model.joblib',
                'logistic_regression': 'logistic_regression_model.joblib',
                'decision_tree': 'decision_tree_model.joblib',
                'random_forest': 'random_forest_model.joblib'
            }

            for name, filename in model_files.items():
                model_path = os.path.join(MODELS_DIR, filename)
                logging.info(f"Loading {name} model from {model_path}")

                # Check if file exists before loading
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                self.models[name] = load(model_path)
                # Update status after each model loads
                self.loading_label.config(text=f"Loading {name}... complete")
                self.root.update_idletasks()

            # Load domain data
            logging.info(f"Loading domain data from {DOMAINS_FILE}")
            try:
                self.domains_data = pd.read_csv(DOMAINS_FILE)
                logging.info(f"Loaded data for {len(self.domains_data)} domains")
                self.loading_label.config(text="Loading domain data... complete")
                self.root.update_idletasks()
            except Exception as e:
                logging.warning(f"Could not load domain data: {str(e)}")
                self.domains_data = None

            logging.info("All models loaded successfully")
            self.models_loaded = True

            # Update UI
            self.root.after(0, self.update_ui_after_loading)

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            self.loading_error = str(e)
            self.root.after(0, self.show_loading_error)

    def update_ui_after_loading(self):
        """Update UI after models are loaded"""
        self.loading_label.config(text="Models loaded successfully")
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_button.config(state=tk.NORMAL)
        self.status_bar.config(text="Ready to analyze news")

    def show_loading_error(self):
        """Show error message if models failed to load"""
        self.loading_label.config(text="Error loading models")
        self.progress.stop()
        messagebox.showerror("Loading Error",
                            f"Failed to load models: {self.loading_error}\n\nThe application will close.")
        self.root.destroy()

    def clean_text(self, text):
        """Clean input text with the same preprocessing as training data"""
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

    def analyze_text(self):
        """Analyze the input text and display results"""
        if not self.models_loaded:
            messagebox.showinfo("Please Wait", "Models are still loading. Please try again in a moment.")
            return

        # Get text from input area
        news_text = self.news_text.get("1.0", tk.END).strip()

        # Check if text is empty
        if not news_text:
            messagebox.showwarning("Empty Input", "Please enter some news text to analyze.")
            return

        # Get domain from input field
        domain = self.domain_entry.get().strip()

        # Update status
        self.status_bar.config(text="Analyzing text...")

        try:
            # Clean the text
            cleaned_text = self.clean_text(news_text)

            # Vectorize
            X = self.vectorizer.transform([cleaned_text])

            # Make predictions with each model
            results = {}
            probabilities = {}

            for name, model in self.models.items():
                # Get prediction
                prediction = model.predict(X)[0]
                results[name] = prediction

                # Get probability if the model supports it
                try:
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = proba
                except:
                    # Some models might not have predict_proba
                    probabilities[name] = [0.5, 0.5] if prediction == 1 else [0.5, 0.5]

            # Calculate voting result
            votes = list(results.values())
            final_prediction = 1 if sum(votes) > len(votes)/2 else 0

            # Calculate average probabilities
            avg_proba = [0, 0]
            for name in probabilities:
                avg_proba[0] += probabilities[name][0]
                avg_proba[1] += probabilities[name][1]

            avg_proba[0] /= len(probabilities)
            avg_proba[1] /= len(probabilities)

            # Apply domain reputation adjustment if domain is provided and domain data is available
            domain_info = None
            if domain and hasattr(self, 'domains_data') and self.domains_data is not None:
                try:
                    # Extract just the domain name without protocol, www, etc.
                    clean_domain = domain.lower()
                    if '://' in clean_domain:
                        clean_domain = clean_domain.split('://')[1]
                    if clean_domain.startswith('www.'):
                        clean_domain = clean_domain[4:]
                    if '/' in clean_domain:
                        clean_domain = clean_domain.split('/')[0]

                    # Look for the domain in our dataset
                    domain_match = self.domains_data[self.domains_data['domain'].str.contains(clean_domain, case=False, na=False)]

                    if not domain_match.empty:
                        # Get the first matching domain info
                        domain_info = domain_match.iloc[0].to_dict()

                        # Calculate adjustment factor based on domain's fake news ratio
                        # Higher fake ratio increases fake probability
                        if 'fake_ratio' in domain_info:
                            fake_ratio = float(domain_info['fake_ratio'])

                            # Apply domain adjustment to the probabilities
                            # Increase fake probability based on domain reputation
                            adjustment = min(0.3, fake_ratio * 0.5)  # Cap the adjustment at 30%

                            # Adjust probabilities while keeping sum at 1.0
                            avg_proba[0] = min(0.95, avg_proba[0] + adjustment)
                            avg_proba[1] = 1.0 - avg_proba[0]

                            # Recalculate prediction based on adjusted probabilities
                            final_prediction = 1 if avg_proba[1] > avg_proba[0] else 0

                            logging.info(f"Applied domain adjustment for {domain}: fake_ratio={fake_ratio}, " +
                                        f"adjustment={adjustment}, new probabilities: fake={avg_proba[0]:.4f}, real={avg_proba[1]:.4f}")
                except Exception as e:
                    logging.error(f"Error processing domain information: {str(e)}")

            # Format results
            result = {
                'prediction': final_prediction,
                'label': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': avg_proba[final_prediction] * 100,
                'real_probability': avg_proba[1] * 100,
                'fake_probability': avg_proba[0] * 100,
                'model_votes': results,
                'probabilities': probabilities,
                'domain_info': domain_info
            }

            # Display results
            self.display_results(result)
            self.status_bar.config(text="Analysis complete")

            # Log the analysis
            self.log_analysis(domain, news_text, result)

        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
            self.status_bar.config(text="Analysis failed")

    def log_analysis(self, domain, news_text, result):
        """Log the analysis to a CSV file"""
        log_file = os.path.join(ASSETS_DIR, "analysis_log.csv")

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Prepare log entry
        log_entry = {
            'domain': domain,
            'news_text': news_text[:2000],
            'prediction': result['label'],
            'confidence': f"{result['confidence']:.2f}",
            'real_probability': f"{result['real_probability']:.2f}",
            'fake_probability': f"{result['fake_probability']:.2f}",
            'model_votes': str(result['model_votes']),
        }

        # Check if file exists
        file_exists = os.path.isfile(log_file)

        try:
            # Open the file in append mode
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                # Create CSV writer
                writer = pd.DataFrame([log_entry])

                # Write header if file doesn't exist
                if not file_exists:
                    writer.to_csv(f, index=False)
                else:
                    writer.to_csv(f, index=False, header=False)

            logging.info(f"Analysis logged to {log_file}")

        except Exception as e:
            logging.error(f"Failed to log analysis: {str(e)}")
            # Continue execution even if logging fails

    def display_results(self, result):
        """Display the analysis results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Show separator
        self.separator.pack(fill=tk.X, pady=20)

        # Show results frame
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Results header
        results_header = tk.Frame(self.results_frame, bg=COLORS['background'])
        results_header.pack(fill=tk.X)

        results_label = tk.Label(
            results_header,
            text="Analysis Results",
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

    def create_probability_chart(self, result):
        """Create a visual chart of the prediction probabilities"""
        chart_frame = tk.Frame(self.results_frame, bg=COLORS['background'])
        chart_frame.pack(fill=tk.BOTH, pady=10)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 2.5))

        # Data
        labels = ['FAKE', 'REAL']
        sizes = [result['fake_probability'], result['real_probability']]
        colors = [COLORS['fake'], COLORS['real']]

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

    def load_example_text(self):
        """Load an example text for analysis"""
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

    def clear_text(self):
        """Clear the text input area and results"""
        self.news_text.delete("1.0", tk.END)
        self.domain_entry.delete(0, tk.END)

        # Hide results
        self.results_frame.pack_forget()
        self.separator.pack_forget()

        # Update status
        self.status_bar.config(text="Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = NewsClassifierUI(root)
    root.mainloop()
