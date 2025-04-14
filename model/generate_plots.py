import pandas as pd
import numpy as np  # Add numpy import
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time
import logging
from visualization import (
    create_visualization_directory,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_learning_curves
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_all_plots():
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv("emails.csv")
        
        # Prepare data
        X = df['text']
        y = df['spam']
        
        # Split data
        logger.info("Splitting data...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=5
        )
        
        # Vectorize text
        logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=20, random_state=5),
            'SVM': LinearSVC(random_state=5),
            'Naive Bayes': MultinomialNB()
        }
        
        # Store results
        accuracies = []
        times = []
        predictions = {}
        probabilities = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Get probabilities (handle SVM separately)
            if isinstance(model, LinearSVC):
                # For SVM, use decision function instead of probabilities
                decision_scores = model.decision_function(X_test)
                # Convert to pseudo-probabilities
                proba = 1 / (1 + np.exp(-decision_scores))
                probabilities[name] = np.column_stack((1 - proba, proba))
            else:
                probabilities[name] = model.predict_proba(X_test)
            
            end_time = time.time()
            
            accuracy = accuracy_score(y_test, y_pred)
            execution_time = end_time - start_time
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, Time: {execution_time:.2f}s")
            
            accuracies.append(accuracy)
            times.append(execution_time)
            predictions[name] = y_pred
        
        # Create visualization directory
        viz_dir = create_visualization_directory()
        logger.info(f"Created visualization directory: {viz_dir}")
        
        # Generate plots
        logger.info("Generating plots...")
        
        # Model comparison plot
        plot_model_comparison(accuracies, times, viz_dir)
        
        # Confusion matrices
        plot_confusion_matrices(
            y_test,
            predictions['Random Forest'],
            predictions['SVM'],
            predictions['Naive Bayes'],
            viz_dir
        )
        
        # ROC curves
        plot_roc_curves(
            y_test,
            probabilities['Random Forest'],
            probabilities['SVM'],
            probabilities['Naive Bayes'],
            viz_dir
        )
        
        # Feature importance (Random Forest only)
        rf_model = models['Random Forest']
        plot_feature_importance(vectorizer, rf_model, viz_dir)
        
        logger.info(f"All plots have been generated in the '{viz_dir}' directory.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    generate_all_plots()