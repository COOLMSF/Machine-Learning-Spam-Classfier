# Import necessary libraries for machine learning, data handling, and utilities
import pandas as pd          # For data manipulation and loading CSV files
import os                   # For checking file existence
import logging              # For logging program execution details
import time                 # For measuring execution time
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.svm import LinearSVC                    # SVM model
from sklearn.naive_bayes import MultinomialNB        # Naive Bayes model
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
from sklearn.model_selection import train_test_split         # Data splitting
from sklearn import metrics                          # For accuracy calculation

# Configure logging to track program execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to display a welcome message to the user
def display_welcome_message() -> str:
    """Display a welcome message explaining the program's purpose."""
    message = (
        "\n" + "="*25 + "\n" +
        "Welcome to the Spam Email Classifier!\n" +
        "This program classifies emails as spam or not using machine learning.\n" +
        "You can choose between Random Forest, SVM, or Naive Bayes methods.\n" +
        "\n" + "="*25 + "\n"
    )
    return message

# Function to check if the input CSV file exists
def check_file_exists(file_path):
    """Verify if the CSV file exists at the specified path."""
    logger.info(f"Checking if file exists: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    logger.info("File check passed.")

# Function to load data from the CSV file
def load_data(file_path):
    """Load email data from the CSV file into a pandas DataFrame."""
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise Exception(f"Error loading data: {e}")

# Function to display a sample of the loaded data
def display_data_sample(df, num_rows=5):
    """Show the first few rows of the DataFrame for user inspection."""
    print("\nHere’s a sample of the loaded data:")
    print("-"*40)
    print(df.head(num_rows))
    print("-"*40)

# Function to summarize the distribution of spam vs. non-spam emails
def summarize_data(df):
    """Provide a summary of spam and non-spam email counts."""
    logger.info("Summarizing data distribution.")
    spam_count = df['spam'].value_counts()
    print("\nData Summary:")
    print(f"Total emails: {len(df)}")
    print(f"Spam emails (1): {spam_count.get(1, 0)}")
    print(f"Non-spam emails (0): {spam_count.get(0, 0)}")

# Function to validate the DataFrame structure
def validate_dataframe(df):
    """Ensure the DataFrame has required columns: 'text' and 'spam'."""
    required_columns = ['text', 'spam']
    logger.info("Validating DataFrame structure.")
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"DataFrame is missing the '{col}' column.")

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.3, random_state=5):
    """Split the dataset into training and testing sets."""
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Data split completed.")
    return X_train, X_test, y_train, y_test

# Function to convert text data into TF-IDF features
def vectorize_text(X_train_text, X_test_text):
    """Transform text data into TF-IDF numerical features."""
    logger.info("Starting text vectorization with TF-IDF.")
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer only on training data to avoid data leakage
    vectorizer.fit(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    logger.info("Text vectorization completed.")
    return vectorizer, X_train, X_test

# Function to prompt the user for email text
def get_user_email():
    """Prompt the user to input an email text for classification."""
    print("\nPlease enter the email text you want to classify:")
    print("(Press Enter after typing your email.)")
    your_email = input("> ")
    logger.info("User entered email text for classification.")
    return your_email

# Function to display the classifier selection menu
def display_menu():
    """Display the menu of available classifiers."""
    print("\n" + "-"*50)
    print("Available Machine Learning Methods:")
    print("1. Random Forest - A robust ensemble method.")
    print("2. SVM - Support Vector Machine with a linear kernel.")
    print("3. Naive Bayes - A probabilistic classifier.")
    print("-"*50)

# Function to get the user's classifier choice
def get_user_choice():
    """Get the user's selection of classifier."""
    display_menu()
    choice = input("Enter your choice (1, 2, or 3): ")
    logger.info(f"User selected option: {choice}")
    return choice

# Function to confirm the user's choice
def confirm_user_choice(choice):
    """Ask the user to confirm their classifier choice."""
    options = {"1": "Random Forest", "2": "SVM", "3": "Naive Bayes"}
    selected = options.get(choice, "Invalid")
    print(f"\nYou selected: {selected}")
    confirm = input("Proceed with this choice? (yes/no): ").lower()
    logger.info(f"User confirmation response: {confirm}")
    return confirm == "yes"

# Function to train and evaluate Random Forest classifier
def run_random_forest(X_train, y_train, X_test, y_test, vectorizer, email, ui):
    """Execute the Random Forest classification process."""
    # logger.info("Starting Random Forest classification.")
    ui.plainTextEdit_modelOutput.appendPlainText("Starting Random Forest classification.")
    start_time = time.time()
    
    # Initialize the Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=20, random_state=5)
    # logger.info("Random Forest classifier initialized.")
    ui.plainTextEdit_modelOutput.appendPlainText("Random Forest classifier initialized.")
    
    # Train the model
    classifier.fit(X_train, y_train)
    ui.plainTextEdit_modelOutput.appendPlainText("Random Forest training completed.")
    
    # Predict on test data
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Accuracy: {accuracy:.4f}")
    
    # Predict on user's email
    email_transformed = vectorizer.transform([email])
    prediction = classifier.predict(email_transformed)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    if prediction[0] == 1:
        print("Result: Your email is classified as SPAM.")
    else:
        print("Result: Your email is classified as NOT SPAM.")
    ui.plainTextEdit_modelOutput.appendPlainText("Random Forest classification completed.")

    is_spam = "True" if prediction[0] == 1 else "False"

    return accuracy, is_spam, end_time - start_time

# Function to train and evaluate SVM classifier
def run_svm(X_train, y_train, X_test, y_test, vectorizer, email, ui):
    """Execute the SVM classification process."""
    ui.plainTextEdit_modelOutput.appendPlainText("Starting SVM classification.")
    start_time = time.time()
    
    # Initialize the SVM classifier
    classifier = LinearSVC(random_state=5)
    ui.plainTextEdit_modelOutput.appendPlainText("SVM classifier initialized.")
    
    # Train the model
    classifier.fit(X_train, y_train)
    ui.plainTextEdit_modelOutput.appendPlainText("SVM training completed.")
    
    # Predict on test data
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"\nSVM Accuracy: {accuracy:.4f}")
    
    # Predict on user's email
    email_transformed = vectorizer.transform([email])
    prediction = classifier.predict(email_transformed)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    if prediction[0] == 1:
        print("Result: Your email is classified as SPAM.")
    else:
        print("Result: Your email is classified as NOT SPAM.")
    ui.plainTextEdit_modelOutput.appendPlainText("SVM classification completed.")

    is_spam = "True" if prediction[0] == 1 else "False"
    return accuracy, is_spam, end_time - start_time


# Function to train and evaluate Naive Bayes classifier
def run_naive_bayes(X_train, y_train, X_test, y_test, vectorizer, email, ui):
    """Execute the Naive Bayes classification process."""
    logger.info("Starting Naive Bayes classification.")
    start_time = time.time()
    
    # Initialize the Naive Bayes classifier
    classifier = MultinomialNB()
    ui.plainTextEdit_modelOutput.appendPlainText("Naive Bayes classifier initialized.")
    
    # Train the model
    classifier.fit(X_train, y_train)
    ui.plainTextEdit_modelOutput.appendPlainText("Naive Bayes training completed.")
    
    # Predict on test data
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"\nNaive Bayes Accuracy: {accuracy:.4f}")
    
    # Predict on user's email
    email_transformed = vectorizer.transform([email])
    prediction = classifier.predict(email_transformed)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    if prediction[0] == 1:
        print("Result: Your email is classified as SPAM.")
    else:
        print("Result: Your email is classified as NOT SPAM.")
    ui.plainTextEdit_modelOutput.appendPlainText("Naive Bayes classification completed.")

    is_spam = "True" if prediction[0] == 1 else "False"
    return accuracy, is_spam, end_time - start_time


# Function to display a farewell message
def display_farewell_message():
    """Display a closing message to the user."""
    print("\n" + "="*50)
    print("Thank you for using the Spam Email Classifier!")
    print("Goodbye.")
    print("="*50)

# Main function to orchestrate the program
def main():
    """Main function to run the spam email classification program."""
    # Display welcome message
    display_welcome_message()
    
    # Define the file path
    file_path = "emails.csv"
    logger.info("Starting program with file: emails.csv")
    
    # Check if the file exists
    try:
        check_file_exists(file_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load the data
    try:
        df = load_data(file_path)
    except Exception as e:
        print(e)
        return
    
    # Validate the DataFrame
    try:
        validate_dataframe(df)
    except ValueError as e:
        print(e)
        return
    
    # Display data sample and summary
    display_data_sample(df)
    summarize_data(df)
    
    # Prepare data for training
    X_text = df['text']  # Email content
    y = df['spam']       # Labels: 1 for spam, 0 for not spam
    
    # Split the data
    X_train_text, X_test_text, y_train, y_test = split_data(X_text, y)
    
    # Vectorize the text data
    vectorizer, X_train, X_test = vectorize_text(X_train_text, X_test_text)
    
    # Get user email input
    user_email = get_user_email()
    
    # Get and confirm user choice
    choice = get_user_choice()
    if not confirm_user_choice(choice):
        print("Operation cancelled by user.")
        display_farewell_message()
        return
    
    # Execute the chosen classifier
    if choice == "1":
        run_random_forest(X_train, y_train, X_test, y_test, vectorizer, user_email)
    elif choice == "2":
        run_svm(X_train, y_train, X_test, y_test, vectorizer, user_email)
    elif choice == "3":
        run_naive_bayes(X_train, y_train, X_test, y_test, vectorizer, user_email)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        logger.warning(f"Invalid choice entered: {choice}")
    
    # Display farewell message
    display_farewell_message()

# Entry point of the program
if __name__ == "__main__":
    logger.info("Program execution started.")
    main()
    logger.info("Program execution completed.")
