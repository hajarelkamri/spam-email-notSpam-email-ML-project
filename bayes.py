
import os
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd



# quelques variables globales utiles
PATH_TO_HAM_DIR = "C:\\Users\\hp\\PycharmProjects\\teest\\emails\\ham"
PATH_TO_SPAM_DIR = "C:\\Users\\hp\\PycharmProjects\\teest\\emails\\spam"

SPAM_TYPE = "SPAM"
HAM_TYPE = "HAM"

# les tableaux X et Y seront ordonnés et de la même taille
# X représente l'input Data (ici les mails)
X = []
# indique s'il s'agit d'un mail ou non
Y = []  # les etiquettes (labels) pour le training set


def readFilesFromDirectory(path, classification):
    os.chdir(path)
    files_name = os.listdir(path)
    for current_file in files_name:
        message = extract_mail_body(current_file)
        X.append(message)
        Y.append(classification)


def extract_mail_body(file_name_str):
    inBody = False
    lines = []
    file_descriptor = io.open(file_name_str, 'r', encoding='latin1')
    for line in file_descriptor:
        if inBody:
            lines.append(line)
        elif line == '\n':
            inBody = True
        message = '\n'.join(lines)
    file_descriptor.close()
    return message


readFilesFromDirectory(PATH_TO_HAM_DIR, HAM_TYPE)
readFilesFromDirectory(PATH_TO_SPAM_DIR, SPAM_TYPE)

# Suppose X contains the email texts and Y contains the corresponding labels (SPAM or HAM)
training_data = {'X': X, 'Y': Y}
training_set = pd.DataFrame(data=training_data)



vectorizer = CountVectorizer(stop_words=None)
counts = vectorizer.fit_transform(training_set['X'].values)


classifier = MultinomialNB()
targets = training_set['Y'].values
classifier.fit(counts, targets)


# Function to test a new email
def test_new_mail(new_mail_path):
    """
    This function takes the path to a new email and predicts its classification (spam or ham).
    """
    new_mail_message = extract_mail_body(new_mail_path)
    new_mail_vectorized = vectorizer.transform([new_mail_message])
    prediction = classifier.predict(new_mail_vectorized)[0]

    if prediction == SPAM_TYPE:
        print("This email is classified as SPAM.")
    else:
        print("This email is classified as HAM.")


# Example usage: Replace 'path/to/new/mail.txt' with the actual path to your new email
test_new_mail('C:\\Users\\hp\\PycharmProjects\\teest\\emails\\test2.txt')