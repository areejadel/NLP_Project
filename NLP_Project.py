import pandas as pd
import os
import nltk
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob                 
from deep_translator import GoogleTranslator   


# Load the dataset
df = pd.read_csv("/Users/areejbaw/Library/CloudStorage/OneDrive-UMMAL-QURAUNIVERSITY/Desktop/معالجات/project/ProjectA/Restaurant_Review.tsv",sep='\t')





def remove_punctuation(text):
    
  text_nopunct ="".join([char for char in text if char not in string.punctuation])
  return text_nopunct



def remove_stopWords(text):
  nltk.download('stopwords')
    
  negative_words = ["no", "not", "don't", "aren't", "couldn't", "didn't", "doesn't", 
                    "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", 
                    "wasn't", "weren't", "won't", "wouldn't"]
  stop_words = set(nltk.corpus.stopwords.words('english'))
  stop_words = stop_words.difference(set(negative_words))
  text_nostop=[char for char in text if char not in stop_words ]
  
  return text_nostop

nltk.download('punkt')

def tokenize(text):
 token = word_tokenize(text)
 return token 






df['CleanData']=df['Review'].apply(lambda x: remove_punctuation(x))

df['CleanData']=df['CleanData'].apply(lambda x:tokenize(x))

df['CleanData']=df['CleanData'].apply(lambda x: remove_stopWords(x))





#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






def summarize_text(text, num_sentences=1):
    # Create a plaintext parser
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Create an LSA summarizer
    summarizer = LsaSummarizer()

    # Summarize the document with the specified number of sentences
    summary = summarizer(parser.document, num_sentences)

    # Combine the sentences into a single string and return it
    return " ".join(str(sentence) for sentence in summary)


df["Summary"] = df['CleanData'].apply(lambda x: summarize_text(" ".join([summarize_text(word,1) for word in x])))




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



df["Arabic"] = df.iloc[:10,:]['Summary'].apply(lambda x: GoogleTranslator(source='en', target='ar').translate((x)))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    # Return 'c' if sentiment score is greater than 0, 'negative' if it's less than 0, and 'neutral' if it's 0
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'


df['Sentiment'] = df['Review'].apply(lambda x : analyze_sentiment(x))






positiveComments=""
negativeComments=""
def calculate_sentiment_percentage(df, column):

  # Create variables to store the number of positive and negative comments
  positive_count = 0
  negative_count = 0


  for index, row in df.iterrows():

      if row[column] and pd.notna(row[column]):

        if row['Sentiment'] == 'positive':
            positive_count += 1
        elif row['Sentiment'] == 'negative':
            negative_count += 1


  total_count = positive_count + negative_count
  positive_percentage = (positive_count / total_count )* 100

  negative_percentage = (negative_count / total_count) * 100

  positiveComments = "The positive comments about the " + column + " was: ", round(positive_percentage,3), "%"
  negativeComments= "The negative comments about the " + column + " was: ", round(negative_percentage,3), "%"
  return str(positiveComments)+"\n"+ str(negativeComments),positive_percentage,negative_percentage




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Create a column for the root of each word in the reviews
ps = PorterStemmer()
df["root"] = df["CleanData"].apply(lambda x: " ".join([ps.stem(word) for word in x]))

# Create new columns for price, service, and food
df["price"] = ""
df["service"] = ""
df["food"] = ""


def extract_descriptions(row):
  # Define some keywords for each category
  
  
  
  price_keywords = ["afford", "budget-friendli", "good valu", "worth the price", "within budget", "within reach", "within mean", "attract", "appeal", "invit", "welcom", "pleasant", "charm", "enjoy", "comfort", "pleasur", "satisfi", "gratifi", "delight", "happi", "joy", "bliss", "sensibl", "practic", "competit", "fair-price", "mid-rang", "standard", "averag", "mass-market", "no-frill", "basic", "simpl", "budget-consciou", "low-budget", "afford luxuri", "low-end", "entry-level", "bargain", "cheap", "expens", "unafford", "overpric", "costli", "premium", "unreason", "inflat", "outrag", "exorbit", "sky-high", "rip-off", "worthless", "shoddi", "poor", "low-qual", "trashi", "tacki", "crappi", "rotten", "defect", "flaw", "faulti", "broken", "damag", "ruin", "subpar", "substandard", "mediocr", "inadequ", "unsatisfactori"]
  service_keywords = ["great", "friend", "help", "welcom", "kind", "gentl", "respect", "smile", "accommod", "knowledg", "polit", "profess", "prompt", "respons", "satisfi", "sincer", "thought", "understand", "welcom", "alert", "help", "impress", "observ", "person", "reliabl", "skill", "speedi", "time", "admir", "adequ", "amic", "assur", "attent", "candid", "charm", "cheer", "clean", "commend", "compass", "compliment", "concern", "confid", "consider", "cordi", "rude", "slow", "fast", "wait", "ineffici", "unprompt", "unsatisfactori", "thoughtless", "unwelcom", "unhelp", "unimpress", "unobserv", "imperson", "unreli", "unskil", "untim", "unobtru", "aloof", "discourteous", "disinterest", "inattent", "incompet", "inconsider", "irrit", "malici", "obnoxi", "sarcastic", "unprofession", "brood", "disdain", "quick", "invit", "accomod", "worst", "unsatisfi", "unsatisfi", "mess", "insult", "poor", "aw", "suck"]
  food_keywords = ["great", "delici", "tasti", "flavor", "fresh", "yummi", "appet", "cook", "crisp", "crunchi", "juici", "mouthwat", "rich", "savori", "season", "spici", "sweet", "tender", "aromat", "balanc", "blend", "bold", "color", "combin", "complex", "creami", "crumb", "decad", "distinct", "divin", "fluffi", "freshest", "full-flavor", "hearti", "homemad", "infus", "ingredi", "innov", "irresist", "layer", "melt", "mild", "nutriti", "perfect", "perfectli", "plenti", "pure", "roast", "robust", "savouri", "sear", "smoki", "bland", "blandi", "blandness", "burnt", "burnt out", "burnt to a crisp", "burnt to a cinder", "burnt to a shell", "chewi", "cold", "dri", "flavorless", "greasi", "hard", "insipid", "overcook", "over-salt", "over-spice", "past it prime", "raw", "tasteless", "tough", "undercook", "under-salt", "under-spice", "wateri", "wilt", "yucki", "delici", "tasti", "spici", "stale", "huge", "recommend", "beauti", "amaz", "delight", "decent", "season", "yummi", "authent", "healthi", "moist", "nasti", "pale", "unhealthi", "soggi", "warm", "overrat", "bad", "disgust", "averag", "sick"]
 # Split the root column into a list of words
  words = row["root"].split()

  # Loop through the words and check if they match any keyword
  for word in words:
    if word in price_keywords:
      row["price"] += word + " "
    elif word in service_keywords:
      row["service"] += word + " "
    elif word in food_keywords:
      row["food"] += word + " "

  # Return the row with the updated columns
  return row

# Apply the function to the dataframe
df = df.apply(extract_descriptions, axis=1)


  
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





# # Preprocess the text data
# df['CleanData'] = df['Review'].apply(preprocess_text)


# vectorizer = CountVectorizer()

X = pd.get_dummies(df['Sentiment'])
# X = pd.concat([pd.DataFrame(X_sentiment)], axis=1)
# X.columns = X.columns.astype(str) #
target = df['Liked']

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

# Train a random forest classifier on the training set
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X_train, y_train)

# Evaluate the classifier on the testing set
score = rfc.score(X_test, y_test)
print('Accuracy:', score)
# Make predictions on the testing set
y_pred = rfc.predict(X_test)




df.to_csv('table.txt', index=False)
df.to_excel("table.xlsx")
print(os.path.join(os.getcwd(), 'table.txt'))


tee0 ,positive_percentagefo ,negative_percentagefo = calculate_sentiment_percentage(df, "food")
tee1,positive_percentageSe,negative_percentageSe  =calculate_sentiment_percentage(df, "service")
tee2 ,positive_percentagePr ,negative_percentagePr =calculate_sentiment_percentage(df, "price")
tee="\n\n\n"+str(tee0)+"\n"+ GoogleTranslator(source='en', target='ar').translate((tee0))+"\n\n\n"+str(tee1)+"\n"+ GoogleTranslator(source='en', target='ar').translate((tee1))+"\n\n\n"+str(tee2)+"\n"+ GoogleTranslator(source='en', target='ar').translate((tee2))+"\n\n\n"+ "For more details about any of them, click on the following buttons:\n"

# textl=calculate_sentiment_percentage(df, "food")+calculate_sentiment_percentage(df, "service")+calculate_sentiment_percentage(df, "price")





df.head(20)




import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk


def show_pie_chart(title,positive_percentage,negative_percentage):
    new_window = tk.Toplevel(window)


    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]
    colors = ['#E8A09A', '#FBE29F']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    canvas = FigureCanvas(fig)
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    fig_image = ImageTk.PhotoImage(master=new_window, image=pil_image)

    chart_label = tk.Label(new_window, image=fig_image)
    chart_label.image = fig_image 
    chart_label.pack()


window = tk.Tk()
window.geometry("600x600")
window.title("Sentiment Analysis")

label = tk.Label(window, text=tee)
label.pack()

food_button = tk.Button(window, text="Food", command=lambda: show_pie_chart("Food",positive_percentagefo,negative_percentagefo))
food_button.pack()
service_button = tk.Button(window, text="Service", command=lambda: show_pie_chart("Service",positive_percentageSe,negative_percentageSe))
service_button.pack()
price_button = tk.Button(window, text="Price", command=lambda: show_pie_chart("Price",positive_percentagePr,negative_percentagePr))
price_button.pack()

window.mainloop()



