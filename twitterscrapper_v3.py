import re
import joblib
import requests

BASE_URL = "https://scraping-api.thesocialproxy.com/twitter/v0/search/top"
BEARER_TOKEN = "ck_5d08e0a7c419af9d7a90c76d1019559b53d9815a"
BEARER_SECRET = "cs_9636ac43e09a78a4e8ab06181d061836ded1af39"

HEADERS = {"Api-Key": f"{BEARER_TOKEN}:{BEARER_SECRET}"}

query = {"query": "climate+change"}

# load the saved model and vectorizer

tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_model.pkl")


# Define function to clean tweets
def clean_tweet(value: str):
    value = re.sub(r"http\S+", "", value)  # Remove URLs
    value = re.sub(r"#\w+", "", value)  # Remove hashtags
    value = re.sub(r"@[A-Za-z0-9_]+", "", value)  # Remove mentions
    value = re.sub(
        r"[^A-Za-z\s]", "", value
    )  # Remove special characters (keep only letters and spaces)
    value = re.sub(r"\s+", " ", value).strip()  # Remove extra spaces
    return value


def get_model_prediction(x):

    x_numpy = tfidf.transform(x).toarray()

    pred = model.predict(x_numpy)

    result = ([*x], [*pred])

    return result

def extract_tweets():
    response = requests.get(BASE_URL, timeout=120, headers=HEADERS, params=query)

    response_json = response.json()

    response_tweets = response_json.get("tweets", [])

    total_tweets = []
    

    for tweet_object in response_tweets:
        tweet = tweet_object["tweet"].get("full_text", "")

        if tweet:
            cleaned_tweet = clean_tweet(value=tweet)
            total_tweets.append(cleaned_tweet)

    return total_tweets

if __name__ == "__main__":

    scrapped_tweets = extract_tweets()

    cleaned_tweets, model_prediction = get_model_prediction(x=scrapped_tweets)

    for tweet_str, prediction in zip(cleaned_tweets, model_prediction):
        print(f"Tweet: {tweet_str}")
        print(f"Prediction: {prediction}\n\n")
