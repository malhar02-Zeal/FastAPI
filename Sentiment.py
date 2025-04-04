import pandas as pd
import json
import os
import traceback
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from transformers import pipeline
import unicodedata
import datetime
# try:
#     import langdetect
# except ImportError:
#     # pip install langdetect

from langdetect import detect, LangDetectException
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

AWARENESS_KEYWORDS = ["researching", "learning about", "just discovered", "heard about", "first time", "considering", "curious about", "found out about", "searching for", "interested in", "exploring", "want to try", "what is", "new to", "looking into", "wondering about", "trying to find", "thinking about", "exploring options", "looking for information", "open to exploring"]
TRIAL_KEYWORDS = ["trying", "testing", "experimenting", "first time using", "not sure about", "getting started", "evaluating", "initial experience", "trying out", "giving it a go", "just installed", "using for the first time", "curious to see how it works", "trying before buying", "on trial", "considering usage", "testing out features", "getting a feel", "exploring the functionality", "initial test"]
PURCHASE_KEYWORDS = ["bought", "purchased", "just ordered", "ordered", "received my order", "finally bought", "made a purchase", "added to cart", "purchased recently", "transaction complete", "excited for delivery", "looking forward to receiving", "confirmed my order", "placed an order", "completed the purchase", "ordered today", "paid for", "just purchased", "purchased now", "bought it", "checkout complete"]
SUPPORT_KEYWORDS = ["issue", "problem", "troubleshooting", "help", "support", "customer service", "question", "concern", "not working", "fix", "error", "refund", "defective", "doesn't work", "broken", "return", "repair", "stuck", "difficulty", "unresolved", "complaint", "customer care", "need assistance", "not as expected", "can't get it to work", "malfunction", "support request", "issues with", "technical support", "waiting for help"]
RENEWAL_KEYWORDS = ["renew", "subscribe", "renewal", "reorder", "would buy again", "plan to renew", "considering renewal", "loyal customer", "subscription end", "looking to renew", "repeat purchase", "continuing with", "extending subscription", "re-up", "re-purchase", "next order", "re-subscribing", "will buy again", "renewing soon", "re-subscribed", "updating subscription", "about to renew", "renewal reminder", "renewed subscription", "reordering"]
COMPLAINT_KEYWORDS = ["issue", "problem", "error", "not working", "dissatisfied", "broken", "faulty", "defective", "refund", "return", "damaged", "disappointed", "hate", "poor quality", "worst", "can't use", "doesn't work", "malfunction", "missing", "delayed", "unresolved", "unsatisfied", "complain", "flaw", "negative experience", "bad experience", "not as expected", "too slow", "inconvenient", "not worth", "low performance", "terrible", "not happy", "not recommended", "unresponsive", "problematic", "not as advertised", "unhappy"]
COMPLIMENT_KEYWORDS = ["great", "good", "excellent", "amazing", "fantastic", "best", "perfect", "satisfied", "happy", "love", "wonderful", "outstanding", "impressed", "highly recommend", "recommend", "quality", "worth it", "superb", "awesome", "affordable", "worth the money", "top-notch", "reliable", "fast", "smooth", "beautiful", "delightful", "perfect fit", "satisfied with", "good value", "effective", "nice", "convenient", "great value", "outstanding service", "very happy", "brilliant", "phenomenal"]

def convert_numpy_types(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def detect_complaint_or_compliment_multilingual(review_text, sentiment, language):
    if language == 'en':
        return detect_complaint_or_compliment(review_text, sentiment)
    if sentiment == 'positive':
        return 'Compliment'
    elif sentiment == 'negative':
        return 'Complaint'
    else:
        return 'Neutral'

def detect_complaint_or_compliment(review_text, sentiment):
    if not isinstance(review_text, str):
        return 'Neutral'
    review_lower = review_text.lower()
    complaint_keyword_match = any(keyword in review_lower for keyword in COMPLAINT_KEYWORDS)
    compliment_keyword_match = any(keyword in review_lower for keyword in COMPLIMENT_KEYWORDS)
    if complaint_keyword_match and not compliment_keyword_match:
        return 'Complaint'
    elif compliment_keyword_match and not complaint_keyword_match:
        return 'Compliment'
    if sentiment == 'negative' and complaint_keyword_match:
        return 'Complaint'
    elif sentiment == 'positive' and compliment_keyword_match:
        return 'Compliment'
    return 'Neutral'

def detect_customer_journey_stage(review_text, sentiment):
    # If review_comment is empty or contains only white spaces, set customer_journey_stage to None
    if not isinstance(review_text, str) or review_text.strip() == "":
        return "Not Detected"  # Return None for empty or whitespace-only reviews

    review_lower = review_text.lower()
    def check_keyword_proximity(keywords, text, max_distance=5):
        words = text.split()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for i, word in enumerate(words):
                if keyword_lower in word:
                    start = max(0, i - max_distance)
                    end = min(len(words), i + max_distance + 1)
                    for j in range(start, end):
                        if j != i and any(key in words[j].lower() for key in keywords):
                            return True
        return False

    all_keywords = {
        "Awareness": AWARENESS_KEYWORDS,
        "Trial": TRIAL_KEYWORDS,
        "Purchase": PURCHASE_KEYWORDS,
        "Support": SUPPORT_KEYWORDS,
        "Renewal": RENEWAL_KEYWORDS
    }

    matched_stages = []
    for stage, keywords in all_keywords.items():
        keyword_match = any(keyword.lower() in review_lower for keyword in keywords)
        proximity_match = check_keyword_proximity(keywords, review_text)
        if keyword_match or proximity_match:
            matched_stages.append(stage)

    if not matched_stages:
        if sentiment == 'positive':
            return "Purchase"
        elif sentiment == 'negative':
            return "Support"
        else:
            return "Awareness"

    if len(matched_stages) > 1:
        stage_priority = ["Awareness", "Trial", "Purchase", "Support", "Renewal"]
        for priority_stage in stage_priority:
            if priority_stage in matched_stages:
                return priority_stage

    return matched_stages[0]


# UPDATED: compute_token_frequencies_by_location now groups tokens by location and by review date.
def compute_token_frequencies_by_location(df):
    location_token_freq = []
    for location_number, group in df.groupby('location_number'):
        email_id = group['email_id'].dropna().unique()[0] if len(group['email_id'].dropna().unique()) > 0 else None
        account_id = group['account_id'].dropna().unique()[0] if len(group['account_id'].dropna().unique()) > 0 else None

        # Compute location-wide token frequencies (fields with "location_wise_")
        all_tokens = [token for tokens in group['tokens'] for token in tokens]
        token_counts = Counter(all_tokens)
        location_wise_token_frequencies = [
            {"word": word, "frequency": count}
            for word, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Group tokens by review date within the location (fields with "datewise_")
        date_grouped_token_freq = []
        # Group by the "create_review_date" field (assumed to be present in flattened data)
        for review_date, date_group in group.groupby('create_review_date'):
            if not review_date or review_date == "":
                continue
            all_date_tokens = [token for tokens in date_group['tokens'] for token in tokens]
            date_token_counts = Counter(all_date_tokens)
            datewise_token_frequencies = [
                {"word": word, "frequency": count}
                for word, count in sorted(date_token_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            date_grouped_token_freq.append({
                "date": review_date,
                "datewise_token_frequencies": datewise_token_frequencies
            })

        location_token_freq.append({
            "location_number": location_number,
            "email_id": email_id,
            "account_id": account_id,
            "location_wise_token_frequencies": location_wise_token_frequencies,
            "date_grouped_token_frequencies": date_grouped_token_freq
        })
    return location_token_freq

def detect_review_language(text):
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'

def detect_aspects_dominant(text, classifier, threshold=0.3, margin=0.1):
    ASPECT_LABELS = [
        "Overall Experience",
        "Product Quality",
        "Service Quality",
        "Pricing / Value",
        "Delivery & Speed",
        "Staff Behavior",
        "Workplace Treatment",
        "Cleanliness"
    ]
    if not isinstance(text, str) or not text.strip():
        return ["Not Detected"]
    try:
        result = classifier(
            text,
            candidate_labels=ASPECT_LABELS,
            multi_label=True
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        if top_label == "Overall Experience" and top_score >= threshold:
            return ["Overall Experience"]
        aspects_found = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= threshold and (top_score - score) <= margin:
                aspects_found.append(label)
        return aspects_found if aspects_found else ["Not Detected"]
    except Exception as e:
        print(f"Error detecting aspects for text: {text[:50]}... - {str(e)}")
        return ["Not Detected"]

def clean_locality(locality):
    if not isinstance(locality, str):
        return ''
    words = locality.split()
    cleaned_words = [word for word in words if not word.isnumeric()]
    return ' '.join(cleaned_words).strip()

def extract_date_from_datetime(datetime_str):
    if not isinstance(datetime_str, str):
        return None
    try:
        dt = datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.date().isoformat()
    except (ValueError, TypeError):
        return None

def flatten_reviews(data):
    flattened_reviews = []
    for email_item in data:
        email_id = email_item.get('email_id')
        for account in email_item.get('accounts', []):
            account_id = account.get('account_id')
            gmb_account_name = account.get('gmb_account_name')
            for profile in account.get('profiles', []):
                location_name = profile.get('location_name')
                location_number = profile.get('location_number')
                locality = clean_locality(profile.get('locality', ''))
                address_lines = profile.get('addressLines', [])
                latitude = profile.get('latitude')
                longitude = profile.get('longitude')
                for review in profile.get('reviews', []):
                    flat_review = {
                        'email_id': email_id,
                        'account_id': account_id,
                        'gmb_account_name': gmb_account_name,
                        'location_name': location_name,
                        'location_number': location_number,
                        'addressLines': address_lines,
                        'locality': locality,
                        'latitude': latitude,
                        'longitude': longitude,
                        'review_id': review.get('review_id', ''),
                        'review_comment': review.get('review_comment', ''),
                        'reviewer_name': review.get('reviewer_name', ''),
                        'review_star_rating': review.get('review_star_rating', ''),
                        'reply_comment': review.get('reply_comment', ''),
                        'reply_update_time': review.get('reply_update_time', ''),
                        'create_review_time': review.get('review_create_time', ''),
                        'create_review_date': extract_date_from_datetime(review.get('review_create_time', '')),
                        'reply_update_date': extract_date_from_datetime(review.get('reply_update_time', ''))
                    }
                    flattened_reviews.append(flat_review)
    return flattened_reviews

def preprocess_text(text, language='english'):
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize('NFKD', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        try:
            stop_words = set(stopwords.words(language))
        except:
            stop_words = set(stopwords.words('english'))
        tokens = re.findall(r'\b\w+\b', text)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        return tokens
    except Exception as e:
        print(f"Tokenization error: {e}")
        return []

def is_extremely_short(review_text, min_word_count=3):
    tokens = preprocess_text(review_text)
    return len(tokens) < min_word_count

def compute_cosine_similarity(text1, text2):
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    try:
        vectorizer = CountVectorizer().fit_transform([text1, text2])
        return cosine_similarity(vectorizer)[0][1]
    except ValueError:
        return 0.0

def check_review_flags(df, review_column='review_comment', user_id_column='reviewer_name', rating_column='review_star_rating'):
    all_flags = []
    for i, row in df.iterrows():
        review_text = row[review_column] if isinstance(row[review_column], str) else ""
        user_id = row[user_id_column] if user_id_column in df.columns and pd.notna(row[user_id_column]) else f"user_{i}"
        rating = row[rating_column] if rating_column in df.columns else ""
        sentiment = row.get('sentiment', '')
        is_short = is_extremely_short(review_text)
        exact_duplicate_with_same_user = False
        near_duplicate_with_same_user = False
        near_duplicate_with_diff_user = False
        for j, other_row in df.iterrows():
            if i == j:
                continue
            other_text = other_row[review_column] if isinstance(other_row[review_column], str) else ""
            other_user_id = other_row[user_id_column] if user_id_column in df.columns and pd.notna(other_row[user_id_column]) else f"user_{j}"
            if user_id == other_user_id and review_text == other_text and review_text != "":
                exact_duplicate_with_same_user = True
                break
            if review_text and other_text:
                similarity = compute_cosine_similarity(review_text, other_text)
                if similarity > 0.8:
                    if user_id == other_user_id:
                        near_duplicate_with_same_user = True
                        break
                    else:
                        near_duplicate_with_diff_user = True
        flag = ""
        if exact_duplicate_with_same_user:
            flag = "Extremely Short, Repeated User with Exactly Duplicate" if is_short else "Repeated User with Exactly Duplicate"
        elif near_duplicate_with_same_user:
            flag = "Extremely Short, Repeated User with Nearly Duplicate" if is_short else "Repeated User with Nearly Duplicate"
        elif near_duplicate_with_diff_user:
            flag = "Extremely Short, Nearly Duplicate" if is_short else "Nearly Duplicate"
        elif is_short and review_text:
            flag = "Extremely Short"
        if sentiment and rating:
            rating_match = False
            if sentiment == 'positive' and (rating in ['FOUR', 'FIVE', 4, 5]):
                rating_match = True
            elif sentiment == 'neutral' and (rating in ['TWO', 'THREE', 2, 3]):
                rating_match = True
            elif sentiment == 'negative' and (rating in ['ZERO', 'ONE', 0, 1]):
                rating_match = True
            if rating_match:
                flag = flag + (", " if flag else "") + "Sentiment vs Rating Matched"
            else:
                flag = flag + (", " if flag else "") + "Sentiment vs Rating Not Matched"
        all_flags.append(flag)
    return all_flags

def load_sentiment_model():
    try:
        print("Loading sentiment analysis model...")
        classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        print("Model loaded successfully")
        return classifier
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None

def analyze_sentiment_batch(texts, classifier, batch_size=500):
    print(f"Analyzing sentiment for texts with batch size {batch_size}")
    valid_texts = []
    indices = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_texts.append(text)
            indices.append(i)
    if not valid_texts:
        return [None] * len(texts)
    try:
        results = classifier(
            valid_texts,
            candidate_labels=['positive', 'negative', 'neutral'],
            multi_label=False
        )
        output = [None] * len(texts)
        for i, result in enumerate(results):
            sentiment = result['labels'][0]
            output[indices[i]] = sentiment
        return output
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        traceback.print_exc()
        return [None] * len(texts)

def aggregate_sentiments_by_profile(df, profile_column='location_number', sentiment_column='sentiment', email_id_column='email_id', account_id_column='account_id', reply_column='reply_comment'):
    required_columns = [profile_column, sentiment_column, reply_column, account_id_column]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in DataFrame")
            return []
    sentiment_counts = df.groupby([profile_column, sentiment_column]).size().reset_index(name='count')
    reply_counts = df.groupby(profile_column).apply(
        lambda x: {
            'has_reply': len(x[x[reply_column].notna() & (x[reply_column] != '')]),
            'no_reply': len(x[x[reply_column].isna() | (x[reply_column] == '')])
        }
    ).reset_index()
    reply_counts.columns = [profile_column, 'reply_status']
    aggregated_sentiments = []
    for profile in sentiment_counts[profile_column].unique():
        profile_sentiments = sentiment_counts[sentiment_counts[profile_column] == profile]
        profile_replies = reply_counts[reply_counts[profile_column] == profile]['reply_status'].iloc[0]
        profile_account_id = df[df[profile_column] == profile][account_id_column].dropna().unique()
        profile_account_id = profile_account_id[0] if len(profile_account_id) > 0 else None
        sentiment_data = {
            'location_number': profile,
            'account_ids': profile_account_id,
            'sentiments': [
                {
                    'type': sentiment.capitalize(),
                    'count': int(count)
                }
                for sentiment, count in zip(
                    profile_sentiments[sentiment_column],
                    profile_sentiments['count']
                )
            ],
            'replies': [
                {'type': 'Has Reply', 'count': profile_replies['has_reply']},
                {'type': 'No Reply', 'count': profile_replies['no_reply']}
            ]
        }
        if email_id_column in df.columns:
            profile_email_ids = df[df[profile_column] == profile][email_id_column].dropna().unique()
            if len(profile_email_ids) > 0:
                sentiment_data['email_id'] = profile_email_ids[0]
        for sentiment_type in ['Positive', 'Neutral', 'Negative']:
            if sentiment_type.lower() not in set(profile_sentiments[sentiment_column]):
                sentiment_data['sentiments'].append({
                    'type': sentiment_type,
                    'count': 0
                })
        aggregated_sentiments.append(sentiment_data)
    return aggregated_sentiments

def load_aspect_model():
    try:
        print("Loading aspect classification model...")
        aspect_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            multi_label=True
        )
        print("Aspect model loaded successfully")
        return aspect_classifier
    except Exception as e:
        print(f"Error loading aspect model: {str(e)}")
        traceback.print_exc()
        return None

def convert_star_rating_to_numeric(rating):
    if isinstance(rating, (int, float)):
        return float(rating)
    if not isinstance(rating, str):
        return None
    rating_upper = rating.upper().strip()
    rating_map = {
        'ZERO': 0.0,
        'ONE': 1.0,
        'TWO': 2.0,
        'THREE': 3.0,
        'FOUR': 4.0,
        'FIVE': 5.0
    }
    if rating_upper in rating_map:
        return rating_map[rating_upper]
    try:
        return float(rating)
    except (ValueError, TypeError):
        return None

def enhance_sentiment_count_json(df, output_file='Sentiment_Count.json'):
    df['review_star_rating_number'] = df['review_star_rating'].apply(convert_star_rating_to_numeric)
    location_metrics = []
    for location in df['location_number'].unique():
        location_df = df[df['location_number'] == location]
        profile_sentiments = aggregate_sentiments_by_profile(
            location_df,
            profile_column='location_number',
            sentiment_column='sentiment',
            email_id_column='email_id',
            account_id_column='account_id',
            reply_column='reply_comment'
        )[0]
        sentiment_counts = location_df['sentiment'].value_counts()
        location_wise_sentiments = []
        for sentiment_type in ['Positive', 'Neutral', 'Negative']:
            count = sentiment_counts.get(sentiment_type.lower(), 0)
            location_wise_sentiments.append({
                'type': sentiment_type,
                'count': int(count)
            })
        profile_sentiments['location_wise_sentiments'] = location_wise_sentiments
        reply_counts = {
            'has_reply': int((location_df['reply_comment'].notna() & (location_df['reply_comment'] != '')).sum()),
            'no_reply': int((location_df['reply_comment'].isna() | (location_df['reply_comment'] == '')).sum())
        }
        profile_sentiments['location_wise_replies'] = [
            {'type': 'Has Reply', 'count': reply_counts['has_reply']},
            {'type': 'No Reply', 'count': reply_counts['no_reply']}
        ]
        complaint_compliment_counts = location_df['complaint_or_compliment'].value_counts()
        profile_sentiments['location_wise_complaint_compliment'] = [
            {
                'type': category.capitalize(),
                'count': int(count)
            }
            for category, count in complaint_compliment_counts.items()
        ]
        date_df = location_df[location_df['create_review_date'].notna() & (location_df['create_review_date'] != '')]
        date_grouped_metrics = []
        for review_date in date_df['create_review_date'].unique():
            date_specific_df = date_df[date_df['create_review_date'] == review_date]
            total_date_reviews = len(date_specific_df)
            sentiment_counts = date_specific_df['sentiment'].value_counts()
            sentiment_rates = {
                'datewise_positive_rate': round(sentiment_counts.get('positive', 0) / total_date_reviews, 2),
                'datewise_neutral_rate': round(sentiment_counts.get('neutral', 0) / total_date_reviews, 2),
                'datewise_negative_rate': round(sentiment_counts.get('negative', 0) / total_date_reviews, 2)
            }
            datewise_sentiments = []
            for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_counts.get(sentiment_type.lower(), 0)
                datewise_sentiments.append({
                    'type': sentiment_type,
                    'count': int(count)
                })
            date_complaint_compliment_counts = date_specific_df['complaint_or_compliment'].value_counts()
            date_data = {
                'date': review_date,
                'datewise_total_reviews': total_date_reviews,
                'datewise_sentiments': datewise_sentiments,
                'datewise_sentiment_rates': sentiment_rates,
                'datewise_complaint_compliment': [
                    {
                        'type': category.capitalize(),
                        'count': int(count)
                    }
                    for category, count in date_complaint_compliment_counts.items()
                ]
            }
            reply_counts = {
                'has_reply': int((date_specific_df['reply_comment'].notna() & (date_specific_df['reply_comment'] != '')).sum()),
                'no_reply': int((date_specific_df['reply_comment'].isna() | (date_specific_df['reply_comment'] == '')).sum())
            }
            date_data['datewise_replies'] = [
                {'type': 'Has Reply', 'count': reply_counts['has_reply']},
                {'type': 'No Reply', 'count': reply_counts['no_reply']}
            ]
            star_rating_sentiment_distribution = date_specific_df.groupby(['review_star_rating', 'sentiment']).size().unstack(fill_value=0)
            date_data['datewise_star_ratings'] = []
            for rating in star_rating_sentiment_distribution.index:
                rating_data = {
                    'rating': str(rating),
                    'total_count': star_rating_sentiment_distribution.loc[rating].sum(),
                    'sentiments': [
                        {'type': sentiment.capitalize(), 'count': count}
                        for sentiment, count in star_rating_sentiment_distribution.loc[rating].items()
                    ]
                }
                date_data['datewise_star_ratings'].append(rating_data)
            language_sentiment_distribution = date_specific_df.groupby(['language', 'sentiment']).size().unstack(fill_value=0)
            date_data['datewise_languages'] = []
            for language in language_sentiment_distribution.index:
                language_data = {
                    'language': str(language),
                    'total_count': language_sentiment_distribution.loc[language].sum(),
                    'sentiments': [
                        {'type': sentiment.capitalize(), 'count': count}
                        for sentiment, count in language_sentiment_distribution.loc[language].items()
                    ]
                }
                date_data['datewise_languages'].append(language_data)
            aspect_df = date_specific_df.assign(aspect=date_specific_df['aspects'].str.split(', ')).explode('aspect')
            aspect_sentiment_distribution = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
            date_data['datewise_aspects'] = []
            for aspect in aspect_sentiment_distribution.index:
                aspect_data = {
                    'aspect': str(aspect),
                    'total_count': aspect_sentiment_distribution.loc[aspect].sum(),
                    'sentiments': [
                        {'type': sentiment.capitalize(), 'count': count}
                        for sentiment, count in aspect_sentiment_distribution.loc[aspect].items()
                    ]
                }
                date_data['datewise_aspects'].append(aspect_data)
            journey_stage_sentiment_distribution = date_specific_df.groupby(['customer_journey_stage', 'sentiment']).size().unstack(fill_value=0)
            date_data['datewise_customer_journey_stages'] = []
            for journey_stage in journey_stage_sentiment_distribution.index:
                journey_stage_data = {
                    'stage': str(journey_stage),
                    'total_count': journey_stage_sentiment_distribution.loc[journey_stage].sum(),
                    'sentiments': [
                        {'type': sentiment.capitalize(), 'count': count}
                        for sentiment, count in journey_stage_sentiment_distribution.loc[journey_stage].items()
                    ]
                }
                date_data['datewise_customer_journey_stages'].append(journey_stage_data)
            date_grouped_metrics.append(date_data)
        profile_sentiments['date_grouped_metrics'] = date_grouped_metrics
        total_reviews = len(location_df)
        sentiment_counts = location_df['sentiment'].value_counts()
        sentiment_rates = {
            'location_wise_positive_rate': round(sentiment_counts.get('positive', 0) / total_reviews, 2),
            'location_wise_neutral_rate': round(sentiment_counts.get('neutral', 0) / total_reviews, 2),
            'location_wise_negative_rate': round(sentiment_counts.get('negative', 0) / total_reviews, 2)
        }
        profile_sentiments['location_wise_sentiment_rates'] = sentiment_rates
        sentiment_by_language = location_df.groupby(['language', 'sentiment']).size().unstack(fill_value=0)
        location_wise_language_sentiment_list = []
        for language in sentiment_by_language.index:
            sentiments = []
            for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_by_language.loc[language].get(sentiment_type.lower(), 0)
                sentiments.append({
                    'type': sentiment_type,
                    'count': int(count)
                })
            language_data = {
                'language': language,
                'sentiments': sentiments
            }
            location_wise_language_sentiment_list.append(language_data)
        profile_sentiments['location_wise_sentiment_by_language'] = location_wise_language_sentiment_list
        sentiment_by_star_rating = location_df.groupby(['review_star_rating', 'sentiment']).size().unstack(fill_value=0)
        location_wise_star_rating_sentiment_list = []
        for star_rating in sentiment_by_star_rating.index:
            sentiments = []
            for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_by_star_rating.loc[star_rating].get(sentiment_type.lower(), 0)
                sentiments.append({
                    'type': sentiment_type,
                    'count': int(count)
                })
            star_rating_data = {
                'star_rating': str(star_rating),
                'sentiments': sentiments
            }
            location_wise_star_rating_sentiment_list.append(star_rating_data)
        profile_sentiments['location_wise_sentiment_by_star_rating'] = location_wise_star_rating_sentiment_list
        aspect_df = location_df.assign(aspect=location_df['aspects'].str.split(', ')).explode('aspect')
        sentiment_by_aspect = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        location_wise_aspect_sentiment_list = []
        for aspect in sentiment_by_aspect.index:
            sentiments = []
            for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_by_aspect.loc[aspect].get(sentiment_type.lower(), 0)
                sentiments.append({
                    'type': sentiment_type,
                    'count': int(count)
                })
            aspect_data = {
                'aspect': str(aspect),
                'sentiments': sentiments
            }
            location_wise_aspect_sentiment_list.append(aspect_data)
        profile_sentiments['location_wise_sentiment_by_aspect'] = location_wise_aspect_sentiment_list
        sentiment_by_journey_stage = location_df.groupby(['customer_journey_stage', 'sentiment']).size().unstack(fill_value=0)
        location_wise_journey_stage_sentiment_list = []
        for journey_stage in sentiment_by_journey_stage.index:
            sentiments = []
            for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_by_journey_stage.loc[journey_stage].get(sentiment_type.lower(), 0)
                sentiments.append({
                    'type': sentiment_type,
                    'count': int(count)
                })
            journey_stage_data = {
                'stage': str(journey_stage),
                'sentiments': sentiments
            }
            location_wise_journey_stage_sentiment_list.append(journey_stage_data)
        profile_sentiments['location_wise_sentiment_by_customer_journey_stage'] = location_wise_journey_stage_sentiment_list
        aspect_counts = location_df['aspects'].str.split(', ', expand=True).stack().value_counts()
        profile_sentiments['location_wise_aspect_counts'] = [
            {'type': aspect, 'count': int(count)}
            for aspect, count in aspect_counts.items()
        ]
        journey_stage_counts = location_df['customer_journey_stage'].value_counts()
        profile_sentiments['location_wise_customer_journey_stages'] = [
            {'type': stage, 'count': int(count)}
            for stage, count in journey_stage_counts.items()
        ]
        # Remove the original "sentiments" and "replies" keys
        if 'sentiments' in profile_sentiments:
            del profile_sentiments['sentiments']
        if 'replies' in profile_sentiments:
            del profile_sentiments['replies']
        # Reorder the dictionary so that email_id comes before location_number
        ordered_profile = {}
        ordered_profile["email_id"] = profile_sentiments.pop("email_id", None)
        ordered_profile["location_number"] = profile_sentiments.pop("location_number", None)
        ordered_profile.update(profile_sentiments)
        profile_sentiments = ordered_profile
        location_metrics.append(profile_sentiments)
    location_metrics = [convert_numpy_types(metric) for metric in location_metrics]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(location_metrics, f, indent=4, ensure_ascii=False)
    print(f"Enhanced metrics saved to {output_file}")

def count_reviews_before_and_after(original_data, processed_data):
    output = []
    for email_item in original_data:
        email_id = email_item.get('email_id', 'Unknown Email')
        output.append(f"Email ID: {email_id}")
        accounts = email_item.get('accounts', [])
        output.append(f"Number of Accounts: {len(accounts)}")
        for account in accounts:
            account_id = account.get('account_id', 'Unknown Account')
            output.append(f"Account ID: {account_id}")
            profiles = account.get('profiles', [])
            output.append(f"Number of Locations: {len(profiles)}")
            for profile in profiles:
                location_number = profile.get('location_number', 'Unknown Location')
                reviews_before = len(profile.get('reviews', []))
                reviews_with_comments_before = len([
                    review for review in profile.get('reviews', [])
                    if review.get('review_comment', '').strip()
                ])
                processed_reviews = [
                    r for r in processed_data
                    if (r.get('location_number') == location_number and
                        r.get('email_id') == email_id and
                        r.get('account_id') == account_id)
                ]
                processed_reviews_count = len(processed_reviews)
                processed_reviews_with_sentiments = len([
                    r for r in processed_reviews
                    if r.get('sentiment') is not None
                ])
                output.append(f"Location Number: {location_number}")
                output.append(f"Total Reviews: {reviews_before}")
                output.append(f"Reviews with Comments: {reviews_with_comments_before}")
                output.append(f"Total Processed Reviews: {processed_reviews_count}")
                output.append(f"Reviews with Sentiments: {processed_reviews_with_sentiments}")
                output.append("")
        output.append("\n")
    return "\n".join(output)

def process_reviews(file_path, output_prefix, classifier, aspect_classifier):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return False
    flattened_reviews = flatten_reviews(data)
    df = pd.DataFrame(flattened_reviews)
    df['review_star_rating_number'] = df['review_star_rating'].apply(convert_star_rating_to_numeric)
    batch_size = 500
    reviews = df['review_comment'].tolist()
    sentiments = analyze_sentiment_batch(reviews, classifier, batch_size=batch_size)
    df['sentiment'] = sentiments
    print("Detecting review languages...")
    df['language'] = [detect_review_language(str(review)) for review in df['review_comment']]
    print("Preprocessing tokens with language detection...")
    df['tokens'] = df.apply(
        lambda row: preprocess_text(
            str(row['review_comment']),
            language=row['language'] if row['language'] != 'unknown' else 'english'
        ),
        axis=1
    )
    print("Detecting aspects...")
    aspects = [detect_aspects_dominant(str(review), aspect_classifier) for review in df['review_comment']]
    df['aspects'] = [', '.join(aspect_list) if aspect_list else '' for aspect_list in aspects]
    print("Detecting customer journey stages...")
    customer_journey_stages = [
        detect_customer_journey_stage(str(review), sentiment)
        for review, sentiment in zip(df['review_comment'], df['sentiment'])
    ]
    df['customer_journey_stage'] = customer_journey_stages
    print("Detecting complaint or compliment...")
    complaint_or_compliment = [
        detect_complaint_or_compliment_multilingual(str(review), sentiment, language)
        for review, sentiment, language in zip(df['review_comment'], df['sentiment'], df['language'])
    ]
    df['complaint_or_compliment'] = complaint_or_compliment
    print("Applying flagging logic...")
    df['flag'] = check_review_flags(df, review_column='review_comment', user_id_column='reviewer_name', rating_column='review_star_rating')
    print("Applying flagging logic...")
    df['flag'] = check_review_flags(df, review_column='review_comment', user_id_column='reviewer_name', rating_column='review_star_rating')
    flag_counts = Counter(df['flag'])
    print("\nFlag distribution:")
    for flag, count in flag_counts.items():
        flag_display = flag if flag else "[No Flag]"
        print(f"{flag_display}: {count}")
    print("\nLanguage Distribution:")
    language_counts = df['language'].value_counts()
    for lang, count in language_counts.items():
        print(f"  - {lang}: {count}")
    journey_stage_counts = df['customer_journey_stage'].value_counts()
    print("\nCustomer Journey Stage Distribution:")
    for stage, count in journey_stage_counts.items():
        print(f"  - {stage}: {count}")
    complaint_compliment_counts = df['complaint_or_compliment'].value_counts()
    print("\nComplaint/Compliment Distribution:")
    for category, count in complaint_compliment_counts.items():
        print(f"  - {category}: {count}")
    json_output = f"{output_prefix}.json"
    print(f"Saving results to JSON: {json_output}")
    output_data = df.to_dict(orient='records')
    output_data = [convert_numpy_types(item) for item in output_data]
    for item in output_data:
        if 'tokens' in item:
            item['tokens'] = list(item['tokens'])
        item['language'] = item.get('language', 'unknown')
        item['complaint_or_compliment'] = item.get('complaint_or_compliment', 'Neutral')
        item['review_star_rating_number'] = item.get('review_star_rating_number', None)
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print("Analyzing token frequencies by location...")
    token_frequencies = compute_token_frequencies_by_location(df)
    token_freq_output = "Word_Count.json"
    with open(token_freq_output, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(token_frequencies), f, indent=4, ensure_ascii=False)
    print("Enhancing Sentiment Count JSON...")
    enhance_sentiment_count_json(df, 'Sentiment_Count.json')
    print(f"Token frequencies saved to: {token_freq_output}")
    print("\nProcessing statistics:")
    print(f"Total records processed: {len(df)}")
    print("\nSentiment distribution:")
    sentiment_counts = df['sentiment'].value_counts(dropna=False)
    for sentiment, count in sentiment_counts.items():
        print(f"  - {sentiment if sentiment is not None else 'Undefined'}: {count}")
    print("\nFlag distribution:")
    for flag, count in flag_counts.items():
        flag_display = flag if flag else "[No Flag]"
        print(f"  - {flag_display}: {count}")
    return {
        'total_records': len(df),
        'sentiment_analysis': {
            'positive': len(df[df['sentiment'] == 'positive']),
            'negative': len(df[df['sentiment'] == 'negative']),
            'neutral': len(df[df['sentiment'] == 'neutral']),
            'undefined': len(df[df['sentiment'].isna()])
        },
        'language_distribution': dict(language_counts),
        'customer_journey_stages': dict(journey_stage_counts),
        'complaint_or_compliment_distribution': dict(complaint_compliment_counts),
        'flags': {flag: count for flag, count in flag_counts.items()},
        'output_files': [json_output, 'Sentiment_Count.json', token_freq_output],
        'profile_sentiments': None,
        'token_frequencies': token_frequencies,
        'original_data': original_data
    }

if __name__ == "__main__":
    file_path = '/content/Json_Acc_To_Dev_Team (2).json'
    output_prefix = '/content/Analyzed_Review_with_Sentiment_Aspect_Rules'
    classifier = load_sentiment_model()
    if not classifier:
        print("Failed to load sentiment analysis model. Exiting.")
        exit(1)
    aspect_classifier = load_aspect_model()
    if not aspect_classifier:
        print("Failed to load aspect classification model. Exiting.")
        exit(1)
    result = process_reviews(file_path, output_prefix, classifier, aspect_classifier)
    if result:
        print("\nAnalysis completed successfully!")
        review_counts = count_reviews_before_and_after(
            result['original_data'],
            json.load(open(result['output_files'][0], 'r', encoding='utf-8'))
        )
        print("\nReview Counts:\n", review_counts)
        print(f"Total records: {result['total_records']}")
        print("\nSentiment Distribution:")
        for sentiment, count in result['sentiment_analysis'].items():
            print(f"  - {sentiment.capitalize()}: {count}")
        print("\nComplaint/Compliment Distribution:")
        for category, count in result['complaint_or_compliment_distribution'].items():
            print(f"  - {category}: {count}")
