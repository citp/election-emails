from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
import string
import numpy as np
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve, plot_precision_recall_curve, brier_score_loss, f1_score
import spacy
from sklearn.calibration import calibration_curve
from statsmodels.stats.proportion import proportion_confint
from nltk.util import ngrams
from pysentimiento import SentimentAnalyzer
from stanza.server import CoreNLPClient
import random
from sklearn.model_selection import KFold

random_state = 2020

client = CoreNLPClient(annotators=['coref'],
                       timeout=30000, endpoint='http://localhost:' + str(random.randint(9000, 13000)),
                       memory='6G')

ngram_range = (1, 3)
ngrams_dep = 3

punctuation_list = list(string.punctuation) + [' ']
analyzer = SentimentAnalyzer(lang="en")

nlp_spacy = spacy.load('en_core_web_lg')


urgency_common_phrases = set()
with open('../data/keywords/common_phrases_urgency', 'r') as f:
    for word in f.readlines():
        if len(word.strip()) > 0:
            urgency_common_phrases.add(word.strip())


fw_referencing_common_phrases = set()
with open('../data/keywords/common_phrases_fw', 'r') as f:
    for word in f.readlines():
        if len(word.strip()) > 0:
            fw_referencing_common_phrases.add(word.strip())


sens_common_phrases = set()
with open('../data/keywords/common_phrases_sens', 'r') as f:
    for word in f.readlines():
        if len(word.strip()) > 0:
            sens_common_phrases.add(word.strip())


hyperbolic_words = set()
with open('../data/keywords/hyperbolic', 'r') as f:
    for word in f.readlines():
        if len(word.strip()) > 0:
            hyperbolic_words.add(word.strip())


content_words = ['video', 'tv', 'clip', 'watch', 'live', 'photo', 'post', 'look', 'visual', 'must see']
content_dictionary = []

for word in content_words:
    content_dictionary.append(word + ':')
    content_dictionary.append(word + '!')
    content_dictionary.append('(' + word + ')')
    content_dictionary.append('[' + word + ']')


def all_pronominal(ann, text):
    values = set()
    mentionType = set()
    for mention in ann.corefChain[0].mention:
        values.add(ann.sentence[mention.sentenceIndex].token[mention.beginIndex].value.lower())
        mentionType.add(mention.mentionType)

    if text not in values:
        return True

    if text in values and len(mentionType) == 1 and list(mentionType)[0] == 'PRONOMINAL':
        return True

    # if text in values and len(mentionType) == 2 and len(list(mentionType - set(['PRONOMINAL', 'NOMINAL']))) == 0 :
    #    return True

    return False


def has_reference(doc):
    value = False
    text = doc.text
    for index, token in enumerate(doc):
        # Subject pronouns and determiners
        if token.text.lower() in ['this', 'he', 'she', 'they', 'it', 'that', 'those', 'them', 'these', 'his', 'her', 'their', 'what']:
            if token.dep_ == 'nsubj':
                ann = client.annotate(text)
                if not (len(ann.corefChain) > 0 and not all_pronominal(ann, token.text)):
                    value = True

        # Object pronouns and determiners
        if token.text.lower() in ['this', 'that', 'those', 'these', 'him', 'her', 'them', 'it']:
            if token.dep_ in ['dobj', 'pobj', 'dative']:
                ann = client.annotate(text)
                if not (len(ann.corefChain) > 0 and not all_pronominal(ann, token.text)):
                    value = True

        # Questions that have an answer in the body
        if token.text.lower() in ['here', 'why', 'where', 'what', 'whose', 'there', 'those', 'these', 'how']:
            if token.pos_ == 'ADV':
                value = True

        # Pronouns and determiners preceding object
        if token.text.lower() in ['this', 'his', 'her', 'their', 'its', 'that', 'those', 'these']:
            if token.head.ent_type_ not in ['DATE', 'TIME'] and token.head.lemma_ not in ['team', 'year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'morning', 'evening', 'midnight', 'noon', 'night']:
                if token.head.dep_ in ['pobj', 'dobj'] or (token.head.pos_ in ['NOUN', 'PROPN']):
                    ann = client.annotate(text)
                    if not (len(ann.corefChain) > 0 and not all_pronominal(ann, token.text)):
                        value = True

        # Pronounds and determiners preceding subject
        if token.text.lower() in ['this', 'his', 'her', 'their', 'whose', 'its', 'that', 'those', 'these']:
            if token.head.dep_ in ['nsubj', 'nsubjpass']:
                value = True

        if token.text.lower() in ['what', 'why', 'where', 'whose']:
            if token.pos_ in ['PRON'] and token.dep_ in ['pobj', 'dobj', 'det']:
                value = True

        # if token.is_sent_start:
        #    if token.pos_ == 'NUM' and token.head.pos_ in ['NOUN', 'PROPN'] and token.head.text.lower() not in ['time', 'donation', 'donations', 'emails', 'email', 'year', 'years', 'days', 'day', 'hour', 'hours', 'week', 'weeks', 'second', 'seconds', '%', 'match', 'chance']:
        #       value = True
        # else:
        #    if token.pos_ == 'NUM' and doc[:index].text.lower() in ['re:', 'fwd:', 're', 'fwd', 'the'] and token.head.pos_ in ['NOUN', 'PROPN'] and token.head.text.lower() not in ['time', 'donation', 'donations', 'emails', 'email', 'year', 'years', 'days', 'day', 'hour', 'hours', 'week', 'weeks', 'second', 'seconds', '%', 'match', 'chance']:
        #        value = True

        if value:
            break

    return 1 if value else 0


def ascii_approx_text(text):
    text = text.replace('❗', '!')
    text = text.replace('❕', '!')
    text = text.replace('❔', '?')
    text = text.replace('❓', '?')
    
    text_decoded = unidecode(text)

    return ''.join(filter(lambda x: x in string.printable, text_decoded)).strip()


def join_ngrams(ngrams_list):
    return map(lambda x: ' '.join(x), ngrams_list)


def transform_and_replace_text(text):
    doc = nlp_spacy(text)
    result = []

    for sent in doc.sents:
        tokens = []
        for token in sent:
            if token.like_email:
                tokens.extend(list(map(lambda x: (x, False), token.text.split('@')[0].split('-'))))
            elif token.like_num:
                tokens.append(('token_cardinal', False))
            elif all(i in punctuation_list for i in token.text) or token.like_url:
                pass
            else:
                if str(token.lemma_) == '-PRON-':
                    tokens.append((token.text, token.is_stop))
                else:
                    tokens.append((token.lemma_, token.is_stop))

        tokens = list(map(lambda x: (x[0].strip(), x[1]), tokens))
        tokens = list(filter(lambda x: len(x[0]) > 0, tokens))

        result.append(tokens)

    return result


def tokenize_text_ngrams(text, ngram_range):
    ngram_tokens = []
    ngram_tokens_stopwords = []

    text_tokens = transform_and_replace_text(text)

    for i in range(ngram_range[0], ngram_range[1]+1):
        for t in text_tokens:
            ngram_tokens.extend(join_ngrams(list(ngrams(map(lambda x: x[0], filter(lambda x: not x[1], t)), i))))
            ngram_tokens_stopwords.extend(join_ngrams(list(ngrams(map(lambda x: x[0], t), i))))

    ngram_tokens = list(filter(lambda x: len(x.strip()) > 1, ngram_tokens))
    ngram_tokens_stopwords = list(filter(lambda x: len(x.strip()) > 1, ngram_tokens_stopwords))

    for i, nt in enumerate(ngram_tokens):
        ngram_tokens[i] = ''.join([x for x in nt if x in set(string.ascii_letters + ' ')]).lower()

    for i, nt in enumerate(ngram_tokens_stopwords):
        ngram_tokens_stopwords[i] = ''.join([x for x in nt if x in set(string.ascii_letters + ' ')]).lower()

    return ngram_tokens, ngram_tokens_stopwords


def tokenize_text(text):
    doc = nlp_spacy.tokenizer(text)
    tokens = []

    for token in doc:
        if token.is_punct or all([x in string.punctuation for x in token.text]):
            continue
        else:
            tokens.append(token.text)

    return tokens


def apply_pos_tags(doc, ngram_range):
    pos_tags = []

    text_pos_tags = [token.pos_ for token in doc]
    text_pos_tags = list(filter(lambda x: x != 'PUNCT', text_pos_tags))

    for i in range(ngram_range[0], ngram_range[1]+1):
        pos_tags.extend(join_ngrams(list(ngrams(text_pos_tags, i))))

    return pos_tags

    
def apply_dep_tags(doc, ngram):
    def traverse(root, depth, max_depth):
        if root is None:
            return []

        if depth >= max_depth:
            return []

        result = []
        for child in root.children:
            tags = traverse(child, depth + 1, max_depth)

            for tag in tags:
                result.append(root.dep_ + ' ' + tag)

        result.append(root.dep_)
        return result

    def depth_first(root, ngram):
        result = traverse(root, 0, ngram)
        for child in root.children:
            result.extend(depth_first(child, ngram))

        return result

    def sentence(doc, ngram):
        result = []

        for sent in doc.sents:
            result.extend(depth_first(sent.root, ngram))

        result = list(filter(lambda x: not 'ROOT' in x and not 'punct' in x, result))

        return result

    return sentence(doc, ngram)


def entities_lowercase(text):
    doc = nlp_spacy(text)
    result_text_replaced = text
    result_text_preserved = text

    if len(doc.ents) != 0:
        result_text_replaced = ''
        result_text_preserved = ''
        start = 0

        for ent in doc.ents:
            result_text_replaced += text[start: ent.start_char].lower()
            result_text_preserved += text[start: ent.start_char].lower()

            if ent.label_ in ['CARDINAL', 'TIME', 'CURRENCY', 'QUANTITY', 'DATE', 'MONEY', 'ORDINAL', 'PERCENT']:
                label = ent.label_

                if label == 'DATE' or label == 'TIME':
                    label = 'TIME'
                    result_text_replaced += (label.lower())
                else:
                    result_text_replaced += ('token_' + label.lower())
            else:
                result_text_replaced += ('token_' + ent.label_.lower())

            result_text_preserved += ent.text

            start = ent.end_char

        result_text_replaced += text[start: len(text)].lower()
        result_text_preserved += text[start: len(text)].lower()

    else:
        result_text_replaced = result_text_replaced.lower()
        result_text_preserved = result_text_preserved.lower()

    result_text_replaced = re.sub('\d+', 'token_cardinal', result_text_replaced)
    result_text_replaced = re.sub('token_cardinal:token_cardinalam', 'time', result_text_replaced)
    result_text_replaced = re.sub('token_cardinal:token_cardinalpm', 'time', result_text_replaced)
    result_text_replaced = re.sub('token_cardinal/token_cardinal/token_cardinal', 'time', result_text_replaced)
    result_text_replaced = re.sub('tomorrow', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('today', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('months', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('month', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('weeks', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('week', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('minutes', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('minute', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('hours', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('hour', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('seconds', 'time', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('second', 'time', result_text_replaced, flags=re.IGNORECASE)

    result_text_replaced = re.sub('time[ ,]+(time[ ,]*)+', 'time ', result_text_replaced, flags=re.IGNORECASE).strip()
    result_text_replaced = re.sub('token_org[ ,]+(token_org[ ,]*)+', 'token_org ', result_text_replaced).strip()
    result_text_replaced = re.sub('token_product[ ,]+(token_product[ ,]*)+', 'token_product ',
                                  result_text_replaced).strip()
    result_text_replaced = re.sub('token_person[ ,]+(token_person[ ,]*)+', 'token_person ',
                                  result_text_replaced).strip()
    result_text_replaced = re.sub('token_gpe[ ,]+(token_gpe[ ,]*)+', 'token_gpe ', result_text_replaced).strip()

    result_text_replaced = re.sub('single', 'token_cardinal', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('double', 'token_cardinal', result_text_replaced, flags=re.IGNORECASE)
    result_text_replaced = re.sub('triple', 'token_cardinal', result_text_replaced, flags=re.IGNORECASE)

    return result_text_replaced, result_text_preserved


def match_proportion(row):
    from_name_decoded = row['from_name_decoded']
    if '@' in row['from_address'] and '@' in from_name_decoded and row['from_address'] != from_name_decoded:
        from_name_decoded = from_name_decoded.split('@')[0]

    name = row['name']

    string1_words = set(x.lower() for x in tokenize_text(from_name_decoded))
    string2_words = set(x.lower() for x in tokenize_text(name))

    return len(string1_words.intersection(string2_words)) / len(string1_words.union(string2_words))


def leading_match(row):
    from_name_decoded = row['from_name_decoded']
    if '@' in row['from_address'] and '@' in from_name_decoded and row['from_address'] != from_name_decoded:
        from_name_decoded = from_name_decoded.split('@')[0]

    name = row['name']

    string1_words = [x.lower() for x in tokenize_text(from_name_decoded)]

    if len(string1_words) == 0:
        return 1

    string2_words = [x for x in tokenize_text(name)]

    # Include acronyms
    if len(string2_words) > 2:
        new_string = ''
        for word in string2_words:
            if word[0].isupper():
                new_string += word[0]

        string2_words.append(new_string)
        if len(new_string) > 2:
            for i in range(2, len(new_string) - 1):
                string2_words.append(new_string[0:i])

        string2_words.append(new_string + 'PAC')

    string2_words = [x.lower() for x in string2_words]

    index = None
    for word in string2_words:
        try:
            index = string1_words.index(word)
            break
        except:
            pass

    return 1 if index is None else index / len(string1_words)


def find_longest_sequence(text):
    list_of_uppercase_runs = re.findall(r'[A-Z\' ]+', text)
    list_of_uppercase_runs = [x.strip() if x.strip != '' else None for x in list_of_uppercase_runs]
    if len(list_of_uppercase_runs) == 0:
        return 0
    else:
        max_run = len(max(list_of_uppercase_runs, key=len))
        if max_run == 1 or max_run == 0:
            return 0
        else:
            return max_run


def count_punct(text,
                punctuation=[',', '.', '$', '%', '&', '/', '\\', '+', '=', '!', '?', '(', ')', '[', ']', '>', '|', ':',
                             '-', ';', '"']):
    count = 0
    for punct in punctuation:
        count += text.count(punct)

    return count


def ends_in_punct(text,
                  punctuation=[',', '.', '$', '%', '&', '/', '\\', '+', '=', '!', '?', '(', ')', '[', ']', '>', '|',
                               ':', '-', ';', '"']):
    for punct in punctuation:
        if text.strip().endswith(punct):
            return 1
    return 0


def get_sentiment(text):
    res = analyzer.predict(text)
    return res.probas['NEG'], res.probas['NEU'], res.probas['POS']


def token_features(text):
    def get_values(text):
        number_digits = number_upper = number_lower = number_stop = number_contractions = number_tokens = word_length = 0

        for token in nlp_spacy.tokenizer(text):
            if token.is_punct or all([x in string.punctuation for x in token.text]):
                continue

            number_tokens += 1

            current_token = token.text
            if token.like_email:
                current_token = ''.join(filter(lambda x: x.isalpha, token.text.split('@')[0]))

            word_length += len(current_token)

            if len(current_token) > 2 and current_token.isupper():
                number_upper += 1

            if len(current_token) > 2 and current_token.islower():
                number_lower += 1

            if token.like_num:
                number_digits += 1

            if token.is_stop:
                number_stop += 1

            if current_token.lower() in ["'ll", "n't", "'re", "'ve", "'d", "'s", "'m"]:
                number_contractions += 1

        return number_digits, number_upper, number_lower, number_stop, number_contractions, word_length, number_tokens

    text1 = text.split('-DELIM-')[0]
    text2 = text.split('-DELIM-')[1]

    nd1, nu1, nl1, ns1, nc1, wl1, d1 = get_values(text1)
    nd2, nu2, nl2, ns2, nc2, wl2, d2 = get_values(text2)

    if (d1 + d2) == 0:
        return 0, 0, 0, 0, nc1+nc2, 0, 0

    return (nd1+nd2)/(d1+d2), (nu1+nu2)/(d1+d2), (nl1+nl2)/(d1+d2), (ns1+ns2)/(d1+d2), nc1+nc2, (wl1+wl2)/(d1+d2), d1+d2


def count_common(text, common_phrases):
    text_list = []
    for token in nlp_spacy.tokenizer(text):
        if token.like_email:
            text_list.append(' '.join(token.text.split('@')[0].split('-')))

        if not token.is_punct and not all([x in string.punctuation for x in token.text]):
            text_list.append(token.text)

    text = ' '.join(text_list).lower()

    count = 0
    for word in common_phrases:
        count += text.count(word)

    return count


def cap_pos(row, value):
    verb_count = 0
    adj_count = 0
    adv_count = 0
    noun_count = 0

    text = row[value]
    text_lowered_doc = row[value + '_entities_preserved_lowercase_spacy']

    for token in text_lowered_doc:
        if len(token.text.strip()) == 1:
            continue
            
        if token.pos_ == 'VERB' and token.text.upper() in text:
            verb_count += 1

        if token.pos_ == 'ADJ' and token.text.upper() in text:
            adj_count += 1

        if token.pos_ == 'ADV' and token.text.upper() in text:
            adv_count += 1

        if token.pos_ == 'NOUN' and token.text.upper() in text:
            noun_count += 1

    return verb_count, adj_count, adv_count, noun_count


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_list):
        self.column_list = column_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column_list].values

    def get_feature_names(self):
        return self.column_list


def nested_cross_validation(model, xData, yData):
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_results = []
    for train_ix, test_ix in cv_outer.split(xData):
        X_train, X_test = xData.iloc[train_ix, :], xData.iloc[test_ix, :]
        y_train, y_test = yData.iloc[train_ix], yData.iloc[test_ix]

        result = model.fit(X_train, y_train)
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)

        f1 = f1_score(y_test, yhat)
        outer_results.append((f1, best_model, result.best_params_))

    return outer_results


def extract_features(coded):
    name_list = list(map(lambda x: str(x), coded['name'].tolist()))
    from_address_list = list(map(lambda x: str(x), coded['from_address'].tolist()))

    subject_list = list(map(lambda x: str(x), coded['subject'].tolist()))
    from_name_list = list(map(lambda x: str(x), coded['from_name'].tolist()))

    subject_decoded_list = map(lambda x: ascii_approx_text(x), subject_list)
    subject_decoded_list = map(lambda x: ' '.join(x.split()), subject_decoded_list)

    from_name_decoded_list = map(lambda x: ascii_approx_text(x), from_name_list)
    from_name_decoded_list = map(lambda x: ' '.join(x.split()), from_name_decoded_list)

    party_affiliation_list = list(map(lambda x: x, coded['party_affiliation'].tolist()))
    source_list = list(map(lambda x: str(x), coded['source'].tolist()))
    office_level_list = list(map(lambda x: str(x), coded['office_level'].tolist()))
    incumbent_list = list(map(lambda x: str(x), coded['incumbent'].tolist()))

    df = pd.DataFrame({'name': name_list,
                       'subject': subject_list,
                       'subject_decoded': subject_decoded_list,
                       'from_name': from_name_list,
                       'from_address': from_address_list,
                       'from_name_decoded': from_name_decoded_list,
                       'party_affiliation': party_affiliation_list,
                       'source': source_list,
                       'office_level': office_level_list,
                       'incumbent': incumbent_list})

    df['subject'] = df['subject'].astype(str)
    df['subject_decoded'] = df['subject_decoded'].astype(str)
    df['from_name'] = df['from_name'].astype(str)
    df['from_name_decoded'] = df['from_name_decoded'].astype(str)

    df['input'] = df['from_name'] + '. ' + df['subject']
    df['input_decoded'] = df['from_name_decoded'] + '. ' + df['subject_decoded']
    df['input_decoded_ngram'] = df['from_name_decoded'] + '-DELIM-' + df['subject_decoded']


    df['from_name_decoded_entities_replaced_lowercase'], df['from_name_decoded_entities_preserved_lowercase'] = zip(
        *df['from_name_decoded'].apply(entities_lowercase))
    df['subject_decoded_entities_replaced_lowercase'], df['subject_decoded_entities_preserved_lowercase'] = zip(
        *df['subject_decoded'].apply(entities_lowercase))

    print('Preserving/replace entities complete')

    df['from_name_decoded_entities_preserved_lowercase_spacy'] = df[
        'from_name_decoded_entities_preserved_lowercase'].apply(nlp_spacy)
    df['subject_decoded_entities_preserved_lowercase_spacy'] = df[
        'subject_decoded_entities_preserved_lowercase'].apply(nlp_spacy)

    print('Preserve spacy object complete')

    df['from_name_decoded_ngram_tags'], df['from_name_decoded_ngram_tags_stopwords'] = zip(
        *df['from_name_decoded_entities_replaced_lowercase'].apply(lambda x: tokenize_text_ngrams(x, ngram_range)))

    df['subject_decoded_ngram_tags'], df['subject_decoded_ngram_tags_stopwords'] = zip(
        *df['subject_decoded_entities_replaced_lowercase'].apply(lambda x: tokenize_text_ngrams(x, ngram_range)))

    df['input_decoded_ngram_tags'] = df['from_name_decoded_ngram_tags'] + df['subject_decoded_ngram_tags']
    df['input_decoded_ngram_tags_stopwords'] = df['from_name_decoded_ngram_tags_stopwords'] + df[
        'subject_decoded_ngram_tags_stopwords']


    df['from_name_decoded_ngram_tags_original'], df['from_name_decoded_ngram_tags_stopwords_original'] = zip(
        *df['from_name_decoded'].apply(lambda x: tokenize_text_ngrams(x, ngram_range)))

    df['subject_decoded_ngram_tags_original'], df['subject_decoded_ngram_tags_stopwords_original'] = zip(
        *df['subject_decoded'].apply(lambda x: tokenize_text_ngrams(x, ngram_range)))

    df['input_decoded_ngram_tags_original'] = df['from_name_decoded_ngram_tags_original'] + df['subject_decoded_ngram_tags_original']
    df['input_decoded_ngram_tags_stopwords_original'] = df['from_name_decoded_ngram_tags_stopwords_original'] + df[
        'subject_decoded_ngram_tags_stopwords_original']

    print('Tokenizing and n-grams complete')

    df['from_name_decoded_pos_tags'] = df['from_name_decoded_entities_preserved_lowercase_spacy'].apply(lambda x: apply_pos_tags(x, ngram_range))
    df['subject_decoded_pos_tags'] = df['subject_decoded_entities_preserved_lowercase_spacy'].apply(lambda x: apply_pos_tags(x, ngram_range))
    df['input_decoded_pos_tags'] = df['from_name_decoded_pos_tags'] + df['subject_decoded_pos_tags']

    print('Part of speech and n-grams complete')

    df['from_name_decoded_dep_tags'] = df['from_name_decoded_entities_preserved_lowercase_spacy'].apply(lambda x: apply_dep_tags(x, ngrams_dep))
    df['subject_decoded_dep_tags'] = df['subject_decoded_entities_preserved_lowercase_spacy'].apply(lambda x: apply_dep_tags(x, ngrams_dep))
    df['input_decoded_dep_tags'] = df['from_name_decoded_dep_tags'] + df['subject_decoded_dep_tags']

    print('Dependency tags and sn-grams complete')

    df['subject_unicode_count'] = df['subject'].apply(
        lambda x: sum([char not in string.printable + '’‘“”' for char in x.strip()]))
    df['from_name_unicode_count'] = df['from_name'].apply(
        lambda x: sum([char not in string.printable + '’‘“”' for char in x.strip()]))
    df['unicode_count'] = df['subject_unicode_count'] + df['from_name_unicode_count']

    print('Unicode counts complete')

    df['subject_raw_length'] = df['subject'].apply(lambda x: len(x.strip()))
    df['from_name_raw_length'] = df['from_name'].apply(lambda x: len(x.strip()))
    df['raw_length'] = df['subject_raw_length'] + df['from_name_raw_length']

    print('Raw length counts complete')

    df['punct_count_subject'] = df['subject_decoded'].apply(count_punct)
    df['punct_count_from_name'] = df['from_name_decoded'].apply(count_punct)
    df['punct_count'] = df['punct_count_subject'] + df['punct_count_from_name']

    print('Punctuation counts complete')

    df['ends_punct_from_name'] = df['from_name_decoded'].apply(ends_in_punct)
    df['ends_punct_subject'] = df['subject_decoded'].apply(ends_in_punct)
    df['ends_punct'] = df['input_decoded'].apply(ends_in_punct)

    print('Ends punct complete')

    #df['sentiment_subject_neg'], df['sentiment_subject_neu'], df['sentiment_subject_pos'] = zip(*df['subject_decoded'].apply(get_sentiment))
    #df['sentiment_from_name_neg'], df['sentiment_from_name_neu'], df['sentiment_from_name_pos'] = zip(*df['from_name_decoded'].apply(get_sentiment))
    df['sentiment_input_neg'], df['sentiment_input_neu'], df['sentiment_input_pos'] = zip(*df['input_decoded'].apply(get_sentiment))

    print('Sentiment complete')

    df['subject_decoded_digits_prop'], df['subject_decoded_upper_prop'], df['subject_decoded_lower_prop'],\
    df['subject_decoded_stop_words_prop'], df['subject_decoded_contraction_words_count'], df['subject_decoded_average_word_length'],\
    df['subject_decoded_word_count'] = zip(*(df['subject_decoded'] + '-DELIM-').apply(token_features))

    df['from_name_decoded_digits_prop'], df['from_name_decoded_upper_prop'], df['from_name_decoded_lower_prop'], \
    df['from_name_decoded_stop_words_prop'], df['from_name_decoded_contraction_words_count'], df['from_name_decoded_average_word_length'],\
    df['from_name_decoded_word_count'] = zip(*(df['from_name_decoded'] + '-DELIM-').apply(token_features))

    df['input_decoded_digits_prop'], df['input_decoded_upper_prop'], df['input_decoded_lower_prop'], \
    df['input_decoded_stop_words_prop'], df['input_decoded_contraction_words_count'], df['average_word_length'], df['word_count'] = zip(
        *(df['input_decoded_ngram']).apply(token_features))

    print('Token features complete')

    df['is_question'] = df.apply(
        lambda x: 1 if (('?' in x['from_name_decoded']) or ('?' in x['subject_decoded'])) else 0, axis=1)
    df['is_exclamation'] = df.apply(
        lambda x: 1 if (('!' in x['from_name_decoded']) or ('!' in x['subject_decoded'])) else 0, axis=1)

    df['has_quote'] = df.apply(lambda x: 1 if ((x['from_name_decoded'].count('"') > 0 and x['from_name_decoded'].count('"') % 2 == 0) or \
                                               (x['from_name_decoded'].count("'") > 0 and x['from_name_decoded'].count("'") % 2 == 0) or \
                                               (x['subject_decoded'].count('"') > 0 and x['subject_decoded'].count('"') % 2 == 0) or \
                                               (x['subject_decoded'].count("'") > 0 and x['subject_decoded'].count("'") % 2 == 0)) else 0, axis=1)

    df['has_full_quote'] = df['subject_decoded'].apply(
        lambda x: 1 if (x.strip().startswith('"') and x.strip().endswith('"')) else 0)

    df['question_or_follow_up_punct'] = df.apply(lambda x: 1 if ((('??' in x['from_name_decoded']) or
                                                                 ('!?' in x['from_name_decoded']) or
                                                                 ('?!' in x['from_name_decoded']) or
                                                                 ('?]' in x['from_name_decoded']) or
                                                                 ('?)' in x['from_name_decoded']) or
                                                                 ('!?]' in x['from_name_decoded']) or
                                                                 ('?!]' in x['from_name_decoded']) or
                                                                 ('..?' in x['from_name_decoded']) or
                                                                 ('!?)' in x['from_name_decoded']) or
                                                                 ('?!)' in x['from_name_decoded'])) or
                                                                 (('??' in x['subject_decoded']) or
                                                                  ('!?' in x['subject_decoded']) or
                                                                  ('?!' in x['subject_decoded']) or
                                                                  ('?]' in x['subject_decoded']) or
                                                                  ('..?' in x['subject_decoded']) or
                                                                  ('?)' in x['subject_decoded']) or
                                                                  ('!?]' in x['subject_decoded']) or
                                                                  ('?!]' in x['subject_decoded']) or
                                                                  ('!?)' in x['subject_decoded']) or
                                                                  ('?!)' in x['subject_decoded']) or
                                                                  ('...' in x['subject_decoded']) or
                                                                  x['subject_decoded'].strip().endswith('..') or
                                                                  x['subject_decoded'].strip().endswith(':'))) else 0, axis=1)

    df['content_word_present'] = df['input_decoded'].apply(
        lambda x: 1 if any([word in x.lower() for word in content_dictionary]) else 0)

    print('Misc features complete')

    df['cap_verb_subject'], df['cap_adj_subject'], df['cap_adv_subject'], df['cap_noun_subject'] = zip(
        *df.apply(lambda x: cap_pos(x, 'subject_decoded'), axis=1))
    df['cap_verb_from_name'], df['cap_adj_from_name'], df['cap_adv_from_name'], df['cap_noun_from_name'] = zip(
        *df.apply(lambda x: cap_pos(x, 'from_name_decoded'), axis=1))
    df['cap_verb'] = df['cap_verb_subject'] + df['cap_verb_from_name']
    df['cap_adj'] = df['cap_adj_subject'] + df['cap_adj_from_name']
    df['cap_adv'] = df['cap_adv_subject'] + df['cap_adv_from_name']
    df['cap_noun'] = df['cap_noun_subject'] + df['cap_noun_from_name']

    print('Part of speech capitalized complete')

    df['has_reference'] = df['from_name_decoded_entities_preserved_lowercase_spacy'].apply(has_reference) | df[
        'subject_decoded_entities_preserved_lowercase_spacy'].apply(has_reference)
    print('Has reference complete')

    df['count_common_urgency'] = df['from_name_decoded_entities_replaced_lowercase'].apply(lambda x: count_common(x, urgency_common_phrases)) + \
                                 df['subject_decoded_entities_replaced_lowercase'].apply(lambda x: count_common(x, urgency_common_phrases))

    df['count_common_fw'] = df['from_name_decoded'].apply(lambda x: count_common(x, fw_referencing_common_phrases)) + \
                                 df['subject_decoded'].apply(lambda x: count_common(x, fw_referencing_common_phrases))

    df['count_common_sens'] = df['from_name_decoded'].apply(lambda x: count_common(x, sens_common_phrases)) + \
                              df['subject_decoded'].apply(lambda x: count_common(x, sens_common_phrases))

    df['count_hyperbolic'] = df['from_name_decoded'].apply(lambda x: count_common(x, hyperbolic_words)) + \
                             df['subject_decoded'].apply(lambda x: count_common(x, hyperbolic_words))

    print('Count phrases complete')

    df['match_proportion'] = df.apply(match_proportion, axis=1)
    df['leading_match'] = df.apply(leading_match, axis=1)

    print('Proportion and leading match complete')

    del df['from_name_decoded_entities_preserved_lowercase_spacy']
    del df['subject_decoded_entities_preserved_lowercase_spacy']

    return df


def extract_labels(coded):
    return coded[['urgency', 'sensationalism', 'forward_referencing', 'obscured_name']]


def print_classifier_output(classifier, x_test, y_test, actual_subjects=None, column_name=None, classifier_name='', filepath=''):
    # classifier is the output of a grid search or randomized search
    #print('Best params: \n', classifier.best_params_)
    #print('\n')

    print('Test set output:')
    predicted_test_labels = classifier.predict(x_test)

    print('1. Classification report:')
    print(classification_report(y_test, predicted_test_labels))
    print('\n')

    print('2. Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_test, predicted_test_labels),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))
    print('\n')

    roc_disp = plot_roc_curve(classifier, x_test, y_test, name=classifier_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.show()
    if len(filepath) > 0:
        plt.savefig(filepath + 'roc.png', bbox_inches='tight', facecolor='white')

    pr_disp = plot_precision_recall_curve(classifier, x_test, y_test, name=classifier_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.show()

    if len(filepath) > 0:
        plt.savefig(filepath + 'pr.png', bbox_inches='tight', facecolor='white')

    if actual_subjects is not None:
        x_test = actual_subjects

    if column_name is not None:
        print('3. False positives:')
        print(x_test[(predicted_test_labels != y_test) & (y_test == False)][column_name].values)
        print('\n')

        print('4. False negatives:')
        print(x_test[(predicted_test_labels != y_test) & (y_test == True)][column_name].values)

    else:
        print('4. False positives:')
        print(x_test[(predicted_test_labels != y_test) & (y_test == False)])
        print('\n')

        print('5. False negatives:')
        print(x_test[(predicted_test_labels != y_test) & (y_test == True)])


def print_calibration_output(classifier, x_test, y_test, groups=None, bins=5):
    def err(x):
        n = len(x)
        p = len(x[x['label']])
        ll, ul = proportion_confint(p, n, method='wilson')
        prop = p/n

        return pd.Series({'prop': prop, 'mean': np.mean(x['score']), 'll': prop - ll, 'ul': ul - prop, 'count': len(x)})

    def func(x, y, h, lb, ub, **kwargs):
        data = kwargs.pop('data')
        errLo = data.pivot(index=x, columns=h, values=lb)
        errHi = data.pivot(index=x, columns=h, values=ub)
        err = []
        for col in errLo:
            err.append([errLo[col].values, errHi[col].values])
        err = np.abs(err)
        p = data.pivot(index=x, columns=h, values=y)
        p.plot(kind='bar', yerr=err, ax=plt.gca(), **kwargs)

    predicted_test_scores = classifier.predict_proba(x_test)[:, 1]

    frame = pd.DataFrame({'score': predicted_test_scores, 'label': y_test, 'group': groups}).sort_values('score')
    frame['category'] = pd.cut(frame['score'], np.linspace(0, 1, bins + 1), labels=list(range(1, bins + 1)),
                               include_lowest=True)

    df = frame.groupby(['category']).apply(err).reset_index()
    print(df)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.errorbar(df['mean'], df['prop'], yerr=df[['ll', 'ul']].values.transpose(), marker='o')
    plt.show()

    df = frame.groupby(['category', 'group']).apply(err).reset_index()
    print(df)

    print('Brier score loss:', brier_score_loss(y_test, predicted_test_scores))

    df = frame.groupby(['category', 'group']).apply(err).reset_index()
    df['v'] = ''

    g = sns.FacetGrid(df, col='v', height=4)
    g.map_dataframe(func, 'category', 'prop', 'group', 'll', 'ul', color=['orange', 'gray'])
    g.add_legend()

    plt.show()
