import re
# import nltk
# from fuzzywuzzy import fuzz
# import string
# import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from os import listdir 
from os.path import isfile, join

def parse_cb_companies():
    with open('dump_c.txt', encoding = 'utf8', errors = 'ignore') as f:
        lines = f.readlines()

        entity_id = []
        parent_id = []
        company_name = []
        company_normalized_name = []

        for l in lines:
            l = l.replace('INSERT INTO \"cb_objects\" VALUES ', "")
            rows = l.split("),(")

            for r in rows:
                r = r.split(',')
                curr_entity_id = r[2].replace('\'', '').strip()
                curr_parent_id = r[3].replace('\'', '').strip()
                curr_company_name = r[4].lower().replace('\'', '').strip()
                curr_company_normalized_name = r[5].lower().replace('\'', '').strip()
                entity_id.append(curr_entity_id)
                parent_id.append(curr_parent_id)
                company_name.append(curr_company_name)
                company_normalized_name.append(curr_company_normalized_name)

        company_df = pd.DataFrame({'entity_id': entity_id, 'parent_id': parent_id, 'company_name': company_name, 'company_normalized_name': company_normalized_name}).drop_duplicates()
        company_df.to_csv('cb_companies.csv')

def parse_cb_headlines():
    with open('dump.txt', encoding = 'utf8', errors = 'ignore') as f:
        lines = f.readlines()

        acquisition_id = []
        acquiring_object_id = []
        acquired_object_id = []
        acquirer = []
        acquired = []
        acquired_at = []
        price_amount = []
        price_currency_code = []
        headline = []

        for l in lines:
            rows = l.split("),(")

            for r in rows:
                r = r.split(',')
                curr_acquisition_id = r[1].replace('\'', '').strip()
                curr_acquiring_object_id = r[2].replace('\'', '').strip()
                curr_acquired_object_id = r[3].replace('\'', '').strip()
                curr_acquired_at = r[7].replace('\'', '').strip()
                curr_price_amt = r[5].replace('\'', '').strip()
                curr_price_currency_code = r[6].replace('\'', '').strip()
                curr_headline = r[9].replace('\'', '').strip().lower()
                acquisition_id.append(curr_acquisition_id)
                acquiring_object_id.append(curr_acquiring_object_id)
                acquired_object_id.append(curr_acquired_object_id)
                acquired_at.append(curr_acquired_at)
                price_amount.append(curr_price_amt)
                price_currency_code.append(curr_price_currency_code)
                headline.append(curr_headline)

        headline_df = pd.DataFrame({
            'acquisition_id': acquisition_id,
            'acquiring_object_id': acquiring_object_id,
            'acquired_object_id': acquired_object_id,
            'acquirer': acquirer,
            'acquired': acquired,
            'acquired_at': acquired_at,
            'price_amount': price_amount,
            'price_currency_code': price_currency_code,
            'headline': headline
        }).drop_duplicates()
        headline_df.to_csv('cb_headlines.csv')

def parse_lexis_nexis_headlines(ln_folder = '157560'):
    # enumerate file names inside LN folder
    ln_files = [f for f in listdir(ln_folder) if isfile(join(ln_folder, f))]
    for ln in ln_files:
        tags = []
        day = []
        month = []
        year = []
        headline = []
        headline_id = []

        with open(join(ln_folder, ln)) as f:
            contents = f.read()
            articles = re.findall('<articleDoc .*>', contents)
            for a in articles:
                curr_headline = re.search('<nitf:hl1>(.*)<\/nitf:hl1>', a)
                if curr_headline: curr_headline = curr_headline.group(1)
                else: continue
                
                curr_headline_id = re.search('<dc:identifier identifierScheme="PGUID">urn:contentItem:(.*)<\/dc:identifier>', a)
                if curr_headline_id: curr_headline_id = curr_headline_id.group(1)
                else: continue 

                curr_date_fmt1 = re.search('<publicationDate month="([0-9]*)" day="([0-9]*)" year="([0-9]*)">', a)
                curr_date_fmt2 = re.search('<publicationDate day="([0-9]*)" month="([0-9]*)" year="([0-9]*)">', a)
                if curr_date_fmt1:
                    curr_month = curr_date_fmt1.group(1) 
                    curr_day = curr_date_fmt1.group(2)
                    curr_year = curr_date_fmt1.group(3)
                elif curr_date_fmt2: 
                    curr_day = curr_date_fmt2.group(1) 
                    curr_month = curr_date_fmt2.group(2)
                    curr_year = curr_date_fmt2.group(3)
                else:
                    curr_day, curr_month, curr_year = "", "", ""

                headline.append(curr_headline)
                headline_id.append(curr_headline_id)
                day.append(curr_day)
                month.append(curr_month)
                year.append(curr_year)
        headline_df = pd.DataFrame({
            'headline_id': headline_id,
            'headline': headline,
            'month': month,
            'day': day,
            'year': year
        }).drop_duplicates()
        headline_df.to_csv('ln_headlines.csv', mode = 'a', header = not os.path.exists('ln_headlines.csv'))

        print('finished', ln)

def reindex_ln_headlines():
    ln_headlines = pd.read_csv('ln_headlines.csv').drop(columns = 'Unnamed: 0').reindex()
    ln_headlines.to_csv('ln_headlines.csv')

def train_brill_tagger():
    pass 

# parse_cb_companies()
# parse_cb_headlines()
# parse_lexis_nexis_headlines()
# reindex_ln_headlines()
    