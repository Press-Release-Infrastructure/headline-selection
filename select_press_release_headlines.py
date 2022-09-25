import re
import nltk
from fuzzywuzzy import fuzz
import string
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd

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

# parse_cb_companies()
# parse_cb_headlines()
    