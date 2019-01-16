from bs4 import BeautifulSoup as soup, SoupStrainer
import requests
import lxml
import pandas as pd
import sys
from io import StringIO
from datetime import datetime
import multiprocessing

def tags(links):
    df_tag = pd.DataFrame()

    for link in links.iterrows():
        try:
            url = link[1]['steam_url']
            r = requests.get(url, cookies=cookies)
            div_tags = SoupStrainer("div", {"class":"glance_tags popular_tags"})
            page_soup = soup(r.content, "lxml", parse_only=div_tags)
            #title = page_soup.find('span', {'itemprop':'name'}).text
            tags = page_soup.find("div", {"class":"glance_tags popular_tags"})
            tags = tags.text
            tags = tags.replace('\t', '')
            tags = tags.replace('\r', '')
            tags = tags.replace('\n', ', ')
            tags = tags.replace('+, ', '')
            tags = tags.replace(', , ', '')
            df_tags = pd.DataFrame([tags])
            df_tag = df_tag.append(df_tags)
        except AttributeError:
            title = "skip"
            tags = "N/A"
            df_tags = pd.DataFrame([tags])
            df_tag = df_tag.append(df_tags)

    df_tag_2 = df_tag.reset_index()
    df_tag_final = df_tag_2.drop(['index'], axis=1)

    steam_tags = pd.concat([df_links, df_tag_final], axis=1, sort=False)
    return steam_tags.to_csv('steam_tags', sep="\t", header=['title', 'steam_url', 'steam_tags'])
    
startTime = datetime.now()
cookies = {'birthtime': '568022401'}
df_links = pd.read_csv('products_test.csv', delimiter="\t")
# tags(df_links)

if __name__ == '__main__':
    for i in range(100):
        p = multiprocessing.Process(target=tags, args=(df_links,))
        p.start()
