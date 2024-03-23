import requests
from bs4 import BeautifulSoup

def check_price_option(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    print(soup)
    price_section = soup.find(string=lambda text: 'maximum 6 names' in text)
    print(price_section)
    if price_section and 'Full' not in price_section:
        print("The $100 ~ $140 option might have availability!")
    else:
        print("The $100 ~ $140 option is still marked as full.")

check_price_option('https://enets.kmspks.org/new/qingming/')
