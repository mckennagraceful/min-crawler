import csv
import glob
import json
import logging
import os
import re
import time
from collections import OrderedDict
from random import randint
from urllib.parse import urlparse, urljoin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import fitz
import numpy as np
import requests
import serpapi
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

class Logger:
    @staticmethod
    def init():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def info(message):
        logging.info(message)

def get_browser():
    option = webdriver.ChromeOptions()
    option.add_argument('--disable-gpu')
    option.add_argument('lang=zh_CN.UTF-8')
    option.add_argument('--disable-popup-blocking')
    option.add_argument("--blink-settings=imagesEnabled=false")

    option.add_argument("--disable-javascript")
    option.add_argument("--disable-preconnect")
    option.add_argument("--disable-webrtc")

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        # 'permissions.default.stylesheet': 2,
        'profile.default_content_settings.popups': 0,
        # change the path to where you want to save files
        'download.default_directory': r"google_scholar/pdfs",
        "profile.default_content_setting_values.automatic_downloads": 1,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    }
    option.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(options=option)
    browser.implicitly_wait(20)
    browser.set_page_load_timeout(20)
    return browser


def get_browser_2():
    option = webdriver.ChromeOptions()
    option.add_argument('--disable-gpu')
    option.add_argument('lang=zh_CN.UTF-8')
    option.add_argument('--disable-popup-blocking')
    option.add_argument("--blink-settings=imagesEnabled=false")

    option.add_argument("--disable-javascript")
    option.add_argument("--disable-preconnect")
    option.add_argument("--disable-webrtc")

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        'profile.default_content_settings.popups': 0,
        # change the path to where you want to save files
        'download.default_directory': r"bing_search/pdfs",
        "profile.default_content_setting_values.automatic_downloads": 1,
        "download.prompt_for_download": False,  # To auto download the file
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    }
    option.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(options=option)
    browser.implicitly_wait(20)
    browser.set_page_load_timeout(20)
    return browser

def get_html(browser, url):
    try:
        browser.get(url)
        WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        html = browser.page_source
        return html
    except Exception as e:
        print("Error:", e)
        return None

class Ask:
    _MAX_RESULTS = 10
    # @formatter:off
    _USER_AGENT = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'} # noqa
    # @formatter:on

    @staticmethod
    def _build_url(query, count):
        return f"https://www.bing.com/search?q={query.replace(' ', '+')}&count={count}"

    @staticmethod
    def _fetch_html(query, count):
        url = Ask._build_url(query, count)
        Logger.info(f"Fetching count{count}: {url}")
        return requests.get(url, headers=Ask._USER_AGENT).text

    @staticmethod
    def _scrape(html):
        result = []
        soup = BeautifulSoup(html, "html.parser")
        b_algo_elements = soup.find_all('li', class_='b_algo')
        #         print(b_algo_elements)

        for b_algo_element in b_algo_elements:
            tpcn_element = b_algo_element.find('div', class_='b_tpcn')
            if (tpcn_element is not None):
                a_element = tpcn_element.find('a')

                href_value = a_element.get('href')
                result.append(href_value)

        return result

    @staticmethod
    def _sleep():
        seconds = randint(5, 10)
        time.sleep(seconds)

    @staticmethod
    def search(query, sleep, browser):
        results = []

        if sleep:
            Ask._sleep()
        html = Ask._fetch_html(query, 30)
        #         print(html)
        page_results = Ask._scrape(html)
        #         print(page_results)
        if len(page_results) == 0:
            html = get_html(browser, Ask._build_url(query, 30))
            soup = BeautifulSoup(html, "html.parser")
            b_algo_elements = soup.find_all('li', class_='b_algo')

            for b_algo_element in b_algo_elements:
                tpcn_element = b_algo_element.find('div', class_='b_tpcn')
                if (tpcn_element is not None):
                    a_element = tpcn_element.find('a')

                    href_value = a_element.get('href')
                    page_results.append(href_value)
        results += page_results
        # results = Ask._find_unique_results(results)
        if len(results) >= Ask._MAX_RESULTS:
            results = results[:Ask._MAX_RESULTS]

        return results

class Task:
    _INPUT_DIR = "input"
    _OUTPUT_DIR = "output"

    def __init__(self):
        Logger.init()
        self._queries_statistics = OrderedDict()
        self._average_statistics = {}

    @staticmethod
    def _get_input_file_path(file_name):
        return f"{Task._INPUT_DIR}/{file_name}"

    @staticmethod
    def _get_output_file_path(file_name):
        return f"{Task._OUTPUT_DIR}/{file_name}"

    @staticmethod
    def _load_results(file_path):
        with open(file_path, "r") as f:
            return json.load(f, object_pairs_hook=OrderedDict)  # noqa

    @staticmethod
    def _dump_results(results, file_path):
        with open(file_path, "w") as f:
            f.write(json.dumps(results, indent=2))

    @staticmethod
    def _fetch_ask_results(query, sleep, browser):
        results = OrderedDict()
        query_results = Ask.search(query, sleep, browser)
        results[query] = query_results
        return results

    @staticmethod
    def _get_ask_results(query, scrape, sleep, browser):
        if scrape:
            Logger.info("Fetching Ask results")
            results = Task._fetch_ask_results(query, sleep, browser)
            return results


    def run(self, query, browser, scrape=True, sleep=True):
        ask_results = Task._get_ask_results(query, scrape, sleep if scrape else None, browser)
        return ask_results


def remove_first_line(text):
    lines = text.splitlines()

    if len(lines) > 1:
        lines = lines[1:]

    return '\n'.join(lines)


def extract_references_from_pdf(pdf_path, start, end):
    doc = fitz.open(pdf_path)

    text = ''

    for page_number in range(len(doc)):
        if page_number >= start and page_number <= end:
            page = doc.load_page(page_number)
            text += remove_first_line(page.get_text())

    doc.close()

    index = text.find('References')
    if index != -1:
        text = text[index + len('References'):]
    references = re.findall(r'([A-Za-z\s.,-]+\d{4}.*?p\.\s*\d+\s*–\d+\s*\.[\n\r]*)', text.replace("\n", ""))
    return references


def fetch_pdfs_by_references(reference, browser):
    pdf_urls = []
    task = Task()
    result_links = task.run(query=reference, browser=browser, scrape=True)
    return result_links[reference]


def get_links(browser, url, domain):

    def is_pdf(url):
        return url.lower().endswith('.pdf') or url.lower().find('.pdf') != -1

    try:
        html = get_html(browser, url)
        #         print(html)
        soup = BeautifulSoup(html, "html.parser")
        # Extract all links from the page
        for link in soup.find_all("a", href=True):
            #             print(len(soup.find_all("a", href=True)))
            next_url = urljoin(url, link["href"])
            #             print(next_url)
            parsed_next_url = urlparse(next_url)
            #             print(parsed_next_url.netloc)

            # Check if the link is within the same domain and is using http or https protocol
            if parsed_next_url.netloc == domain and parsed_next_url.scheme in ["http", "https"]:
                if is_pdf(next_url):
                    browser.get(next_url)
    except Exception as e:
        print("Error:", e)

def get_page_content(url):
    _USER_AGENT = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    try:
        response = requests.get(url, headers=_USER_AGENT)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text()
            return page_text
        else:
            print("Error:", response.status_code)
            return None
    except Exception as e:
        print("Error fetching page:", e)
        return None

def download_webpage(url, filename):
    _USER_AGENT = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    try:
        response = requests.get(url,  headers=_USER_AGENT)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print("Webpage downloaded successfully.")
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)


def search_google_scholar(query, driver):
    res = []

    driver.get("https://scholar.google.com/")

    search_box = driver.find_element(By.ID, "gs_hdr_tsi")
    search_box.send_keys(query)
    search_box.submit()

    driver.implicitly_wait(10)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    #     print(soup)
    results = soup.find_all('div', class_='gs_ri')
    #     print(results)
    for result in results:
        #         print(result)
        a_element = result.find('a')
        href_value = a_element.get('href')
        #         print(href_value)
        if href_value.startswith("http"):
            #             requests.get(href_value)
            res.append(href_value)
    #             driver.get(href_value)

    #     driver.quit()
    return res

def get_webpages(browser, mineral_names):
    webpages = []
    for name in mineral_names:
        query = name + "lead-zinc mine data"
        pages = fetch_pdfs_by_references(query, browser)
        webpages.append(pages)
    return webpages

def get_web_content(web_pages):
    web_content = []
    for web_page in web_pages:
        web_content.append(get_page_content(web_page))
    return web_content


def get_pdf_list(path):
    def extract_text_from_pdf(pdf_path):

        doc = fitz.open(pdf_path)

        text = ''

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text += page.get_text()

        doc.close()
        return text

    pdf_files = []
    for file_path in glob.glob(os.path.join(path, '*.pdf')):
        pdf_files.append(extract_text_from_pdf(file_path))
    return pdf_files

def get_file_list(path):
    res = []
    for file_path in glob.glob(os.path.join(path, '*.pdf')):
        res.append(file_path)
    return res


if __name__ == '__main__':
    mineral_names = ["Mehdiabad Iran",
                     "Admiral Bay",
                     "Navan",
                     "Reocin",
                     "Fankou",
                     "Pavlovskoye",
                     "Buick",
                     "Elura",
                     "Daliangzi",
                     "Touissit-Bou Beker",
                     "Polaris",
                     "Bleiberg",
                     "Lisheen",
                     "Prairie",
                     "San Vicente",
                     "Florida Canyon",
                     "Gayna River",
                     "Sumsar",
                     "Tianbaoshan",
                     "Cadjebut Trend",
                     "El Abed",
                     "Pillara",
                     "Black Angel",
                     "Urultun",
                     "Magmont",
                     "Nanisivik",
                     "Huayuan",
                     "Silvermines",
                     "Galmoy",
                     "Emarat"
                     ]

    # Load initial PDF references, change the file for other initial PDF
    initial_pdf_path = "USGS Report on MVT Zinc.pdf"
    initial_references = extract_references_from_pdf(initial_pdf_path, 42, 52)

    browser = get_browser()

    params = {
        "engine": "google_scholar",
        "q": "biology",
        "api_key": "682af9627880b2a155e9416d6125a4b9f217fdff4c0355c52560b677e6870d62"
    }

    # Search and fetch PDFs for each reference from Google Scholar
    for reference in initial_references:
        mineral_pdf_urls = fetch_pdfs_by_references(reference, browser)
        params["q"] = reference

        search_results = serpapi.search(params)
        links = [result.get("link") for result in search_results["organic_results"]]
        if len(links) != 1 or links[0] is not None:
            links = links[:3]
            mineral_pdf_urls.extend(links)
        print(mineral_pdf_urls)
        for url in mineral_pdf_urls:
            if url is None:
                continue
            try:
                if url.lower().endswith('.pdf'):
                    browser.get(url)
                    time.sleep(5)
                else:
                    parsed_start_url = urlparse(url)
                    # print(domain)
                    domain = parsed_start_url.netloc
                    # Start crawling links
                    get_links(browser, url, domain)
            except TimeoutException:
                print("Time Out")

    browser.quit()

    # crawl from direct bing search with keywords
    browser = get_browser()
    webpages = get_webpages(browser, mineral_names)
    browser.quit()

    # deal with output from bing search, download those pdfs and save others in the web_pages list
    web_pages = []
    browser_2 = get_browser_2()
    for webpage_for_each_mineral in webpages:
        for webpage in webpage_for_each_mineral:
            if webpage.lower().endswith('.pdf') or webpage.find(".pdf") != -1:
                browser_2.get(url)
                time.sleep(5)
            else:
                web_pages.append(webpage)

    keywords_list = ["grade", "inferred", "indicated", "measured", "resource", "probable", "proven", "reserve",
                     "tonnage"]

    pdf_documents_list = get_pdf_list("google_scholar/pdfs")
    pdf_documents_list.extend(get_pdf_list("bing_search/pdfs"))
    file_list = get_file_list("google_scholar/pdfs")
    file_list.extend(get_file_list("bing_search/pdfs"))

    # calculate TF-IDF value for all the pdfs
    vectorizer = TfidfVectorizer(vocabulary=keywords_list)
    tfidf_matrix = vectorizer.fit_transform(pdf_documents_list)

    # normalize
    scaler = MinMaxScaler()
    normalized_tfidf_matrix = scaler.fit_transform(tfidf_matrix.toarray())

    # calculate doc score
    document_scores = np.sum(normalized_tfidf_matrix, axis=1)

    # adjust to 0-100
    min_score = np.min(document_scores)
    max_score = np.max(document_scores)
    document_scores = ((document_scores - min_score) / (max_score - min_score)) * 100

    data = zip(file_list, document_scores)

    output_file = "output/document_scores.csv"

    # write CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Score'])
        writer.writerows(data)

    # calculate TF-IDF value for all the web pages
    web_content_list = get_web_content(web_pages)

    vectorizer = TfidfVectorizer(vocabulary=keywords_list)
    tfidf_matrix = vectorizer.fit_transform(web_content_list)

    scaler = MinMaxScaler()
    normalized_tfidf_matrix = scaler.fit_transform(tfidf_matrix.toarray())

    web_scores = np.sum(normalized_tfidf_matrix, axis=1)

    min_score = np.min(web_scores)
    max_score = np.max(web_scores)
    web_scores = ((web_scores - min_score) / (max_score - min_score)) * 100

    data = zip(web_pages, web_scores)

    output_file = "output/webpages_scores.csv"

    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Web Url', 'Score'])
        writer.writerows(data)

