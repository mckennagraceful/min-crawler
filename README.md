# Introduction
It's a crawler program that can start from a original pdf and automatically extract the references within and crwal pdfs and webpages that may be useful from the Internet.

We crwal pdfs and webpages from mainly two resources. One is Google Scholar, another is direct Bing keywords search. The outputs are some pdfs and two cvs files which contains file name/webpage url and the scores we calculate using TF-IDF. The scores are normalized and adjust between 0~100.

# Usage
- Put the original pdf in the path as same as crawler.py change the initial_pdf_path to the name of your pdf file.
- Change the passing values of the function extract_references_from_pdf(), specifically, start and end to the start and end page numbers of your orignal pdf.
- We use SerpApi to crawl results from Google Scholar. The api_key we are using now binds to my account. However, it is chargeable. Therefore, if you want to crwal more in the future, you should pay your own account and change the api_key in the script. You can refer to https://serpapi.com/google-scholar-api for more information. If you don't want to pay, I've written a function named search_google_scholar() in the script you can refer to. But the problem is that frequent visits to Google Scholar may result in CAPTCHA.
- You can add more keywords into keywords_list or remove some out of it to get your own criterion.
- mineral_names may also be adjusted according to actual conditions (mineral_names is used for direct Bing search by concatenating each mine name in mineral_names with "lead-zinc mine data" to search in Bing). You can also change the concatenating string to what you expect.
- Remember to download all required libraries before running the program.