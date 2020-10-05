from selenium import webdriver
import time
import pickle

def get_issue_url(driver, base_url, year_list):
    driver.get(base_url)
    time.sleep(1)

    issue_items = driver.find_elements_by_xpath('//article[starts-with(@class, "journal-preview ")]//div//a')
    print(len(issue_items))

    issue_url_list = []
    for item in issue_items:
        for year in year_list:
            if year in item.text:
                print(item.text)
                issue_url_list.append( item.get_attribute('href') )
    
    return issue_url_list


def get_article_url(driver, issue_url):
    driver.get(issue_url)
    time.sleep(1)

    articles = driver.find_elements_by_xpath('//article[starts-with(@class, "journal-article ")]//h3//a')

    article_url_list = []
    for item in articles:
        article_url_list.append( item.get_attribute('href') )

    return article_url_list


if __name__ == "__main__":
    # base url for journal
    base_url = "https://www.aeaweb.org/journals/aer/issues"

    # set up browser driver
    browser = webdriver.Chrome()
    options = webdriver.ChromeOptions()
    prefs = {
        "download.prompt_for_download": False,
        'download.default_directory': './data',
        "plugins.always_open_pdf_externally": True
    }
    options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(executable_path='./chromedriver',
                              chrome_options=options)

    # get url of all issues
    year_list = ["2011", "2012", "2013", "2014"]
    issue_url_list = get_issue_url(driver, base_url, year_list)
    print(issue_url_list)

    # get the url of all articles
    # issue_url = "https://www.aeaweb.org/issues/570"
    # article_url_list = get_article_url(driver, issue_url)
    article_url_list = []
    for issue_url in issue_url_list:
        article_url_list += get_article_url(driver, issue_url)

    # get url of all pdf
    pdf_url_list = []
    for url in article_url_list:
        elements = url.split("/")
        journal_id = elements[-2][12:]
        article_id = elements[-1]
        pdf_url = f"https://www.aeaweb.org/articles/pdf/doi/{journal_id}/{article_id}"
        pdf_url_list.append(pdf_url)
        # print(pdf_url)

    with open("./pdf_url.pkl", 'wb') as f:
        pickle.dump(pdf_url_list, f)
