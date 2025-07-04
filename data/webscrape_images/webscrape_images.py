from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

service = Service(executable_path = './chromedriver')
driver = webdriver.Chrome(service = service)

driver.get('https://app.box.com/folder/325323078385?s=m39ejz87u7k3iesoasu359w4yloox1ic&sortColumn=name&sortDirection=ASC')

time.sleep(60)

driver.quit()