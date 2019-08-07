from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class GoogleImages():

    def __init__(self):
        opts = Options()
        opts.set_headless()

        self.browser = webdriver.Firefox(
            executable_path='/home/atish/Downloads/geckodriver-v0.24.0-linux64/geckodriver', options=opts)
        assert opts.headless  # Operating in headless mode
        self.browser.get('https://duckduckgo.com')
        self.search_form = self.browser.find_element_by_id(
            'search_form_input_homepage')
        self.search_form.send_keys('Google Image')
        self.search_form.submit()
        print(self.search_form.find_element_by_class_name('result__a'))

    def get_google_images(self):
        # temp = self.search_form.find_element_by_class_name("result__a")
        # temp = self.search_form.find_element_by_link_text("Google Images")
        print(temp)
        self.browser.close()
        quit()


if __name__ == "__main__":
    g_img_obj = GoogleImages()
    g_img_obj.get_google_images()
