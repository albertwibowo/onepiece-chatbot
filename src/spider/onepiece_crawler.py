import scrapy
import re 
from scrapy.settings import BaseSettings

def clean_text(text):
    """Clean unwanted characters and normalize text."""
    # Remove Unicode escape sequences (e.g., \uXXXX)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

    # Replace non-breaking spaces (\u00a0) and other weird characters with regular spaces
    text = text.replace('\u00a0', ' ').replace('\u2260', '').replace('\u2021', '').replace('\u2020', '')

    # Replace newlines, tabs, and other whitespace with a single space
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Remove double quotes
    text = text.replace('"', '')

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove empty strings
    return text if text else None

class OnePieceCrawler(scrapy.Spider):
    name = 'onepiece-crawler'
    allowed_domains = ["https://onepiece.fandom.com/wiki/One_Piece_Wiki", "https://en.wikipedia.org/wiki/One_Piece"]
    
    def __init__(self, url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls.append(url)

    @classmethod
    def update_settings(cls, settings: BaseSettings) -> None:
        super().update_settings(settings)

        # old log will be replaced by new one
        settings.set("LOG_FILE_APPEND", False, priority="spider")
        # add delay to avoid getting banned
        settings.set("DOWNLOAD_DELAY", 1.5, priority="spider")

    def parse(self, response):
        self.logger.info("Parse function is running on %s", response.url)

        # Extract main information, including hyperlink text if present
        main_information = []
        for paragraph in response.css("p"):
            # Combine text and link text
            full_text = ''.join(paragraph.css("*::text").getall()).strip()
            main_information.append(clean_text(full_text))

        # Extract additional information, including hyperlink text if present
        additional_information = []
        for li in response.css("ul li"):
            # Combine text and link text
            full_text = ''.join(li.css("*::text").getall()).strip()
            additional_information.append(clean_text(full_text))

        # Yield the combined results
        yield {
            'main_information': main_information,
            'additional_information': additional_information
        }
