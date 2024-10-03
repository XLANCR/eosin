import pdfplumber
from utils import combine_text_objects
import dateparser
from dateparser.search import search_dates

class Parser:
    def __init__(self, statement):
        self.statement = statement
        self.pdf_object = None
        self.words_list = None
        self.headers = []
        self.table_date = None
        self.date_column = None

    def parse(self):
        with pdfplumber.open(self.statement) as pdf:
            pdf_objects = pdf.pages[0]
            self.pdf_object = pdf_objects
            self._get_words()
            return self.words_list

    def _get_words(self):
        words_list = self.pdf_object.extract_words()
        for index in range(len(words_list)):
            words_list[index]["index"] = index

        self.words_list = words_list

        # return self.text
    
    def find_nearby_headers(self, text_date_object, date_start, top_padding = 15, bottom_padding = 10):
        potential_headers = []
        words_list = self.words_list
        for i in range(date_start, len(self.words_list)):
            if words_list[i]["top"] > (text_date_object["top"] - top_padding) and words_list[i]["bottom"] < (text_date_object["bottom"] + bottom_padding):
                potential_headers.append(words_list[i])

        return potential_headers

    def is_table_date(self, date):
        DESIRABLE_HEADERS = ["deposit", "withdrawal", "credit", "detail", "particular", "reference", "chq", "cheque", "narration"]
        nearby_headers = self.find_nearby_headers(date)
        print([nearby_header["text"].lower() for nearby_header in nearby_headers])

        for nearby_header in nearby_headers:
            if any(desirable_header in nearby_header["text"].lower() for desirable_header in DESIRABLE_HEADERS):
                print("GOODLIST", [text_object["text"] for text_object in nearby_headers])
                return True
        else:
            return False
    
    def find_date_header_padding(self, date_header):
        
        header_list = self.find_nearby_headers(date_header)

        horizontal_gaps = []
        for x in range(len(header_list) - 1):
            horizontal_gaps.append(header_list[x+1]["x0"] - header_list[x]["x1"])
        
        vertical_gaps = []
        for x in range(len(header_list)):
            vertical_gaps.append(header_list[x]["bottom"])

        # print(horizontal_gaps)
        # print(vertical_gaps)
        return [10, 10]
    
    def is_valid_date(date_string):
        # Parse the date
        parsed = dateparser.parse(date_string)
        
        # Check if parsing was successful
        if parsed:
            # Get detailed information about the parsed date
            info = search_dates(date_string, settings={'REQUIRE_PARTS': ['day', 'month', 'year'], 'STRICT_PARSING': True})
            
            if info:
                # Date was complete (has day, month, and year)
                return "Valid"
            else:
                # Date was incomplete
                # print("Incomplete date")
                return "Incomplete"
        else:
            # print("Not a date")
            return "Invalid"
    
    def find_date_header(self):
        words_list = self.words_list
        for i in range(len(words_list)):
            # print(words_list[i]["text"])
            if "date" in words_list[i]["text"].lower():
                # print(words_list[i-1])
                # print(words_list[i])
                # print(words_list[i+1])
                # padding = (words_list[i+1]["x0"] - words_list[i]["x1"]) / 2
                potential_date = words_list[i]
                print(potential_date)

                if self.is_table_date(potential_date):
                    
                    table_date = potential_date

                    adjacent_headers = self.find_nearby_headers(table_date)

                    headers = self.group_adjacent_headers(adjacent_headers)
                    print([header["text"] for header in headers])

                    # for header in headers:
                    #     if "date" in header["text"].lower():
                    #         date_column = header
                    #         break
                    padding = self.find_padding(table_date)

                    date_columns = (
                        table_date["x0"] - padding[0], 
                        table_date["x1"] + padding[1]
                    )
                    print(date_columns)

                    self.table_date = table_date
                    self.date_columns = date_columns                    
                    break
        else:
            raise Exception("Date header not found")
        
    
        

parser = Parser("test.pdf")
print(parser.parse()[0])
    