import pdfplumber
from utils import combine_text_objects, group_adjacent_text
import dateparser
from dateparser.search import search_dates
import pandas as pd

class Parser:
    def __init__(self, statement):
        self.statement = statement
        self.pdf_object = None
        self.words_list = None
        self.headers = []
        self.table_date = None
        self.date_column_dimension = None
        self.date_rows = None
        self.data = None

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
    
    def find_header_padding(self, date_header):
        return [25, 25]
    
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
                    padding = self.find_header_padding(table_date)

                    date_column_dimensions = (
                        table_date["x0"] - padding[0], 
                        table_date["x1"] + padding[1]
                    )
                    print(date_column_dimensions)

                    self.table_date = table_date
                    self.date_column_dimensions = date_column_dimensions
                    break
        else:
            raise Exception("Date header not found")
    
    # TODO: Refactor this monstrosity of a method
    def find_date_rows(self):
        words_list = self.words_list
        date_column_dimensions = self.date_column_dimension

        potential_dates = []

        table_date_index = self.table_date["index"]

        for i in range(table_date_index + 1, len(words_list)):
            if words_list[i]["x0"] > date_column_dimensions[0] and words_list[i]["x1"] < date_column_dimensions[1]:
                print(words_list[i]["x0"],words_list[i]["x1"])
                    # nearby = find_nearby_headers(words_list[i])
                    # values = group_adjacent_headers(nearby)
                    # print([value["text"] for value in values])
                potential_dates.append(words_list[i])

        print([date["text"] for date in potential_dates])
        dates = []
        i = 0

        while i in range(len(potential_dates)):
            match self.is_valid_date(potential_dates[i]["text"]):
                case "Valid":
                    dates.append(potential_dates[i])
                case "Incomplete":
                    if i + 1 < len(potential_dates):
                        match self.is_valid_date(combine_text_objects([potential_dates[i], potential_dates[i+1]])["text"]):
                            case "Valid":
                                dates.append(combine_text_objects([potential_dates[i], potential_dates[i+1]]))
                                i += 1
                            case "Incomplete":
                                match self.is_valid_date(combine_text_objects([potential_dates[i], potential_dates[i+1], potential_dates[i+2]])["text"]):
                                    case "Valid":
                                        dates.append(combine_text_objects([potential_dates[i], potential_dates[i+1], potential_dates[i+2]]))
                                        i += 2
                                    case "Incomplete":
                                        pass
                                    case "Invalid":
                                        pass
                            case "Invalid":
                                pass
                case "Invalid":
                    pass
            i += 1
        
        print([date["text"] for date in dates])

        self.date_rows = dates
    
    def categorize_text_into_headers(self, text_objects):
        headers = self.headers
        categorized_text = {header["text"]: "" for header in headers}

        for text in text_objects:
            for header in headers:
                if max(header["x1"], text["x1"]) - min(header["x0"], text["x0"]) < header["width"] + text["width"]:
                    categorized_text[header["text"]] += text["text"]
                    break
            else:
                for header in headers:
                    if header["x0"] > text["x0"]:
                        categorized_text[header["text"]] = text["text"]
                        break
        return categorized_text
    
    def parse_dates_top_aligned(self):
        dates = self.date_rows
        words_list = self.words_list
        headers = self.headers
        table_date_index = self.table_date["index"]

        gaps_between_rows = []
        table = {}

        # print([date["text"] for date in dates])
        for i in range(len(dates) - 1):
            gaps_between_rows.append(dates[i+1]["top"] - dates[i]["bottom"])
        
        print(gaps_between_rows)
        
        for i in range(len(dates) - 1):
            potential_headers = [dates[i]]
            top = dates[i]["top"] - 2
            bottom = dates[i]["bottom"] + gaps_between_rows[i] + 2
            print(dates[i]["text"], gaps_between_rows[i], (dates[i]["top"] - 2),(dates[i]["bottom"] + gaps_between_rows[i] + 2)  )

            for j in range(table_date_index, len(words_list)):
                if words_list[j]["top"] > top and words_list[j]["bottom"] <  bottom and words_list[j]["x0"] > dates[i]["x0"]:
                    potential_headers.append(words_list[j])
        
            # print([header["text"] for header in potential_headers])
            grouped_row_text = group_adjacent_text(potential_headers)
            # print([header["text"] for header in bruh])
            categorized_text = self.categorize_text_into_headers(grouped_row_text)
            for ctext in categorized_text:
                if ctext not in table:
                    table[ctext] = [categorized_text[ctext]]
                else:
                    table[ctext].append(categorized_text[ctext])
            # if i >1:
            #     break
        # print(table)
        df = pd.DataFrame(table)
        self.data = df
        return df
        

parser = Parser("test.pdf")
print(parser.parse()[0])
    