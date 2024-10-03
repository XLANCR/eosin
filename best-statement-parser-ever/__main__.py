import arrow.util
import pdfplumber
import pandas as pd
import dateparser
from dateparser.search import search_dates

with pdfplumber.open("test5.pdf") as pdf:
    first_page = pdf.pages[1]
    # print([first_page.chars[i]["text"] for i in range(1000)])
    im = first_page.to_image(resolution=150)
    # im.draw_rects(first_page.extract_words())
    # im.debug_tablefinder()
    # im.show()
    # table = first_page.extract_table(
    #     table_settings={
    #         "vertical_strategy": "text",
    #         "horizontal_strategy": "lines",
    #     }
    # )
    # print(table)
    words_list = first_page.extract_words()
    date_columns = []
    date_rows = []
    date_start = 0
    for count in range(len(words_list)):
        words_list[count]["index"] = count


    def find_nearby_headers(text_date_object):
        potential_headers = []
        for i in range(date_start, len(words_list)):
            if words_list[i]["top"] > (text_date_object["top"] - 15) and words_list[i]["bottom"] < (text_date_object["bottom"] + 10):
                potential_headers.append(words_list[i])

        return potential_headers
    
    def is_good_date(inp):
        desired_headers = ["deposit", "withdrawal", "credit", "detail", "particular", "reference", "chq", "cheque", "narration"]
        found_headers = find_nearby_headers(inp)
        print([fheader["text"].lower() for fheader in found_headers])

        for x in found_headers:
            if any(header in x["text"].lower() for header in desired_headers):
                print("GOODLIST", [text_object["text"] for text_object in found_headers])
                return True
        else:
            return False
    
    def combine_text_objects(text_objects):
        text = ' '.join(x["text"] for x in text_objects)

        left_most = min(x["x0"] for x in text_objects)
        right_most = max(x["x1"] for x in text_objects)
        top = min(x["top"] for x in text_objects)
        doc_top = min(x["doctop"] for x in text_objects)
        bottom = min(x["bottom"] for x in text_objects)

        height = bottom - top # Since (0,0) is the top left of the page, bottom value is higher than top
        width = right_most - left_most # Same for this, righter value is greater than lefter

        return {
            'text': text,
            'x0': left_most,
            'x1': right_most,
            'top': top,
            'doctop': doc_top,
            'bottom': bottom,
            'upright': True,
            'height': height,
            'width': width,
            'direction': 'ltr'
        }

    def group_adjacent_headers(headers, expected_gap=5):
        nearby_headers = sorted(headers, key=lambda x: x["x0"])
        grouped_headers = []

        nearest_x1 = nearby_headers[0]["x1"]
        words_list_grouping = [nearby_headers[0]]
        for i in range(1, len(nearby_headers)):
            if nearby_headers[i]["x0"] - nearest_x1 < expected_gap:
                words_list_grouping.append(nearby_headers[i])
                nearest_x1 = nearby_headers[i]["x1"]
            else:
                # here we are sorting so the header that is higher on the page ( has lower y value) gets shown first, otherwise it would get shown second
                words_list_grouping.sort(key=lambda x: x["bottom"])
                grouped_headers.append(words_list_grouping)
                words_list_grouping = [nearby_headers[i]]
                nearest_x1 = nearby_headers[i]["x1"]

        grouped_headers.append(words_list_grouping) # Append the last grouping since that doesn't get appended above

        headers = [combine_text_objects(group) for group in grouped_headers]
        # print(grouped_headers)
        return headers
    
    def find_padding(inp):
        
        header_list = find_nearby_headers(inp)

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
                print("Incomplete date")
                return "Incomplete"
        else:
            print("Not a date")
            return "Invalid"


    headers = []
    for i in range(0, len(words_list)):
        # print(words_list[i]["text"])
        if "date" in words_list[i]["text"].lower():
            # print(words_list[i-1])
            # print(words_list[i])
            # print(words_list[i+1])
            # padding = (words_list[i+1]["x0"] - words_list[i]["x1"]) / 2
            print(words_list[i])
            if is_good_date(words_list[i]):
                
                adjacent_headers = find_nearby_headers(words_list[i])
                headers = group_adjacent_headers(adjacent_headers)
                print([header["text"] for header in headers])
                date_column = words_list[i]
                # for header in headers:
                #     if "date" in header["text"].lower():
                #         date_column = header
                #         break
                padding = find_padding(date_column)
                date_columns = [date_column["x0"] - padding[0], date_column["x1"]+ padding[1]]
                print(date_columns)
                
                date_start = i
                break
    

    def detect_cell_alignment(dates):

        arr = []
        for date in dates:
            same_alignment_text = []
            print(date["text"])

            for i in range(0, len(words_list)):
                if abs(words_list[i]["bottom"] - date["bottom"]) < 2 and words_list[i]["x0"] > date["x0"] + 10:
                    same_alignment_text.append(words_list[i])
                if words_list[i]["bottom"] > date["bottom"] + 3:
                    break
            
            if len(same_alignment_text) > 3:
                arr.append(True)
                print([text["text"] for text in same_alignment_text])
        
        if all(arr):
            print(True)

        print(arr)

    temp_dates = []
    for i in range(date_start + 1, len(words_list)):
        if words_list[i]["x0"] > date_columns[0] - 15 and words_list[i]["x1"] < date_columns[1] + 15:
            print(words_list[i]["x0"],words_list[i]["x1"])
                # nearby = find_nearby_headers(words_list[i])
                # values = group_adjacent_headers(nearby)
                # print([value["text"] for value in values])
            temp_dates.append(words_list[i])

    print([date["text"] for date in temp_dates])
    dates = []
    i = 0
    while i in range(len(temp_dates)):
        match is_valid_date(temp_dates[i]["text"]):
            case "Valid":
                dates.append(temp_dates[i])
            case "Incomplete":
                if i + 1 < len(temp_dates):
                    match is_valid_date(combine_text_objects([temp_dates[i], temp_dates[i+1]])["text"]):
                        case "Valid":
                            dates.append(combine_text_objects([temp_dates[i], temp_dates[i+1]]))
                            i += 1
                        case "Incomplete":
                            match is_valid_date(combine_text_objects([temp_dates[i], temp_dates[i+1], temp_dates[i+2]])["text"]):
                                case "Valid":
                                    dates.append(combine_text_objects([temp_dates[i], temp_dates[i+1], temp_dates[i+2]]))
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
    def group_adjacent_text(text_objects, expected_gap=3):
        nearby_text_objects = sorted(text_objects, key=lambda x: x["x0"])
        grouped_text = []
        # print([text["text"] for text in nearby_text_objects])

        nearest_x1 = nearby_text_objects[0]["x1"]
        words_list_grouping = [nearby_text_objects[0]]
        # for i in range(1, len(nearby_text_objects)):

  
        for i in range(1, len(nearby_text_objects)):
            if nearby_text_objects[i]["x0"] - nearest_x1 < expected_gap:
                # print("NEAREST X1", nearest_x1)
                # print(nearby_text_objects[i]["x0"])
                # print("We should never be here", nearby_text_objects[i]["x0"] - nearest_x1)
                words_list_grouping.append(nearby_text_objects[i])
                nearest_x1 = nearby_text_objects[i]["x1"]
            else:
                # here we are sorting so the header that is higher on the page ( has lower y value) gets shown first, otherwise it would get shown second
                words_list_grouping.sort(key=lambda x: x["bottom"])
                grouped_text.append(words_list_grouping)
                words_list_grouping = [nearby_text_objects[i]]
                nearest_x1 = nearby_text_objects[i]["x1"]

        grouped_text.append(words_list_grouping) # Append the last grouping since that doesn't get appended above

        output_text = [combine_text_objects(group) for group in grouped_text]
        # print(grouped_headers)
        return output_text

    def categorize_text_from_headers(text_objects, headers):
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


    def parse_dates_top_aligned(dates):
        gaps = []

        table = {}

        

        # print([date["text"] for date in dates])
        for i in range(len(dates) - 1):
            gaps.append(dates[i+1]["top"] - dates[i]["bottom"])
        
        print(gaps)
        
        for i in range(len(dates) - 1):
            potential_headers = [dates[i]]
            top = dates[i]["top"] - 2
            bottom = dates[i]["bottom"] + gaps[i] + 2
            print(dates[i]["text"], gaps[i], (dates[i]["top"] - 2),(dates[i]["bottom"] + gaps[i] + 2)  )

            for j in range(date_start, len(words_list)):
                if words_list[j]["top"] > top and words_list[j]["bottom"] <  bottom and words_list[j]["x0"] > dates[i]["x0"]:
                    potential_headers.append(words_list[j])
        
            # print([header["text"] for header in potential_headers])
            bruh = group_adjacent_text(potential_headers)
            # print([header["text"] for header in bruh])
            categorized_text = categorize_text_from_headers(bruh, headers)
            for ctext in categorized_text:
                if ctext not in table:
                    table[ctext] = [categorized_text[ctext]]
                else:
                    table[ctext].append(categorized_text[ctext])
            # if i >1:
            #     break
        # print(table)
        df = pd.DataFrame(table)
        print(df)
    
    
    def parse_dates_center_aligned(dates):
        gaps = []
        table = {}

        # print([date["text"] for date in dates])
        for i in range(len(dates) - 1):
            gaps.append(dates[i+1]["top"] - dates[i]["bottom"])
        
        print(gaps)
        
        for i in range(len(dates) - 1):
            potential_headers = []
            if i == 0:
                top = dates[i]["top"] - (gaps[0]//2 + 2)
            else:
                top = dates[i]["top"] - (gaps[i-1]//2 + 2)

            bottom = dates[i]["bottom"] + gaps[i]//2 + 2
            print(dates[i]["text"], gaps[i], (dates[i]["top"] - 2),(dates[i]["bottom"] + gaps[i] + 2)  )

            for j in range(date_start, len(words_list)):
                if words_list[j]["top"] > top and words_list[j]["bottom"] <  bottom and words_list[j]["x0"] > dates[i]["x0"]:
                    potential_headers.append(words_list[j])
        
            # print([header["text"] for header in potential_headers])
            # print(potential_headers)

            bruh = group_adjacent_text(potential_headers)
            # print([header["text"] for header in bruh])
            categorized_text = categorize_text_from_headers(bruh, headers)
            for ctext in categorized_text:
                if ctext not in table:
                    table[ctext] = [categorized_text[ctext]]
                else:
                    table[ctext].append(categorized_text[ctext])
            # if i >1:
            #     break
        # print(table)
        df = pd.DataFrame(table)
        print(df)
    
    def parse_dates_bottom_aligned(dates):
        gaps = []
        table = {}

        # print([date["text"] for date in dates])
        for i in range(len(dates) - 1):
            gaps.append(dates[i+1]["top"] - dates[i]["bottom"])
        
        print(gaps)
        
        for i in range(len(dates) - 1):
            potential_headers = []
            if i == 0:
                top = dates[i]["top"] - (gaps[0]//2 + 4)
            else:
                top = dates[i]["top"] - (gaps[i-1]//2 + 4)
            bottom = dates[i]["bottom"] + 2
            print(dates[i]["text"], gaps[i], (dates[i]["top"] - 2),(dates[i]["bottom"] + gaps[i] + 2)  )

            for j in range(date_start, len(words_list)):
                if words_list[j]["top"] > top and words_list[j]["bottom"] <  bottom and words_list[j]["x0"] > dates[i]["x0"]:
                    potential_headers.append(words_list[j])
        
            # print([header["text"] for header in potential_headers])
            bruh = group_adjacent_text(potential_headers)
            # print([header["text"] for header in bruh])
            categorized_text = categorize_text_from_headers(bruh, headers)
            for ctext in categorized_text:
                if ctext not in table:
                    table[ctext] = [categorized_text[ctext]]
                else:
                    table[ctext].append(categorized_text[ctext])
            # if i >1:
            #     break
        
        df = pd.DataFrame(table)
        print(df)
        
            

    # find_cell_height(dates)
    # print(dates[0])
    # detect_cell_alignment(dates)
    parse_dates_top_aligned(dates)
    # parse_dates_center_aligned(dates)
    # parse_dates_bottom_aligned(dates)


if __name__ == "__main__":
    pass
