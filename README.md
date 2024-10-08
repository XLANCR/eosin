# Why eosin?

Eosin is a coloring agent used to distinguish cells. 
This package is used to differentiate cells from bank statement pdfs.

Bank statements vary enormously when it comes to lines, spacing, column heights/widths, row heights/widths, borders, colors, column names, date formats, cell spacing, fields, and much much more.

It is fairly easy to create a parser for a particular bank statement but creating one that'll work on any statement regardless of the above mentioned inconsistencies requires one to consider the hardest version of the problem,
so that all subcases/variants are resolved automatically.

We assume that the statement has no borders, no spacing consistency, no text consistency, no standard date format, no column format, no row format, no colors, distributed tables, multiple pages, random text etc. If we solve this then we can safely say it'll work for almost any bank statement.

Here is a list of all problems I found - 
- Headers are never consistent
- Cell size is inconsistent between adjacent cells
- Row size is inconsistent between adjacent rows
- Column size is inconsistent between adjacent columns
- Can't rely on borders, as they may or may not be present
- Dates can be in a single line or even multiline
- Date format isnt consistent
- First page has headers, second page may or may not
- Each date in its cell can be at the top, bottom, or middle of the cell
- Some columns may be empty on certain rows (This one really sucks to implement) :/
- There may be random rows in between with unneccesary information
- text in cells may be center alighned, left aligned or right aligned
- There may be multiple date headers in the table
- Table headers vary a lot between statements
- Certain statements have no spacing between nearby rows
- Currency format varies wildly
- Sometimes the dates are spread across multiple lines
- Some statements are hard for even humans to read, much less an algorithm

Here is a list of assumptions we make - 

- We can use the date header as a source of truth, along with checking other lateral headers to see if its the table date header. We then use this header as the basis for the columns and use the dates found within theis column as the basis ofr rows
- We find all dates and use them as reference for the cell, ignoring text that doesn't fit the date format, while also trying to align spread out/broken dates
- If the date is not valid, we try to manipulate values above and below to see if its because its incomplete, otherwise we discard
- Table headers dont overlap with each other's fields i.e there is no overlapping text
- We assume x and y spacing remains almost consistent between different pages
