# Eosin: A Comprehensive Bank Statement Cell Parsing Tool

**Eosin** is a tool built to tackle one of the trickiest problems in data extraction—parsing bank statements. If you've ever looked at bank statements from different institutions, you know how wildly they can vary in structure, format, and content. Eosin aims to make sense of that chaos by using clever techniques to extract data from these complex PDFs, no matter how inconsistent or irregular they are.

## Why the Name Eosin?

In biology, _eosin_ is a dye that helps differentiate cells under a microscope. In a similar way, this package is designed to differentiate and extract data from the messy structures of bank statements. While creating a parser for a single, specific statement format is easy, Eosin is built to handle the hardest version of the problem—working with all kinds of inconsistent formats and messy data.

## Production Modal endpoint

Use the deployed Modal web endpoint as `https://<workspace>--bank-parser.modal.run`. When proxy auth is enabled, callers must send `Modal-Key` and `Modal-Secret`; Modal reports this as `requires_proxy_auth`.

Production secrets are provisioned with `modal secret create eosin-tailscale` for Tailscale access and `modal secret create eosin-metrics-push` for push metrics credentials. pull-based monitoring is intentionally disabled for the Modal worker; use the configured push endpoint instead.

Eosin does not impose application-level PDF byte-size or page-count limits.
Uploads must still have a PDF filename and signature and must open successfully
with PyMuPDF. Production work remains protected by Modal proxy authentication,
bounded container concurrency, parser-pool admission timeout, maximum container
count, and function timeout. `/health` is liveness; `/ready` checks
parser-service readiness and returns `503` when the service cannot accept work.

### What Makes Bank Statements So Hard to Parse?

Bank statements are notorious for being a nightmare to automate due to:

1. **Inconsistent Headers**: Each statement has its own unique headers, which can even change across pages.
2. **Cell Size Variations**: Adjacent cells aren’t always the same size.
3. **Irregular Rows and Columns**: Rows and columns often don’t follow consistent heights and widths.
4. **No Reliable Borders**: Borders may or may not exist, so we can’t rely on them.
5. **Multiline Dates**: Dates might be crammed into one line or spread across two or more.
6. **Date Format Chaos**: There’s no consistent way dates are presented—every statement seems to have its own idea.
7. **Missing Data**: Some rows might have empty columns, especially for certain transactions.
8. **Random Rows**: There are often irrelevant or random rows of data that throw everything off.
9. **Alignment Problems**: Text inside cells might be aligned in any direction—center, left, or right.
10. **Varying Headers**: Table headers can change or overlap as you go from page to page.
11. **No Consistent Row Spacing**: Nearby rows might be squeezed together or spaced far apart.
12. **Currency Format Mess**: Currency symbols and formats can be completely different between statements.
13. **Unreadable Statements**: Some statements are just hard to read—even for a human.

### The Assumptions We Make

To manage all these headaches, we made a few assumptions:

- **Dates Are Key**: We treat the date header as the most reliable thing on the page. We use it to figure out the structure of the table and align everything else around it.
- **Smart Date Parsing**: Eosin will try to pull together broken or spread-out dates and align them. If it still doesn’t make sense, we’ll ignore it and move on.
- **Headers Don’t Overlap**: We assume headers don’t interfere with each other, making them useful to anchor the rest of the data.
- **Spacing is Reasonably Consistent Across Pages**: While row and column spacing might be all over the place on one page, we assume it doesn’t change too wildly across the different pages.

### Known Issues and TODOs

- Last row of the table is clipped out intentionally currently for testing purposes.
- The date parser library is quite slow and also accepts incorrect dates sometimes (for example '01/01/2024 d' is accepted as valid)
- The parser currently does not differentiate between dates that are top aligned, center aligned, or bottom aligned.
- Padding between the date header and adjacent headers isn't calculated correctly, currently we assume a fixed padding.
- Whenever we need to search for a certain word/property within a word, we currently iterate over every word in the document. Should implement a hashmap type structure for this.
- Only the first page of the document is parsed currently for testing purposes.
- The parser currently does not differentiate between different types of transactions (credit, debit, etc.) within the text itself (for example '15CR' or '15DR')
