# mailparser version github.com/aruneshmathur/mail-parser#egg=mail-parser
import mailparser
import pandas as pd
import re
from html.parser import HTMLParser
from email.header import decode_header
import os
import glob
from bs4 import BeautifulSoup
import html2text
from collections import Counter
from unidecode import unidecode
from datetime import datetime
import uuid


# see https://stackoverflow.com/a/44611484
INVISIBLE_ELEMS = ('style', 'script', 'head', 'title')
RE_SPACES = re.compile(r' {2,}')

EMAIL_REGEX_1 = r'<[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+>'
EMAIL_REGEX_2 = r'<?[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+>?'
# Server domain
# Replace this with your own domain name
EMAIL_DOMAIN = ['gmail.com']

TERMINALS = '.?!:'

acc_dic = {}


def visible_texts(soup):
    """ get visible text from a document """
    text = ' '.join([
        s for s in soup.strings
        if s.parent.name not in INVISIBLE_ELEMS
    ])
    return RE_SPACES.sub(' ', text)


def get_preheader(big_html):

    def isolate_piece(html, element):
        html = html.strip()
        keep_elements = ["<p", "<spa", "<a", "<div", "<h", "<img"]
        important_elements = keep_elements + [element]

        start = html.find(">") + 1
        end = html[1:].find("<") + 1
        while sum([elem in html[end:][:6] for elem in important_elements]) == 0 and start == end: # skips over not important elements
            start = html.find(">", end) + 1
            end = html[1:].find("<", end) + 1
        if start == end: # for the cases like <p><p>hello</p> or <p><span>hello</span></p>
            return find_inner(html[end:])
        else: # for the cases like <p>hello</p> or <p><\p><p>hello</p>
            if html[start:end].strip() == "" or html[start:end].strip() == "-->":
                return find_inner(html[end:])
            elif sum([elem in html[end:][:6] for elem in keep_elements]) == 0 and html[end:end+len(element)] != element:
                keep, rest = isolate_piece(html[end:], element)
                return html[start:end]+keep, rest
            return html[start:end], html[end:]

    def find_inner(html):
        p = html.find("<p ")
        p1 = html.find("<p>")
        span = html.find("<span ")
        span1 = html.find("<span>")
        href = html.find("<a ")
        href1 = html.find("<a>")
        div = html.find("<div ")
        div1 = html.find("<div>")
        td = html.find("<td ")
        td1 = html.find("<td>")
        h = [html.find("<h{} ".format(i)) for i in range(1, 7)]
        h = [float("inf") if ind == -1 else ind for ind in h]
        h = min(h)
        h1 = [html.find("<h{}>".format(i)) for i in range(1, 7)]
        h1 = [float("inf") if ind == -1 else ind for ind in h1]
        h1 = min(h1)
        img = html.find("<img ")

        indices = [p, p1, span, span1, href, href1, div, div1, td, td1, h, h1, img]
        indices = [float("inf") if ind == -1 else ind for ind in indices]

        first = min(indices)

        if first == float("inf"):
            return "", ""
        this_chunk = html[first:]
        if p == first or p1 == first:
            return isolate_piece(this_chunk, '</p')
        elif span == first or span1 == first:
            return isolate_piece(this_chunk, '</span')
        elif href == first or href1 == first:
            return isolate_piece(this_chunk, '</a')
        elif div == first or div1 == first:
            return isolate_piece(this_chunk, "</div")
        elif td == first or td1 == first:
            return isolate_piece(this_chunk, "</td")
        elif h == first or h1 == first:
            return isolate_piece(this_chunk, "</h")
        elif img == first:
            alt = this_chunk.find("alt")
            end = this_chunk.find(">")
            if end < alt: # for the case of older HTML versions that don't require alt
                return isolate_piece(this_chunk[end+1:], '')
            quotation1 = this_chunk[alt:].find('"')
            quotation2 = this_chunk[alt:].find("'")
            if quotation1 != -1 and quotation1 < quotation2:
                second_quotation1 = this_chunk[alt:][quotation1+1:].find('"')
                if second_quotation1 > 0:
                    return this_chunk[alt:][quotation1+1:quotation1+1+second_quotation1], this_chunk[alt:][quotation1+1+second_quotation1:]
                else:
                    return find_inner(this_chunk[this_chunk[alt:].find(">") + 1:])
            else:
                second_quotation2 = this_chunk[alt:][quotation1+1:].find("'")
                if second_quotation2 > 0:
                    return this_chunk[alt:][quotation2+1:quotation2+1+second_quotation2], this_chunk[alt:][quotation2+1+second_quotation2:]
                else:
                    return find_inner(this_chunk[this_chunk[alt:].find(">") + 1:])

    to_remove = ['&nbsp;', '&zwnj;', '<br>', '<br />', '\n', '\t', '\r', u'\u200c']
    for remove in to_remove:
        big_html = big_html.replace(remove, ' ')

    big_result = ''
    small_result = ''

    while len(big_result) < 150:

        uncleaned_result, big_html = find_inner(big_html)
        result = HTMLParser().unescape(uncleaned_result).rstrip()
        for remove in to_remove:
            result = result.replace(remove, '')
        big_result += ' ' + result

        if small_result == '':
            small_result = big_result.rstrip()

    return small_result, big_result.rstrip()


def only_alphanum(words):
    return re.sub(r'\W+', '', words)


def extract_recipient(to):
    # This function returns the recipient's address from the 'to' output of
    # mailparser. This output of mailparser is a list of lists. This function
    # assumes the list has only one list item, and searches its items for a
    # string resembling an email address. If the length of 'from_' is greater
    # than two, this will return the first match.
    for to_list_item in to:
        for to_item in to_list_item:
            to_item = str(to_item)
            # Regular expression that matches emails.
            if re.match(EMAIL_REGEX_2, to_item):
                to_item = to_item.strip().strip('<>').strip().lower()
                if to_item.split('@')[1] in EMAIL_DOMAIN:
                    return to_item

    return None


def get_emails_from_folder(base_dir, index_file, source=None):

    def get_emails(data, val=None):
        email_sender, from_name, from_address, to_address, email_timestamp,\
        headers, subject, body_text,\
        preheader_long, preheader_short, email_path_list = [], [], [], [], [], [], [], [], [], [], []

        raw_dir = 'raw'
        html_dir = 'html'
        plain_dir = 'plain'
        metadata_dir = 'metadata'
        httrack_dir = 'httrack'

        if data is None:
            recipient = val
        else:
            recipient = data.name

        recipient_dir = os.path.join(base_dir, raw_dir, recipient)

        if not os.path.isdir(recipient_dir):
            return None

        for sender_dir in os.listdir(recipient_dir):
            for email_dir in os.listdir(os.path.join(recipient_dir, sender_dir)):
                for email_path in glob.glob(os.path.join(recipient_dir, sender_dir, email_dir, 'email.eml')):

                    html_file = os.path.join(base_dir, html_dir, recipient, sender_dir, email_dir, 'email.html')
                    plain_file = os.path.join(base_dir, plain_dir, recipient, sender_dir, email_dir, 'email.txt')
                    header_file = os.path.join(base_dir, metadata_dir, recipient, sender_dir, email_dir, 'headers.txt')
                    httrack_file = os.path.join(base_dir, httrack_dir, recipient, sender_dir, email_dir, 'email.png')

                    if not os.path.isfile(header_file):
                        continue

                    if (not os.path.isfile(html_file)) or (os.path.isfile(html_file) and not os.path.isfile(httrack_file)):
                        date_email = datetime.fromtimestamp(float(email_dir))
                        today = datetime.today()
                        if abs((today - date_email).days) < 3:
                            continue

                    email_sender.append(sender_dir)
                    email_timestamp.append(email_dir)
                    email_path_list.append(email_path)

                    with open(email_path, 'rb') as f:
                        parsed_email = mailparser.parse_from_bytes(f.read())

                        header_dict = dict((k.lower(), v) for k,v in parsed_email.headers.items())
                        from_header = header_dict.get('from', '')

                        fa = re.search(EMAIL_REGEX_1, from_header)
                        if fa is None:
                            fa = re.search(EMAIL_REGEX_2, from_header)

                        if fa is not None:
                            fa = fa.group().strip().strip('<>').strip()
                        else:
                            fa = ''

                        from_address.append(fa.lower())

                        fn = re.sub(EMAIL_REGEX_1, '', from_header)
                        if fn == from_header:
                            fn = re.sub(EMAIL_REGEX_2, '', from_header)
                        fn = fn.strip().strip('"').strip()
                        if len(fn) == 0 or fn is None:
                            from_name.append(fa)
                        else:
                            from_name.append(fn)

                        ta = extract_recipient(parsed_email.to)
                        if ta is None:
                            ta = extract_recipient(parsed_email.cc)
                            if ta is None:
                                ta = extract_recipient(parsed_email.bcc)
                                if ta is None:
                                    for received in parsed_email.received:
                                        ta = extract_recipient(received.items())
                                        if ta is not None:
                                            break

                        if ta is None:
                            ta = ''

                        to_address.append(ta.lower())

                        body_content = None

                        if os.path.isfile(html_file):
                            with open(html_file, 'rb') as hf:
                                content = hf.read()
                                charset_file = os.path.join(base_dir, html_dir, recipient, sender_dir, email_dir, 'charset.txt')
                                with open(charset_file, 'rb') as ef:
                                    charset = ef.read().decode('utf-8')
                                    content = content.decode(charset)
                                    # body_html.append(content)
                                    try:
                                        body_content = '\n'.join(get_email_lines(content))
                                    except:
                                        body_content = None
                                        print('Warning: Unable to extract HTML text')
                                    ps, pl = get_preheader(content)
                                    preheader_short.append(ps)
                                    preheader_long.append(pl)
                        else:
                            # body_html.append('')
                            preheader_short.append('')
                            preheader_long.append('')

                        if os.path.isfile(plain_file):
                            with open(plain_file, 'rb') as pf:
                                content = pf.read()
                                charset_file = os.path.join(base_dir, plain_dir, recipient, sender_dir, email_dir, 'charset.txt')
                                with open(charset_file, 'rb') as ef:
                                    charset = ef.read().decode('utf-8')
                                    content = content.decode(charset)
                                    # body_plain.append(content)

                                    if body_content is None:
                                        body_content = '\n'.join(remove_empty_lines(content))
                        else:
                            pass
                            # body_plain.append('')

                        if body_content is None:
                            body_text.append('')
                        else:
                            body_text.append(body_content)

                        if os.path.isfile(header_file):
                            with open(header_file, 'rb') as hef:
                                content = hef.read().decode('utf-8')
                                headers.append(content)
                                content = eval(content)
                                subject_string = ''
                                for c in content:
                                    if c[0] == 'Subject':
                                        for tup in decode_header(c[1]):
                                            if tup[1] is None:
                                                try:
                                                    subject_string += tup[0].decode('utf-8')
                                                except:
                                                    try:
                                                        subject_string += tup[0].encode('utf-8', errors='ignore').decode('utf-8')
                                                    except:
                                                        subject_string += unidecode(tup[0])
                                            else:
                                                subject_string += tup[0].decode(tup[1])

                                subject.append(subject_string)
                        else:
                            headers.append('')
                            subject.append('')

        return pd.DataFrame({'email_sender': email_sender,
                             'from_name': from_name,
                             'from_address': from_address,
                             'to_address': to_address,
                             'email_timestamp': email_timestamp,
                             'email_path': email_path_list,
                             'headers': headers,
                             'subject': subject,
                             #'body_plain': body_plain,
                             #'body_html': body_html,
                             'body_text': body_text,
                             'preheader_long': preheader_long,
                             'preheader_short': preheader_short})

    iframe = pd.read_csv(index_file)

    if source is not None:
        iframe = iframe[iframe['source'].isin(source)]
    iframe['email_recipient'] = iframe['query_data'].apply(lambda x: dict(eval(x))['email'].split('@')[0])
    iframe['to_address'] = iframe['query_data'].apply(lambda x: dict(eval(x))['email'])
    eframe = iframe.groupby('email_recipient').apply(get_emails).reset_index()
    del eframe['level_1']
    del iframe['email_recipient']

    df = None
    for unknown_dir in glob.glob(os.path.join(base_dir, 'raw', 'unknown*')):
        udir = os.path.basename(os.path.normpath(unknown_dir))

        if df is None:
            df = get_emails(None, udir)
            df['email_recipient'] = udir
        else:
            edf = get_emails(None, udir)
            edf['email_recipient'] = udir
            df = df.append(edf)

    eframe = eframe.append(df)

    eframe['body_text_sents'] = eframe['body_text'].apply(clean_email_lines)

    all_accs = eframe['to_address'].unique().tolist()

    for acc in all_accs:
        acc_dic[acc] = {}
        eframe_acc = eframe[eframe['to_address'] == acc]
        obs = len(eframe_acc)
        sents = eframe_acc['body_text_sents'].tolist()
        lines = [sent.split('\n') for sent in sents if sent != None]
        flat_list = [item.strip() for sublist in lines for item in sublist]
        acc_dic[acc]['lines'] = Counter(flat_list)
        acc_dic[acc]['obs'] = obs

    eframe['body_text_sents_filtered'] = eframe.apply(filter_email_lines, axis=1)

    eframe['uid'] = eframe.apply(lambda x: uuid.uuid5(uuid.NAMESPACE_DNS,
                                                      x['email_recipient'] + x['email_sender'] + x['email_timestamp']).hex,
                                axis=1)

    return eframe.merge(iframe, on='to_address', how='inner')


def remove_empty_lines(text):
    lines = text.split('\n')
    lines = map(lambda x: x.replace('\u200c', ' ').strip(), lines)
    lines = list(filter(lambda x: x.lower().strip() != '', lines))

    return lines


def get_email_lines(email_html):
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.images_to_alt = True
    h.skip_internal_links = True
    h.ignore_tables = True
    h.body_width = 0

    email_html_text = h.handle(str(BeautifulSoup(email_html, 'lxml')))

    return remove_empty_lines(email_html_text)


def clean_email_lines(row):
    """keeps 'proper' sentences, removes duplicates"""
    if row is None or len(row) == 0:
        return ''

    sents = row.split('\n')

    clean_sents = []
    for sent in sents:
        prox = sent.lower().strip()
        last = prox[-1]
        if last in TERMINALS and 'unsubscribe' not in prox:
            # looking specifically for the 'unsubscribe' term
            clean_sents.append(sent)

    # concatenate, remove duplicate lines (preheader etc.)
    clean_sents = unique(clean_sents)

    return '\n'.join(clean_sents)


def filter_email_lines(row, boundary=0.9):
    """
    boundary = cutoff for removing lines.
    example: if boundary = 0.9, then all lines which are included
    in more than 90% of emails for a corresponding inbox are removed.

    note that these shares can reach values > 1.0 because a line
    could for instance be included 4 times across 3 emails.
    """
    to_check = acc_dic[row['to_address']]

    final_lines = []
    current_lines = row['body_text_sents'].split('\n')
    if not current_lines:
        return row
    if to_check['obs'] > 3:
        # removing accounts with very few observations
        for l in current_lines:
            if l in dict(to_check['lines']).keys():
                # compute share
                rel = to_check['lines'][l] / to_check['obs']
                if rel < boundary:
                    final_lines.append(l)
        return '\n'.join(final_lines)
    else:
        return '\n'.join(current_lines)


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]