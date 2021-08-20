import os
import subprocess
import re
from bs4 import BeautifulSoup
from bs4.element import Comment
from urlextract import URLExtract
import random
import time

# Server domain
# Replace this with your own domain name
EMAIL_DOMAIN = ['gmail.com']

# User agent for email client.
EMAIL_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Thunderbird/68.1.0 Lightning/68.1.0'


def sender(from_):
    # This function returns the sender's address from the 'from_' output of
    # mailparser. This output of mailparser is a list of lists. This function
    # assumes the list has only one list item, and searches its items for a
    # string resembling an email address. If the length of 'from_' is greater
    # than two, this will return the first match.
    for from_list_item in from_:
        for from_item in from_list_item:
            # Regular expression that matches emails.
            if re.match(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)', from_item):
                return from_item

    return None


def recipient(to):
    # This function returns the recipient's address from the 'to' output of
    # mailparser. This output of mailparser is a list of lists. This function
    # assumes the list has only one list item, and searches its items for a
    # string resembling an email address. If the length of 'from_' is greater
    # than two, this will return the first match.
    for to_list_item in to:
        for to_item in to_list_item:
            # Regular expression that matches emails.
            if re.match(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)', to_item):
                if to_item.split('@')[1] in EMAIL_DOMAIN:
                    return to_item

    return None


def create_dir(directory):
    # Creates the given directory if it does not already exist.
    if not os.path.isdir(directory):
        os.makedirs(directory)


def move_file(file_path, destination):
    # Move file_path to the destination directory.
    if not os.path.isdir(destination):
        raise Exception('The destination is not a directory')

    if not os.path.isfile(file_path):
        raise Exception('File not found')

    mv_ecode = subprocess.call(['mv', file_path, destination])
    return mv_ecode


def delete_file(file_path):
    # Delete file_path.
    if os.path.isfile(file_path):
        del_ecode = subprocess.call(['rm', file_path])
        return del_ecode
    else:
        raise Exception('File not found')


def delete_folder(folder_path):
    # Delete folder_path.
    if os.path.isdir(folder_path):
        del_ecode = subprocess.call(['rm', '-r', folder_path])
        return del_ecode
    else:
        raise Exception('Directory not found')


def httrack(url, output_dir, ignore_file_path, cw_directory):
    # Execute HTTrack for the provided URL from the directory specified in
    # cw_directory.
    # HTTrack expects a cookie.txt file in this directory.
    # The output is stored in the directory specified by output_dir.
    httrack_ecode = subprocess.call(['httrack', url, '-O', '\"' + output_dir + '\"',
                                     '-s0', '-n', '-T10', '-Q', '-q',
                                     '-b1', '-F', EMAIL_USER_AGENT,
                                     '-%F', '<!-- ignore -->', '-u2', '-%e50', '-%S', ignore_file_path,
                                     '-%P0', '-%H'], cwd=cw_directory)

    # Return the exit code of the process call.
    return httrack_ecode


def matches(match_string, match_list):
    # Does match_string contain any of the expressions in match_list?
    match_string_list = match_string.split(' ')
    for ms in match_string_list:
        ms = ms.strip()
        for ml in match_list:
            if ms == ml:
                return True

    return False


def matches_fuzzy(match_string, match_list):
    # Does match_string contain any of the expressions in match_list?
    for ml in match_list:
        if ml in match_string:
            return True

    return False


def screenshot(url, output_directory, screenshot_file_name):
    # Dumps a screenshot of the HTML file file_path to the directory.
    chrome_profile_dir = '/tmp/' + str(time.time() + random.randint(0, 1000))

    chrome_ecode = subprocess.call(['google-chrome', '--disable-gpu', '--headless', '--user-data-dir=' + chrome_profile_dir, '--hide-scrollbars',
                                    '--screenshot', '--window-size=1680,10000', url], cwd=output_directory)

    rename_ecode = subprocess.call(
        ['mv', 'screenshot.png', screenshot_file_name], cwd=output_directory)
    trim_ecode = subprocess.call(
        ['convert', '-trim', screenshot_file_name, screenshot_file_name], cwd=output_directory)
    border_ecode = subprocess.call(['convert', '-bordercolor', 'White', '-border',
                                    '50x50', screenshot_file_name, screenshot_file_name], cwd=output_directory)

    delete_code = delete_folder(chrome_profile_dir)

    return chrome_ecode | rename_ecode | trim_ecode | border_ecode | delete_code


def transform_html(content, encoding):
    soup = BeautifulSoup(content, 'html5lib', from_encoding=encoding)

    # Change the href tag name so HTTrack avoids it.
    for anchor in soup.findAll('a'):
        if 'href' not in anchor.attrs.keys():
            continue
        else:
            anchor.attrs['modhref'] = anchor.attrs['href']
            anchor.attrs.pop('href', None)

    # Strip all script tags because these are not supported by email clients.
    for script in soup.findAll('script'):
        script.decompose()

    return soup.html.encode(encoding, errors='ignore')


def reverse_transform_html(content, encoding):
    soup = BeautifulSoup(content, 'html5lib', from_encoding=encoding)

    # Change the href tag name back.
    for anchor in soup.findAll('a'):
        if 'modhref' not in anchor.attrs.keys():
            continue
        else:
            anchor.attrs['href'] = anchor.attrs['modhref']
            anchor.attrs.pop('modhref', None)

    return soup.html.encode(encoding, errors='ignore')


def get_links_from_html(content):
    # Retrieve anchors links from the HTML content.
    soup = BeautifulSoup(content, 'html.parser')
    links = []

    for anchor in soup.findAll('a'):
        if 'href' in anchor.attrs.keys():
            if anchor.text is None:
                links.append((anchor.attrs['href'], ''))
            else:
                links.append((anchor.attrs['href'], anchor.text))

    return links


def get_links_from_text(content):
    # Retrieve links from the plain text content.
    extractor = URLExtract()
    return extractor.find_urls(content)


def get_body_from_html(content):
    # Filter visible tags.
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False

        if isinstance(element, Comment):
            return False

        return True

    # Retrieve the textual body of the HTML content.
    soup = BeautifulSoup(content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u' '.join(t.strip() for t in visible_texts)
