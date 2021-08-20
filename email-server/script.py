#!/usr/bin/python3
import sys
import mailparser
import os
import time
import logging
import random
from utils import *
import email

# Base server directory.
# This needs to be an absolute path, and also the location of this script.
BASE_DIR = '/mnt/volume_nyc3_01/emails/'

# Raw emails go here.
RAW_DIR = os.path.join(BASE_DIR, 'raw/')

# HTML email parts go here.
HTML_DIR = os.path.join(BASE_DIR, 'html/')

# Plain email parts go here.
PLAIN_DIR = os.path.join(BASE_DIR, 'plain/')

# Non plan and text email parts go here.
OTHER_DIR = os.path.join(BASE_DIR, 'other/')

# Attachments go here.
ATT_DIR = os.path.join(BASE_DIR, 'attachments/')

# Logs go here.
LOGS_DIR = os.path.join(BASE_DIR, 'logs/')

# Error emails go here.
ERRORS_DIR = os.path.join(BASE_DIR, 'errors/')

# HTML with resources go here.
HTT_DIR = os.path.join(BASE_DIR, 'httrack/')

# Confirmation link output will go here.
CONFIRMATION_DIR = os.path.join(BASE_DIR, 'confirmation/')

# Email metadata will go here.
METADATA_DIR = os.path.join(BASE_DIR, 'metadata/')

# File that will store the cookies from opening the emails.
# Do not change the name of this file. This is the name HTTrack expects.
COOKIE_FILE = 'cookies.txt'

# Label for the email files.
EMAIL_PREFIX = 'email'

# Label for the modified HTML files.
MODIFIED_HTML_LABEL = 'mht'

# Prefix for the screenshot of the confirmation page.
CONFIRMATION_PREFIX = 'confirmation'

# Keywords for email confirmation.
EMAIL_CONFIRMATION_KEYWORDS = ['confirm', 'verify', 'validate', 'activate']

# Additional keywords in link text for email confirmation.
EMAIL_CONFIRMATION_LINK_KEYWORDS = ['subscribe', 'activate']

# Blacklisted keywords in link text for email confirmation.
EMAIL_CONFIRMATION_LINK_BLACKLIST = [
    'unsubscribe', 'view', 'cancel', 'deactivate']

# Blacklisted keywords in link URLs for email confirmation.
EMAIL_CONFIRMATION_LINK_URL_BLACKLIST = ['unsubscribe', 'deactivate']

# Blacklisted keywords in email subject for email confirmation.
EMAIL_CONFIRMATION_SUBJECT_BLACKLIST = ['confirmed', 'subscribed', 'activated']

# Filters used by HTTrack.
# This file is used with HTTrack.
IGNORE_FILE = 'ignore.txt'
IGNORE_FILE_PATH = os.path.join(BASE_DIR, IGNORE_FILE)

# Ignore emails to the test account?
DISABLE_TEST_ACCOUNT = False

def dump_email(raw_mail):
    try:
        # Create the base directory.
        create_dir(BASE_DIR)

        # Create the directory to store logs for this email.
        create_dir(LOGS_DIR)

        # Create the directory to store emails that cannot be parsed.
        create_dir(ERRORS_DIR)

        # Create the directory to store the response from confirmation links.
        create_dir(CONFIRMATION_DIR)

        # Create the directory to store the email metadata.
        create_dir(METADATA_DIR)
    except:
        return

    # Register this email's timestamp.
    # Add a random number to distinguish emails arriving simultaneously.
    tstamp = str(round(time.time(), 2) + random.randint(0, 10000))

    # Should we download the assets?
    download_assets = False

    # Set up the logger.
    logger = logging.getLogger(__name__)
    lf_handler = logging.FileHandler(os.path.join(LOGS_DIR, tstamp + '.log'))
    lf_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    lf_handler.setFormatter(lf_format)
    logger.addHandler(lf_handler)
    logger.setLevel(logging.INFO)

    # HTTrack needs this file to ignore certain kinds of requests.
    try:
        # Create file if it does not exist.
        if not os.path.isfile(IGNORE_FILE_PATH):
            with open(IGNORE_FILE_PATH, 'w') as f:
                f.write('''-mime:application/javascript\n''')

        logger.info('Successfully created ignore file for HTTrack')
    except:
        logger.exception('Error creating ignore file for HTTrack')

    # Parse the email.
    try:
        mail = mailparser.parse_from_bytes(raw_mail)
        logger.info('Successfully parsed email using mailparser')
    except:
        try:
            logger.exception('Error while parsing email using mailparser')
            with open(os.path.join(ERRORS_DIR, tstamp + '.eml'), 'wb') as f:
                f.write(raw_mail)
            logger.info('Dumped raw email to the errors/ dir')
        except:
            logger.exception(
                'Error while dumping the raw email to the errors/ dir')
        return

    # mail.from_ is a list, see the official API for the behavior.
    if len(mail.from_) > 1:
        logger.warning('More than one sender, picking #1 match: %s' %
                       str(mail.from_))

    # Extract the sender's email address.
    try:
        sender_addr = sender(mail.from_)
        if sender_addr is None:
            sender_addr = 'unknown' + tstamp
    except:
        logger.exception('Error while retrieving the sender, setting to unknown')
        sender_addr = 'unknown' + tstamp

    logger.info('Email sender: %s' % sender_addr)

    # mail.to is a list, see the official API for the behavior.
    if len(mail.to) > 1:
        logger.warning('More than one recipient, picking #1 match: %s' %
                       str(mail.to))

    # Extract the recipient's email address.
    recipient_addr = None
    try:
        recipient_addr = recipient(mail.to)
        if recipient_addr is None:
            recipient_addr = recipient(mail.cc)
            if recipient_addr is None:
                recipient_addr = recipient(mail.bcc)
                if recipient_addr is None:
                    recipient_addr = 'unknown' + tstamp
    except:
        logger.exception('Error while retrieving the recipient')
        recipient_addr = 'unknown' + tstamp

    logger.info('Email recipient: %s' % recipient_addr)
    recipient_addr = recipient_addr.split('@')[0]

    if recipient_addr == 'test' and DISABLE_TEST_ACCOUNT:
        return

    # Create the directory where email metadata will be stored. These directories
    # are indexed by recipient email addresses.
    metadata_recipient_dir = os.path.join(METADATA_DIR, recipient_addr)
    metadata_sender_dir = os.path.join(metadata_recipient_dir, sender_addr)
    metadata_mail_dir = os.path.join(metadata_sender_dir, tstamp)
    try:
        create_dir(metadata_recipient_dir)
        create_dir(metadata_sender_dir)
        create_dir(metadata_mail_dir)
    except:
        logger.exception('Error while creating the metadata dir')

    try:
        dump_file = dump_file = os.path.join(metadata_mail_dir, 'headers.txt')
        headers = email.message_from_bytes(raw_mail)._headers
        with open(dump_file, 'wb') as f:
            f.write(str(headers).encode('utf-8'))
        logger.info('Dumped metadata to %s' % dump_file)
    except:
        logger.exception('Error while dumping the metadata')


    # Create the directory where the raw email will be stored. These directories
    # are indexed by recipient email addresses.
    raw_recipient_dir = os.path.join(RAW_DIR, recipient_addr)
    raw_sender_dir = os.path.join(raw_recipient_dir, sender_addr)
    raw_mail_dir = os.path.join(raw_sender_dir, tstamp)
    try:
        create_dir(raw_recipient_dir)
        create_dir(raw_sender_dir)
        create_dir(raw_mail_dir)
        logger.info('Created raw email dir %s' % raw_mail_dir)
    except:
        logger.exception('Error while creating the raw email dir')

    try:
        dump_file = os.path.join(raw_mail_dir, EMAIL_PREFIX + '.eml')
        with open(dump_file, 'wb') as f:
            f.write(raw_mail)
        logger.info('Dumped raw email to %s' % dump_file)
    except:
        logger.exception('Error while dumping the raw email')


    # Dump the text/HTML email bodies.
    logger.info('Number of HTML parts %d' % len(mail.text_html))
    if len(mail.text_html) != 0:
        # Create the directory where all text/html email bodies will be stored.
        # These directories are indexed by recipient email addresses.
        html_mail_recipient_dir = os.path.join(HTML_DIR, recipient_addr)
        html_mail_sender_dir = os.path.join(html_mail_recipient_dir, sender_addr)
        html_mail_dir = os.path.join(html_mail_sender_dir, tstamp)
        try:
            create_dir(html_mail_recipient_dir)
            create_dir(html_mail_sender_dir)
            create_dir(html_mail_dir)
            logger.info('Created html email dir %s' % html_mail_dir)
        except:
            logger.exception('Error while creating the html email dir')

        try:
            # Dump the file and its charset.
            dump_file = os.path.join(html_mail_dir, EMAIL_PREFIX + '.html')
            dump_file_charset = os.path.join(html_mail_dir, 'charset.txt')

            # Even if the mail has multiple HTML parts, we only pick the first.
            html = mail.text_html[0]
            content = html['payload']
            charset = html['charset']
            logger.info('HTML charset is %s' % charset)

            with open(dump_file, 'wb') as f:
                f.write(content.encode(charset, errors='ignore'))

            with open(dump_file_charset, 'wb') as f:
                f.write(charset.encode('utf-8', errors='ignore'))

            logger.info('Dumped html email to %s' % dump_file)
        except:
            logger.exception('Error while dumping the html email')

        # Dump a version of the file where anchor tag hrefs are renamed as
        # well as script tags are removed.
        # We do this to prevent HTTrack from following and crawling these
        # links.
        try:
            dump_file = os.path.join(html_mail_dir, '_'.join(
                [EMAIL_PREFIX, MODIFIED_HTML_LABEL + '.html']))

            html = mail.text_html[0]
            content = html['payload']
            charset = html['charset']

            modified_content = transform_html(content.encode(charset, errors='ignore'), charset)
            logger.info('Successfully modified html email for HTTrack')

            logger.info('HTTrack modified HTML email charset is %s' % charset)
            with open(dump_file, 'w', encoding=charset, errors='surrogateescape') as f:
                f.write(modified_content.decode(charset))
            logger.info('Dumped modified html email to %s' % dump_file)

            download_assets = True
        except:
            logger.exception('Error while dumping the modified html email')
    else:
        logger.info('No html email to dump')

    # Dump all text/plain email bodies.
    logger.info('Number of Plain parts %d' % len(mail.text_plain))
    if len(mail.text_plain) != 0:
        # Create the directory where the text/plain email bodies will be stored.
        # These directories are indexed by recipient email addresses.
        plain_mail_recipient_dir = os.path.join(PLAIN_DIR, recipient_addr)
        plain_mail_sender_dir = os.path.join(plain_mail_recipient_dir, sender_addr)
        plain_mail_dir = os.path.join(plain_mail_sender_dir, tstamp)
        try:
            create_dir(plain_mail_recipient_dir)
            create_dir(plain_mail_sender_dir)
            create_dir(plain_mail_dir)
            logger.info('Created plain email dir %s' % plain_mail_dir)
        except:
            logger.exception('Error while creating the plain email dir')

        try:
            dump_file = os.path.join(plain_mail_dir, EMAIL_PREFIX + '.txt')
            dump_file_charset = os.path.join(plain_mail_dir, 'charset.txt')

            # Even if the mail has multiple Plain parts, we only pick the first.
            plain = mail.text_plain[0]
            content = plain['payload']
            charset = plain['charset']
            logger.info('Plain charset is %s' % charset)

            with open(dump_file, 'wb') as f:
                f.write(content.encode(charset, errors='ignore'))

            with open(dump_file_charset, 'wb') as f:
                f.write(charset.encode('utf-8', errors='ignore'))

            logger.info('Dumped plain email to %s' % dump_file)
        except:
            logger.exception('Error while dumping the plain email')
    else:
        logger.info('No plain email to dump')

    # Dump all non text/html and text/plain content.
    logger.info('Number of Other parts %d' % len(mail.other_content))
    if len(mail.other_content) != 0:
        # Create the directory where this content will be stored.
        # These directories are indexed by recipient email addresses.
        other_content_recipient_dir = os.path.join(OTHER_DIR, recipient_addr)
        other_content_sender_dir = os.path.join(other_content_recipient_dir, sender_addr)
        other_content_dir = os.path.join(other_content_sender_dir, tstamp)
        try:
            create_dir(other_content_recipient_dir)
            create_dir(other_content_sender_dir)
            create_dir(other_content_dir)
            logger.info('Created other content dir %s' % other_content_dir)
        except:
            logger.exception('Error while creating the other content dir')

        for i, other in enumerate(mail.other_content):
            dump_file = os.path.join(
                other_content_dir, '_'.join(['other', str(i)]))
            dump_file_info = os.path.join(
                other_content_dir, '_'.join(['other', str(i), 'info.txt']))

            try:
                with open(dump_file, 'wb') as fp:
                    fp.write(other['payload'].encode(other['charset']))
                    logger.info('Dumped other content to %s' % dump_file)

                with open(dump_file_info, 'wb') as fp:
                    other.pop('payload', None)
                    fp.write(str(other).encode('utf-8'))
                    logger.info('Dumped other content information to %s' % dump_file_info)
            except:
                logger.exception('Error dumping other content')
    else:
        logger.info('No other content to dump')

    # Dump attachments.
    logger.info('Number of attachments %d' % len(mail.attachments))
    if len(mail.attachments) != 0:
        # Create the directory where the attachments will be stored. These
        # directories are indexed by recipient email addresses.
        attachment_recipient_dir = os.path.join(ATT_DIR, recipient_addr)
        attachment_sender_dir = os.path.join(attachment_recipient_dir, sender_addr)
        attachment_dir = os.path.join(attachment_sender_dir, tstamp)
        try:
            create_dir(attachment_recipient_dir)
            create_dir(attachment_sender_dir)
            create_dir(attachment_dir)
            logger.info('Created attachment dir %s' % attachment_dir)
        except:
            logger.exception('Error while creating the attachment dir')

        logger.info('Number of attachments: %d' % len(mail.attachments))

        for i, attachment in enumerate(mail.attachments):
            dump_file = os.path.join(
                attachment_dir, '_'.join(['attachment', str(i)]))

            dump_file_info = os.path.join(
                attachment_dir, '_'.join(['attachment', str(i), 'info.txt']))

            # Dump the json to file.
            try:
                with open(dump_file, 'wb') as fp:
                    if 'charset' in attachment.keys() and attachment['charset'] != None:
                        fp.write(attachment['payload'].encode(attachment['charset']))
                    else:
                        fp.write(attachment['payload'].encode('utf-8'))
                    logger.info('Dumped attachment to %s' % dump_file)

                with open(dump_file_info, 'wb') as fp:
                    attachment.pop('payload', None)
                    fp.write(str(attachment).encode('utf-8'))
                    logger.info('Dumped attachment information to %s' % dump_file_info)
            except:
                logger.exception('Error dumping attachment')
    else:
        logger.info('No attachments to dump')

    # Save the entire HTML file locally along with assets.
    # Take a screenshot of the saved file.
    # We use the modified HTML file for this step.
    if download_assets:
        # The file for which we need to save the resources.
        html_file_url = 'file://' + os.path.join(html_mail_dir, '_'.join(
            [EMAIL_PREFIX, MODIFIED_HTML_LABEL + '.html']))

        charset = mail.text_html[0]['charset']

        # Create the recipient's mailbox directory.
        httrack_recipient_dir = os.path.join(HTT_DIR, recipient_addr)
        httrack_sender_dir = os.path.join(httrack_recipient_dir, sender_addr)
        httrack_dir = os.path.join(httrack_sender_dir, tstamp)
        try:
            create_dir(httrack_recipient_dir)
            create_dir(httrack_sender_dir)
            # Normally we wouldn't have to create the following directory.
            # However, if we want the hts-io.txt, this is necessary.
            # Bug: https://forum.httrack.com/readmsg/2215/index.html
            create_dir(httrack_dir)
            logger.info('Created HTTrack recipient directory %s' % httrack_recipient_dir)
        except:
            logger.exception('Error while creating the HTTrack recipient dir')

        # The output directory is where the HTTrack output will be stored for
        # this HTML file. HTTrack will create this directory for us.
        logger.info('HTTrack file url: %s' % html_file_url)
        logger.info('HTTrack output directory: %s' % httrack_dir)

        # Run HTTrack from within the recipient directory. The recipient
        # directory will contain the cookies.txt file.
        try:
            # Sleep for random time before opening email so it doesn't seem
            # suspicious that the emails are opened immediately _all_ the time.
            # This matters only in the HTML version of emails.
            time.sleep(random.randint(0, 10))

            ec = httrack(html_file_url, httrack_dir,
                         IGNORE_FILE_PATH, httrack_recipient_dir)
            logger.info('HTTrack exit code: %d' % ec)
        except:
            logger.exception('Error executing HTTrack')

        # Move the cookies.txt file from the output directory to the recipient
        # directory.
        try:
            cookie_f = os.path.join(httrack_dir, COOKIE_FILE)

            if os.path.exists(cookie_f):
                ec = move_file(cookie_f, httrack_recipient_dir)
                logger.info('Move cookie file exit code: %d' % ec)
            else:
                logger.info('No cookie file to move')
        except:
            logger.exception('Error moving cookie file')

        # Delete the unnecessary HTTrack files.
        try:
            f = os.path.join(httrack_dir, 'index.html')

            if os.path.exists(f):
                ec = delete_file(f)
                logger.info('Delete index.html exit code: %d' % ec)
            else:
                logger.info('No index.html file to delete')
        except:
            logger.exception('Error deleting index.html')

        try:
            f = os.path.join(httrack_dir, 'backblue.gif')

            if os.path.exists(f):
                ec = delete_file(f)
                logger.info('Delete backblue.gif exit code: %d' % ec)
            else:
                logger.info('No backblue.gif file to delete')
        except:
            logger.exception('Error deleting backblue.gif')

        try:
            f = os.path.join(httrack_dir, 'fade.gif')

            if os.path.exists(f):
                ec = delete_file(f)
                logger.info('Delete fade.gif exit code: %d' % ec)
            else:
                logger.info('No fade.gif file to delete')
        except:
            logger.exception('Error deleting fade.gif')

        try:
            f = os.path.join(httrack_dir, 'hts-cache/')

            if os.path.isdir(f):
                ec = delete_folder(f)
                logger.info('Delete hts-cache/ exit code: %d' % ec)
            else:
                logger.info('No hts-cache/ to delete')
        except:
            logger.exception('Error deleting hts-cache/')

        # Take a screenshot of the downloaded HTTrack file with assets.
        try:
            dump_file_dir = os.path.join(httrack_dir)
            dump_file_name = EMAIL_PREFIX + '.png'

            # httrack_file = os.path.join(httrack_output_dir, 'file', html_mail_dir[1:], '_'.join(
            #    [sender_addr, MODIFIED_HTML_LABEL, tstamp + '.html']))

            # The slice is necessary to make os.path.join work.
            # Rather than constructing the file like above, we list the directory's
            # content, which will (should) have only one file. We then pick hat file
            # as the one to reverse transform.
            # HTTrack has a bug in which if the file path is too long, the file name
            # is truncated.
            httrack_file_dir = os.path.join(httrack_dir, 'file')
            current_dir = httrack_file_dir
            httrack_file = ''

            while True:
                dir_content = os.listdir(current_dir)[0]
                current_file = os.path.join(current_dir, dir_content)
                if os.path.isfile(current_file):
                    httrack_file = current_file
                    httrack_file_dir = current_dir
                    break
                else:
                    current_dir = current_file

            if os.path.isfile(httrack_file):
                with open(httrack_file, 'r', encoding=charset, errors='surrogateescape') as htfr:
                    # Modify the href tags back before taking a screenshot.
                    content = reverse_transform_html(htfr.read().encode(charset, errors='ignore'), charset)

                    rev_mod_file = os.path.join(httrack_file_dir, EMAIL_PREFIX + '.html')
                    with open(rev_mod_file, 'w', encoding=charset, errors='surrogateescape') as f:
                        f.write(content.decode(charset))

                    logger.info('Removed all modhrefs in %s' % rev_mod_file)

                    if os.path.isfile(rev_mod_file):
                        logger.info('Taking screenshot of %s', rev_mod_file)
                        ecode = screenshot(rev_mod_file, dump_file_dir, dump_file_name)
                        if ecode != 0:
                            logger.error('Error dumping screenshot')
                        else:
                            logger.info('Dumped screenshot to %s' % os.path.join(dump_file_dir, dump_file_name))
                    else:
                        logger.info('No modref edited file to take screenshot')
            else:
                logger.info('No HTTrack output to take screenshot')
        except:
            logger.exception('Error dumping screenshot')

    else:
        logger.info('No assets to download')

    # Discover confirmation link.
    if len(mail.text_html) == 1 or len(mail.text_plain) == 1:
        confirmation_link = None
        links = []
        body = ''
        subject = ''

        if mail.subject is not None:
            subject = mail.subject.lower()

        logger.info('Email subject: %s' % subject)

        if len(mail.text_html) == 1:
            try:
                content = mail.text_html[0]['payload']
                body = get_body_from_html(content)
                links = get_links_from_html(content)
                logger.info(
                    '# Links extracted from the HTML email: %d' % len(links))
            except:
                logger.exception(
                    'Error while retrieving links or reading the HTML email text body')

        elif len(mail.text_plain) == 1:
            try:
                body = mail.text_plain[0]['payload']
                links = get_links_from_text(body)
                logger.info(
                    '# Links extracted from the Plain email: %d' % len(links))
            except:
                logger.exception(
                    'Error while retrieving links or reading the Plain email text body')

        if len(links) != 0:
            try:
                body_matches = matches(
                    body.lower(), EMAIL_CONFIRMATION_KEYWORDS)
                logger.info('Body matches? %s' % str(body_matches))

                subject_matches = matches(subject.lower(), EMAIL_CONFIRMATION_KEYWORDS) and not matches(
                    subject, EMAIL_CONFIRMATION_SUBJECT_BLACKLIST)
                logger.info('Subject matches? %s' % str(subject_matches))

                if body_matches or subject_matches:
                    if len(mail.text_html) == 1:
                        for link in links:
                            link_href, link_text = link

                            link_text = link_text.lower()

                            if matches(link_text, EMAIL_CONFIRMATION_LINK_BLACKLIST) or matches_fuzzy(link_href, EMAIL_CONFIRMATION_LINK_URL_BLACKLIST):
                                continue

                            if matches(link_text, EMAIL_CONFIRMATION_KEYWORDS):
                                confirmation_link = link_href
                                break

                        if confirmation_link is None:
                            for link in links:
                                link_href, link_text = link

                                link_text = link_text.lower()

                                if matches(link_text, EMAIL_CONFIRMATION_LINK_BLACKLIST) or matches_fuzzy(link_href, EMAIL_CONFIRMATION_LINK_URL_BLACKLIST):
                                    continue

                                if matches(link_text, EMAIL_CONFIRMATION_LINK_KEYWORDS):
                                    confirmation_link = link_href
                                    break

                    elif len(mail.text_plain) == 1:
                        for link in links:
                            if matches(link, EMAIL_CONFIRMATION_LINK_BLACKLIST) or matches(link, EMAIL_CONFIRMATION_LINK_URL_BLACKLIST):
                                continue

                            if matches(link, EMAIL_CONFIRMATION_LINK_KEYWORDS):
                                confirmation_link = link_href
                                break
                else:
                    logger.info(
                        'Exiting because neither body nor subject match confirmation keywords')
            except:
                logger.exception(
                    'Error retrieving the confirmation link from the list of links')
        else:
            logger.info(
                'Exiting because no links found to analyze as confirmation links')

        logger.info('Confirmation link: %s' % confirmation_link)

        # Click confirmation link here.
        if confirmation_link is not None:
            try:
                confirmation_recipient_dir = os.path.join(
                    CONFIRMATION_DIR, recipient_addr)
                confirmation_sender_dir = os.path.join(
                    confirmation_recipient_dir, sender_addr)
                confirmation_mail_dir = os.path.join(confirmation_sender_dir, tstamp)

                try:
                    create_dir(confirmation_recipient_dir)
                    create_dir(confirmation_sender_dir)
                    create_dir(confirmation_mail_dir)
                    logger.info('Created confirmation dir %s' %
                                confirmation_mail_dir)
                except:
                    logger.exception('Error creating confirmation directory')

                dump_file_name = CONFIRMATION_PREFIX + '.png'

                ecode = screenshot(confirmation_link, confirmation_mail_dir, dump_file_name)

                if ecode != 0:
                    logger.error('Error clicking confirmation link')
                else:
                    logger.info('Dumped confirmation link output to %s' % os.path.join(confirmation_mail_dir, dump_file_name))

            except:
                logger.exception('Error dumping confirmation link output')
    else:
        logger.info(
            'Ignoring confirmation link analysis because of multiple HTML/plain parts')

    return


if __name__ == '__main__':
    # Read in the raw email from standard input.
    # Read in as raw bytes, not string.
    raw_mail = sys.stdin.buffer.read()

    # Process email.
    dump_email(raw_mail)
