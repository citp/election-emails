from selenium.webdriver.common.keys import Keys
from selenium import webdriver as wd
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common.exceptions import TimeoutException
from urllib import urlencode
from urllib2 import Request, urlopen, URLError
from urlparse import urljoin
import random
import time
import timeit
import datetime
import re

from ..MPLogger import loggingclient
from ..utilities import domain_utils
from utils.webdriver_extensions import wait_until_loaded
from browser_commands import get_website, bot_mitigation, save_screenshot, dump_page_source

# Link text ranking
_TYPE_TEXT = 'text'
_TYPE_HREF = 'href'
_FLAG_NONE = 0
_FLAG_STAY_ON_PAGE = 1
_FLAG_IN_NEW_URL_ONLY = 2
_LINK_TEXT_RANK = [
    # probably newsletters
    (_TYPE_TEXT, 'newsletter', 10, _FLAG_NONE),
    (_TYPE_TEXT, 'stay informed',   9, _FLAG_NONE),
    (_TYPE_TEXT, 'subscribe',   9, _FLAG_NONE),
    (_TYPE_TEXT, 'inbox',       8, _FLAG_NONE),
    (_TYPE_TEXT, 'keep in touch',  6, _FLAG_NONE),

    # sign-up links (for something?)
    (_TYPE_TEXT, 'signup',     5, _FLAG_NONE),
    (_TYPE_TEXT, 'sign up',    5, _FLAG_NONE),
    (_TYPE_TEXT, 'sign me up', 5, _FLAG_NONE),
    (_TYPE_TEXT, 'join',       5, _FLAG_NONE),
    (_TYPE_TEXT, 'take action',       5, _FLAG_NONE),
    (_TYPE_TEXT, 'get involved',       5, _FLAG_NONE),
    (_TYPE_TEXT, 'get engaged',       5, _FLAG_NONE),
    (_TYPE_TEXT, 'volunteer',   4, _FLAG_NONE),
    (_TYPE_TEXT, 'register',   4, _FLAG_NONE),
    (_TYPE_TEXT, 'email',       4, _FLAG_NONE),
    (_TYPE_TEXT, 'create',     4, _FLAG_NONE),
    (_TYPE_TEXT, 'petition',       4, _FLAG_NONE),
    (_TYPE_TEXT, 'connect',       4, _FLAG_NONE),
    (_TYPE_TEXT, 'contact',       4, _FLAG_NONE),

    # articles (sometimes sign-up links are on these pages...)
    (_TYPE_HREF, '/article', 3, _FLAG_NONE),
    (_TYPE_HREF, '/home', 3, _FLAG_STAY_ON_PAGE | _FLAG_IN_NEW_URL_ONLY),
    (_TYPE_HREF, 'news/',    3, _FLAG_IN_NEW_URL_ONLY),
    (_TYPE_HREF, '/' + str(datetime.datetime.now().year), 2, _FLAG_NONE),

    # country selectors (for country-selection landing pages)
    (_TYPE_HREF, '/us/',  1, _FLAG_STAY_ON_PAGE | _FLAG_IN_NEW_URL_ONLY),
    (_TYPE_HREF, '=us&',  1, _FLAG_STAY_ON_PAGE | _FLAG_IN_NEW_URL_ONLY),
    (_TYPE_HREF, 'en-us', 1, _FLAG_STAY_ON_PAGE | _FLAG_IN_NEW_URL_ONLY),
]
_LINK_RANK_SKIP = 6  # minimum rank to select immediately (skipping the rest of the links)
_LINK_MATCH_TIMEOUT = 20  # maximum time to match links, in seconds
_LINK_TEXT_BLACKLIST = ['unsubscribe', 'mobile', 'phone']

# Keywords
_KEYWORDS_EMAIL  = ['email', 'e-mail', 'subscribe', 'newsletter']
_KEYWORDS_EMAIL_BLACKLIST = ['contact us', 'contact me', 'message', 'subject', 'your message', 'comment']
_KEYWORDS_SUBMIT = ['submit', 'sign up', 'sign-up', 'signup', 'sign me up', 'subscribe', 'register', 'join', 'i\'m in']
_KEYWORDS_SELECT = ['yes', 'ny', 'new york', 'united states', 'usa', '1990', 'english']
_DOMAIN_EXCEPTIONS = ['actionnetwork.org', 'mailchi.mp', 'myngp.com', 'bsd.net', 'webaction.org', 'ngpvan.com', 'actnow.io', 'myngp.com', 'list-manage.com', 'wiredforchange.com', 'ipetitions.com', 'eepurl.com']

# Other constants
_PAGE_LOAD_TIME = 7  # time to wait for pages to load (in seconds)
_FORM_SUBMIT_SLEEP = 2  # time to wait after submitting a form (in seconds)
_FORM_CONTAINER_SEARCH_LIMIT = 4  # number of parents of input fields to search

# The maximum number of popups we will dismiss.
_MAX_POPUP_DISMISS = 2

# User information to supply to forms
def _get_user_info(email):
    """Returns a dictionary of user information."""
    return {
        'email': email,
        'first_name': 'Bob',
        'last_name': 'Smith',
        'full_name': 'Bob Smith',
        'user': 'bobsmith' + str(random.randrange(0,1000)),
        'password': 'p4S$w0rd123',
        'tel': '212' + '555' + '01' + str(random.randrange(0,10)) + str(random.randrange(0,10)),
        'company': 'Smith & Co.',
        'title': 'Mr.',
        'zip': '12345',
        'street1': '101 Main St.',
        'street2': 'Apt. 101',
        'city': 'Schenectady',
        'state': 'New York',
    }

def fill_forms(url, user_data, num_links, page_timeout, debug, visit_id,
               webdriver, proxy_queue, browser_params, manager_params, extension_socket):
    """Finds a newsletter form on the page. If not found, visits <num_links>
    internal links and scans those pages for a form. Submits the form if found.
    """
    # load the site
    webdriver.set_page_load_timeout(page_timeout)
    get_website(url, 0, visit_id, webdriver, proxy_queue, browser_params, extension_socket)

    # connect to the logger
    logger = loggingclient(*manager_params['logger_address'])

    # sleep before proceeding, let popups (if any, appear)
    time.sleep(_PAGE_LOAD_TIME)

    # take a screenshot, and try to find a newsletter form on the landing page
    if debug:
        save_screenshot(str(visit_id) + '_landing_page', webdriver, browser_params, manager_params)

    if _find_and_fill_form(webdriver, user_data, visit_id, debug, browser_params, manager_params, logger):
        if debug: logger.debug('Done searching and submitting forms, exiting')
        return

    if debug: logger.debug('Could not find and submit a newsletter form on the landing page; scanning more pages..')

    # otherwise, scan more pages
    main_handle = webdriver.current_window_handle
    visited_links = set()
    for i in xrange(num_links):
        # get all links on the page
        links = webdriver.find_elements_by_tag_name('a')
        random.shuffle(links)

        current_url = webdriver.current_url
        current_ps1 = domain_utils.get_ps_plus_1(current_url)

        # find links to click
        match_links = []
        start_time = timeit.default_timer()
        for link in links:
            try:
                if not link.is_displayed():
                    continue

                # check if link is valid and not already visited
                href = link.get_attribute('href')
                if href is None or href in visited_links:
                    continue

                # check if this is an internal link
                if not _is_internal_link(href, current_url, current_ps1) and not _whitelisted_links(href, current_url):
                    continue

                link_text = link.get_attribute('text').lower().strip()

                # skip links with blacklisted text
                blacklisted = False
                for bl_text in _LINK_TEXT_BLACKLIST:
                    if bl_text in link_text:
                        blacklisted = True
                        break
                if blacklisted:
                    continue

                # should we click this link?
                link_rank = 0
                for type, s, rank, flags in _LINK_TEXT_RANK:
                    if (type == _TYPE_TEXT and s in link_text) or (type == _TYPE_HREF and s in href):
                        if flags & _FLAG_IN_NEW_URL_ONLY:
                            # don't use this link if the current page URL already matches too
                            if type == _TYPE_HREF and s in current_url:
                                continue

                        # link matches!
                        link_rank = rank
                        match_links.append((link, rank, link_text, href, flags))
                        break
                if link_rank >= _LINK_RANK_SKIP:  # good enough, stop looking
                    break
            except:
                logger.error('Error while looping through links...')

            # quit if too much time passed (for some reason, this is really slow...)
            if match_links and timeit.default_timer() - start_time > _LINK_MATCH_TIMEOUT:
                logger.warning('Too much time passed, quiting')
                break

        # find the best link to click
        if not match_links:
            if debug: logger.debug('No more links to click')
            break  # no more links to click
        match_links.sort(key=lambda l: l[1])
        next_link = match_links[-1]
        visited_links.add(next_link[3])

        # click the link
        try:
            # load the page
            logger.info("Clicking on link '%s' - %s" % (next_link[2], next_link[3]))
            next_link[0].click()
            wait_until_loaded(webdriver, _PAGE_LOAD_TIME)
            if browser_params['bot_mitigation']:
                bot_mitigation(webdriver)

            # find newsletter form
            if _find_and_fill_form(webdriver, user_data, visit_id, debug, browser_params, manager_params, logger):
                if debug: logger.debug('Found and submitted newsletter form on this page')
                return

            # should we stay on this page?
            if next_link[4] & _FLAG_STAY_ON_PAGE:
                continue

            # go back
            webdriver.back()
            if debug: logger.debug('Pressing the back button')
            wait_until_loaded(webdriver, _PAGE_LOAD_TIME)

            # check other windows (ex. pop-ups)
            windows = webdriver.window_handles
            if len(windows) > 1:
                form_found_in_popup = False
                for window in windows:
                    if window != main_handle:
                        webdriver.switch_to_window(window)
                        wait_until_loaded(webdriver, _PAGE_LOAD_TIME)

                        # find newsletter form
                        if _find_and_fill_form(webdriver, user_data, visit_id, debug, browser_params, manager_params, logger):
                            if debug: logger.debug('Found and submitted newsletter form in a popup on this page')
                            form_found_in_popup = True

                        webdriver.close()

                if form_found_in_popup:
                    return

                webdriver.switch_to_window(main_handle)
                time.sleep(1)

        except:
            pass

    if debug: logger.debug('Failed to find and submit a newsletter form')

def _is_internal_link(href, url, ps1=None):
    """Returns whether the given link is an internal link."""
    if ps1 is None:
        ps1 = domain_utils.get_ps_plus_1(url)
    return domain_utils.get_ps_plus_1(urljoin(url, href)) == ps1

def _whitelisted_links(href, url):
    """Returns whether the given link is whitelisted."""
    return domain_utils.get_ps_plus_1(urljoin(url, href)) in _DOMAIN_EXCEPTIONS

def _find_and_fill_form(webdriver, user_data, visit_id, debug, browser_params, manager_params, logger):
    """Finds and fills a form, and returns True if accomplished."""
    current_url = webdriver.current_url
    current_site_title = webdriver.title.encode('ascii', 'replace')
    main_handle = webdriver.current_window_handle
    in_iframe = False

    if debug: logger.debug('The current URL is %s' % current_url)

    # debug: save before/after screenshots and page source
    debug_file_prefix = str(visit_id) + '_'
    debug_form_pre_initial = debug_file_prefix + 'form_initial_presubmit'
    debug_form_post_initial = debug_file_prefix + 'form_initial_result'
    debug_form_pre_followup = debug_file_prefix + 'form_followup_presubmit'
    debug_form_post_followup = debug_file_prefix + 'form_followup_result'
    debug_page_source_initial = debug_file_prefix + 'src_initial'
    debug_page_source_followup = debug_file_prefix + 'src_followup'

    newsletter_form = None

    # Search for a modal dialog, and for a form in the modal dialog
    # Search for no more than two modal dialogs
    try:
        search_count = 0
        while (search_count < _MAX_POPUP_DISMISS):
            if debug: logger.debug('Round %d of modal dialog search...' % search_count)
            dialog_container = _get_dialog_container(webdriver)
            if dialog_container:
                if debug: logger.debug('Modal dialog found, searching for newsletter form in dialog...')
                newsletter_form = _find_newsletter_form(dialog_container, webdriver, debug, logger)

                if newsletter_form is None:
                    clicked = _dismiss_dialog(webdriver, dialog_container)
                    if debug:
                        if int(clicked) > 0:
                            if debug: logger.debug('No newsletter form in dialog, dismissed it')
                        else:
                            if debug:
                                logger.debug('Made no clicks to dismiss the dialog')
                                webdriver.find_element_by_tag_name('html').send_keys(Keys.ESCAPE)
                                logger.debug('Pressed ESC to dismiss the dialog')
                else:
                    if debug: logger.debug('Found a newsletter form in the dialog')
                    break
            else:
                if debug: logger.debug('No dialog on the page')
                break

            search_count += 1
    except Exception as e:
        logger.error('Error while examining for modal dialogs: %s' % str(e))

    # try to find newsletter forms on landing page after dismissing the dialog
    if newsletter_form is None:
        if debug: logger.debug('Searching the rest of the page for a newsletter form')
        newsletter_form = _find_newsletter_form(None, webdriver, debug, logger)

    # Search for newsletter forms in iframes
    if newsletter_form is None:
        if debug: logger.debug('No newsletter form found on this page, searching for forms in iframes...')

        # search for forms in iframes (if present)
        iframes = webdriver.find_elements_by_tag_name('iframe') + webdriver.find_elements_by_tag_name('frame')
        if debug: logger.debug('Searching in %d iframes' % len(iframes))

        for iframe in iframes:
            try:
                # switch to the iframe
                webdriver.switch_to_frame(iframe)

                # is there a form?
                newsletter_form = _find_newsletter_form(None, webdriver, debug, logger)
                if newsletter_form is not None:
                    if debug:
                        dump_page_source(debug_page_source_initial, webdriver, browser_params, manager_params)
                        logger.debug('Found a newsletter in an iframe on this page')
                    in_iframe = True
                    break  # form found, stay on the iframe

                # switch back
                webdriver.switch_to_default_content()
            except Exception as e:
                if debug: logger.error('Error while analyzing an iframe: %s' % str(e))
                webdriver.switch_to_default_content()

        # still no form?
        if newsletter_form is None:
            if debug: logger.debug('None of the iframes have newsletter forms')
            return False
    elif debug:
        dump_page_source(debug_page_source_initial, webdriver, browser_params, manager_params)

    email = user_data['email']
    user_info = user_data
    _form_fill_and_submit(newsletter_form, user_info, webdriver, True, browser_params, manager_params, debug_form_pre_initial if debug else None)
    logger.info('Submitted form on [%s] with email [%s] on visit_id [%d]', current_url, email, visit_id)
    time.sleep(_FORM_SUBMIT_SLEEP)
    _dismiss_alert(webdriver)

    if debug:
        save_screenshot(debug_form_post_initial, webdriver, browser_params, manager_params)
        logger.debug('The current URL is %s' % webdriver.current_url)
        logger.debug('Filling any follow-up forms on this page...')

    # fill any follow-up forms...
    wait_until_loaded(webdriver, _PAGE_LOAD_TIME)  # wait if we got redirected
    follow_up_form = None

    # first check other windows (ex. pop-ups)
    windows = webdriver.window_handles
    if len(windows) > 1:
        if debug: logger.debug('Found %d windows (e.g., popups)' % len(windows))
        form_found_in_popup = False
        for window in windows:
            if window != main_handle:
                webdriver.switch_to_window(window)

                # find newsletter form
                if follow_up_form is None:
                    follow_up_form = _find_newsletter_form(None, webdriver, debug, logger)
                    if follow_up_form is not None:
                        if debug:
                            dump_page_source(debug_page_source_followup, webdriver, browser_params, manager_params)
                            logger.debug('Found a newsletter form in another window')
                        _form_fill_and_submit(follow_up_form, user_info, webdriver, True, browser_params, manager_params, debug_form_pre_followup if debug else None)

                        logger.info('Submitted form on [%s] with email [%s] on visit_id [%d]', webdriver.current_url, email, visit_id)

                        time.sleep(_FORM_SUBMIT_SLEEP)
                        _dismiss_alert(webdriver)
                        if debug: save_screenshot(debug_form_post_followup, webdriver, browser_params, manager_params)

                webdriver.close()
        webdriver.switch_to_window(main_handle)
        time.sleep(1)

    # else check current page
    if follow_up_form is None:
        if debug: logger.debug('Found no follow-up forms in other windows, checking current page')
        follow_up_form = _find_newsletter_form(None, webdriver, debug, logger)
        if follow_up_form is not None:
            if debug:
                dump_page_source(debug_page_source_followup, webdriver, browser_params, manager_params)
                logger.debug('Found a follow-up form in this page')

            _form_fill_and_submit(follow_up_form, user_info, webdriver, True, browser_params, manager_params, debug_form_pre_followup if debug else None)

            logger.info('Submitted form on [%s] with email [%s] on visit_id [%d]', webdriver.current_url, email, visit_id)

            time.sleep(_FORM_SUBMIT_SLEEP)
            _dismiss_alert(webdriver)
            if debug: save_screenshot(debug_form_post_followup, webdriver, browser_params, manager_params)
        else:
            if debug: logger.debug('No follow-up forms on the current page')

	# switch back
    if in_iframe:
        if debug: logger.debug('We were in an iframe, switching back to the main window')
        webdriver.switch_to_default_content()

    # close other windows (ex. pop-ups)
    windows = webdriver.window_handles
    if len(windows) > 1:
        if debug: logger.debug('Closing %d windows (e.g., popups)' % len(windows))
        for window in windows:
            if window != main_handle:
                webdriver.switch_to_window(window)
                webdriver.close()
        webdriver.switch_to_window(main_handle)
        time.sleep(1)

    return True

def _find_newsletter_form(container, webdriver, debug, logger):
    """Tries to find a form element on the page for newsletter sign-up.
    Returns None if no form was found.
    """
    if container is None:
        container = webdriver

    # find all forms that match
    newsletter_forms = []
    forms = container.find_elements_by_tag_name('form')

    if debug: logger.debug('Found %d forms on this page' % len(forms))
    for form in forms:
        if not form.is_displayed():
            continue

        # find email keywords in the form HTML (preliminary filtering)
        form_html = form.get_attribute('outerHTML').lower()

        match = False
        for s in _KEYWORDS_EMAIL:
            if s in form_html:
                match = True
                if debug: logger.debug('Form matches keywords so match is True')
                break

        if _check_form_blacklist(form):
            match = False
            if debug: logger.debug('Form matches blacklist so match is False')
            break

        if not match:
            continue

        # check if an input field contains an email element
        input_fields = form.find_elements_by_tag_name('input')
        match = False
        for input_field in input_fields:
            if input_field.is_displayed() and _is_email_input(input_field):
                match = True
                if debug: logger.debug('Form contains an email input field')
                break

        if not match:
            if debug: logger.debug('Form does not contain an email input field')
            continue

        # form matched, get some other ranking criteria:
        # - rank modal/pop-up/dialogs higher, since these are likely to be sign-up forms
        z_index = _get_z_index(form, webdriver)
        has_modal_text = 'modal' in form_html or 'dialog' in form_html
        # - rank login dialogs lower, in case better forms exist
        #   (count occurrences of these keywords, since they might just be in a URL)
        login_text_count = -sum([form_html.count(s) for s in ['login', 'log in', 'sign in']])
        # - rank forms with more input elements higher
        input_field_count = len([x for x in input_fields if x.is_displayed()])
        newsletter_forms.append((form, (z_index, int(has_modal_text), login_text_count, input_field_count)))

    if debug: logger.debug('%d are newsletter forms' % len(newsletter_forms))

    # return highest ranked form
    if newsletter_forms:
        newsletter_forms.sort(key=lambda x: x[1], reverse=True)
        if debug: logger.debug('Returning highest ranked form')
        return newsletter_forms[0][0]

    # try to find any container with email input fields and a submit button
    input_fields = container.find_elements_by_tag_name('input')
    if debug:
        logger.debug('Searching for input fields in form-like containers')
        logger.debug('Found %d input fields' % len(input_fields))

    visited_containers = set()
    for input_field in input_fields:
        if not input_field.is_displayed() or not _is_email_input(input_field):
            continue

        if debug:
            logger.debug('Found a visible input email field')
            logger.debug('Checking whether its parent container has a submit button')

        # email input field found, check parents for container with a submit button
        try:
            e = input_field
            for i in xrange(_FORM_CONTAINER_SEARCH_LIMIT):
                e = e.find_element_by_xpath('..')  # get parent
                if e is None or e.id in visited_containers:
                    continue  # already visited

                # is this a container type? (<div> or <span>)
                tag_name = e.tag_name.lower()
                if tag_name == 'div' or tag_name == 'span':
                    # does this contain a submit button?
                    if _has_submit_button(e):
                        if _check_form_blacklist(e):
                            if debug: logger.debug('Parent container matches blacklist, ignoring')
                            raise Exception()

                        if debug: logger.debug('Found a form to submit, returning')
                        return e  # yes, we're done

                visited_containers.add(e.id)
        except:
            pass

    # still no matches?
    return None

def _is_email_input(input_field):
    """Returns whether the given input field is an email input field."""
    type = input_field.get_attribute('type').lower()
    if type == 'email':
        return True
    elif type == 'text':
        if _element_contains_text(input_field, _KEYWORDS_EMAIL):
            return True
    return False

def _has_submit_button(container):
    """Returns whether the given container has a submit button."""
    # check <input> tags
    input_fields = container.find_elements_by_tag_name('input')
    for input_field in input_fields:
        if not input_field.is_displayed():
            continue

        type = input_field.get_attribute('type').lower()
        if type == 'submit' or type == 'button' or type == 'image':
            if _element_contains_text(input_field, _KEYWORDS_SUBMIT):
                return True

    # check <button> tags
    buttons = container.find_elements_by_tag_name('button')
    for button in buttons:
        if not button.is_displayed():
            continue

        type = button.get_attribute('type').lower()
        if type is None or (type != 'reset' and type != 'menu'):
            if _element_contains_text(button, _KEYWORDS_SUBMIT):
                return True

    return False

def _get_z_index(element, webdriver):
    """Tries to find the actual z-index of an element, otherwise returns 0."""
    e = element
    while e is not None:
        try:
            # selenium is usually wrong, don't bother with this
            #z = element.value_of_css_property('z-index')

            # get z-index with javascript
            script = 'return window.document.defaultView.getComputedStyle(arguments[0], null).getPropertyValue("z-index")'
            z = webdriver.execute_script(script, e)
            if z != None and z != 'auto':
                try:
                    return int(z)
                except ValueError:
                    pass

            # try the parent...
            e = e.find_element_by_xpath('..')  # throws exception when parent is the <html> tag
        except:
            break
    return 0

def _dismiss_alert(webdriver):
    """Dismisses an alert, if present."""
    try:
        WebDriverWait(webdriver, 0.5).until(expected_conditions.alert_is_present())
        alert = webdriver.switch_to_alert()
        alert.dismiss()
    except TimeoutException:
        pass

def _form_fill_and_submit(form, user_info, webdriver, clear, browser_params, manager_params, screenshot_filename):
    """Fills out a form and submits it, then waits for the response."""
    # try to fill all input fields in the form...
    input_fields = form.find_elements_by_tag_name('input')
    submit_button = None
    text_field = None
    for input_field in input_fields:
        type = input_field.get_attribute('type').lower()

        # execptions for checkbox and radio elements since these can be invisble but still visible
        # because of superimposed elements
        if not input_field.is_displayed() and type != 'checkbox' and type != 'radio':
            continue

        if type == 'email':
            # using html5 "email" type, this is probably an email field
            _type_in_field(input_field, user_info['email'], clear)
            text_field = input_field
        elif type == 'text':
            # try to decipher this based on field attributes
            if _element_contains_text(input_field, 'company'):
                _type_in_field(input_field, user_info['company'], clear)
            elif _element_contains_text(input_field, 'title'):
                _type_in_field(input_field, user_info['title'], clear)
            elif _element_contains_text(input_field, 'name'):
                if _element_contains_text(input_field, ['first', 'forename', 'fname']):
                    _type_in_field(input_field, user_info['first_name'], clear)
                elif _element_contains_text(input_field, ['last', 'surname', 'lname']):
                    _type_in_field(input_field, user_info['last_name'], clear)
                elif _element_contains_text(input_field, ['user', 'account']):
                    _type_in_field(input_field, user_info['user'], clear)
                else:
                    _type_in_field(input_field, user_info['full_name'], clear)
            elif _element_contains_text(input_field, ['zip', 'postal']):
                _type_in_field(input_field, user_info['zip'], clear)
            elif _element_contains_text(input_field, 'city'):
                _type_in_field(input_field, user_info['city'], clear)
            elif _element_contains_text(input_field, 'state'):
                _type_in_field(input_field, user_info['state'], clear)
            elif _element_contains_text(input_field, _KEYWORDS_EMAIL):
                _type_in_field(input_field, user_info['email'], clear)
            elif _element_contains_text(input_field, ['street', 'address']):
                if _element_contains_text(input_field, ['2', 'number']):
                    _type_in_field(input_field, user_info['street2'], clear)
                elif _element_contains_text(input_field, '3'):
                    pass
                else:
                    _type_in_field(input_field, user_info['street1'], clear)
            elif _element_contains_text(input_field, ['phone', 'tel', 'mobile']):
                _type_in_field(input_field, user_info['tel'], clear)
            elif _element_contains_text(input_field, 'search'):
                pass
            else:
                # skip if visibly marked "optional"
                placeholder = input_field.get_attribute('placeholder')
                if placeholder is not None and 'optional' in placeholder.lower():
                    pass

                # default: assume email
                else:
                    _type_in_field(input_field, user_info['email'], clear)
            text_field = input_field
        elif type == 'number':
            if _element_contains_text(input_field, ['phone', 'tel', 'mobile']):
                _type_in_field(input_field, user_info['tel'], clear)
            elif _element_contains_text(input_field, ['zip', 'postal']):
                _type_in_field(input_field, user_info['zip'], clear)
            else:
                _type_in_field(input_field, user_info['zip'], clear)
        elif type == 'checkbox' or type == 'radio':
            # check anything/everything
            if input_field.is_displayed():
                if not input_field.is_selected():
                    try:
                        input_field.click()
                    except:
                        webdriver.execute_script('return arguments[0].click()', input_field)
            else:
                try:
                    checked = webdriver.execute_script('return arguments[0].checked', input_field)
                    if checked == False:
                        webdriver.execute_script('return arguments[0].click()', input_field)
                except:
                    pass
        elif type == 'password':
            _type_in_field(input_field, user_info['password'], clear)
        elif type == 'tel':
            # exceptions
            if _element_contains_text(input_field, ['zip', 'postal',]):
                _type_in_field(input_field, user_info['zip'], clear)
            else:
                _type_in_field(input_field, user_info['tel'], clear)
        elif type == 'submit' or type == 'button' or type == 'image':
            if _element_contains_text(input_field, _KEYWORDS_SUBMIT):
                submit_button = input_field
        elif type == 'reset' or type == 'hidden' or type == 'search':
            # common irrelevant input types
            pass
        else:
            # default: assume email
            _type_in_field(input_field, user_info['email'], clear)

    # find 'button' tags (if necessary)
    if submit_button is None:
        buttons = form.find_elements_by_tag_name('button')
        for button in buttons:
            if not button.is_displayed():
                continue

            # filter out non-submit button types
            type = button.get_attribute('type').lower()
            if type is not None and (type == 'reset' or type == 'menu'):
                continue

            # pick first matching button
            if _element_contains_text(button, _KEYWORDS_SUBMIT):
                submit_button = button
                break

    # Check for 'div' tags that are buttons
    if submit_button is None:
        div_buttons = form.find_elements_by_xpath('.//div[@role="button"]')
        for dbutton in div_buttons:
            if not dbutton.is_displayed():
                continue

            role = dbutton.get_attribute('role').lower()
            if role is not None and role == 'button':
                if _element_contains_text(dbutton, _KEYWORDS_SUBMIT):
                    submit_button = dbutton
                    break

    # fill in 'select' fields
    select_fields = form.find_elements_by_tag_name('select')
    for select_field in select_fields:
        if not select_field.is_displayed():
            continue

        # select an appropriate element if possible,
        # otherwise second element (to skip blank fields),
        # falling back on the first
        select = Select(select_field)
        select_options = select.options
        selected_index = None
        for i, opt in enumerate(select_options):
            opt_text = opt.text.strip().lower()
            if opt_text in _KEYWORDS_SELECT:
                selected_index = i
                break
        if selected_index is None:
            selected_index = min(1, len(select_options) - 1)
        select.select_by_index(selected_index)

    # debug: save screenshot
    if screenshot_filename: save_screenshot(screenshot_filename, webdriver, browser_params, manager_params)

    # submit the form
    if submit_button is not None:
        try:
            submit_button.click()  # trigger javascript events if possible
            return
        except:
            pass
    if text_field is not None:
        try:
            text_field.send_keys(Keys.RETURN)  # press enter
        except:
            pass
    try:
        if form.tag_name.lower() == 'form':
            form.submit()  # submit() form
    except:
        pass

def _element_contains_text(element, text):
    """Scans various element attributes for the given text."""
    attributes = ['name', 'class', 'id', 'placeholder', 'value', 'for', 'title', 'innerHTML', 'aria-label']
    text_list = text if type(text) is list else [text]
    for s in text_list:
        for attr in attributes:
            e = element.get_attribute(attr)
            if e is not None and s in e.lower():
                return True
    return False

def _type_in_field(input_field, text, clear):
    """Types text into an input field."""
    if clear:
        input_field.send_keys(Keys.CONTROL, 'a')
        input_field.send_keys(Keys.BACKSPACE)
    input_field.send_keys(text)

def _get_dialog_container(webdriver):
    """If there exists a modal popup, return its container."""
    try:
        script = open('common.js').read() + ';' + open('dismiss_dialogs.js').read() + ';' + 'return getPopupContainer();'
        return webdriver.execute_script(script)
    except Exception as e:
        raise Exception('Script to extract popup dialog container crashed: %s' % str(e))

def _dismiss_dialog(webdriver, container):
    """Dismisses the popup with parent container."""
    try:
        script = open('common.js').read() + ';' + open('dismiss_dialogs.js').read() + ';' + 'return closeDialog(arguments[0]);'
        return webdriver.execute_script(script, container)
    except Exception as e:
        raise Exception('Script to dismiss popup dialog crashed: %s' % str(e))

def _check_form_blacklist(form):
    """Checks whether the form should be blacklisted and ignored."""
    form_text = []
    for line in form.text.lower().split('\n'):
        form_text.append(re.sub('[^A-Za-z ]', '', line).strip())

    for s in _KEYWORDS_EMAIL_BLACKLIST:
        if s in form_text:
            return True

    return False
