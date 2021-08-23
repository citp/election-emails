# Email signup bot
This folder contains the code of our automated email signup bot. This repository contains [code](https://github.com/citp/email_tracking) from [Englehardt et al](https://petsymposium.org/2018/files/papers/issue1/paper42-2018-1-source.pdf) and is built upon [OpenWPM](https://github.com/mozilla/openwpm).

Start by running the `install.sh` shell script to install OpenWPM. Once installed, run the `crawl_mailinglist_signup.py` file to perform a signup on the input file specified by the `site_list` variable.

The input file should be in a CSV format and contain at least two columns: `final_website`, which indicates the website that must be visited, and `query_data` which contains a Python dictionary in string format with the following fields:

	   { 
	   	   'email': email,
	   	   'first_name': 'Bob',
	   	   'last_name': 'Smith',
	   	   'full_name': 'Bob Smith',
	   	   'user': 'bobsmith123',
	   	   'password': 'p4S$w0rd123',
	   	   'tel': '2125551234',
	   	   'company': 'Smith & Co.',
	   	   'title': 'Mr.','zip': '12345',
	   	   'street1': '101 Main St.',
	   	   'street2': 'Apt. 101',
	   	   'city': 'Schenectady',
	   	   'state': 'New York'
	  }

The crawler visits the websites in sequential order and attempts to fill a mailing list form with the information provided by the `query_data` variable.

The [custom_commands.py](https://github.com/citp/election-emails/blob/main/email-signup/automation/Commands/custom_commands.py) file drives the entire crawler. Read the file by starting at the `fill_forms` method and following the sequence of commands. 
