# Sever setup and email archival
This folder contains the necessary steps and files to set up a server and archive emails. The following instructions below have been tested on Ubuntu 18.04.3 LTS.

## Postfix setup

The server uses a Postfix mail agent to receive emails. Once you've purchased your domain name, first point its mail exchanger record (MX record) to this server's IP address.

Then, follow [these](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-postfix-on-ubuntu-16-04) steps to install Postfix on your server.

## Postfix hook

Once the Postfix server is installed, add a hook in Postfix's `master.cf` configuration file to execute the `script.sh` shell script.

    myhook unix - n n - - pipe
    flags=F user=amathur argv=/mnt/volume_nyc3_01/script.sh ${sender} ${size} ${recipient}

Replace `user=amathur` with the name of your user. The `master.cf` file available in this repository describes how to incorporate the hook in the Postfix configuraiton file.

## Python setup
Create a virtual environment with the packages listed in the `requirements.txt` file.

    pip install -r requirements.txt
    
Edit the `script.sh` script to invoke the virtual environment.

## Archiving emails
The `script.py` is responsible for archiving emails. The file runs through the `script.sh` script when the Postfix hook is activated on an incoming email.

The `script.py` file contains further documentation on the various folder structure that result from the archival of a single email. Replace the `BASE_DIR` variable in the `script.py` file with the destination directory where you would like to archive the emails. Also replace the `EMAIL_DOMAIN` variable in the `utils.py` file with the domain name of your server.

Note that the file also expects [HTTrack](https://www.httrack.com/page/2/) and [Google Chrome](https://www.google.com/chrome/) to be installed on the server.

We used HTTrack v3.49-2 and Google Chrome v78.0.3904.97 for our project.
