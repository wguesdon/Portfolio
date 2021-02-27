Create a Machine Learning Ubuntu Server on Google Cloud
================

# Introduction

The goal is to set up a working environment with a jupyter notebook and compatible with the Python scripts from Approaching (almost) any machine learning problem by abhishek thakur.pdf alias.

This tutorial is adaptated from: https://medium.com/analytics-vidhya/setting-up-jupyter-lab-instance-on-google-cloud-platform-3a7acaa732b7

# 1. Create an instance

``` bash

# Create a new users with sudo privilege
sudo adduser will
sudo usermod -aG sudo will

# Disconnet from the root account and reconect to the new user
su - will

# Parameter the firewall
sudo su
ufw app list
ufw allow OpenSSH
ufw enable
ufw status
exit
```

# 2. Set up a static IP adress

``` bash

su - will

sudo apt update
sudo apt upgrade #Provide Y/yes as input when prompted.
sudo apt install python3-pip
sudo pip3 install jupyterlab

sudo su
ufw allow 8888
ufw enable
exit

sudo jupyter serverextension enable --py jupyterlab --sys-prefix

# see https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52

jupyter notebook --generate-config

vim /home/will/.jupyter/jupyter_notebook_config.py

c = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

jupyter-notebook --no-browser --port=8888
```

http://35.226.32.49:8888

Enter token: 43708e14a51b1271c55a5ef388cd43ab517d565be330af56

5261b0cefe65a5f4c0f0267f0cbcbe0ebb67f6eb7b9c45b7

# 3. Install and confifure Ananconda

see https://medium.com/google-cloud/set-up-anaconda-under-google-cloud-vm-on-windows-f71fc1064bd7


``` bash
su - will

sudo apt-get update
sudo apt-get install bzip2 libxml2-dev

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source .bashrc
conda
conda create --name ml
conda activate ml
conda install scikit-learn pandas jupyter ipython
```

The command `!pip install kaggle` now work

``` python
# Copy kaggle.json file content to the kaggle.json file
! echo '{"username":"wguesdon","key":"YOUR TOKEN"}' > kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# List files
api.competitions_data_list_files('titanic')

# Download all files

# Signature: competition_download_files(competition, path=None, force=False, quiet=True)
api.competition_download_files('titanic')

# Download single file for a competition
# Signature: competition_download_file(competition, file_name, path=None, force=False, quiet=False)
api.competition_download_file('titanic','gender_submission.csv')
```


# 4. Install Tmux to run operation on server after ssh disconection

``` bash
sudo apt-get update
sudo apt-get install tmux
```

# 5. Install git

``` bash
sudo apt install git
```

# 6. Configure the ftp server

see https://zatackcoder.com/how-to-setup-ftp-server-vsftpd-on-google-cloud-compute-engine/

``` bash
sudo apt install vsftpd
sudo ufw allow ftp
sudo nano /etc/vsftpd.conf
```

```
write_enable=YES
chroot_local_user=YES
local_umask=022

#define your ftp directory
local_root=/home/ftpuser
#to allow upload in root directory
allow_writeable_chroot=YES
# passive mode min port number
pasv_min_port=40000
# passive mode max port number
pasv_max_port=50000
```

``` bash
sudo systemctl restart vsftpd
sudo systemctl status vsftpd

sudo adduser ftpuser
sudo chown ftpuser /home/ftpuser
sudo chmod 755 /home/ftpuser


sudo su
ufw allow 20:21/tcp
ufw allow 990/tcp
ufw allow 40000:50000/tcp
ufw enable
exit
```

# To Do 

* Change notebook config to allow connection from my IP only
* Donwload Kaggle dataset in python: https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python





