Set Up an Ubuntu Server for Bioinformatic Analysis
================

## Abstract

Next Generation Sequencing and B cells receptors clonal analysis are
ressources intensive and I run these on a remote server. Digital Ocean
is a good choice for me since it allow users to generate an image of
your server. If yo have a ponctual need to run ana anlysis you can
restore your server aloowing you to pay only for the days you need to
run your analysis.

## 1\. Create a server on Digital Ocean

  - Select the Ubuntu 18.04.03 Droplet.
  - Select the amount of CPUs, RAM and disk space needed. 16 GB / 6 CPUs
    / 320 GB is often a good starting point for me.

## 2\. Parameter Users and permissions

Create and paramters news users with or without sudo users.

``` bash
# Connect to your server using SSH inside the mac terminal, Linux terminal or Windows Ubuntu terminal.
ssh root@206.189.19.132 # Connect to the root account

# Create a new users with sudo privilege
sudo adduser will
usermod -aG sudo will

# Disconnet from the root account and reconect to the new user
su - will

# Update the sytem
sudo apt-get update
sudo apt-get upgrade
sudo reboot

# Parameter the firewall
sudo su
ufw app list
ufw allow OpenSSH
ufw enable
ufw status
```

## 3\. Install R

``` bash
# Install R
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9  
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' 
sudo apt update
sudo apt install r-base
```

## 4\. Install the Rstudio Server

``` bash
sudo apt-get install libapparmor1 gdebi-core
wget https://download2.rstudio.org/rstudio-server-0.99.491-amd64.deb
sudo gdebi rstudio-server-0.99.491-amd64.deb
sudo rstudio-server start
sudo rstudio-server verify-installation
```

## 5\. Install the ftp server

``` bash
sudo apt install vsftpd
sudo ufw allow ftp
```

## 6\. Install Git

``` bash
sudo apt update
sudo apt install git
git --version

# Clone the first repository
git clone git@github.com:wguesdon/Got_10X_CD19.git
```

## 7\. Install Docker

``` bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker
# exit with control + c
```

## 8\. Install Tmux to run operation on server after ssh disconection

``` bash
sudo apt-get update
sudo apt-get install tmux
```

## 9\. Install Jupyter Notebook

``` bash
sudo apt update
sudo apt install python3-pip python3-dev
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
mkdir ~/python_project
cd ~/python_project
virtualenv python_project_env
source python_project_env/bin/activate
pip install jupyter
```

Acess to the jupyter notebook via ssh tuneling.

``` bash
ssh -L 8888:localhost:8888 will@206.189.19.132
cd python_project/
source python_project_env/bin/activate
jupyter notebook
```

## 10\. Backup the server

In the Digital Ocean perform a liver snapshot of the server. It
recommended to turn off the Droplet before perforling a backup but I
experienced connection issues with this method and live snapshot have
worked better for me so far. Digital Ocean offer options to resize the
droplet depending allowing to set up a server from your server image
with different CPUs and RAM configurations.

## References

Digital Ocean provide a large collection of tutorials to set up your
server. Below is the list of the tutorial that were used to set up this
server.

1.  <https://www.digitalocean.com/docs/droplets/how-to/create/>
2.  <https://www.howtogeek.com/336775/how-to-enable-and-use-windows-10s-built-in-ssh-commands/>
3.  <https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-18-04>
4.  <https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-18-04>
5.  <https://www.digitalocean.com/community/tutorials/how-to-set-up-rstudio-on-an-ubuntu-cloud-server>
6.  <https://www.digitalocean.com/community/tutorials/how-to-install-git-on-ubuntu-18-04>
7.  <https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04>
8.  <https://machiine.com/2013/pulling-a-git-repo-from-github-to-your-ubuntu-server/>
