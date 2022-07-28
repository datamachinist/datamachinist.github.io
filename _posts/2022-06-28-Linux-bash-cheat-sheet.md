---
title:  "Linux bash cheat sheet"
excerpt: "Some useful commands and shortcuts for the Linux Bash command line interpreter"
toc: true
category:
  - cheat sheet
---

![linux-logo]({{ site.url }}{{ site.baseurl }}/assets/images/linux_logo.png)


## Some definitions

- **Linux** is an open-source operating system. It comes under different "flavours" or **distributions** such as Ubuntu, Arch Linux, Fedora, etc... 
- **Bash** - for Bourne Again SHell - is Linux' command line interpreter (or **shell**) that provides a command line interface inside a terminal emulator.

## Cheat sheet

You can download a printable Linux command line cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/Linux-bash-cheat-sheet.pdf) from [Cheatography](https://cheatography.com/davechild/cheat-sheets/linux-command-line/).

## 1. Find files


- Find files match a pattern 
 
```bash 
find -name "query"
```

    
- Find files match a pattern, case insensitive 
 

```bash
find -iname "query" 
```

    
- Find all files that don't contain the pattern 
 
```bash
find -not -name "query_to_avoid"
```


- Find all files in current directory 
 
```bash
find . -type f
```   

    
- Find all directories in current directory 
 
```bash
find . -type d
```   

- Find all symbolic links in current directory 
 
```bash
find . -type l 
```

    
- Find all items ending in ".conf" in current directory 
 
```bash
find . -name *.conf
```

    
- Find all files in the current directory starting with "pro"
 
```bash
find . -name pro* 
```

- Find all items containing "yolo" in current directory 
 
```bash  
find . -name *yolo*
```   


- Find pattern 'tensorflow' in python files in the current directory 
 
```bash
find . -name *.py | grep -rnw tensorflow 
```   

Option explained:
- -r or -R : recursive
- -n : show line number
- -w : match the whole word
- -l (lower-case L) : give the file name of matching files (optional). 
 

    
- Find pattern in file recursively 
 

```bash
grep -rnw '/path/to/somewhere/' -e 'pattern'
```   

```bash
grep -r 'pattern'
```   

    
- Move file with extension recursively to a directory 
 
```bash
find /thisdir -type f -name '*.ogg' -exec mv -i {} /somedir \;
```   


## 2. Backup installed packages


- Write list of packages 
 
```bash
sudo dpkg --get-selections "*" > my_packages.txt
```   

    
- Install list of packages from txt file 
 

```bash
sudo dpkg --set-selections < my_packages.txt
sudo apt-get -u dselect-upgrade
```   

    
-  
## 3. Command line history

    
- Search command line history (shortcut)  

```bash
CTRL + r
```   

    
- View command line history 
 

```bash
history
```   

    
## 4. Copy to/from SSH server
    
- When logged in locally, copy from local to remote 
 

```bash
scp /file/to/send username@remote:/where/to/put
# OR
rsync -avzh /file/to/send username@remote:/where/to/put
```  
where the rsync options are: 

- -a : archive mode, which preserves permissions, ownership, and modification times
- -v : enables verbose 
- -z : enables compression during transfer
- -h : outputs numbers in human-readable format 
    
- When logged in locally, from remote to local 
 

```bash
scp username@remote:/file/to/send /where/to/put
# OR
rsync -avzh username@remote:/file/to/send /where/to/put
```   


- When logged in remotely, copy from remote to local 
 

```bash  
sftp username@remote
get -r "remote/dir/path" "local/dir/path"
```   

    
- When logged in remotely, copy from local to remote 
 

```bash
sftp username@remote
put -r "local/dir/path" "remote/dir/path"
```   

(note: in Ubuntu, you may encounter this error: "Couldn't canonicalise: No such file or directory". Solution: create a directory on the remote first using mkdir) 
 

    
- Launch a job on an SSH server and be able to log off without killing the job 
 

```bash  
nohup python my_script.py &gt;&amp; log.run &amp;
```   

    

## 5. File ownership / make executable

    
- Change directory ownership 
 

```bash
chown -R user:group /directory/of/interest
```   


- Make script executable 
 

```bash  
chmod +x "file name"
```   

    

## 6. System specs

    
- Disk usage of the entire filesystem 
 

```bash  
df -h --total
df -h
```   

    
- Disk usage of current directory 
 

```bash  
du -sh .
```   

    
- Show disk usage of the 40 biggest subfolders in the current directory (ranked) 
 

```bash  
du -hsx * | sort -rh | head -n 40
```   

    
- Show disk usage of all the subfolders in the current directory one by one 
 

```bash  
du -hsx *
```   

    
- Show disk usage of the biggest subfolders in the current directory (ranked), with hidden directories 
 

```bash  
du -sch .[!.]* * | sort -rh
```   

    
- In Ubuntu, this nice tool can explore the disk usage of your entire system quickly 
 

```bash
sudo apt-get install ncdu
ncdu
```   

    
- View system specs (and export them to a html file) 
 

```bash  
sudo lshw
sudo lshw -html > specs.html
```   

    
- Check Linux Kernel 
 

```bash  
uname -a
```   

    
- Check Ubuntu version 
 

```bash  
lsb_release -a
```   

    
- Print total, free and used RAM memory 
 

```bash  
free -h
```   

    
- List CPU and processor info 
 

```bash  
lscpu
```   

    
- Print generic hardware information 
 

```bash  
hwinfo
```   

    
- Print PCI buses, including graphics card, network adapter 
 

```bash  
lspci
```   

    
- List block devices (storage and partitions) 
 

```bash  
lsblk
```   

    
- List devices connected via USB 
 

```bash  
cd /dev
ls
```   

    
- View running processes 
 

```bash  
top  
htop
```   

    
- Print the process ID of a running task 
 

```bash  
pidof firefox
```   

    
## 7. Kill a running job


```bash  
kill 'process ID'
```   

    
Note, you can find process ID using the 'top' command or using pidof.
 

    

## 8. Use SLURM (job scheduler for computing clusters)

    
- Submit job 
 

```bash  
sbatch job.sh
```   

    
- View queue 
 

```bash  
squeue -u "user name"  
squeue
```   


## 9. Resolve merging conflicts with Git

```bash  
git mergetool --tool=emerge
```   

It will open 3 windows: version a on top left, version b on top right and final version at the bottom.
- press n for next change
- press a or b to choose which version I want to keep
- press q to quit and save 
 

## 10. Manipulate PDF documents with pdftk


    
- Cut pdf pages 
 

```bash
pdftk full-pdf.pdf cat 12-15 output outfile_p12-15.pdf
```   

    
- Merge pdf pages 
 
```bash
pdftk file1.pdf file2.pdf file3.pdf cat output newfile.pdf
```   

## 11. Make Gifs and videos from images using ffmpeg
 

- Create video from images

```bash
ffmpeg -i input_0%d.png -vcodec libx264 ouput.mp4
ffmpeg -i input_%d.png -filter:v "setpts=10.0*PTS" -vcodec libx264 ouput.mp4
ffmpeg -start_number 1 -i input%d.png -filter:v "setpts=2*PTS" -filter:v "crop=490:360:15:15" -r 48 -c:v libx264 -crf 0 -profile:v high444 -preset slow -b:v 15M ouput.mp4
```   


- Create gif from video

```bash
ffmpeg -t 30 -i input.mp4 -vf "fps=10,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
ffmpeg -t 30 -i input.mp4 -vf "fps=24,scale=300:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
```

Create gif from images using Convert (from ImageMagik)

```bash
convert -resize 50% -delay 20 -loop 0 input_{0..99}.jpg output.gif
convert -resize 50% -delay 20 -loop 0 *.jpg output.gif
convert -resize 50% -delay 20 -loop 0 `ls -v` output.gif
```   

Create gif from images using Gimp (GUI, potentially more robust than Convert)
1. Select File 
Menu > Open as Layers > Select all images you want to be in the GIF > Open

2. Select Filters from main 
Menu > Animation > Click Optimize for GIF

3. Save GIF
Select File > click Export as > Select File Type as gif > Select ‘As Animation’ > Select ‘Loop Forever'


## 12. Useful shortcuts

    
- Open a terminal new widow 
 
```bash
CTRL + ALT +T
```   

    
- Open a new terminal tab 
 
```bash
CTRL + SHIFT + T
```   

    
- Close the current terminal tab 
 
```bash
CTRL + SHIFT + W
```   

    
- Kill the current running process 
 

```bash
CTRL + C
```   

    
- Exit Python interpreter / log out from SSH 
 

```bash
CTRL + D 
```   

    
- Copy-paste in terminal with a mouse 
1. select text
2. move cursor where you want to paste
3. paste with a middle mouse click
    


- Insert the euro sign (€)

```bash
CTRL+SHIFT+u
20ac
"enter"
```   

 
- Put task in background 
 
```bash
CTRL + C 
```   

    
- Bring it back to the foreground 
 

```bash
fg
```   

- Nicely formated ls command

```bash
ls -lrt
```   

    
- Download from website

```bash
wget http://www.website-name.com
curl http://www.website-name.com
```   

- Execute the previous command with sudo 
```bash
sudo !!
``` 

- Login as root
```bash
sudo -i
```

- Logout as root
```bash
CTRL + D
```
