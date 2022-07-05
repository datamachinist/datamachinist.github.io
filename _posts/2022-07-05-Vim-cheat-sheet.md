---
title:  "Vim cheat sheet"
excerpt: "Some useful commands and shortcuts for the Vim text editor"
toc: true
category:
  - cheat sheet
---


![vim_logo]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/vim_logo.png)

## Definition

**Vim** is a free and open-source text editor program, popular on UNIX systems.


## Cheat sheet

You can download a printable Vim command line cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/vim-cheat-sheet.pdf) ([Source](https://cheatography.com/zoltan/cheat-sheets/basic-vim/)).


| Command                | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| ESC                    | Command mode                                                        |
| i                      | insert mode (write)                                                 |
| v                      | visual mode (search in document)                                    |
| :shell                 | open command line prompt                                            |
| :e filename            | open a new file                                                     |
| :w                     | save changes                                                        |
| :q                     | exit vim                                                            |
| :q!                    | exit vim without changing changes                                   |
| :wq                    | save changes and exit vim                                           |
| :u                     | undo last action                                                    |
| CTRL + r               | redo                                                                |
| o                      | open new line BELOW the cursor + insert mode                        |
| O                      | open new line ABOVE the cursor + insert mode                        |
| A                      | go to the end of the line + insert mode                             |
| yy                     | yank the current line                                               |
| y                      | yank the highlighted text                                           |
| dd                     | delete the current line                                             |
| D                      | delete to the end of the line                                       |
| d                      | delete the highlighted text                                         |
| dw                     | delete word                                                         |
| p                      | paste text after cursor position, paste line below the current line |
| x                      | delete current character                                            |
| :vs                    | split the current window vertically                                 |
| :split file            | opens file in second window                                         |
| CTRL + l               | move to the right window                                            |
| CTRL + h               | move to the left window                                             |
| :q                     | close window                                                        |
| h                      | move LEFT                                                           |
| j                      | move DOWN                                                           |
| k                      | move UP                                                             |
| l                      | move RIGHT                                                          |
| gg                     | go to the beginning of the document                                 |
| G                      | go to the end of the document                                       |
| w                      | move to the NEXT word                                               |
| :n [line_nb]           | jump to line number [line_nb]                                       |
| qf\<commands recorded\>q | record macro and associate it to the character "f"                  |
| @f                     | execute macro associated to the character "f"                       |
| :%s/foo/bar/g          | find each occurrence of 'foo' and replace it with 'bar'             |
| :%s/foo/bar/gc         | change each 'foo' to 'bar', but ask for confirmation first          |