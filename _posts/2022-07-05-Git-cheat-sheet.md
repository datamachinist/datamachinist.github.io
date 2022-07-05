---
title:  "Git cheat sheet"
excerpt: "Some useful commands and shortcuts for the Git version-control system"
toc: true
category:
  - cheat sheet
---


![git_logo]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/git_logo.png)


## Some definitions

**Git** is a version-control tool for tracking changes in source code during software development.

A **commit** records a snapshot of your changes and associate it with a message and a commit id.

A **branch** is a separate version of the main repository that allows the developper to test a new feature for example without affecting the main repository.

**Staging** means adding a file to the Git index before commiting it.

**Remote repositories** are versions of your source code that are hosted on the Internet, mainly to share the code or collaborate with others. Examples of Git-based source code repository hosting service include [Github](https://github.com/), [Bitbucket](https://bitbucket.org/product/) or [Gitlab](https://about.gitlab.com/). 

You can download a printable Git command line cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/git-cheat-sheet.pdf) ([Source](https://www.jrebel.com/blog/git-cheat-sheet)).


![git workflow]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/git_flow.png)
<sub><sup>*[Source](https://www.jrebel.com/blog/git-cheat-sheet)*</sup></sub>

## Create a repository

- create a new local repository

```bash
git init my_project_name
```

- Download from an existing repository

```bash
git clone my_url
```


## Observe the repository

- list files not yet commited

```bash
git status
```

- Show full change history

```bash
git log
```

- Show change history for file/directory including diffs

```bash
git log -p [file/directory]
```

- Show the changes to files not yet staged

```bash
git diff
```

- Show the changes between two commit ids

```bash
git diff commit1 commit2
```


## Working with branches


- List all local branches

```bash
git branch
```

- Create a new branch

```bash
git branch new_branch
```

- Switch to a branch and update working directory

```bash
git checkout my_branch
```

- Delete a branch

```bash
git branch -d my_branch
```

- Merge branch_a into branch_b

```bash
git checkout branch_b
git merge branch_a
```

## Make a change

- Stage all changed files in current directory

```bash
git add .
```

- Commit all your tracked files to versioned history

```bash
git commit -am "commit message"
```

- Unstages file, keeping the file changes

```bash
git reset [file]
```

- Revert everything to the last commit

```bash
git reset --hard
```

## Synchronize with remote repository

- Get the latest changes from origin (no merge)

```bash
git fetch
```

- Fetch the latest changes from origin and merge

```bash
git pull
```

- Push local changes to the origin

```bash
git push
```
