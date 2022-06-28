---
permalink: /cheat_sheets/
title: "Cheat sheets"
excerpt: "A number of handy cheat sheets to remember commands and syntax for various programming tools."
last_modified_at: 2022-06-28T11:59:26-04:00
toc: true
---

Remembering commands and syntax for the various tools used in programming can be a daunting task. Here are some handy cheat sheets that can help you with this.


<!-- Create array of posts with category 'cheat sheet' and sort them alphabetically -->

{% assign sortedPosts = site.categories['cheat sheet'] | sort: 'title' %}

<!-- Create a list of post using the array defined earlier -->

<ul>
  {% for post in sortedPosts %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

