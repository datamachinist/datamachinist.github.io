---
permalink: /reinforcement_learning/
title: "Reinforcement Learning"
excerpt: "A series of tutorials on reinforcement learning, mainly for robotics applications."
last_modified_at: 2022-06-20T11:59:26-04:00
toc: true
---


In this tutorial series, we will learn about reinforcement learning and its application in robotics.

<!-- Create array of posts with category 'Reinforcement Learning' and sort them alphabetically -->

{% assign sortedPosts = site.categories['reinforcement learning'] | sort: 'title' %}

<!-- Create a list of post using the array defined earlier -->

<ul>
  {% for post in sortedPosts %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

