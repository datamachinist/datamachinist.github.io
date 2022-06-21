---
permalink: /reinforcement_learning/
title: "Reinforcement Learning"
excerpt: "A series of tutorials on reinforcement learning, mainly for robotics applications."
last_modified_at: 2022-06-20T11:59:26-04:00
toc: true
---


In this tutorial series, we will learn about reinforcement learning and its application in robotics.

<!-- List all the posts with the category 'reinforcement learning' -->

<ul>
  {% for post in site.categories['reinforcement learning'] %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>



