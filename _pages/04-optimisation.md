---
permalink: /optimisation/
title: "Optimisation"
excerpt: "A blog series is about optimisation, including evolutionary optimisation and Bayesian optimisation."
last_modified_at: 2022-07-12T11:59:26-04:00
toc: true
---

Optimisation is the general process of selecting the best element with regards of some defined criteria.

In machine learning, the problems to solve are generally described as a function that maps input data to output data. The goal is to approximate this function as best as possible. An optimisation algorithm is used to determine the function parameters that minimise the error when mapping inputs to outputs.


In this blog series, we will dive into different optimisation techniques, such as evolutionary optimisation and Bayesian optimisation.


<!-- Create array of posts with category 'cheat sheet' and sort them alphabetically -->

{% assign sortedPosts = site.categories['optimisation'] | sort: 'title' %}

<!-- Create a list of post using the array defined earlier -->

<ul>
  {% for post in sortedPosts %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>
